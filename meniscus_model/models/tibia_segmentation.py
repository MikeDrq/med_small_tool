"""Coronal U2Net adapter used by Module 4 tibia-width measurement."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch.nn.functional as F


def _load_model_definition(path: str | Path) -> ModuleType:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"U2Net model definition not found: {path}")
    spec = importlib.util.spec_from_file_location("module4_u2net_definition", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import U2Net model definition: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CoronalTibiaU2Net:
    """Run the supplied three-class coronal U2Net and return a tibia mask.

    The segmentation repository's multiclass convention is background=0,
    femur=1, and tibia=2. Input MRI slices are already scaled to [0, 1] by the
    meniscus DICOM reader, matching the segmentation code's ToTensor path.
    """

    source = "coronal_u2net_class2"

    def __init__(
        self,
        model: torch.nn.Module,
        device: str | torch.device,
        input_size: int = 320,
        tibia_class: int = 2,
        batch_size: int = 4,
        amp: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.input_size = int(input_size)
        self.tibia_class = int(tibia_class)
        self.batch_size = int(batch_size)
        self.amp = bool(amp) and self.device.type == "cuda"

    @classmethod
    def from_config(cls, config: dict[str, Any], device: str | torch.device) -> "CoronalTibiaU2Net":
        definition = _load_model_definition(config["model_definition"])
        factory_name = str(config.get("factory", "u2net_full"))
        factory = getattr(definition, factory_name)
        model = factory(out_ch=int(config.get("num_classes", 3)))

        checkpoint_path = Path(config["checkpoint"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"U2Net checkpoint not found: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = payload.get("model", payload) if isinstance(payload, dict) else payload
        if not isinstance(state, dict):
            raise TypeError(f"U2Net checkpoint has no state dictionary: {checkpoint_path}")
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        return cls(
            model=model,
            device=device,
            input_size=int(config.get("input_size", 320)),
            tibia_class=int(config.get("tibia_class", 2)),
            batch_size=int(config.get("batch_size", 4)),
            amp=bool(config.get("amp", True)),
        )

    @torch.inference_mode()
    def predict_mask(self, volume: torch.Tensor) -> torch.Tensor:
        """Return a CPU boolean tibia mask with shape ``[T,H,W]``."""

        frames = torch.as_tensor(volume, dtype=torch.float32)
        if frames.ndim == 4:
            if frames.shape[0] != 1:
                raise ValueError(f"Expected [1,T,H,W] volume, got {tuple(frames.shape)}")
            frames = frames[0]
        if frames.ndim != 3:
            raise ValueError(f"Expected [T,H,W] or [1,T,H,W] volume, got {tuple(frames.shape)}")
        _, height, width = frames.shape
        output: list[torch.Tensor] = []
        for start in range(0, frames.shape[0], self.batch_size):
            batch = frames[start : start + self.batch_size, None].repeat(1, 3, 1, 1).to(self.device)
            batch = F.interpolate(batch, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                probabilities = self.model(batch)
            if isinstance(probabilities, (tuple, list)):
                probabilities = probabilities[0]
            labels = probabilities.float().argmax(dim=1, keepdim=True)
            mask = (labels == self.tibia_class).float()
            mask = F.interpolate(mask, size=(height, width), mode="nearest")[:, 0]
            output.append(mask.bool().cpu())
        return torch.cat(output, dim=0)
