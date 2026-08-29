"""End-to-end inference entrypoint for the four-module pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from engine.validate_module1 import build_module1_model
from engine.validate_module2 import build_module2_model
from models.pipeline import MeniscusDiagnosticPipeline
from models.tibia_segmentation import CoronalTibiaU2Net
from utils.checkpoint import load_checkpoint
from utils.logging import get_logger


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_pipeline(config: dict[str, Any]) -> MeniscusDiagnosticPipeline:
    """Build modules and load checkpoints from ``pipeline.yaml``."""

    pipeline_cfg = config["pipeline"]
    device_name = pipeline_cfg.get("device", "cuda")
    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")

    module1_cfg = load_yaml(pipeline_cfg["module1_config"])
    module2_cfg = load_yaml(pipeline_cfg["module2_config"])
    module3_cfg = load_yaml(pipeline_cfg["module3_config"])
    module1 = build_module1_model(module1_cfg)
    module2 = build_module2_model(module2_cfg)
    module3 = build_module2_model(module3_cfg)

    load_checkpoint(pipeline_cfg["module1_checkpoint"], module1, map_location=device)
    load_checkpoint(pipeline_cfg["module2_checkpoint"], module2, map_location=device)
    module3_metadata = load_checkpoint(pipeline_cfg["module3_checkpoint"], module3, map_location=device)
    if pipeline_cfg.get("module3_use_checkpoint_threshold", True):
        checkpoint_threshold = module3_metadata.get("decision_threshold")
        if checkpoint_threshold is not None:
            config.setdefault("thresholds", {})["bucket_handle_score"] = float(checkpoint_threshold)
    tibia_segmenter = None
    segmenter_cfg = config.get("module4", {}).get("tibia_segmenter", {})
    if bool(segmenter_cfg.get("enabled", False)):
        tibia_segmenter = CoronalTibiaU2Net.from_config(segmenter_cfg, device=device)
    return MeniscusDiagnosticPipeline(
        module1,
        module2,
        module3,
        config=config,
        device=device,
        tibia_segmenter=tibia_segmenter,
    )


def infer_pipeline(
    config_path: str,
    case_id: str,
    sagittal_dicom_dir: str,
    coronal_dicom_dir: str,
    output_dir: str,
    tibia_width: float | None = None,
    tibia_mask_path: str | None = None,
) -> dict[str, Any]:
    """Run full meniscus inference for one case and write ``result.json``."""

    config = load_yaml(config_path)
    logger = get_logger("infer_pipeline")
    pipeline = build_pipeline(config)
    pipe_cfg = config["pipeline"]
    case = MeniscusDiagnosticPipeline.load_case(
        case_id=case_id,
        sagittal_dicom_dir=sagittal_dicom_dir,
        coronal_dicom_dir=coronal_dicom_dir,
        output_size=int(pipe_cfg.get("input_size", 224)),
        num_slices=int(pipe_cfg.get("num_slices", 16)),
        patch_size=int(pipe_cfg.get("patch_size", 16)),
        slice_selection=pipe_cfg.get("slice_selection"),
    )
    result = pipeline.infer_case(case, output_dir=output_dir, tibia_width=tibia_width or pipe_cfg.get("tibia_width"), tibia_mask_path=tibia_mask_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_path = output_path / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info(f"saved pipeline result: {result_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--sagittal-dicom-dir", required=True)
    parser.add_argument("--coronal-dicom-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tibia-width", type=float, default=None, help="Optional tibia width for Module 4.")
    parser.add_argument("--tibia-mask-path", default=None, help="Reserved Module 4 tibia mask input.")
    args = parser.parse_args()
    infer_pipeline(
        config_path=args.config,
        case_id=args.case_id,
        sagittal_dicom_dir=args.sagittal_dicom_dir,
        coronal_dicom_dir=args.coronal_dicom_dir,
        output_dir=args.output_dir,
        tibia_width=args.tibia_width,
        tibia_mask_path=args.tibia_mask_path,
    )


if __name__ == "__main__":
    main()
