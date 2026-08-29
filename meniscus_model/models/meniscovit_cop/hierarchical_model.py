"""Paper-aligned hierarchical MeniscoViT-CoP backbone.

The manuscript figures specify four UniFormer stages with 3D patch embedding,
channels ``64/128/320/512``, spatial strides ``4/8/16/32``, temporal stride 2,
and a coordinate-aware sagittal/coronal fusion layer at every stage. Regional
attention is evaluated on pooled stage features to keep 512x512 MRI tractable,
then its cross-plane residual is broadcast back to the native feature grid.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .patch_embed import PatchEmbedShape
from .uniformer_blocks import DropPath


def _pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return int(value), int(value)
    if len(value) != 2:
        raise ValueError("expected an int or two spatial dimensions")
    return int(value[0]), int(value[1])


class ConvMlp3D(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Conv3d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Conv3d(hidden_dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class LocalUniFormerBlock(nn.Module):
    """Official UniFormer-style local MHRA block used in stages 1 and 2."""

    def __init__(self, dim: int, mlp_ratio: float, dropout: float, drop_path: float):
        super().__init__()
        self.dpe = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm3d(dim)
        self.proj_in = nn.Conv3d(dim, dim, kernel_size=1)
        self.mhra = nn.Conv3d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.proj_out = nn.Conv3d(dim, dim, kernel_size=1)
        self.norm2 = nn.BatchNorm3d(dim)
        self.mlp = ConvMlp3D(dim, int(dim * mlp_ratio), dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dpe(x)
        x = x + self.drop_path(self.proj_out(self.mhra(self.proj_in(self.norm1(x)))))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class TokenAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float, attn_dropout: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_dropout = float(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(out))


class LinearMlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class GlobalUniFormerBlock(nn.Module):
    """Global MHRA with memory-efficient shared space/time attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
        drop_path: float,
        factorized: bool = True,
    ):
        super().__init__()
        self.factorized = bool(factorized)
        self.dpe = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = TokenAttention(dim, num_heads, dropout, attn_dropout)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = LinearMlp(dim, int(dim * mlp_ratio), dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dpe(x)
        batch, channels, depth, height, width = x.shape
        tokens = x.permute(0, 2, 3, 4, 1).contiguous()
        if self.factorized:
            temporal = tokens.permute(0, 2, 3, 1, 4).reshape(batch * height * width, depth, channels)
            temporal = temporal + self.drop_path(self.attn(self.norm1(temporal)))
            tokens = temporal.reshape(batch, height, width, depth, channels).permute(0, 3, 1, 2, 4).contiguous()
            spatial = tokens.reshape(batch * depth, height * width, channels)
            spatial = spatial + self.drop_path(self.attn(self.norm1(spatial)))
            tokens = spatial.reshape(batch, depth, height, width, channels)
        else:
            flat = tokens.reshape(batch, depth * height * width, channels)
            flat = flat + self.drop_path(self.attn(self.norm1(flat)))
            tokens = flat.reshape(batch, depth, height, width, channels)
        tokens = tokens + self.drop_path(self.mlp(self.norm2(tokens)))
        return tokens.permute(0, 4, 1, 2, 3).contiguous()


class HierarchicalPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, spatial_kernel: int, temporal_downsample: bool):
        super().__init__()
        temporal_kernel = 3 if temporal_downsample else 1
        temporal_stride = 2 if temporal_downsample else 1
        self.proj = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(temporal_kernel, spatial_kernel, spatial_kernel),
            stride=(temporal_stride, spatial_kernel, spatial_kernel),
            padding=(1 if temporal_downsample else 0, 0, 0),
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3).contiguous()


class CoordinatePositionEmbedding3D(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        hidden = max(32, dim // 2)
        self.proj = nn.Sequential(
            nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, dim), nn.Dropout(dropout)
        )

    def forward(
        self,
        sagittal: torch.Tensor,
        coronal: torch.Tensor,
        sagittal_coords: torch.Tensor,
        coronal_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([sagittal_coords, coronal_coords], dim=1).float()
        mean = combined.mean(dim=1, keepdim=True)
        std = combined.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)

        def add(features: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
            batch, channels, depth, height, width = features.shape
            normalized = (coords.float() - mean) / std
            embedded = self.proj(normalized.to(dtype=features.dtype))
            embedded = embedded.transpose(1, 2).reshape(batch, channels, depth, height, width)
            return features + embedded

        return add(sagittal, sagittal_coords), add(coronal, coronal_coords)


class RegionalCrossDirection(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float, attn_dropout: float, drop_path: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.norm_q = nn.LayerNorm(dim, eps=1e-6)
        self.norm_kv = nn.LayerNorm(dim, eps=1e-6)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = LinearMlp(dim, int(dim * mlp_ratio), dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    @staticmethod
    def _gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        batch, heads, _, head_dim = values.shape
        queries, nearest = indices.shape[1:]
        expanded_values = values.unsqueeze(2).expand(-1, -1, queries, -1, -1)
        expanded_indices = indices[:, None, :, :, None].expand(batch, heads, queries, nearest, head_dim)
        return torch.gather(expanded_values, dim=3, index=expanded_indices)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, neighbor_indices: torch.Tensor) -> torch.Tensor:
        batch, query_count, channels = query.shape
        key_count = key_value.shape[1]
        query_norm = self.norm_q(query)
        key_value_norm = self.norm_kv(key_value)
        q = self.q(query_norm).reshape(batch, query_count, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(key_value_norm).reshape(batch, key_count, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(key_value_norm).reshape(batch, key_count, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        selected_k = self._gather(k, neighbor_indices)
        selected_v = self._gather(v, neighbor_indices)
        logits = (q.unsqueeze(3) * selected_k).sum(dim=-1) * (self.head_dim**-0.5)
        attention = self.attn_drop(logits.softmax(dim=-1))
        out = (attention.unsqueeze(-1) * selected_v).sum(dim=3)
        out = out.permute(0, 2, 1, 3).reshape(batch, query_count, channels)
        query = query + self.drop_path(self.proj_drop(self.proj(out)))
        return query + self.drop_path(self.mlp(self.norm2(query)))


class RegionalCrossPlaneLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        k_nearest: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
        drop_path: float,
        max_tokens: int,
    ):
        super().__init__()
        self.k_nearest = int(k_nearest)
        self.max_tokens = int(max_tokens)
        self.sagittal_from_coronal = RegionalCrossDirection(dim, num_heads, mlp_ratio, dropout, attn_dropout, drop_path)
        self.coronal_from_sagittal = RegionalCrossDirection(dim, num_heads, mlp_ratio, dropout, attn_dropout, drop_path)

    def _pool(self, features: torch.Tensor, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, depth, height, width = features.shape
        if depth * height * width <= self.max_tokens:
            return features, coords
        spatial_limit = max(1, int(math.sqrt(self.max_tokens / max(depth, 1))))
        target_h, target_w = min(height, spatial_limit), min(width, spatial_limit)
        pooled_features = F.adaptive_avg_pool3d(features, (depth, target_h, target_w))
        coord_grid = coords.transpose(1, 2).reshape(batch, 3, depth, height, width)
        pooled_coords = F.adaptive_avg_pool3d(coord_grid, (depth, target_h, target_w))
        return pooled_features, pooled_coords.flatten(2).transpose(1, 2)

    @staticmethod
    def _tokens(features: torch.Tensor) -> torch.Tensor:
        return features.flatten(2).transpose(1, 2)

    @staticmethod
    def _grid(tokens: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
        batch, _, channels = tokens.shape
        return tokens.transpose(1, 2).reshape(batch, channels, *shape)

    def forward(
        self,
        sagittal: torch.Tensor,
        coronal: torch.Tensor,
        sagittal_coords: torch.Tensor,
        coronal_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sag_pool, sag_coords_pool = self._pool(sagittal, sagittal_coords)
        cor_pool, cor_coords_pool = self._pool(coronal, coronal_coords)
        sag_tokens, cor_tokens = self._tokens(sag_pool), self._tokens(cor_pool)
        distances = torch.cdist(sag_coords_pool.float(), cor_coords_pool.float())
        sag_neighbors = distances.topk(min(self.k_nearest, cor_tokens.shape[1]), dim=-1, largest=False).indices
        cor_neighbors = distances.transpose(1, 2).topk(min(self.k_nearest, sag_tokens.shape[1]), dim=-1, largest=False).indices
        del distances
        sag_fused = self.sagittal_from_coronal(sag_tokens, cor_tokens, sag_neighbors)
        cor_fused = self.coronal_from_sagittal(cor_tokens, sag_tokens, cor_neighbors)
        sag_delta = self._grid(sag_fused - sag_tokens, tuple(sag_pool.shape[-3:]))
        cor_delta = self._grid(cor_fused - cor_tokens, tuple(cor_pool.shape[-3:]))
        if sag_delta.shape[-3:] != sagittal.shape[-3:]:
            sag_delta = F.interpolate(sag_delta, size=sagittal.shape[-3:], mode="trilinear", align_corners=False)
        if cor_delta.shape[-3:] != coronal.shape[-3:]:
            cor_delta = F.interpolate(cor_delta, size=coronal.shape[-3:], mode="trilinear", align_corners=False)
        return sagittal + sag_delta, coronal + cor_delta


def _initial_stage_coords(
    coords: torch.Tensor,
    input_depth: int,
    input_height: int,
    input_width: int,
    patch_size: tuple[int, int],
    output_depth: int,
) -> torch.Tensor:
    grid_h, grid_w = input_height // patch_size[0], input_width // patch_size[1]
    expected = input_depth * grid_h * grid_w
    if coords.shape[1] != expected:
        raise ValueError(
            f"coordinate token count {coords.shape[1]} does not match expected {expected}; "
            "dataset patch_size must equal the initial model patch size"
        )
    grid = coords.reshape(coords.shape[0], input_depth, grid_h, grid_w, 3)
    temporal_indices = (torch.arange(output_depth, device=coords.device) * 2).clamp_max(input_depth - 1)
    return grid.index_select(1, temporal_indices).reshape(coords.shape[0], -1, 3)


def _downsample_coords(coords: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    batch = coords.shape[0]
    depth, height, width = shape
    grid = coords.transpose(1, 2).reshape(batch, 3, depth, height, width)
    grid = F.avg_pool3d(grid, kernel_size=(1, 2, 2), stride=(1, 2, 2))
    return grid.flatten(2).transpose(1, 2)


class HierarchicalMeniscoViTCoP(nn.Module):
    """Four-stage, dual-plane, coordinate-aware MeniscoViT-CoP backbone."""

    def __init__(
        self,
        image_size: int | Sequence[int] = 512,
        in_channels: int = 3,
        embed_dims: Sequence[int] = (64, 128, 320, 512),
        depths: Sequence[int] = (3, 4, 8, 3),
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        cross_attention_k: int = 32,
        cross_attention_max_tokens: int = 4096,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        use_checkpoint: bool = True,
        factorized_global_attention: bool = True,
        output_original_depth: bool = True,
        normalize_input: bool = False,
        input_mean: Sequence[float] = (0.45, 0.45, 0.45),
        input_std: Sequence[float] = (0.225, 0.225, 0.225),
    ):
        super().__init__()
        if len(embed_dims) != 4 or len(depths) != 4:
            raise ValueError("paper hierarchical MeniscoViT requires four embed_dims and four depths")
        self.image_size = _pair(image_size)
        self.in_channels = int(in_channels)
        self.embed_dims = tuple(int(value) for value in embed_dims)
        self.depths = tuple(int(value) for value in depths)
        self.out_dim = self.embed_dims[-1]
        self.initial_patch_size = (4, 4)
        self.use_checkpoint = bool(use_checkpoint)
        self.output_original_depth = bool(output_original_depth)
        self.normalize_input = bool(normalize_input)
        if len(input_mean) != self.in_channels or len(input_std) != self.in_channels:
            raise ValueError("input_mean/input_std length must equal in_channels")
        if any(float(value) <= 0 for value in input_std):
            raise ValueError("input_std values must be positive")
        self.register_buffer(
            "input_mean",
            torch.tensor(list(input_mean), dtype=torch.float32).view(1, self.in_channels, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "input_std",
            torch.tensor(list(input_std), dtype=torch.float32).view(1, self.in_channels, 1, 1, 1),
            persistent=False,
        )
        dpr = torch.linspace(0, float(drop_path_rate), sum(self.depths)).tolist()
        stage_offset = 0
        self.sagittal_patch_embeds = nn.ModuleList()
        self.coronal_patch_embeds = nn.ModuleList()
        self.sagittal_stages = nn.ModuleList()
        self.coronal_stages = nn.ModuleList()
        self.rcpe_layers = nn.ModuleList()
        self.cross_layers = nn.ModuleList()

        for stage_index, (dim, depth) in enumerate(zip(self.embed_dims, self.depths)):
            input_dim = self.in_channels if stage_index == 0 else self.embed_dims[stage_index - 1]
            spatial_kernel = 4 if stage_index == 0 else 2
            temporal_downsample = stage_index == 0
            self.sagittal_patch_embeds.append(HierarchicalPatchEmbed(input_dim, dim, spatial_kernel, temporal_downsample))
            self.coronal_patch_embeds.append(HierarchicalPatchEmbed(input_dim, dim, spatial_kernel, temporal_downsample))
            num_heads = max(1, dim // int(head_dim))

            def make_stage() -> nn.ModuleList:
                blocks: list[nn.Module] = []
                for block_index in range(depth):
                    drop_path = float(dpr[stage_offset + block_index])
                    if stage_index < 2:
                        block = LocalUniFormerBlock(dim, mlp_ratio, dropout, drop_path)
                    else:
                        block = GlobalUniFormerBlock(
                            dim, num_heads, mlp_ratio, dropout, attn_dropout, drop_path,
                            factorized=factorized_global_attention,
                        )
                    blocks.append(block)
                return nn.ModuleList(blocks)

            self.sagittal_stages.append(make_stage())
            self.coronal_stages.append(make_stage())
            self.rcpe_layers.append(CoordinatePositionEmbedding3D(dim, dropout))
            self.cross_layers.append(
                RegionalCrossPlaneLayer(
                    dim, num_heads, cross_attention_k, mlp_ratio, dropout,
                    attn_dropout, float(dpr[stage_offset]), cross_attention_max_tokens,
                )
            )
            stage_offset += depth
        self.sagittal_norm = nn.BatchNorm3d(self.out_dim)
        self.coronal_norm = nn.BatchNorm3d(self.out_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _prepare_channels(self, volume: torch.Tensor) -> torch.Tensor:
        if volume.shape[1] == self.in_channels:
            prepared = volume
        elif volume.shape[1] == 1 and self.in_channels == 3:
            prepared = volume.repeat(1, 3, 1, 1, 1)
        else:
            raise ValueError(f"cannot map input channels={volume.shape[1]} to model in_channels={self.in_channels}")
        if self.normalize_input:
            prepared = (
                prepared - self.input_mean.to(dtype=prepared.dtype)
            ) / self.input_std.to(dtype=prepared.dtype)
        return prepared

    @staticmethod
    def _uniformer_target_suffix(source_key: str) -> tuple[str, str] | None:
        """Map one official UniFormer-S tensor key to this branch layout."""

        patch_match = re.fullmatch(r"patch_embed([1-4])\.(.+)", source_key)
        if patch_match:
            stage_index = int(patch_match.group(1)) - 1
            return f"patch_embeds.{stage_index}.{patch_match.group(2)}", "patch"

        block_match = re.fullmatch(r"blocks([1-4])\.(\d+)\.(.+)", source_key)
        if block_match:
            stage_index = int(block_match.group(1)) - 1
            block_index = int(block_match.group(2))
            remainder = block_match.group(3)
            if remainder.startswith("pos_embed."):
                remainder = "dpe." + remainder.removeprefix("pos_embed.")
            if stage_index < 2:
                local_names = {"conv1.": "proj_in.", "attn.": "mhra.", "conv2.": "proj_out."}
                for source_prefix, target_prefix in local_names.items():
                    if remainder.startswith(source_prefix):
                        remainder = target_prefix + remainder.removeprefix(source_prefix)
                        break
            return f"stages.{stage_index}.{block_index}.{remainder}", "block"

        norm_match = re.fullmatch(r"norm\.(.+)", source_key)
        if norm_match:
            return f"norm.{norm_match.group(1)}", "norm"
        return None

    def load_uniformer_pretrained(
        self,
        checkpoint_path: str | Path,
        expected_sha256: str | None = None,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Initialize both plane branches from official UniFormer-S weights.

        The paper backbone uses the official UniFormer-S dimensions/depths.
        Patch embeddings, all 18 intra-plane blocks, and the final BN therefore
        map shape-for-shape.  Co-plane attention, RCPE, and detection heads are
        intentionally left task-random.  One source backbone is duplicated to
        sagittal and coronal branches.
        """

        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"UniFormer checkpoint not found: {path}")
        if expected_sha256:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest.lower() != str(expected_sha256).lower():
                raise ValueError(
                    f"UniFormer checkpoint SHA256 mismatch: expected {expected_sha256}, got {digest}"
                )

        raw = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
            raw = raw["model"]
        elif isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
            raw = raw["state_dict"]
        if not isinstance(raw, dict):
            raise TypeError("UniFormer checkpoint must contain a tensor state dict")

        source = {
            str(key).removeprefix("module."): value
            for key, value in raw.items()
            if isinstance(value, torch.Tensor)
        }
        target_state = self.state_dict()
        mapped: dict[str, torch.Tensor] = {}
        ignored_source_keys: list[str] = []
        for source_key, value in source.items():
            mapping = self._uniformer_target_suffix(source_key)
            if mapping is None:
                ignored_source_keys.append(source_key)
                continue
            suffix, kind = mapping
            for plane in ("sagittal", "coronal"):
                if kind == "norm":
                    target_key = f"{plane}_norm.{suffix.removeprefix('norm.')}"
                else:
                    target_key = f"{plane}_{suffix}"
                if target_key not in target_state:
                    raise KeyError(f"mapped UniFormer tensor has no target: {source_key} -> {target_key}")
                if tuple(value.shape) != tuple(target_state[target_key].shape):
                    raise ValueError(
                        f"shape mismatch for {source_key} -> {target_key}: "
                        f"{tuple(value.shape)} != {tuple(target_state[target_key].shape)}"
                    )
                mapped[target_key] = value

        expected_targets = {
            key
            for key in target_state
            if key.startswith((
                "sagittal_patch_embeds.", "coronal_patch_embeds.",
                "sagittal_stages.", "coronal_stages.",
                "sagittal_norm.", "coronal_norm.",
            ))
        }
        missing_targets = sorted(expected_targets - set(mapped))
        if strict and missing_targets:
            raise KeyError(
                "official UniFormer checkpoint did not cover all intra-plane tensors; "
                f"first missing keys: {missing_targets[:10]}"
            )
        self.load_state_dict(mapped, strict=False)
        report = {
            "checkpoint": str(path),
            "mapped_source_tensors": len(mapped) // 2,
            "loaded_target_tensors": len(mapped),
            "missing_target_tensors": len(missing_targets),
            "ignored_source_keys": sorted(ignored_source_keys),
        }
        self.pretrained_load_report = report
        return report

    def _run_stage(self, features: torch.Tensor, blocks: nn.ModuleList) -> torch.Tensor:
        for block in blocks:
            features = checkpoint(block, features, use_reentrant=False) if self.use_checkpoint and self.training else block(features)
        return features

    @staticmethod
    def _flatten(features: torch.Tensor) -> torch.Tensor:
        return features.permute(0, 2, 3, 4, 1).reshape(features.shape[0], -1, features.shape[1])

    def forward(
        self,
        sagittal_volume: torch.Tensor,
        coronal_volume: torch.Tensor,
        sagittal_coords: torch.Tensor,
        coronal_coords: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, Any]]:
        input_depth, input_height, input_width = map(int, sagittal_volume.shape[2:])
        if coronal_volume.shape[2:] != sagittal_volume.shape[2:]:
            raise ValueError("sagittal and coronal volumes must share T/H/W after preprocessing")
        sagittal, coronal = self._prepare_channels(sagittal_volume), self._prepare_channels(coronal_volume)
        sag_stage_coords, cor_stage_coords = sagittal_coords, coronal_coords
        for stage_index in range(4):
            previous_shape = tuple(sagittal.shape[-3:])
            sagittal = self.sagittal_patch_embeds[stage_index](sagittal)
            coronal = self.coronal_patch_embeds[stage_index](coronal)
            if stage_index == 0:
                sag_stage_coords = _initial_stage_coords(
                    sag_stage_coords, input_depth, input_height, input_width,
                    self.initial_patch_size, sagittal.shape[2],
                )
                cor_stage_coords = _initial_stage_coords(
                    cor_stage_coords, input_depth, input_height, input_width,
                    self.initial_patch_size, coronal.shape[2],
                )
            else:
                sag_stage_coords = _downsample_coords(sag_stage_coords, previous_shape)
                cor_stage_coords = _downsample_coords(cor_stage_coords, previous_shape)
            # Figure 9 and the methods place the co-plane Layer before the
            # modified intra-plane Stage.  Real-coordinate position embedding
            # is therefore injected after regional cross-attention and before
            # the stage's MHRA/FFN blocks.
            sagittal, coronal = self.cross_layers[stage_index](
                sagittal, coronal, sag_stage_coords, cor_stage_coords
            )
            sagittal, coronal = self.rcpe_layers[stage_index](
                sagittal, coronal, sag_stage_coords, cor_stage_coords
            )
            sagittal = self._run_stage(sagittal, self.sagittal_stages[stage_index])
            coronal = self._run_stage(coronal, self.coronal_stages[stage_index])
        sagittal, coronal = self.sagittal_norm(sagittal), self.coronal_norm(coronal)
        if self.output_original_depth and sagittal.shape[2] != input_depth:
            sagittal = F.interpolate(sagittal, size=(input_depth, sagittal.shape[3], sagittal.shape[4]), mode="trilinear", align_corners=False)
            coronal = F.interpolate(coronal, size=(input_depth, coronal.shape[3], coronal.shape[4]), mode="trilinear", align_corners=False)
        sag_tokens, cor_tokens = self._flatten(sagittal), self._flatten(coronal)
        grid_depth, grid_height, grid_width = map(int, sagittal.shape[-3:])
        patch_height, patch_width = input_height // grid_height, input_width // grid_width

        def shape_metadata() -> dict[str, Any]:
            return asdict(PatchEmbedShape(
                batch_size=int(sagittal.shape[0]), channels=1, slices=grid_depth,
                height=input_height, width=input_width, patch_height=patch_height,
                patch_width=patch_width, grid_height=grid_height, grid_width=grid_width,
                patches_per_slice=grid_height * grid_width,
            ))

        metadata = {
            "sagittal": shape_metadata(), "coronal": shape_metadata(),
            "tokens_per_plane": int(sag_tokens.shape[1]),
            "fused_tokens": int(sag_tokens.shape[1] + cor_tokens.shape[1]),
            "architecture": "paper_hierarchical_uniformer", "native_temporal_stride": 2,
        }
        per_token = torch.cat([sag_tokens, cor_tokens], dim=1)
        return {
            "sagittal_tokens_fused": sag_tokens,
            "coronal_tokens_fused": cor_tokens,
            "per_token_features": per_token,
            "fused_global_feature": per_token.mean(dim=1),
            "shape_metadata": metadata,
        }
