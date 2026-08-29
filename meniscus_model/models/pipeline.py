"""End-to-end inference pipeline for the four-module meniscus system.

paper-specified:
    The framework integrates four task-specific modules: MeniscoViT-CoP for
    detection/localization/laterality/abnormality, Video Swin for degeneration
    vs tear grading, Video Swin for bucket-handle tear detection, and a beta-rule
    discoid detector.

paper-inferred implementation:
    The paper does not publish end-to-end post-processing code. This pipeline
    uses highest-confidence token boxes per side/plane/slice, configurable
    missing-slice box filling, ROI crops, and optional activation heatmaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from datasets.coordinate_utils import volume_patch_grid_to_world_coordinates
from datasets.dicom_reader import read_dicom_series
from models.module4_rule_based import (
    WidthMeasurement,
    detect_discoid_meniscus,
    lateral_coronal_predicted_meniscus_width_from_roi_cache,
    paired_widths_from_unet_and_roi_cache,
    tibia_width_from_mask,
)
from scripts.generate_rois import crop_roi_sequence, fill_missing_boxes
from utils.metrics import filter_module1_predictions
from utils.visualization import save_module4_measurement_plot, save_roi_preview, save_slice_overlay, save_token_heatmap


STATUS_NAMES = ["normal", "degeneration", "tear"]
SIDES = ("medial", "lateral")
PLANES = ("sagittal", "coronal")
ID_TO_SIDE = {0: "medial", 1: "lateral"}


@dataclass
class PipelineCase:
    """Pipeline input after DICOM loading.

    Tensors are unbatched:
        sagittal_volume/coronal_volume: ``[1,T,H,W]``
        sagittal_coords/coronal_coords: ``[T*N,3]``
    """

    case_id: str
    sagittal_volume: torch.Tensor
    coronal_volume: torch.Tensor
    sagittal_coords: torch.Tensor
    coronal_coords: torch.Tensor
    sagittal_metadata: dict[str, Any]
    coronal_metadata: dict[str, Any]


class MeniscusDiagnosticPipeline:
    """Runs Module 1, ROI generation, Module 2, Module 3, and Module 4."""

    def __init__(
        self,
        module1: torch.nn.Module,
        module2: torch.nn.Module,
        module3: torch.nn.Module,
        config: dict[str, Any],
        device: str | torch.device = "cuda",
        tibia_segmenter: Any | None = None,
    ):
        self.config = config
        self.device = torch.device(device if str(device) == "cuda" and torch.cuda.is_available() else "cpu")
        self.module1 = module1.to(self.device).eval()
        self.module2 = module2.to(self.device).eval()
        self.module3 = module3.to(self.device).eval()
        self.tibia_segmenter = tibia_segmenter
        self.amp = bool(config.get("pipeline", {}).get("amp", True)) and self.device.type == "cuda"
        self.device_type = "cuda" if self.device.type == "cuda" else "cpu"

    @staticmethod
    def load_case(
        case_id: str,
        sagittal_dicom_dir: str | Path,
        coronal_dicom_dir: str | Path,
        output_size: int,
        num_slices: int,
        patch_size: int,
        slice_selection: dict[str, Any] | None = None,
    ) -> PipelineCase:
        """Read DICOM, normalize/resize/sample, and compute patch coordinates."""

        sagittal = read_dicom_series(sagittal_dicom_dir, output_size=output_size, target_slices=num_slices, slice_selection=slice_selection)
        coronal = read_dicom_series(coronal_dicom_dir, output_size=output_size, target_slices=num_slices, slice_selection=slice_selection)

        def coords(series) -> torch.Tensor:
            _, height, width = series.volume.shape
            values = volume_patch_grid_to_world_coordinates((height, width), patch_size, series.volume.shape[0], series.metadata)
            return torch.from_numpy(values).float()

        return PipelineCase(
            case_id=case_id,
            sagittal_volume=torch.from_numpy(sagittal.volume).unsqueeze(0).float(),
            coronal_volume=torch.from_numpy(coronal.volume).unsqueeze(0).float(),
            sagittal_coords=coords(sagittal),
            coronal_coords=coords(coronal),
            sagittal_metadata=sagittal.metadata,
            coronal_metadata=coronal.metadata,
        )

    @torch.inference_mode()
    def infer_case(
        self,
        case: PipelineCase,
        output_dir: str | Path,
        tibia_width: float | None = None,
        tibia_mask_path: str | None = None,
    ) -> dict[str, Any]:
        """Run the full pipeline and write visualizations.

        Returns:
            JSON-serializable dict matching the requested output schema.
        """

        output_dir = Path(output_dir)
        vis_dir = output_dir / "visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)
        vis_paths = {"bbox_images": [], "attention_heatmaps": [], "roi_images": [], "module4_images": []}

        module1_pred = self._run_module1(case)
        raw_boxes = self._collect_module1_boxes(module1_pred, case)
        filled_boxes = {
            side: {
                plane: fill_missing_boxes(raw_boxes[side][plane], strategy=self.config["pipeline"].get("roi_fill_strategy", "nearest"))
                for plane in PLANES
            }
            for side in SIDES
        }
        rois = self._make_rois(case, filled_boxes)

        result = {
            "case_id": case.case_id,
            "menisci": {
                "medial": {"detected": False, "status": "normal", "bucket_handle": False, "boxes": [], "confidence": {}},
                "lateral": {
                    "detected": False,
                    "status": "normal",
                    "bucket_handle": False,
                    "discoid": False,
                    "discoid_available": False,
                    "discoid_beta": None,
                    "tibia_width": None,
                    "tibia_width_source": None,
                    "tibia_width_slice_index": None,
                    "meniscus_width": None,
                    "meniscus_width_slice_index": None,
                    "boxes": [],
                    "confidence": {},
                },
            },
            "visualizations": vis_paths,
        }

        for side in SIDES:
            detected = any(item is not None for plane in PLANES for item in raw_boxes[side][plane])
            abnormal_probability = max(
                [
                    float(item.get("abnormal_probability", 0.0))
                    for plane in PLANES
                    for item in raw_boxes[side][plane]
                    if item is not None
                ]
                or [0.0]
            )
            result["menisci"][side]["detected"] = bool(detected)
            result["menisci"][side]["boxes"] = self._boxes_for_json(filled_boxes[side])
            result["menisci"][side]["confidence"]["abnormality"] = abnormal_probability
            is_abnormal = abnormal_probability >= float(self.config["thresholds"].get("abnormal_score", 0.4))
            if detected and is_abnormal:
                status_name, status_conf, status_probs = self._run_module2(rois[side]["sagittal_roi"], rois[side]["coronal_roi"])
                if status_name == "tear":
                    bucket, bucket_conf = self._run_module3(rois[side]["sagittal_roi"], rois[side]["coronal_roi"])
                else:
                    bucket, bucket_conf = False, 0.0
                result["menisci"][side]["status"] = status_name
                result["menisci"][side]["bucket_handle"] = bucket
                result["menisci"][side]["confidence"].update({"status": status_conf, "status_probabilities": status_probs, "bucket_handle": bucket_conf})
            else:
                result["menisci"][side]["confidence"].update({"status": 0.0, "status_probabilities": {name: 0.0 for name in STATUS_NAMES}, "bucket_handle": 0.0})

        self._run_module4(result, filled_boxes, tibia_width=tibia_width, tibia_mask_path=tibia_mask_path, case=case)

        if self.config.get("outputs", {}).get("save_visualizations", True):
            vis_paths.update(self._save_visualizations(case, filled_boxes, rois, module1_pred, result, vis_dir))
        return result

    def _run_module1(self, case: PipelineCase) -> dict[str, Any]:
        sagittal = case.sagittal_volume.unsqueeze(0).to(self.device)
        coronal = case.coronal_volume.unsqueeze(0).to(self.device)
        sagittal_coords = case.sagittal_coords.unsqueeze(0).to(self.device)
        coronal_coords = case.coronal_coords.unsqueeze(0).to(self.device)
        with torch.autocast(device_type=self.device_type, dtype=torch.float16, enabled=self.amp):
            return self.module1(sagittal, coronal, sagittal_coords, coronal_coords)

    @staticmethod
    def _empty_boxes(num_slices: int) -> list[dict[str, Any] | None]:
        return [None for _ in range(num_slices)]

    def _collect_module1_boxes(self, pred: dict[str, Any], case: PipelineCase) -> dict[str, dict[str, list[dict[str, Any] | None]]]:
        result = {
            side: {"sagittal": self._empty_boxes(case.sagittal_volume.shape[1]), "coronal": self._empty_boxes(case.coronal_volume.shape[1])}
            for side in SIDES
        }
        pipeline_cfg = self.config["pipeline"]
        score_threshold = float(self.config["thresholds"].get("detection_score", 0.5))
        filtered = filter_module1_predictions(
            pred,
            score_threshold=score_threshold,
            topk_per_image=pipeline_cfg.get("module1_topk_per_image"),
            nms_threshold=pipeline_cfg.get("module1_nms_threshold"),
            final_score_mode=str(pipeline_cfg.get("module1_score_mode", "objectness")),
            quality_score_alpha=float(pipeline_cfg.get("module1_quality_score_alpha", 1.0)),
            quality_score_beta=float(pipeline_cfg.get("module1_quality_score_beta", 1.0)),
        )[0]
        side_probs = torch.softmax(pred["side_logits"][0], dim=-1).detach().float().cpu()
        abnormal_probs = torch.softmax(pred["abnormality_logits"][0], dim=-1).detach().float().cpu()

        for box, score_value, plane_id, slice_index, side_id, token_index in zip(
            filtered["boxes"],
            filtered["scores"],
            filtered["plane_ids"],
            filtered["slice_indices"],
            filtered["side_ids"],
            filtered["indices"],
        ):
            token_index_int = int(token_index)
            side_id_int = int(side_id)
            if float(side_probs[token_index_int, side_id_int]) < float(
                self.config["thresholds"].get("side_score", 0.5)
            ):
                continue
            side = ID_TO_SIDE[side_id_int]
            plane = PLANES[int(plane_id)]
            slice_index_int = int(slice_index)
            score = float(score_value)
            current = result[side][plane][slice_index_int]
            if current is None or score > float(current["score"]):
                result[side][plane][slice_index_int] = {
                    "box": box.detach().float().cpu().clamp(0, 1),
                    "score": score,
                    "source": "module1",
                    "abnormal_probability": float(abnormal_probs[token_index_int, 1]),
                }
        return result

    def _make_rois(self, case: PipelineCase, boxes: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, dict[str, torch.Tensor]]:
        crop_size = int(self.config["pipeline"].get("roi_crop_size", 112))
        return {
            side: {
                "sagittal_roi": crop_roi_sequence(case.sagittal_volume, boxes[side]["sagittal"], crop_size),
                "coronal_roi": crop_roi_sequence(case.coronal_volume, boxes[side]["coronal"], crop_size),
            }
            for side in SIDES
        }

    @torch.inference_mode()
    def _run_module2(self, sagittal_roi: torch.Tensor, coronal_roi: torch.Tensor) -> tuple[str, float, dict[str, float]]:
        with torch.autocast(device_type=self.device_type, dtype=torch.float16, enabled=self.amp):
            logits = self.module2(sagittal_roi.unsqueeze(0).to(self.device), coronal_roi.unsqueeze(0).to(self.device))
        probs = torch.softmax(logits[0], dim=-1).detach().cpu()
        idx = int(probs.argmax())
        return STATUS_NAMES[idx], float(probs[idx]), {name: float(probs[i]) for i, name in enumerate(STATUS_NAMES)}

    @torch.inference_mode()
    def _run_module3(self, sagittal_roi: torch.Tensor, coronal_roi: torch.Tensor) -> tuple[bool, float]:
        with torch.autocast(device_type=self.device_type, dtype=torch.float16, enabled=self.amp):
            logits = self.module3(sagittal_roi.unsqueeze(0).to(self.device), coronal_roi.unsqueeze(0).to(self.device)).squeeze(-1)
        prob = float(torch.sigmoid(logits[0]).detach().cpu())
        return prob >= float(self.config["thresholds"].get("bucket_handle_score", 0.5)), prob

    def _run_module4(
        self,
        result: dict[str, Any],
        boxes: dict[str, dict[str, list[dict[str, Any]]]],
        tibia_width: float | None,
        tibia_mask_path: str | None,
        case: PipelineCase,
    ) -> None:
        spacing_col = float(case.coronal_metadata["slices"][0].get("PixelSpacing", [1.0, 1.0])[1])
        tibia_measurement: WidthMeasurement | None = None
        meniscus_measurement: WidthMeasurement | None = None
        roi_payload = {"lateral": {"boxes": {"coronal": boxes["lateral"]["coronal"]}}}
        if tibia_width is not None:
            tibia_measurement = WidthMeasurement(float(tibia_width), source="inference_argument")
        elif tibia_mask_path is not None:
            tibia_measurement = tibia_width_from_mask(tibia_mask_path, pixel_spacing_col=spacing_col)
        elif self.tibia_segmenter is not None:
            paired = paired_widths_from_unet_and_roi_cache(
                case.coronal_volume,
                self.tibia_segmenter,
                roi_payload,
                image_width=int(case.coronal_volume.shape[-1]),
                pixel_spacing_col=spacing_col,
                min_slice_pixels=int(self.config["module4"].get("min_tibia_slice_pixels", 1)),
                min_plateau_width_fraction=float(self.config["module4"].get("min_plateau_width_fraction", 0.95)),
                aggregation_strategy=str(self.config["module4"].get("beta_aggregation", "max_w2")),
                min_candidate_slices=int(self.config["module4"].get("min_candidate_slices", 1)),
                central_window_radius=int(self.config["module4"].get("central_window_radius", 1)),
            )
            if paired is not None:
                tibia_measurement = paired.tibia
                meniscus_measurement = paired.meniscus
        if tibia_measurement is None:
            result["menisci"]["lateral"]["confidence"]["discoid"] = None
            return
        lateral_result = result["menisci"]["lateral"]
        lateral_result["tibia_width"] = float(tibia_measurement.width)
        lateral_result["tibia_width_source"] = tibia_measurement.source
        lateral_result["tibia_width_slice_index"] = tibia_measurement.slice_index
        result["menisci"]["lateral"]["discoid_available"] = True
        if not result["menisci"]["lateral"]["detected"]:
            result["menisci"]["lateral"]["confidence"]["discoid"] = 0.0
            return
        if meniscus_measurement is None:
            meniscus_measurement = lateral_coronal_predicted_meniscus_width_from_roi_cache(
                roi_payload,
                image_width=int(case.coronal_volume.shape[-1]),
                pixel_spacing_col=spacing_col,
            )
        if meniscus_measurement is None:
            lateral_result["discoid_available"] = False
            lateral_result["confidence"]["discoid"] = None
            return
        lateral_result["meniscus_width"] = meniscus_measurement.width
        lateral_result["meniscus_width_slice_index"] = meniscus_measurement.slice_index
        measurement = detect_discoid_meniscus(
            meniscus_width=meniscus_measurement.width,
            tibia_width=float(tibia_measurement.width),
            threshold=float(self.config["module4"].get("threshold", self.config["thresholds"].get("discoid_beta", 0.4))),
            probability_temperature=float(self.config["module4"].get("probability_temperature", 0.05)),
            meniscus_source=meniscus_measurement.source,
            tibia_source=tibia_measurement.source,
            unit="mm",
        )
        result["menisci"]["lateral"]["discoid"] = bool(measurement.predicted_label)
        result["menisci"]["lateral"]["discoid_beta"] = float(measurement.beta)
        result["menisci"]["lateral"]["confidence"]["discoid"] = float(measurement.probability)

    @staticmethod
    def _boxes_for_json(side_boxes: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        output = []
        for plane, boxes in side_boxes.items():
            for slice_idx, item in enumerate(boxes):
                output.append(
                    {
                        "plane": plane,
                        "slice_index": slice_idx,
                        "bbox": [float(v) for v in item["box"].tolist()],
                        "score": float(item["score"]),
                        "source": item["source"],
                    }
                )
        return output

    def _save_visualizations(
        self,
        case: PipelineCase,
        boxes: dict[str, dict[str, list[dict[str, Any]]]],
        rois: dict[str, dict[str, torch.Tensor]],
        module1_pred: dict[str, Any],
        result: dict[str, Any],
        vis_dir: Path,
    ) -> dict[str, list[str]]:
        paths = {"bbox_images": [], "attention_heatmaps": [], "roi_images": [], "module4_images": []}
        for side in SIDES:
            for plane in PLANES:
                volume = case.sagittal_volume if plane == "sagittal" else case.coronal_volume
                mid = volume.shape[1] // 2
                out = vis_dir / "bbox" / f"{case.case_id}_{side}_{plane}_bbox.png"
                save_slice_overlay(volume, out, boxes_xyxy=[boxes[side][plane][mid]["box"].tolist()], slice_index=mid, title=f"{case.case_id} {side} {plane}")
                paths["bbox_images"].append(str(out))
            roi_out = vis_dir / "roi" / f"{case.case_id}_{side}_roi.png"
            save_roi_preview(rois[side]["sagittal_roi"], rois[side]["coronal_roi"], roi_out, title=f"{case.case_id} {side}")
            paths["roi_images"].append(str(roi_out))

        paths["attention_heatmaps"].extend(self._save_objectness_heatmaps(case, module1_pred, vis_dir / "heatmaps"))

        lateral_result = result["menisci"]["lateral"]
        if lateral_result["discoid_beta"] is not None:
            out = vis_dir / "module4" / f"{case.case_id}_module4.png"
            save_module4_measurement_plot(
                out,
                float(lateral_result["tibia_width"]),
                float(lateral_result["meniscus_width"]),
                float(lateral_result["discoid_beta"]),
            )
            paths["module4_images"].append(str(out))
        return paths

    def _save_objectness_heatmaps(self, case: PipelineCase, pred: dict[str, Any], out_dir: Path) -> list[str]:
        out_paths: list[str] = []
        scores = torch.sigmoid(pred["objectness_logits"][0]).detach().cpu()
        shape = pred["shape_metadata"]
        offsets = {"sagittal": 0, "coronal": int(shape["sagittal"]["slices"] * shape["sagittal"]["patches_per_slice"])}
        for plane in PLANES:
            meta = shape[plane]
            per_slice = int(meta["patches_per_slice"])
            mid = int(meta["slices"]) // 2
            start = offsets[plane] + mid * per_slice
            heat = scores[start : start + per_slice].reshape(int(meta["grid_height"]), int(meta["grid_width"]))
            heat = F.interpolate(heat[None, None], size=(int(meta["height"]), int(meta["width"])), mode="bilinear", align_corners=False)[0, 0]
            volume = case.sagittal_volume if plane == "sagittal" else case.coronal_volume
            out = out_dir / f"{case.case_id}_{plane}_objectness_heatmap.png"
            save_token_heatmap(volume, heat, out, slice_index=mid, title=f"{case.case_id} {plane} objectness")
            out_paths.append(str(out))
        return out_paths
