import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .backends import build_lite_backends
from .backends.base import BackendNotConfigured
from .fusion import consumption_ratio, fuse_volume_ml
from .geometry import (
    build_path_a_mesh,
    build_path_b_mesh,
    estimate_plate_scale_m_per_px,
    mask_area,
)
from .mesh import scale_mesh_to_volume, write_obj
from .types import FoodItem, ItemResult, SegmentationResult, StateReconstruction, VideoFrame
from .utils import ensure_dir, ml_to_m3, normalize_name
from .video import discover_videos, sample_video_frames


class CrossReconPipeline:
    def __init__(self, config: Dict[str, object]):
        self.config = config
        mode = str(config.get("pipeline", {}).get("mode", "lite"))
        if mode != "lite":
            raise BackendNotConfigured(
                "Only the lite backend is runnable in this repository snapshot. "
                "Full mode is reserved for the official Pi3/SAM2/GroundingDINO/GPT Image/Hunyuan3D weights."
            )
        self.backends = build_lite_backends(config)

    def run(
        self,
        video_dir: Path,
        output_dir: Path,
        item_ids: Optional[Iterable[int]] = None,
        limit: Optional[int] = None,
    ) -> List[ItemResult]:
        output_dir = ensure_dir(output_dir)
        items = discover_videos(video_dir)
        if item_ids is not None:
            keep = {int(x) for x in item_ids}
            items = [item for item in items if item.item_id in keep]
        if limit is not None:
            items = items[: int(limit)]
        if not items:
            raise RuntimeError(f"No challenge videos found in {video_dir}")

        results = []
        for item in items:
            results.append(self.process_item(item, output_dir=output_dir))
        self.write_volume_csv(results, output_dir / "volumes.csv")
        self.write_summary(results, output_dir / "reports" / "run_summary.json")
        return results

    def process_item(self, item: FoodItem, output_dir: Path) -> ItemResult:
        observed = {
            "before": self._prepare_observed_state(item, "before", output_dir),
            "after": self._prepare_observed_state(item, "after", output_dir),
        }

        before_area = max(1, mask_area(observed["before"]["segmentation"].food_mask))
        after_area = max(1, mask_area(observed["after"]["segmentation"].food_mask))
        area_ratio = min(1.0, float(after_area) / float(before_area))

        before_b_mesh, before_b_ml = build_path_b_mesh(
            food_name=item.name,
            state="before",
            path_a_mesh=observed["before"]["path_a_mesh"],
            path_a_volume_ml=observed["before"]["path_a_volume_ml"],
            before_path_b_volume_ml=0.0,
            area_ratio_to_before=1.0,
            config=self.config,
        )
        after_b_mesh, after_b_ml = build_path_b_mesh(
            food_name=item.name,
            state="after",
            path_a_mesh=observed["after"]["path_a_mesh"],
            path_a_volume_ml=observed["after"]["path_a_volume_ml"],
            before_path_b_volume_ml=before_b_ml,
            area_ratio_to_before=area_ratio,
            config=self.config,
        )

        before = self._make_state_reconstruction(observed["before"], before_b_mesh, before_b_ml)
        after = self._make_state_reconstruction(observed["after"], after_b_mesh, after_b_ml)

        before.fused_volume_ml = fuse_volume_ml(
            before.path_a_volume_ml,
            before.path_b_volume_ml,
            before.coverage,
            "before",
            before.path_b_volume_ml,
            self.config,
        )
        after.fused_volume_ml = fuse_volume_ml(
            after.path_a_volume_ml,
            after.path_b_volume_ml,
            after.coverage,
            "after",
            before.path_b_volume_ml,
            self.config,
        )
        after.fused_volume_ml = min(after.fused_volume_ml, before.fused_volume_ml)

        before.final_mesh = scale_mesh_to_volume(before.path_b_mesh, ml_to_m3(before.fused_volume_ml))
        after.final_mesh = scale_mesh_to_volume(after.path_b_mesh, ml_to_m3(after.fused_volume_ml))

        mesh_dir = ensure_dir(output_dir / "meshes")
        prefix = f"{item.item_id:02d}_{item.slug}"
        write_obj(before.final_mesh, mesh_dir / f"{prefix}_before.obj")
        write_obj(after.final_mesh, mesh_dir / f"{prefix}_after.obj")

        ratio = consumption_ratio(before.fused_volume_ml, after.fused_volume_ml)
        return ItemResult(item=item, before=before, after=after, consumption_ratio=ratio)

    def _prepare_observed_state(self, item: FoodItem, state: str, output_dir: Path) -> Dict[str, object]:
        pipe_cfg = self.config.get("pipeline", {})
        window = pipe_cfg["before_window"] if state == "before" else pipe_cfg["after_window"]
        frames = sample_video_frames(
            video_path=item.video_path,
            window=window,
            num_frames=int(pipe_cfg.get("candidate_frames_per_state", 16)),
            max_size=int(pipe_cfg.get("frame_max_size", 512)),
        )
        if not frames:
            raise RuntimeError(f"No frames sampled for {item.name} ({state})")

        scored: List[Tuple[VideoFrame, SegmentationResult]] = []
        for frame in frames:
            seg = self.backends.segmenter.segment(frame.image_bgr, item.name)
            frame.score = seg.confidence * 1000.0 + min(frame.score, 2000.0) / 10.0
            scored.append((frame, seg))
        scored.sort(key=lambda pair: pair[0].score, reverse=True)

        k = int(pipe_cfg.get("keyframes_per_state", 8))
        keyframes = [pair[0] for pair in scored[: max(1, k)]]
        representative, segmentation = scored[0]
        scale_m_per_px = estimate_plate_scale_m_per_px(
            segmentation,
            plate_diameter_cm=float(pipe_cfg.get("plate_diameter_cm", 20.0)),
        )
        path_a_mesh, path_a_volume_ml, coverage = build_path_a_mesh(
            image_bgr=representative.image_bgr,
            segmentation=segmentation,
            scale_m_per_px=scale_m_per_px,
            food_name=item.name,
            config=self.config,
        )

        if bool(pipe_cfg.get("debug_visualizations", True)):
            debug_dir = ensure_dir(output_dir / "debug" / f"{item.item_id:02d}_{item.slug}")
            self._save_debug_overlay(representative.image_bgr, segmentation, debug_dir / f"{state}_mask.jpg")

        return {
            "state": state,
            "representative": representative,
            "segmentation": segmentation,
            "keyframes": keyframes,
            "scale_m_per_px": scale_m_per_px,
            "path_a_mesh": path_a_mesh,
            "path_a_volume_ml": path_a_volume_ml,
            "coverage": coverage,
        }

    def _make_state_reconstruction(
        self,
        observed: Dict[str, object],
        path_b_mesh,
        path_b_volume_ml: float,
    ) -> StateReconstruction:
        return StateReconstruction(
            state=str(observed["state"]),
            representative=observed["representative"],
            segmentation=observed["segmentation"],
            keyframes=observed["keyframes"],
            scale_m_per_px=float(observed["scale_m_per_px"]),
            path_a_mesh=observed["path_a_mesh"],
            path_b_mesh=path_b_mesh,
            path_a_volume_ml=float(observed["path_a_volume_ml"]),
            path_b_volume_ml=float(path_b_volume_ml),
            coverage=float(observed["coverage"]),
        )

    @staticmethod
    def _save_debug_overlay(image_bgr, segmentation: SegmentationResult, path: Path) -> None:
        overlay = image_bgr.copy()
        food = segmentation.food_mask.astype(bool)
        support = segmentation.support_mask.astype(bool)
        overlay[support] = (0.65 * overlay[support] + 0.35 * np.array([255, 180, 0])).astype(np.uint8)
        overlay[food] = (0.45 * overlay[food] + 0.55 * np.array([0, 40, 255])).astype(np.uint8)
        x0, y0, x1, y1 = segmentation.bbox_xyxy
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 255), 2)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), overlay)

    @staticmethod
    def write_volume_csv(results: List[ItemResult], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "id",
                    "food",
                    "before_volume_ml",
                    "after_volume_ml",
                    "consumption_ratio",
                    "before_path_a_ml",
                    "before_path_b_ml",
                    "after_path_a_ml",
                    "after_path_b_ml",
                    "before_coverage",
                    "after_coverage",
                ],
            )
            writer.writeheader()
            for result in results:
                writer.writerow(
                    {
                        "id": result.item.item_id,
                        "food": result.item.name,
                        "before_volume_ml": f"{result.before.fused_volume_ml:.3f}",
                        "after_volume_ml": f"{result.after.fused_volume_ml:.3f}",
                        "consumption_ratio": f"{result.consumption_ratio:.6f}",
                        "before_path_a_ml": f"{result.before.path_a_volume_ml:.3f}",
                        "before_path_b_ml": f"{result.before.path_b_volume_ml:.3f}",
                        "after_path_a_ml": f"{result.after.path_a_volume_ml:.3f}",
                        "after_path_b_ml": f"{result.after.path_b_volume_ml:.3f}",
                        "before_coverage": f"{result.before.coverage:.4f}",
                        "after_coverage": f"{result.after.coverage:.4f}",
                    }
                )

    @staticmethod
    def write_summary(results: List[ItemResult], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "num_items": len(results),
            "items": [
                {
                    "id": r.item.item_id,
                    "food": r.item.name,
                    "before_volume_ml": r.before.fused_volume_ml,
                    "after_volume_ml": r.after.fused_volume_ml,
                    "consumption_ratio": r.consumption_ratio,
                    "meshes": {
                        "before": f"meshes/{r.item.item_id:02d}_{r.item.slug}_before.obj",
                        "after": f"meshes/{r.item.item_id:02d}_{r.item.slug}_after.obj",
                    },
                }
                for r in results
            ],
            "note": (
                "This summary was produced by the lite backend. Challenge reproduction "
                "requires the full Pi3/SAM2/GroundingDINO/GPT Image/Hunyuan3D stack."
            ),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

