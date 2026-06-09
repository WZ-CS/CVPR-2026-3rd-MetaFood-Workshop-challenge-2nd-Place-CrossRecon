from typing import Dict, Tuple

import cv2
import numpy as np
from scipy import ndimage

from .mesh import ellipsoid_mesh, heightfield_mesh, mesh_volume_m3, scale_mesh_to_volume
from .types import Mesh, SegmentationResult
from .utils import m3_to_ml, ml_to_m3, normalize_name


def crop_with_padding(mask: np.ndarray, padding: int = 16) -> Tuple[slice, slice]:
    ys, xs = np.where(mask)
    h, w = mask.shape
    if len(xs) == 0:
        return slice(0, h), slice(0, w)
    x0 = max(0, int(xs.min()) - padding)
    x1 = min(w, int(xs.max()) + padding + 1)
    y0 = max(0, int(ys.min()) - padding)
    y1 = min(h, int(ys.max()) + padding + 1)
    return slice(y0, y1), slice(x0, x1)


def estimate_plate_scale_m_per_px(segmentation: SegmentationResult, plate_diameter_cm: float) -> float:
    ys, xs = np.where(segmentation.support_mask)
    if len(xs) == 0:
        return 1.0
    diameter_px = 0.5 * ((xs.max() - xs.min() + 1) + (ys.max() - ys.min() + 1))
    if diameter_px <= 0:
        return 1.0
    return float(plate_diameter_cm) / 100.0 / float(diameter_px)


def food_prior(config: Dict[str, object], food_name: str) -> Dict[str, float]:
    priors = config.get("food_priors", {})
    if not isinstance(priors, dict):
        return {"volume_ml": 180.0, "height_cm": 3.0}
    slug = normalize_name(food_name)
    prior = priors.get(slug) or priors.get("default") or {}
    return {
        "volume_ml": float(prior.get("volume_ml", 180.0)),
        "height_cm": float(prior.get("height_cm", 3.0)),
    }


def build_observed_heightfield(
    image_bgr: np.ndarray,
    segmentation: SegmentationResult,
    scale_m_per_px: float,
    food_name: str,
    config: Dict[str, object],
) -> Tuple[np.ndarray, float, float, float]:
    grid_size = int(config["pipeline"].get("grid_size", 96))
    prior = food_prior(config, food_name)
    y_slice, x_slice = crop_with_padding(segmentation.food_mask, padding=18)
    crop_mask = segmentation.food_mask[y_slice, x_slice].astype(np.uint8)
    crop_img = image_bgr[y_slice, x_slice]
    if crop_mask.sum() == 0:
        crop_mask = np.ones((grid_size, grid_size), dtype=np.uint8)

    resized_mask = cv2.resize(crop_mask, (grid_size, grid_size), interpolation=cv2.INTER_NEAREST) > 0
    if resized_mask.sum() == 0:
        resized_mask = np.ones((grid_size, grid_size), dtype=bool)

    crop_h, crop_w = crop_mask.shape[:2]
    step_x = max(1.0, crop_w - 1) * scale_m_per_px / max(1, grid_size - 1)
    step_y = max(1.0, crop_h - 1) * scale_m_per_px / max(1, grid_size - 1)

    dist = ndimage.distance_transform_edt(resized_mask)
    if dist.max() > 0:
        dist = dist / dist.max()

    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) if crop_img.size else np.zeros_like(crop_mask)
    gray = cv2.resize(gray, (grid_size, grid_size), interpolation=cv2.INTER_AREA).astype(np.float32)
    texture = 1.0 - np.clip((gray - gray[resized_mask].mean()) / 160.0 if resized_mask.any() else 0.0, -0.25, 0.25)

    base_height_m = prior["height_cm"] / 100.0
    height = resized_mask.astype(np.float64) * base_height_m * (0.22 + 0.78 * dist) * texture
    height = np.maximum(height, 0.0)

    raw_volume = float(height.sum() * step_x * step_y)
    food_support_ratio = float(segmentation.food_mask.sum()) / float(max(1, segmentation.support_mask.sum()))
    area_cap_factor = min(1.8, max(0.22, 0.25 + 1.45 * food_support_ratio))
    max_reasonable = ml_to_m3(prior["volume_ml"] * area_cap_factor)
    if raw_volume > max_reasonable > 0:
        height *= max_reasonable / raw_volume
        raw_volume = max_reasonable

    evidence_cells = float(resized_mask.sum())
    support_cells = float(max(1, cv2.resize(segmentation.support_mask.astype(np.uint8), (grid_size, grid_size), interpolation=cv2.INTER_NEAREST).sum()))
    coverage = max(0.05, min(1.0, segmentation.confidence * min(1.0, evidence_cells / support_cells * 6.0)))
    return height, step_x, step_y, coverage


def build_path_a_mesh(
    image_bgr: np.ndarray,
    segmentation: SegmentationResult,
    scale_m_per_px: float,
    food_name: str,
    config: Dict[str, object],
) -> Tuple[Mesh, float, float]:
    height, step_x, step_y, coverage = build_observed_heightfield(
        image_bgr=image_bgr,
        segmentation=segmentation,
        scale_m_per_px=scale_m_per_px,
        food_name=food_name,
        config=config,
    )
    mesh = heightfield_mesh(
        height=height,
        step_x=step_x,
        step_y=step_y,
        metadata={"path": "A", "description": "lite observed heightfield"},
    )
    return mesh, m3_to_ml(mesh.metadata["volume_m3"]), coverage


def build_path_b_mesh(
    food_name: str,
    state: str,
    path_a_mesh: Mesh,
    path_a_volume_ml: float,
    before_path_b_volume_ml: float,
    area_ratio_to_before: float,
    config: Dict[str, object],
) -> Tuple[Mesh, float]:
    prior = food_prior(config, food_name)
    lite_cfg = config.get("lite_backend", {})
    prior_weight = float(lite_cfg.get("path_b_prior_weight", 0.65))
    completion_bias = float(lite_cfg.get("completion_bias_after", 0.65))

    if state == "before":
        target_ml = prior_weight * prior["volume_ml"] + (1.0 - prior_weight) * path_a_volume_ml
    else:
        hallucinated_fraction = max(completion_bias, min(1.0, area_ratio_to_before ** 0.5))
        target_ml = max(path_a_volume_ml, before_path_b_volume_ml * hallucinated_fraction)

    vertices = path_a_mesh.vertices
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    rx = max(0.01, float(extent[0]) * 0.5)
    ry = max(0.01, float(extent[1]) * 0.5)
    rz = max(0.006, prior["height_cm"] / 200.0)
    mesh = ellipsoid_mesh(
        radius_x=rx,
        radius_y=ry,
        radius_z=rz,
        metadata={"path": "B", "description": "lite generative closed-shape prior"},
    )
    mesh = scale_mesh_to_volume(mesh, ml_to_m3(target_ml))
    return mesh, target_ml


def mask_area(mask: np.ndarray) -> int:
    return int(np.asarray(mask, dtype=bool).sum())
