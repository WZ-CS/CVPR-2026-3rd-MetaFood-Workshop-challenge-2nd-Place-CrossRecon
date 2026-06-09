from typing import Dict, Tuple

import cv2
import numpy as np
from scipy import ndimage

from .types import SegmentationResult


def centered_ellipse_mask(shape: Tuple[int, int], scale: float = 0.82) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    axes = (max(4, int(w * scale * 0.5)), max(4, int(h * scale * 0.5)))
    center = (w // 2, h // 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask.astype(bool)


def ellipse_from_bbox(shape: Tuple[int, int], bbox: Tuple[int, int, int, int], expand: float = 1.25) -> np.ndarray:
    h, w = shape
    x0, y0, x1, y1 = bbox
    cx = int(round((x0 + x1) * 0.5))
    cy = int(round((y0 + y1) * 0.5))
    ax = max(8, int(round((x1 - x0 + 1) * 0.5 * expand)))
    ay = max(8, int(round((y1 - y0 + 1) * 0.5 * expand)))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    return mask.astype(bool)


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        h, w = mask.shape
        return 0, 0, w - 1, h - 1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def largest_components(mask: np.ndarray, max_components: int = 3) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
    keep = np.argsort(sizes)[::-1][:max_components] + 1
    return np.isin(labels, keep)


class LiteSegmenter:
    """Deterministic image heuristics used by the runnable offline backend."""

    def __init__(self, config: Dict[str, object]):
        self.min_food_area_ratio = float(config.get("min_food_area_ratio", 0.015))

    def segment(self, image_bgr: np.ndarray, food_name: str) -> SegmentationResult:
        h, w = image_bgr.shape[:2]
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        sat = hsv[:, :, 1].astype(np.float32)
        val = hsv[:, :, 2].astype(np.float32)
        l_chan = lab[:, :, 0].astype(np.float32)
        a_chan = lab[:, :, 1].astype(np.float32)
        b_chan = lab[:, :, 2].astype(np.float32)

        support = self._detect_support(sat=sat, val=val, l_chan=l_chan)

        border = support & ~centered_ellipse_mask((h, w), scale=0.58)
        if int(border.sum()) < 100:
            eroded = cv2.erode(support.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1).astype(bool)
            border = support & ~eroded
        if int(border.sum()) < 100:
            border = support & (sat < 95)
        if int(border.sum()) < 50:
            border = support
        plate_l = float(np.median(l_chan[border]))
        plate_a = float(np.median(a_chan[border]))
        plate_b = float(np.median(b_chan[border]))

        color_delta = np.sqrt((l_chan - plate_l) ** 2 + (a_chan - plate_a) ** 2 + (b_chan - plate_b) ** 2)
        food = self._food_candidate(support, sat, val, color_delta, plate_l, strict=False)
        food = self._clean_food_mask(food)

        area_ratio = float(food.sum()) / float(max(1, support.sum()))
        if area_ratio > 0.62:
            strict_food = self._food_candidate(support, sat, val, color_delta, plate_l, strict=True)
            strict_food = self._clean_food_mask(strict_food)
            strict_ratio = float(strict_food.sum()) / float(max(1, support.sum()))
            if self.min_food_area_ratio <= strict_ratio < area_ratio:
                food = strict_food
                area_ratio = strict_ratio

        if area_ratio < self.min_food_area_ratio:
            y_slice, x_slice = self._support_slices(support)
            food = np.zeros((h, w), dtype=bool)
            food[y_slice, x_slice] = centered_ellipse_mask(
                (y_slice.stop - y_slice.start, x_slice.stop - x_slice.start),
                scale=0.25,
            )
            area_ratio = float(food.sum()) / float(max(1, support.sum()))

        bbox = bbox_from_mask(food)
        confidence = max(0.05, min(1.0, 0.15 + area_ratio * 1.6))
        return SegmentationResult(
            food_mask=food.astype(bool),
            support_mask=support.astype(bool),
            confidence=confidence,
            bbox_xyxy=bbox,
        )

    @staticmethod
    def _food_candidate(
        support: np.ndarray,
        sat: np.ndarray,
        val: np.ndarray,
        color_delta: np.ndarray,
        plate_l: float,
        strict: bool,
    ) -> np.ndarray:
        if strict:
            food = support & (
                (color_delta > 42)
                | ((sat > 135) & (color_delta > 22))
                | ((val < 55) & (color_delta > 18))
            )
            plate_like = support & (sat < 70) & (val > 105) & (color_delta < 58)
            return food & ~plate_like
        return support & (
            (color_delta > 34)
            | ((sat > 125) & (color_delta > 18))
            | ((val < 60) & (color_delta > 16))
        )

    @staticmethod
    def _clean_food_mask(food: np.ndarray) -> np.ndarray:
        food_u8 = cv2.morphologyEx(food.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        food_u8 = cv2.morphologyEx(food_u8, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        return largest_components(food_u8.astype(bool), max_components=3)

    def _detect_support(self, sat: np.ndarray, val: np.ndarray, l_chan: np.ndarray) -> np.ndarray:
        h, w = sat.shape
        hough = self._detect_hough_plate(sat=sat, val=val)
        if hough is not None:
            return hough

        yy, xx = np.mgrid[:h, :w]
        lower_center = (
            (yy > h * 0.34)
            & (yy < h * 0.92)
            & (np.abs(xx - w * 0.50) < w * 0.34)
        )
        plate_pixels = lower_center & (sat < 95) & (val > 75) & (l_chan > 70)
        plate_pixels = cv2.morphologyEx(
            plate_pixels.astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((11, 11), np.uint8),
        )
        plate_pixels = cv2.morphologyEx(
            plate_pixels,
            cv2.MORPH_OPEN,
            np.ones((5, 5), np.uint8),
        )
        labels, count = ndimage.label(plate_pixels.astype(bool))
        best_bbox = None
        best_score = -1.0
        image_area = float(h * w)
        target = np.array([w * 0.50, h * 0.66], dtype=np.float32)

        for label in range(1, count + 1):
            component = labels == label
            area = float(component.sum())
            area_ratio = area / image_area
            if area_ratio < 0.002 or area_ratio > 0.18:
                continue
            x0, y0, x1, y1 = bbox_from_mask(component)
            bw = max(1, x1 - x0 + 1)
            bh = max(1, y1 - y0 + 1)
            if bw < w * 0.06 or bh < h * 0.035:
                continue
            center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float32)
            dist = float(np.linalg.norm((center - target) / np.array([w, h], dtype=np.float32)))
            aspect = bw / float(bh)
            aspect_penalty = abs(np.log(max(0.25, min(4.0, aspect)) / 1.45))
            score = area_ratio * 10.0 - dist * 2.5 - aspect_penalty
            if score > best_score:
                best_score = score
                best_bbox = (x0, y0, x1, y1)

        if best_bbox is None:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(
                mask,
                (int(w * 0.50), int(h * 0.68)),
                (max(16, int(w * 0.16)), max(12, int(h * 0.11))),
                0,
                0,
                360,
                255,
                -1,
            )
            return mask.astype(bool)
        return ellipse_from_bbox((h, w), best_bbox, expand=1.35)

    @staticmethod
    def _detect_hough_plate(sat: np.ndarray, val: np.ndarray):
        h, w = sat.shape
        gray = val.astype(np.uint8)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(24, int(min(h, w) * 0.06)),
            param1=80,
            param2=28,
            minRadius=max(32, int(min(h, w) * 0.07)),
            maxRadius=max(28, int(min(h, w) * 0.18)),
        )
        if circles is None:
            return None

        target = np.array([w * 0.52, h * 0.67], dtype=np.float32)
        best = None
        best_score = -1e9
        for x, y, r in circles[0]:
            if y < h * 0.38 or y > h * 0.92:
                continue
            if abs(x - w * 0.5) > w * 0.38:
                continue
            disk = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(disk, (int(round(x)), int(round(y))), int(round(r)), 255, -1)
            disk_bool = disk.astype(bool)
            if disk_bool.sum() == 0:
                continue
            low_sat_fraction = float((sat[disk_bool] < 105).mean())
            bright_fraction = float((val[disk_bool] > 65).mean())
            dist = float(np.linalg.norm((np.array([x, y], dtype=np.float32) - target) / np.array([w, h], dtype=np.float32)))
            score = low_sat_fraction + bright_fraction - 2.8 * dist + float(r) / max(h, w)
            if score > best_score:
                best_score = score
                best = (x, y, r)

        if best is None:
            return None
        x, y, r = best
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            mask,
            (int(round(x)), int(round(y))),
            (int(round(r * 1.12)), int(round(r * 0.92))),
            0,
            0,
            360,
            255,
            -1,
        )
        return mask.astype(bool)

    @staticmethod
    def _support_slices(support: np.ndarray) -> Tuple[slice, slice]:
        x0, y0, x1, y1 = bbox_from_mask(support)
        return slice(y0, y1 + 1), slice(x0, x1 + 1)
