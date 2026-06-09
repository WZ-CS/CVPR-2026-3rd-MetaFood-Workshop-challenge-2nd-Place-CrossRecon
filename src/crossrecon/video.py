import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .types import FoodItem, VideoFrame


VIDEO_RE = re.compile(r"^(?P<id>\d+)_(?P<name>.+)\.mp4$", re.IGNORECASE)


def parse_food_item(path: Path) -> Optional[FoodItem]:
    match = VIDEO_RE.match(path.name)
    if not match:
        return None
    item_id = int(match.group("id"))
    name = match.group("name").replace("_", " ").strip()
    return FoodItem(item_id=item_id, name=name, video_path=path)


def discover_videos(video_dir: Path) -> List[FoodItem]:
    items = []
    for path in sorted(video_dir.glob("*.mp4")):
        item = parse_food_item(path)
        if item is not None:
            items.append(item)
    return sorted(items, key=lambda x: x.item_id)


def resize_frame(image: np.ndarray, max_size: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, float(max_size) / float(max(h, w)))
    if scale == 1.0:
        return image
    new_size = (int(round(w * scale)), int(round(h * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def frame_sharpness(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def sample_video_frames(
    video_path: Path,
    window: Sequence[float],
    num_frames: int,
    max_size: int,
) -> List[VideoFrame]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 10.0)
    if frame_count <= 0:
        raise RuntimeError(f"Video has no readable frames: {video_path}")

    start_frac, end_frac = float(window[0]), float(window[1])
    start = max(0, min(frame_count - 1, int(round(frame_count * start_frac))))
    end = max(start + 1, min(frame_count, int(round(frame_count * end_frac))))
    indices = np.linspace(start, end - 1, max(1, int(num_frames))).round().astype(int)

    frames: List[VideoFrame] = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, image = cap.read()
        if not ok:
            continue
        image = resize_frame(image, max_size=max_size)
        sharpness = frame_sharpness(image)
        frames.append(
            VideoFrame(
                frame_index=int(index),
                timestamp_sec=float(index) / max(fps, 1e-6),
                image_bgr=image,
                score=sharpness,
            )
        )
    cap.release()
    return frames


def select_keyframes(frames: Iterable[VideoFrame], k: int) -> List[VideoFrame]:
    ordered = sorted(frames, key=lambda f: f.score, reverse=True)
    return ordered[: max(1, int(k))]

