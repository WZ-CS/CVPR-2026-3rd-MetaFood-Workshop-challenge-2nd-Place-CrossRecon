from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class FoodItem:
    item_id: int
    name: str
    video_path: Path

    @property
    def slug(self) -> str:
        return self.name.lower().replace(" ", "_").replace("-", "_")


@dataclass
class VideoFrame:
    frame_index: int
    timestamp_sec: float
    image_bgr: Array
    score: float = 0.0


@dataclass
class SegmentationResult:
    food_mask: Array
    support_mask: Array
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]


@dataclass
class Mesh:
    vertices: Array
    faces: Array
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class StateReconstruction:
    state: str
    representative: VideoFrame
    segmentation: SegmentationResult
    keyframes: List[VideoFrame]
    scale_m_per_px: float
    path_a_mesh: Mesh
    path_b_mesh: Mesh
    path_a_volume_ml: float
    path_b_volume_ml: float
    coverage: float
    fused_volume_ml: Optional[float] = None
    final_mesh: Optional[Mesh] = None


@dataclass
class ItemResult:
    item: FoodItem
    before: StateReconstruction
    after: StateReconstruction
    consumption_ratio: float

