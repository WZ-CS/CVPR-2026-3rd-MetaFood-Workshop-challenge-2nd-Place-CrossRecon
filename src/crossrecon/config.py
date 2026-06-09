from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .utils import deep_update


DEFAULT_CONFIG: Dict[str, Any] = {
    "pipeline": {
        "mode": "lite",
        "frame_max_size": 512,
        "candidate_frames_per_state": 16,
        "keyframes_per_state": 8,
        "before_window": [0.0, 0.25],
        "after_window": [0.75, 1.0],
        "grid_size": 96,
        "plate_diameter_cm": 20.0,
        "debug_visualizations": True,
    },
    "fusion": {
        "gamma": 8.0,
        "tau": 0.6,
        "beta_before": 1.0,
        "beta_after": 1.3,
        "discrepancy_threshold": 0.5,
    },
    "lite_backend": {
        "min_food_area_ratio": 0.015,
        "completion_bias_after": 0.65,
        "path_b_prior_weight": 0.65,
    },
    "food_priors": {
        "default": {"volume_ml": 180.0, "height_cm": 3.0},
    },
}


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    config = yaml.safe_load(yaml.safe_dump(DEFAULT_CONFIG))
    if path:
        with open(path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
        deep_update(config, user_config)
    return config


def save_config(config: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

