from typing import Any, Dict

from .base import BackendNotConfigured, GenerativeBackend, GeometryBackend, SegmentationBackend


class GroundingDINOSAM2Backend(SegmentationBackend):
    """Integration point for GroundingDINO + SAM2 video masks."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        if not config.get("weights"):
            raise BackendNotConfigured(
                "GroundingDINO/SAM2 weights are not configured. "
                "Set backends.segmentation.weights in configs/full_model_template.yaml."
            )

    def segment(self, image_bgr: Any, food_name: str) -> Any:
        raise NotImplementedError("Wire the official GroundingDINO + SAM2 inference code here.")


class Pi3GeometryBackend(GeometryBackend):
    """Integration point for Pi3 feed-forward video geometry."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        if not config.get("weights"):
            raise BackendNotConfigured(
                "Pi3 weights are not configured. Set backends.pi3.weights before using full mode."
            )

    def reconstruct(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Wire Pi3 pointmap/camera inference here.")


class GPTImageHunyuanBackend(GenerativeBackend):
    """Integration point for GPT Image multi-view generation + Hunyuan3D mesh decoding."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        if not config.get("hunyuan_weights"):
            raise BackendNotConfigured(
                "Hunyuan3D weights are not configured. Set backends.generative.hunyuan_weights."
            )

    def reconstruct(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Wire GPT Image view synthesis and Hunyuan3D reconstruction here.")

