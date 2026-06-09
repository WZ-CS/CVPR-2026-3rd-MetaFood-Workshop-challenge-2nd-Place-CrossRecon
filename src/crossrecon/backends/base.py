from dataclasses import dataclass
from typing import Any


class BackendNotConfigured(RuntimeError):
    pass


class SegmentationBackend:
    def segment(self, image_bgr: Any, food_name: str) -> Any:
        raise NotImplementedError


class GeometryBackend:
    def reconstruct(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class GenerativeBackend:
    def reconstruct(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


@dataclass
class BackendBundle:
    segmenter: SegmentationBackend
    geometry: GeometryBackend
    generator: GenerativeBackend

