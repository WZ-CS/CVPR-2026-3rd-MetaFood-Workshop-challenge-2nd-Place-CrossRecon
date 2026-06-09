from dataclasses import dataclass
from typing import Any, Dict

from ..segmentation import LiteSegmenter
from .base import BackendBundle, GenerativeBackend, GeometryBackend


@dataclass
class LiteGeometryBackend(GeometryBackend):
    config: Dict[str, Any]


@dataclass
class LiteGenerativeBackend(GenerativeBackend):
    config: Dict[str, Any]


def build_lite_backends(config: Dict[str, Any]) -> BackendBundle:
    return BackendBundle(
        segmenter=LiteSegmenter(config.get("lite_backend", {})),
        geometry=LiteGeometryBackend(config=config),
        generator=LiteGenerativeBackend(config=config),
    )

