from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np


@dataclass
class ScaleCandidate:
    category: str
    measured_value: float
    prior_mean: float
    prior_std: float
    dimension: int

    @property
    def linear_scale(self) -> float:
        if self.measured_value <= 0:
            return 1.0
        return float((self.prior_mean / self.measured_value) ** (1.0 / self.dimension))


def tukey_biweight_location(values: Iterable[float], c: float = 4.685) -> Optional[float]:
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return None
    logs = np.log(values)
    med = float(np.median(logs))
    mad = float(np.median(np.abs(logs - med))) + 1e-9
    u = (logs - med) / (c * 1.4826 * mad)
    weights = (1.0 - u**2) ** 2
    weights[np.abs(u) >= 1.0] = 0.0
    if float(weights.sum()) <= 1e-12:
        return float(np.exp(med))
    return float(np.exp(np.sum(weights * logs) / np.sum(weights)))


def vote_metric_scale(candidates: List[ScaleCandidate]) -> float:
    values = [candidate.linear_scale for candidate in candidates]
    scale = tukey_biweight_location(values)
    return 1.0 if scale is None else float(scale)

