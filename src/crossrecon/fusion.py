from typing import Dict

from .utils import sigmoid


def path_a_weight(coverage: float, state: str, config: Dict[str, object]) -> float:
    params = config.get("fusion", {})
    gamma = float(params.get("gamma", 8.0))
    tau = float(params.get("tau", 0.6))
    beta = float(params.get("beta_after" if state == "after" else "beta_before", 1.0))
    alpha = sigmoid(gamma * (float(coverage) - tau)) * beta
    return max(0.0, min(1.0, alpha))


def fuse_volume_ml(
    path_a_volume_ml: float,
    path_b_volume_ml: float,
    coverage: float,
    state: str,
    before_path_b_volume_ml: float,
    config: Dict[str, object],
) -> float:
    alpha = path_a_weight(coverage=coverage, state=state, config=config)
    bounded_b = min(float(path_b_volume_ml), float(before_path_b_volume_ml))
    fused = alpha * float(path_a_volume_ml) + (1.0 - alpha) * bounded_b

    threshold = float(config.get("fusion", {}).get("discrepancy_threshold", 0.5))
    denom = max(abs(float(path_a_volume_ml)), 1e-6)
    discrepancy = abs(float(path_a_volume_ml) - float(path_b_volume_ml)) / denom
    if discrepancy > threshold and coverage >= 0.65:
        fused = float(path_a_volume_ml)
    return max(0.0, fused)


def consumption_ratio(before_volume_ml: float, after_volume_ml: float) -> float:
    if before_volume_ml <= 1e-9:
        return 0.0
    return max(0.0, min(1.0, (before_volume_ml - after_volume_ml) / before_volume_ml))

