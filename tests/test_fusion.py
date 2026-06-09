from crossrecon.fusion import consumption_ratio, fuse_volume_ml, path_a_weight


CONFIG = {
    "fusion": {
        "gamma": 8.0,
        "tau": 0.6,
        "beta_before": 1.0,
        "beta_after": 1.3,
        "discrepancy_threshold": 0.5,
    }
}


def test_after_state_trusts_path_a_more():
    before = path_a_weight(0.7, "before", CONFIG)
    after = path_a_weight(0.7, "after", CONFIG)
    assert after >= before
    assert 0.0 <= after <= 1.0


def test_fusion_uses_path_a_when_discrepant_and_visible():
    fused = fuse_volume_ml(50.0, 160.0, 0.8, "after", 180.0, CONFIG)
    assert fused == 50.0


def test_consumption_ratio_is_bounded():
    assert consumption_ratio(100.0, 40.0) == 0.6
    assert consumption_ratio(100.0, 140.0) == 0.0

