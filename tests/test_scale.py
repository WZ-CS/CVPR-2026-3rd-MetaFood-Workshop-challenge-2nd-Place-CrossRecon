from crossrecon.scale import ScaleCandidate, vote_metric_scale


def test_scale_vote_rejects_large_outlier():
    candidates = [
        ScaleCandidate("plate", measured_value=10.0, prior_mean=20.0, prior_std=2.0, dimension=1),
        ScaleCandidate("fork", measured_value=9.5, prior_mean=19.0, prior_std=2.0, dimension=1),
        ScaleCandidate("bad", measured_value=1.0, prior_mean=80.0, prior_std=10.0, dimension=1),
    ]
    scale = vote_metric_scale(candidates)
    assert 1.5 < scale < 2.5

