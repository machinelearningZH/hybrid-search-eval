import pytest

from generate_evals import compute_metrics, get_max_k, get_metric_k_values


def test_get_metric_k_values_supports_independent_and_legacy_configuration() -> None:
    assert get_metric_k_values(
        {
            "search": {
                "top_k": [20],
                "metrics": {"mrr_k": [1, 5], "hit_rate_k": [10]},
            }
        }
    ) == {"mrr": [1, 5], "hit_rate": [10]}
    assert get_metric_k_values({"search": {"top_k": 7}}) == {
        "mrr": [7],
        "hit_rate": [7],
    }
    assert get_metric_k_values({}) == {"mrr": [10], "hit_rate": [10]}


def test_get_max_k_finds_maximum() -> None:
    assert get_max_k({"mrr": [1, 5], "hit_rate": [10]}) == 10


@pytest.mark.parametrize(
    "cutoffs",
    [
        {},
        {"mrr": [10]},
        {"mrr": [10], "hit_rate": []},
        {"mrr": "10", "hit_rate": [10]},
        {"mrr": [0], "hit_rate": [10]},
        {"mrr": [True], "hit_rate": [10]},
        {"mrr": [1.5], "hit_rate": [10]},
        {"mrr": [10], "hit_rate": [10], "ndcg": [10]},
    ],
)
def test_metric_helpers_reject_malformed_cutoff_dictionaries(
    cutoffs: object,
) -> None:
    with pytest.raises(ValueError, match="metric cutoffs"):
        get_max_k(cutoffs)  # type: ignore[arg-type]


def test_get_metric_k_values_rejects_malformed_config() -> None:
    with pytest.raises(ValueError, match="metric cutoffs"):
        get_metric_k_values({"search": {"metrics": {"mrr_k": [], "hit_rate_k": [10]}}})


def test_compute_metrics_respects_each_cutoff() -> None:
    retrieved = [
        (["d1", "d2", "d3"], ["d2"]),
        (["d4", "d5", "d6"], ["d6"]),
    ]

    metrics = compute_metrics(retrieved, {"mrr": [1, 3], "hit_rate": [2, 3]})

    assert metrics == pytest.approx(
        {
            "mrr@1": 0.0,
            "mrr@3": (1 / 2 + 1 / 3) / 2,
            "hit_rate@2": 0.5,
            "hit_rate@3": 1.0,
        }
    )


def test_compute_metrics_rejects_empty_query_results() -> None:
    with pytest.raises(ValueError, match="query result collection.*empty"):
        compute_metrics([], {"mrr": [10], "hit_rate": [10]})
