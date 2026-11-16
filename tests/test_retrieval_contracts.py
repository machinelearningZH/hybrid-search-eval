from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import generate_evals


def capture_rankings(
    monkeypatch: pytest.MonkeyPatch,
) -> list[list[str]]:
    rankings: list[list[str]] = []
    compute_metrics = generate_evals.compute_metrics

    def capture(
        results: list[tuple[list[str], list[str]]], cutoffs: dict[str, list[int]]
    ) -> dict[str, float]:
        rankings.extend(retrieved for retrieved, _relevant in results)
        return compute_metrics(results, cutoffs)

    monkeypatch.setattr(generate_evals, "compute_metrics", capture)
    return rankings


@pytest.mark.parametrize("backend", ["huggingface", "colbert"])
def test_all_lexical_searches_restrict_matching_to_text(
    evaluation_runtime: SimpleNamespace,
    backend: str,
) -> None:
    runtime = evaluation_runtime
    if backend == "colbert":
        runtime.config["embeddings"]["colbert"] = runtime.config["embeddings"].pop(
            "huggingface"
        )
    runtime.config["search"]["include_bm25_baseline"] = True

    def search(**kwargs: Any) -> SimpleNamespace:
        runtime.searched.append(kwargs)
        return SimpleNamespace(
            objects=[
                SimpleNamespace(
                    properties={"document_id": "d1"},
                    metadata=SimpleNamespace(score=1.0),
                )
            ]
        )

    runtime.collection.query.hybrid = search
    runtime.collection.query.bm25 = search

    assert generate_evals.main() == 0
    assert runtime.searched
    assert all(call.get("query_properties") == ["text"] for call in runtime.searched)


@pytest.mark.parametrize(
    ("semantic", "lexical", "alpha", "expected"),
    [
        ([2.0, 2.0], [], 1.0, ["d1", "d2"]),
        ([2.0, 2.0], [], 0.5, ["d1", "d2"]),
        ([2.0, 2.0], [("d2", 7.0)], 0.5, ["d2", "d1"]),
        ([20.0, 10.0], [("d1", 100.0), ("d2", 10.0)], 0.75, ["d2", "d1"]),
        ([20.0, 10.0], [("d1", 100.0), ("d2", 10.0)], 0.25, ["d1", "d2"]),
        ([20.0, 10.0], [("d1", 7.0), ("d2", 7.0)], 0.5, ["d2", "d1"]),
    ],
)
def test_colbert_fusion_normalization_missing_candidates_and_ties(
    evaluation_runtime: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    semantic: list[float],
    lexical: list[tuple[str, float]],
    alpha: float,
    expected: list[str],
) -> None:
    runtime = evaluation_runtime
    runtime.config["embeddings"]["colbert"] = runtime.config["embeddings"].pop(
        "huggingface"
    )
    runtime.config["search"]["alpha"] = [alpha]
    runtime.rank.rerank = lambda **kwargs: [
        [{"id": "d2", "score": semantic[0]}, {"id": "d1", "score": semantic[1]}]
        for _query in kwargs["queries_embeddings"]
    ]
    runtime.collection.query.bm25 = lambda **_kwargs: SimpleNamespace(
        objects=[
            SimpleNamespace(
                properties={"document_id": document_id},
                metadata=SimpleNamespace(score=score),
            )
            for document_id, score in lexical
        ]
    )
    rankings = capture_rankings(monkeypatch)

    assert generate_evals.main() == 0
    assert rankings == [expected, expected]


@pytest.mark.parametrize(
    ("candidate_limit", "expected_limit"), [(None, 2), (10, 2), (1, 1)]
)
def test_colbert_fusion_bounds_bm25_candidates(
    evaluation_runtime: SimpleNamespace,
    candidate_limit: int | None,
    expected_limit: int,
) -> None:
    runtime = evaluation_runtime
    runtime.config["embeddings"]["colbert"] = runtime.config["embeddings"].pop(
        "huggingface"
    )
    runtime.config["search"]["alpha"] = [0.5]
    runtime.config["search"]["bm25_candidate_limit"] = candidate_limit

    def search(**kwargs: Any) -> SimpleNamespace:
        runtime.searched.append(kwargs)
        return SimpleNamespace(objects=[])

    runtime.collection.query.bm25 = search

    assert generate_evals.main() == 0
    assert [call["limit"] for call in runtime.searched] == [
        expected_limit,
        expected_limit,
    ]


@pytest.mark.parametrize(
    ("backend", "legacy_policy"),
    [("huggingface", "query_properties"), ("colbert", "colbert_tie_break")],
)
def test_updated_retrieval_policy_recomputes_legacy_results_but_reuses_embeddings(
    evaluation_runtime: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    legacy_policy: str,
) -> None:
    runtime = evaluation_runtime
    runtime.config["search"]["include_bm25_baseline"] = True
    if backend == "colbert":
        runtime.config["embeddings"]["colbert"] = runtime.config["embeddings"].pop(
            "huggingface"
        )
        runtime.config["search"]["alpha"] = [1.0]
    rankings = capture_rankings(monkeypatch)
    generate_cache_key = generate_evals.generate_eval_cache_key

    def old_retrieval_key(*args: Any, **kwargs: Any) -> str:
        identity = dict(kwargs["cache_identity"])
        identity.pop(legacy_policy, None)
        kwargs["cache_identity"] = identity
        return generate_cache_key(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(generate_evals, "generate_eval_cache_key", old_retrieval_key)
        assert generate_evals.main() == 0
    first_encoded = len(runtime.encoded)
    first_ranked = len(rankings)
    old_results = set(Path(runtime.config["evals"]["cache_dir"]).glob("*.json"))

    assert generate_evals.main() == 0
    assert len(runtime.encoded) == first_encoded
    assert len(rankings) > first_ranked
    assert old_results <= set(Path(runtime.config["evals"]["cache_dir"]).glob("*.json"))
