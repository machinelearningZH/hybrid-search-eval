from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from _core.utils import (
    MTEBRetrievalData,
    calculate_hit,
    calculate_pareto_flags,
    calculate_reciprocal_rank,
    colbert_embeddings_exist,
    create_html_dashboard,
    create_tradeoff_visualization,
    embeddings_exist,
    eval_results_exist,
    generate_cache_key,
    generate_eval_cache_key,
    load_colbert_embeddings,
    load_embeddings,
    load_eval_results,
    load_mteb_retrieval_data_from_dir,
    parse_model_configs,
    repair_snowflake_position_ids,
    save_colbert_embeddings,
    save_embeddings,
    save_eval_results,
    select_pareto_quality_metric,
    validate_config,
)


def test_parse_model_configs_preserves_provider_behavior_and_unique_names() -> None:
    config = {
        "embeddings": {
            "huggingface": {
                "shared": {
                    "model": "intfloat/multilingual-e5-small",
                    "use_query_prefix": True,
                    "use_passage_prefix": True,
                    "query_prompt_name": "search_query",
                    "passage_prompt_name": "search_document",
                }
            },
            "colbert": {"shared": "answerdotai/answerai-colbert-small-v1"},
            "openrouter": {
                "models": {
                    "shared": "openai/text-embedding-3-small",
                    "remote": "qwen/qwen3-embedding-8b",
                }
            },
        }
    }

    model_configs = parse_model_configs(config)

    assert [model["model_name"] for model in model_configs] == [
        "shared_hf",
        "shared_colbert",
        "shared_or",
        "remote",
    ]
    assert [
        (model["is_openrouter"], model["is_colbert"]) for model in model_configs
    ] == [(False, False), (False, True), (True, False), (True, False)]
    assert model_configs[0]["cache_identity"] == {
        "query_prefix": "query: ",
        "query_encode_kwargs": {"prompt_name": "search_query"},
        "passage_prefix": "passage: ",
        "passage_encode_kwargs": {"prompt_name": "search_document"},
    }


@pytest.mark.parametrize("invalid_value", [123, "", "   ", None])
def test_validate_config_rejects_invalid_prompt_names(
    valid_config: dict[str, Any], invalid_value: object
) -> None:
    config = deepcopy(valid_config)
    config["embeddings"]["huggingface"]["mini"] = {
        "model": "microsoft/harrier-oss-v1-270m",
        "query_prompt_name": invalid_value,
    }

    errors = validate_config(config)

    assert len(errors) == 1
    assert "mini.query_prompt_name" in errors[0]


def test_validate_config_accepts_complete_config(
    valid_config: dict[str, Any],
) -> None:
    assert validate_config(valid_config) == []


def test_validate_config_rejects_obsolete_visualization_keys(
    valid_config: dict[str, Any],
) -> None:
    config = deepcopy(valid_config)
    config["visualization"] = {"recall_dynamic_xlim": True}

    errors = validate_config(config)

    assert len(errors) == 1
    assert "metric_dynamic_xlim" in errors[0]


def test_validate_config_rejects_unavailable_pareto_metric(
    valid_config: dict[str, Any],
) -> None:
    config = deepcopy(valid_config)
    config["visualization"] = {"pareto_quality_metric": "mrr@100"}

    errors = validate_config(config)

    assert len(errors) == 1
    assert "configured search metric" in errors[0]


def test_embedding_cache_key_is_stable_and_input_sensitive() -> None:
    identity = {
        "query_encode_kwargs": {"prompt_name": "search"},
        "query_prefix": "query: ",
    }

    key = generate_cache_key("project", "org/model", "queries", identity)

    assert key == generate_cache_key(
        "project",
        "org/model",
        "queries",
        dict(reversed(identity.items())),
    )
    assert key != generate_cache_key("project", "org/model", "documents", identity)
    assert key != generate_cache_key("project", "org/model", "queries", {})


def test_eval_cache_key_normalizes_metric_and_identity_order() -> None:
    metrics = {"mrr": [10, 1], "hit_rate": [10]}
    identity = {
        "query_encode_kwargs": {"prompt_name": "search"},
        "query_prefix": "query: ",
    }

    key = generate_eval_cache_key(
        "project", "org/model", 0.5, 10, "model", metrics, identity
    )

    assert key == generate_eval_cache_key(
        "project",
        "org/model",
        0.5,
        10,
        "model",
        {"hit_rate": [10], "mrr": [1, 10]},
        dict(reversed(identity.items())),
    )
    assert key != generate_eval_cache_key(
        "project", "org/model", 0.7, 10, "model", metrics, identity
    )
    assert key != generate_eval_cache_key(
        "project", "org/model", 0.5, 10, "other", metrics, identity
    )


def test_mteb_data_builds_string_keyed_views_and_filters_negative_qrels() -> None:
    data = MTEBRetrievalData(
        corpus=pd.DataFrame(
            [
                {"id": 10, "title": "First", "text": "alpha"},
                {"id": 20, "title": "Second", "text": "beta"},
            ]
        ),
        queries=pd.DataFrame([{"id": 1, "text": "find alpha"}]),
        qrels=pd.DataFrame(
            [
                {"query-id": 1, "corpus-id": 10, "score": 2},
                {"query-id": 1, "corpus-id": 20, "score": 0},
            ]
        ),
    )

    assert data.corpus_dict["10"] == {
        "id": "10",
        "text": "alpha",
        "title": "First",
    }
    assert data.get_documents_list() == [
        {"id": "10", "text": "alpha"},
        {"id": "20", "text": "beta"},
    ]
    assert data.get_queries_list() == [
        {"id": "1", "query": "find alpha", "relevant_ids": ["10"]}
    ]
    assert (data.num_documents, data.num_queries, data.num_qrels) == (2, 1, 2)


def test_mteb_data_defaults_missing_relevance_scores() -> None:
    qrels = pd.DataFrame([{"query-id": "q1", "corpus-id": "d1"}])

    data = MTEBRetrievalData(
        corpus=pd.DataFrame([{"id": "d1", "text": "document"}]),
        queries=pd.DataFrame([{"id": "q1", "text": "query"}]),
        qrels=qrels,
    )

    assert "score" not in qrels.columns
    assert data.qrels["score"].tolist() == [1]
    assert data.get_queries_list()[0]["relevant_ids"] == ["d1"]


@pytest.mark.parametrize(
    ("table", "corpus", "queries", "qrels"),
    [
        (
            "corpus",
            pd.DataFrame(columns=["id", "text"]),
            pd.DataFrame([{"id": "q1", "text": "query"}]),
            pd.DataFrame([{"query-id": "q1", "corpus-id": "d1"}]),
        ),
        (
            "queries",
            pd.DataFrame([{"id": "d1", "text": "document"}]),
            pd.DataFrame(columns=["id", "text"]),
            pd.DataFrame([{"query-id": "q1", "corpus-id": "d1"}]),
        ),
        (
            "qrels",
            pd.DataFrame([{"id": "d1", "text": "document"}]),
            pd.DataFrame([{"id": "q1", "text": "query"}]),
            pd.DataFrame(columns=["query-id", "corpus-id"]),
        ),
    ],
)
def test_mteb_data_rejects_empty_tables(
    table: str,
    corpus: pd.DataFrame,
    queries: pd.DataFrame,
    qrels: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match=rf"{table}.*empty"):
        MTEBRetrievalData(corpus, queries, qrels)


@pytest.mark.parametrize("table", ["corpus", "queries"])
@pytest.mark.parametrize("bad_text", [None, 12])
def test_mteb_data_rejects_null_or_non_string_text(
    table: str, bad_text: object
) -> None:
    corpus = pd.DataFrame([{"id": "d1", "text": "document"}])
    queries = pd.DataFrame([{"id": "q1", "text": "query"}])
    if table == "corpus":
        corpus.loc[0, "text"] = bad_text
    else:
        queries.loc[0, "text"] = bad_text

    with pytest.raises(ValueError, match=rf"{table}.*text.*1"):
        MTEBRetrievalData(
            corpus,
            queries,
            pd.DataFrame([{"query-id": "q1", "corpus-id": "d1"}]),
        )


@pytest.mark.parametrize("table", ["corpus", "queries"])
def test_mteb_data_rejects_duplicate_ids_after_string_normalization(
    table: str,
) -> None:
    corpus = pd.DataFrame([{"id": "d1", "text": "document"}])
    queries = pd.DataFrame([{"id": "q1", "text": "query"}])
    if table == "corpus":
        corpus = pd.DataFrame(
            [{"id": 1, "text": "first"}, {"id": "1", "text": "second"}]
        )
        corpus_id = 1
    else:
        queries = pd.DataFrame(
            [{"id": 1, "text": "first"}, {"id": "1", "text": "second"}]
        )
        corpus_id = "d1"

    with pytest.raises(ValueError, match=rf"{table}.*id.*duplicate.*1"):
        MTEBRetrievalData(
            corpus,
            queries,
            pd.DataFrame([{"query-id": 1, "corpus-id": corpus_id}]),
        )


@pytest.mark.parametrize("score", ["high", float("nan"), float("inf")])
def test_mteb_data_rejects_non_numeric_or_non_finite_scores(score: object) -> None:
    with pytest.raises(ValueError, match=r"qrels.*score.*1"):
        MTEBRetrievalData(
            pd.DataFrame([{"id": "d1", "text": "document"}]),
            pd.DataFrame([{"id": "q1", "text": "query"}]),
            pd.DataFrame(
                [{"query-id": "q1", "corpus-id": "d1", "score": score}]
            ),
        )


@pytest.mark.parametrize(
    ("qrel_field", "qrel_id", "missing_id"),
    [("query-id", "missing-query", "missing-query"), ("corpus-id", "missing-doc", "missing-doc")],
)
def test_mteb_data_rejects_qrels_with_missing_references(
    qrel_field: str, qrel_id: str, missing_id: str
) -> None:
    qrel = {"query-id": "q1", "corpus-id": "d1", "score": 1}
    qrel[qrel_field] = qrel_id

    with pytest.raises(ValueError, match=rf"qrels.*{qrel_field}.*{missing_id}"):
        MTEBRetrievalData(
            pd.DataFrame([{"id": "d1", "text": "document"}]),
            pd.DataFrame([{"id": "q1", "text": "query"}]),
            pd.DataFrame([qrel]),
        )


def test_mteb_data_rejects_queries_without_positive_qrels() -> None:
    with pytest.raises(ValueError, match=r"queries.*positive qrels.*q2"):
        MTEBRetrievalData(
            pd.DataFrame([{"id": "d1", "text": "document"}]),
            pd.DataFrame(
                [{"id": "q1", "text": "one"}, {"id": "q2", "text": "two"}]
            ),
            pd.DataFrame(
                [
                    {"query-id": "q1", "corpus-id": "d1", "score": 2.5},
                    {"query-id": "q2", "corpus-id": "d1", "score": 0},
                ]
            ),
        )


def test_mteb_data_accepts_binary_and_graded_qrels() -> None:
    binary = MTEBRetrievalData(
        pd.DataFrame([{"id": "d1", "text": "document"}]),
        pd.DataFrame([{"id": "q1", "text": "query"}]),
        pd.DataFrame([{"query-id": "q1", "corpus-id": "d1"}]),
    )
    graded = MTEBRetrievalData(
        pd.DataFrame([{"id": "d1", "text": "document"}]),
        pd.DataFrame([{"id": "q1", "text": "query"}]),
        pd.DataFrame([{"query-id": "q1", "corpus-id": "d1", "score": 2.5}]),
    )

    assert binary.qrels_dict == {"q1": {"d1": 1}}
    assert graded.qrels_dict == {"q1": {"d1": 2.5}}


@pytest.mark.parametrize(
    ("corpus", "queries", "qrels", "message"),
    [
        (
            pd.DataFrame({"id": ["d1"]}),
            pd.DataFrame({"id": ["q1"], "text": ["query"]}),
            pd.DataFrame({"query-id": ["q1"], "corpus-id": ["d1"]}),
            "Corpus must have",
        ),
        (
            pd.DataFrame({"id": ["d1"], "text": ["document"]}),
            pd.DataFrame({"text": ["query"]}),
            pd.DataFrame({"query-id": ["q1"], "corpus-id": ["d1"]}),
            "Queries must have",
        ),
        (
            pd.DataFrame({"id": ["d1"], "text": ["document"]}),
            pd.DataFrame({"id": ["q1"], "text": ["query"]}),
            pd.DataFrame({"query-id": ["q1"]}),
            "Qrels must have",
        ),
    ],
)
def test_mteb_data_rejects_missing_required_columns(
    corpus: pd.DataFrame,
    queries: pd.DataFrame,
    qrels: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MTEBRetrievalData(corpus, queries, qrels)


def test_load_mteb_data_from_directory(tmp_path: Path) -> None:
    pd.DataFrame([{"id": "d1", "text": "document"}]).to_parquet(
        tmp_path / "corpus.parquet"
    )
    pd.DataFrame([{"id": "q1", "text": "query"}]).to_parquet(
        tmp_path / "queries.parquet"
    )
    pd.DataFrame([{"query-id": "q1", "corpus-id": "d1"}]).to_parquet(
        tmp_path / "qrels.parquet"
    )

    data = load_mteb_retrieval_data_from_dir(str(tmp_path))

    assert data.get_queries_list() == [
        {"id": "q1", "query": "query", "relevant_ids": ["d1"]}
    ]


def test_embedding_cache_round_trips_array_and_metadata(tmp_path: Path) -> None:
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    metadata = {"model": "example", "count": 2}

    saved_path = save_embeddings(embeddings, "cache", tmp_path, metadata)

    loaded_embeddings, loaded_metadata = load_embeddings("cache", tmp_path)
    assert saved_path == tmp_path / "cache.npy"
    assert embeddings_exist("cache", tmp_path)
    np.testing.assert_array_equal(loaded_embeddings, embeddings)
    assert loaded_metadata == metadata


def test_colbert_cache_round_trips_variable_length_arrays(tmp_path: Path) -> None:
    embeddings = [
        np.ones((2, 3), dtype=np.float32),
        np.zeros((4, 3), dtype=np.float32),
    ]

    save_colbert_embeddings(embeddings, "cache", tmp_path, {"model": "colbert"})

    loaded, metadata = load_colbert_embeddings("cache", tmp_path)
    assert colbert_embeddings_exist("cache", tmp_path)
    assert [array.shape for array in loaded] == [(2, 3), (4, 3)]
    assert metadata == {"model": "colbert", "is_colbert": True}


def test_eval_cache_round_trips_results(tmp_path: Path) -> None:
    results = {"mrr@10": 0.75, "hit_rate@10": 1.0}

    save_eval_results(results, "cache", tmp_path)

    assert eval_results_exist("cache", tmp_path)
    assert load_eval_results("cache", tmp_path) == results


@pytest.mark.parametrize(
    ("retrieved", "relevant", "expected"),
    [
        (["d1", "d2"], ["d1"], 1.0),
        (["d1", "d2", "d3"], ["d2", "d3"], 0.5),
        (["d1"], ["missing"], 0.0),
        ([], ["d1"], 0.0),
    ],
)
def test_calculate_reciprocal_rank(
    retrieved: list[str], relevant: list[str], expected: float
) -> None:
    assert calculate_reciprocal_rank(retrieved, relevant) == expected


@pytest.mark.parametrize(
    ("retrieved", "relevant", "expected"),
    [
        (["d1", "d2"], ["d2"], 1),
        (["d1"], ["missing"], 0),
        ([], ["d1"], 0),
    ],
)
def test_calculate_hit(
    retrieved: list[str], relevant: list[str], expected: int
) -> None:
    assert calculate_hit(retrieved, relevant) == expected


def test_select_pareto_quality_metric_honors_config_and_uses_numeric_cutoffs() -> None:
    results = [
        {"mrr@3": 0.5, "mrr@10": 0.6, "hit_rate@100": 0.8},
    ]

    assert select_pareto_quality_metric(results, {}) == "mrr@10"
    assert (
        select_pareto_quality_metric(
            results, {"visualization": {"pareto_quality_metric": "hit_rate@100"}}
        )
        == "hit_rate@100"
    )

    with pytest.raises(ValueError, match="pareto_quality_metric"):
        select_pareto_quality_metric(
            results, {"visualization": {"pareto_quality_metric": "ndcg@10"}}
        )


def test_calculate_pareto_flags_uses_quality_and_embedding_latency() -> None:
    results = [
        {
            "model": "fast",
            "model_short": "fast",
            "mrr@10": 0.70,
            "avg_embed_time_ms": 2.0,
        },
        {
            "model": "slow-dominated",
            "model_short": "slow-dominated",
            "mrr@10": 0.60,
            "avg_embed_time_ms": 4.0,
        },
        {
            "model": "accurate",
            "model_short": "accurate",
            "mrr@10": 0.85,
            "avg_embed_time_ms": 6.0,
        },
        {
            "model": "BM25",
            "model_short": "Baseline_BM25",
            "mrr@10": 0.65,
            "avg_embed_time_ms": 0.0,
        },
        {
            "model": "missing",
            "model_short": "missing",
            "mrr@10": None,
            "avg_embed_time_ms": 3.0,
        },
    ]

    assert calculate_pareto_flags(results, "mrr@10") == [
        True,
        False,
        True,
        None,
        None,
    ]


def test_calculate_pareto_flags_keeps_tied_configurations() -> None:
    results = [
        {"model": "a", "mrr@10": 0.7, "avg_embed_time_ms": 2.0},
        {"model": "b", "mrr@10": 0.7, "avg_embed_time_ms": 2.0},
    ]

    assert calculate_pareto_flags(results, "mrr@10") == [True, True]


def test_dashboard_uses_shared_pareto_flags_and_preserves_unknown_costs(
    tmp_path: Path,
) -> None:
    results = [
        {
            "model": "fast",
            "model_short": "fast",
            "alpha": 1.0,
            "mrr@10": 0.8,
            "avg_embed_time_ms": 2.0,
            "total_embed_time_ms": 20.0,
            "num_queries": 2,
            "num_documents": 10,
        },
        {
            "model": "slow",
            "model_short": "slow",
            "alpha": 1.0,
            "mrr@10": 0.7,
            "avg_embed_time_ms": 4.0,
            "total_embed_time_ms": 40.0,
            "num_queries": 2,
            "num_documents": 10,
        },
        {
            "model": "BM25",
            "model_short": "Baseline_BM25",
            "alpha": 0.0,
            "mrr@10": 0.6,
            "avg_embed_time_ms": 0.0,
            "total_embed_time_ms": 0.0,
            "num_queries": 2,
            "num_documents": 10,
        },
    ]

    create_html_dashboard(
        results,
        memory_data={},
        output_dir=tmp_path,
        timestamp="20260101_120000",
        config={"project_id": "test", "visualization": {}},
    )

    dashboard = (tmp_path / "dashboard_20260101_120000.html").read_text()
    assert dashboard.count('"is_pareto": true') == 1
    assert dashboard.count('"is_pareto": false') == 1
    assert dashboard.count('"is_pareto": null') == 1
    assert dashboard.count('"memory_mb": null') == 3
    assert '"model": "BM25"' in dashboard
    assert '"avg_embed_time_ms": null' in dashboard


def test_tradeoff_chart_supports_missing_memory_and_excludes_bm25_cost(
    tmp_path: Path,
) -> None:
    results = [
        {
            "model": "local",
            "model_short": "local",
            "alpha": 1.0,
            "mrr@10": 0.8,
            "avg_embed_time_ms": 2.0,
            "total_embed_time_ms": 20.0,
            "num_queries": 2,
            "num_documents": 10,
        },
        {
            "model": "api",
            "model_short": "api",
            "alpha": 0.5,
            "mrr@10": 0.75,
            "avg_embed_time_ms": 3.0,
            "total_embed_time_ms": 30.0,
            "num_queries": 2,
            "num_documents": 10,
        },
        {
            "model": "BM25",
            "model_short": "Baseline_BM25",
            "alpha": 0.0,
            "mrr@10": 0.7,
            "avg_embed_time_ms": 0.0,
            "total_embed_time_ms": 0.0,
            "num_queries": 2,
            "num_documents": 10,
        },
    ]

    create_tradeoff_visualization(
        results,
        memory_data={"local": {"peak_memory_mb": 100.0}},
        output_dir=tmp_path,
        timestamp="20260101_120000",
        config={"visualization": {"pareto_quality_metric": "mrr@10"}},
    )

    assert (tmp_path / "tradeoff_20260101_120000.png").is_file()


class _FakeEmbeddings(torch.nn.Module):
    def __init__(self, position_ids: object) -> None:
        super().__init__()
        if isinstance(position_ids, torch.Tensor):
            self.register_buffer("position_ids", position_ids, persistent=False)
        else:
            self.position_ids = position_ids


class _FakeSentenceTransformer:
    def __init__(self, position_ids: object) -> None:
        embeddings = _FakeEmbeddings(position_ids)
        auto_model = SimpleNamespace(embeddings=embeddings)
        self.module = SimpleNamespace(auto_model=auto_model)

    def _first_module(self) -> object:
        return self.module


def test_repair_snowflake_position_ids_resets_corrupted_buffer() -> None:
    model = _FakeSentenceTransformer(
        torch.tensor([0, 4320601048, -1], dtype=torch.long)
    )

    repaired = repair_snowflake_position_ids(
        model,
        "Snowflake/snowflake-arctic-embed-m-v2.0",
    )

    assert repaired is True
    assert model.module.auto_model.embeddings.position_ids.tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    ("model_id", "position_ids"),
    [
        ("sentence-transformers/all-MiniLM-L6-v2", torch.tensor([5, -1])),
        ("Snowflake/model", torch.arange(3)),
        ("Snowflake/model", torch.ones((2, 2))),
        ("Snowflake/model", [0, 1]),
    ],
)
def test_repair_snowflake_position_ids_leaves_inapplicable_buffers_unchanged(
    model_id: str, position_ids: object
) -> None:
    model = _FakeSentenceTransformer(position_ids)

    assert repair_snowflake_position_ids(model, model_id) is False
