import json
from types import SimpleNamespace

import pandas as pd
import pytest

import download_mteb_datasets
from download_mteb_datasets import (
    DatasetConfigs,
    detect_dataset_structure,
    get_id_column,
    resolve_dataset_configs,
    sample_retrieval_frames,
    select_dataset_splits,
)
from list_retrieval_datasets import (
    _filter_tasks_by_type,
    _normalize_lang_code,
    _version_rank,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("de", "deu"),
        (" GER ", "deu"),
        ("zh", "cmn"),
        ("eng", "eng"),
        ("xx", "xx"),
    ],
)
def test_normalize_lang_code(value: str, expected: str) -> None:
    assert _normalize_lang_code(value) == expected


def test_version_rank_orders_supported_benchmark_versions() -> None:
    assert _version_rank("v3") > _version_rank("v2") > _version_rank("classic")
    assert _version_rank("classic") > _version_rank("unknown")
    assert _version_rank("preview") == 0


def test_filter_tasks_by_type_reads_dict_and_object_metadata() -> None:
    tasks = [
        SimpleNamespace(metadata={"type": "Retrieval"}),
        SimpleNamespace(metadata=SimpleNamespace(type="InstructionRetrieval")),
        SimpleNamespace(metadata={"type": "Classification"}),
    ]

    assert (
        _filter_tasks_by_type(tasks, ("Retrieval", "InstructionRetrieval")) == tasks[:2]
    )


def test_get_id_column_prefers_mteb_id_and_reports_missing_column() -> None:
    assert get_id_column(SimpleNamespace(column_names=["id", "_id"])) == "_id"
    assert get_id_column(SimpleNamespace(column_names=["id"])) == "id"
    with pytest.raises(ValueError, match="No ID column found"):
        get_id_column(SimpleNamespace(column_names=["text"]))


def test_detect_dataset_structure_recognizes_complete_language_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        download_mteb_datasets,
        "get_dataset_config_names",
        lambda _: [
            "en-corpus",
            "en-queries",
            "en-qrels",
            "de-corpus",
            "de-queries",
            "incomplete-corpus",
        ],
    )

    assert detect_dataset_structure("example/dataset") == (True, ["en"])


def test_detect_dataset_structure_handles_standard_and_failed_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        download_mteb_datasets,
        "get_dataset_config_names",
        lambda _: ["corpus", "queries", "qrels"],
    )
    assert detect_dataset_structure("example/standard") == (False, [])

    def raise_error(_: str) -> list[str]:
        raise RuntimeError("offline")

    monkeypatch.setattr(download_mteb_datasets, "get_dataset_config_names", raise_error)
    assert detect_dataset_structure("example/unavailable") == (False, [])


def test_resolve_dataset_configs_supports_standard_and_multilingual_layouts() -> None:
    assert resolve_dataset_configs(
        ["corpus", "queries", "qrels"], language=None
    ) == DatasetConfigs("corpus", "queries", "qrels", None)
    assert resolve_dataset_configs(
        ["en-corpus", "en-queries", "en-qrels"], language="en"
    ) == DatasetConfigs("en-corpus", "en-queries", "en-qrels", "en")


def test_resolve_dataset_configs_rejects_missing_qrels_and_ambiguous_language() -> None:
    with pytest.raises(ValueError, match="qrels configuration"):
        resolve_dataset_configs(["corpus", "queries"], language=None)
    with pytest.raises(ValueError, match="multiple languages.*--language"):
        resolve_dataset_configs(
            [
                "de-corpus",
                "de-queries",
                "de-qrels",
                "en-corpus",
                "en-queries",
                "en-qrels",
            ],
            language=None,
        )


def test_select_dataset_splits_uses_one_explicit_evaluation_split() -> None:
    selected = select_dataset_splits(
        corpus_datasets={"corpus": "corpus-data"},
        queries_datasets={"train": "train-queries", "test": "test-queries"},
        qrels_datasets={"train": "train-qrels", "test": "test-qrels"},
        evaluation_split="test",
    )

    assert selected.corpus == "corpus-data"
    assert selected.queries == "test-queries"
    assert selected.qrels == "test-qrels"
    assert (selected.corpus_split, selected.query_split, selected.qrels_split) == (
        "corpus",
        "test",
        "test",
    )


def test_select_dataset_splits_rejects_ambiguous_qrels() -> None:
    with pytest.raises(ValueError, match="multiple evaluation splits.*--split"):
        select_dataset_splits(
            corpus_datasets={"corpus": "corpus-data"},
            queries_datasets={"queries": "queries-data"},
            qrels_datasets={"train": "train-qrels", "test": "test-qrels"},
            evaluation_split=None,
        )


def _retrieval_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    corpus = pd.DataFrame(
        [{"id": f"d{i}", "text": f"document {i}"} for i in range(1, 13)]
    )
    queries = pd.DataFrame([{"id": f"q{i}", "text": f"query {i}"} for i in range(1, 7)])
    qrels = pd.DataFrame(
        [{"query-id": f"q{i}", "corpus-id": f"d{i}", "score": 1} for i in range(1, 7)]
    )
    return corpus, queries, qrels


def test_query_led_sampling_is_deterministic_and_has_no_orphans() -> None:
    corpus, queries, qrels = _retrieval_frames()

    first = sample_retrieval_frames(corpus, queries, qrels, corpus_size=5, seed=7)
    repeated = sample_retrieval_frames(corpus, queries, qrels, corpus_size=5, seed=7)
    other_seed = sample_retrieval_frames(corpus, queries, qrels, corpus_size=5, seed=8)

    assert first.corpus["id"].tolist() == repeated.corpus["id"].tolist()
    assert first.queries["id"].tolist() == repeated.queries["id"].tolist()
    assert first.queries["id"].tolist() != other_seed.queries["id"].tolist()
    assert len(first.queries) > 0
    assert set(first.qrels["query-id"]) <= set(first.queries["id"])
    assert set(first.qrels["corpus-id"]) <= set(first.corpus["id"])
    assert set(first.queries["id"]) != set(queries.head(5)["id"])


def test_query_led_sampling_keeps_all_positives_even_above_requested_size() -> None:
    corpus = pd.DataFrame(
        [{"id": f"d{i}", "text": f"document {i}"} for i in range(1, 5)]
    )
    queries = pd.DataFrame([{"id": "q1", "text": "query"}])
    qrels = pd.DataFrame(
        [{"query-id": "q1", "corpus-id": f"d{i}", "score": 1} for i in range(1, 4)]
    )

    sampled = sample_retrieval_frames(corpus, queries, qrels, corpus_size=2, seed=1)

    assert sampled.corpus["id"].tolist() == ["d1", "d2", "d3"]
    assert sampled.stats["requested_corpus_size"] == 2
    assert sampled.stats["sampled_corpus_size"] == 3
    assert sampled.stats["positive_document_coverage"] == 1.0


class FakeDataset:
    def __init__(self, rows: list[dict[str, object]], fingerprint: str):
        self._frame = pd.DataFrame(rows)
        self._fingerprint = fingerprint
        self.column_names = self._frame.columns.tolist()

    def to_pandas(self) -> pd.DataFrame:
        return self._frame.copy()


def test_download_writes_selection_and_sampling_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    corpus, queries, qrels = _retrieval_frames()
    datasets = {
        "corpus": {"corpus": FakeDataset(corpus.to_dict("records"), "corpus-fp")},
        "queries": {"queries": FakeDataset(queries.to_dict("records"), "queries-fp")},
        "qrels": {"test": FakeDataset(qrels.to_dict("records"), "qrels-fp")},
    }
    monkeypatch.setattr(
        download_mteb_datasets,
        "get_dataset_config_names",
        lambda *_args, **_kwargs: list(datasets),
    )
    monkeypatch.setattr(
        download_mteb_datasets,
        "load_dataset",
        lambda _name, name, revision: datasets[name],
    )

    dataset_dir = download_mteb_datasets.download_mteb_dataset(
        "example/retrieval",
        tmp_path,
        sample_size=5,
        seed=17,
        revision="abc123",
    )

    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    assert manifest["repository"] == "example/retrieval"
    assert manifest["revision"] == "abc123"
    assert manifest["configurations"] == {
        "corpus": "corpus",
        "queries": "queries",
        "qrels": "qrels",
    }
    assert manifest["splits"] == {
        "corpus": "corpus",
        "queries": "queries",
        "qrels": "test",
        "evaluation": "test",
    }
    assert manifest["fingerprints"] == {
        "corpus": "corpus-fp",
        "queries": "queries-fp",
        "qrels": "qrels-fp",
    }
    assert manifest["sampling"]["seed"] == 17
