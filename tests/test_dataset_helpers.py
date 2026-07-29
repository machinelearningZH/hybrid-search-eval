from types import SimpleNamespace

import pytest

import download_mteb_datasets
from download_mteb_datasets import detect_dataset_structure, get_id_column
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
