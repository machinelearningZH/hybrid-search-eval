import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pandas as pd
import pytest
from openai import APIConnectionError

import generate_queries
from generate_queries import generate_queries_for_document, load_documents_from_file


@pytest.mark.parametrize("error", [RuntimeError("bug"), TypeError("bug")])
def test_generate_queries_propagates_programming_errors(error: Exception) -> None:
    def fail(**kwargs: Any) -> object:
        raise error

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fail))
    )
    with pytest.raises(type(error), match="bug"):
        generate_queries_for_document(client, "document", 1, "test", 2, 100, 0.2, 50)


def test_generate_queries_retries_api_connection_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    waits: list[int] = []

    def fail_once(**kwargs: Any) -> object:
        calls.append(kwargs)
        if len(calls) == 1:
            raise APIConnectionError(
                request=httpx.Request("POST", "https://example.test")
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="query"))]
        )

    monkeypatch.setattr(generate_queries.time, "sleep", waits.append)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fail_once))
    )
    assert generate_queries_for_document(
        client, "document", 1, "test", 2, 100, 0.2, 50
    ) == ["query"]
    assert len(calls) == 2
    assert waits == [1]


class _FakeCompletions:
    def __init__(self, content: str | None) -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_load_documents_from_csv_generates_string_ids_and_selects_columns(
    tmp_path: Path,
) -> None:
    path = tmp_path / "documents.csv"
    pd.DataFrame({"text": ["first", "second"], "ignored": ["x", "y"]}).to_csv(
        path, index=False
    )

    documents = load_documents_from_file(path)

    assert documents.to_dict("records") == [
        {"id": "doc_0", "text": "first"},
        {"id": "doc_1", "text": "second"},
    ]


def test_load_documents_rejects_unsupported_or_incomplete_input(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_documents_from_file(tmp_path / "documents.txt")
    with pytest.raises(ValueError, match="Supported formats: .csv, .parquet, .pq"):
        load_documents_from_file(tmp_path / "documents.xlsx")

    path = tmp_path / "documents.csv"
    pd.DataFrame({"body": ["missing text column"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="must have a 'text' column"):
        load_documents_from_file(path)


def test_load_documents_rejects_null_ids(tmp_path: Path) -> None:
    path = tmp_path / "documents.csv"
    pd.DataFrame({"id": [None], "text": ["document"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="non-null"):
        load_documents_from_file(path)


def test_generate_queries_parses_numbering_blanks_and_requested_limit() -> None:
    completions = _FakeCompletions(
        "\n1. first search\n- second search\n\n3. third search\n4. ignored\n"
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )

    queries = generate_queries_for_document(
        client=client,
        document_text="A source document",
        num_queries=3,
        model="test-model",
        max_retries=1,
        max_document_tokens=100,
        temperature=0.2,
        max_output_tokens=50,
    )

    assert queries == ["first search", "second search", "third search"]
    assert completions.calls[0]["model"] == "test-model"
    assert completions.calls[0]["temperature"] == 0.2
    assert "A source document" in completions.calls[0]["messages"][1]["content"]


def test_generate_queries_handles_empty_model_response() -> None:
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=_FakeCompletions(None)),
    )

    assert (
        generate_queries_for_document(
            client=client,
            document_text="document",
            num_queries=2,
            model="test-model",
            max_retries=1,
            max_document_tokens=100,
            temperature=0.2,
            max_output_tokens=50,
        )
        == []
    )


def test_generate_all_queries_preserves_source_order_and_ids_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = pd.DataFrame(
        {"id": ["first", "second"], "text": ["one", "two"]}, index=[8, 3]
    )
    fail_first = False

    def worker(idx: int, doc_id: str, *_args: Any) -> tuple[int, str, list[str]]:
        if fail_first and doc_id == "first":
            raise OSError("offline failure")
        return idx, doc_id, [f"{doc_id} query", f"{doc_id} another"]

    monkeypatch.setattr(generate_queries, "process_document_worker", worker)
    monkeypatch.setattr(
        generate_queries, "as_completed", lambda futures: reversed(list(futures))
    )
    kwargs = {
        "documents_df": documents,
        "num_queries_per_doc": 2,
        "api_key": "test",
        "model": "test",
        "max_workers": 2,
        "max_retries": 1,
        "max_document_tokens": 100,
        "temperature": 0.2,
        "max_output_tokens": 50,
    }

    queries, qrels = generate_queries.generate_all_queries(**kwargs)

    assert queries["id"].tolist() == [
        "first::query_0",
        "first::query_1",
        "second::query_0",
        "second::query_1",
    ]
    assert queries["text"].tolist() == [
        "first query",
        "first another",
        "second query",
        "second another",
    ]
    assert qrels["corpus-id"].tolist() == ["first", "first", "second", "second"]
    assert qrels["query-id"].tolist() == queries["id"].tolist()

    fail_first = True
    remaining_queries, remaining_qrels = generate_queries.generate_all_queries(**kwargs)
    assert remaining_queries["id"].tolist() == ["second::query_0", "second::query_1"]
    assert remaining_qrels["corpus-id"].tolist() == ["second", "second"]


@pytest.mark.parametrize(
    ("ids", "num_queries", "message"),
    [
        ([], 1, "nonempty"),
        (["first"], 0, "positive"),
        ([None], 1, "non-null"),
        ([1, "1"], 1, "unique"),
    ],
)
def test_generate_all_queries_rejects_invalid_identity_before_requests(
    monkeypatch: pytest.MonkeyPatch, ids: list[Any], num_queries: int, message: str
) -> None:
    def unexpected_request(*_args: Any) -> None:
        pytest.fail("invalid input must not reach the provider")

    monkeypatch.setattr(generate_queries, "process_document_worker", unexpected_request)
    with pytest.raises(ValueError, match=message):
        generate_queries.generate_all_queries(
            pd.DataFrame({"id": ids, "text": ["document"] * len(ids)}),
            num_queries,
            "test",
            "test",
            1,
            1,
            100,
            0.2,
            50,
        )


@pytest.mark.parametrize(
    ("responses", "exit_status", "status"),
    [
        ({"first": ["one", "two"], "second": ["three", "four"]}, 0, "complete"),
        ({"first": ["one"], "second": ["three", "four"]}, 1, "partial"),
        ({"first": [], "second": []}, 1, "failed"),
    ],
)
def test_query_cli_records_completeness_and_returns_matching_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    responses: dict[str, list[str]],
    exit_status: int,
    status: str,
) -> None:
    input_path = tmp_path / "input.csv"
    output_dir = tmp_path / "output"
    existing_files = {}
    if status == "failed":
        output_dir.mkdir()
        existing_files = {
            name: b"existing dataset"
            for name in ("corpus.parquet", "queries.parquet", "qrels.parquet")
        }
        for name, content in existing_files.items():
            (output_dir / name).write_bytes(content)
    pd.DataFrame({"id": ["first", "second"], "text": ["one", "two"]}).to_csv(
        input_path, index=False
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_queries.py",
            str(input_path),
            "--provider",
            "ollama",
            "--num-queries",
            "2",
            "--output-dir",
            str(output_dir),
        ],
    )
    monkeypatch.setattr(generate_queries, "load_config", lambda _path: {})
    monkeypatch.setattr(generate_queries, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        generate_queries,
        "process_document_worker",
        lambda idx, doc_id, *_args: (idx, doc_id, responses[doc_id]),
    )

    assert generate_queries.main() == exit_status
    summary = json.loads((output_dir / "generation_summary.json").read_text())
    assert summary == {
        "status": status,
        "requested_queries": 4,
        "actual_queries": sum(map(len, responses.values())),
        "incomplete_documents": [
            {
                "source_id": source_id,
                "requested_queries": 2,
                "actual_queries": len(queries),
            }
            for source_id, queries in responses.items()
            if len(queries) != 2
        ],
    }
    if status == "failed":
        assert {
            name: (output_dir / name).read_bytes() for name in existing_files
        } == existing_files
    else:
        queries = pd.read_parquet(output_dir / "queries.parquet")
        assert len(queries) == summary["actual_queries"]


def test_query_cli_returns_failure_on_output_write_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"id": ["first"], "text": ["one"]}).to_csv(input_path, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_queries.py",
            str(input_path),
            "--provider",
            "ollama",
            "--num-queries",
            "1",
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )
    monkeypatch.setattr(generate_queries, "load_config", lambda _path: {})
    monkeypatch.setattr(generate_queries, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        generate_queries,
        "process_document_worker",
        lambda idx, doc_id, *_args: (idx, doc_id, ["query"]),
    )

    def fail_save(**_kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(generate_queries, "save_mteb_dataset", fail_save)
    assert generate_queries.main() == 1
    output = capsys.readouterr().out
    assert "disk full" in output
    assert "Done!" not in output
