from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from generate_queries import generate_queries_for_document, load_documents_from_file


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
