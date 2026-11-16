from types import SimpleNamespace

import pandas as pd
import pytest

from _core.utils import MTEBRetrievalData
from generate_queries import generate_queries_for_document


@pytest.mark.parametrize("scores", [[1, 0], [0, 1], [1, 1]])
def test_duplicate_normalized_qrels_are_rejected(scores: list[int]) -> None:
    with pytest.raises(ValueError, match="qrels.*duplicate.*query/document"):
        MTEBRetrievalData(
            pd.DataFrame({"id": [1], "text": ["document"]}),
            pd.DataFrame({"id": [2], "text": ["query"]}),
            pd.DataFrame(
                {
                    "query-id": [2, "2"],
                    "corpus-id": [1, "1"],
                    "score": scores,
                },
                dtype=object,
            ),
        )


def test_query_parsing_preserves_digits_and_drops_empty_list_items() -> None:
    content = "\n3D printing\n2026 tax rules\n3.14 approximation\n.NET runtime\n-5 degrees\n1. numbered\n2) parenthesized\n- bullet\n1. \n- \n"
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                )
            )
        )
    )

    assert generate_queries_for_document(
        client, "document", 20, "test", 1, 100, 0.2, 50
    ) == [
        "3D printing",
        "2026 tax rules",
        "3.14 approximation",
        ".NET runtime",
        "-5 degrees",
        "numbered",
        "parenthesized",
        "bullet",
    ]
