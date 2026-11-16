import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def valid_config(tmp_path: Path) -> dict[str, Any]:
    """Return the smallest configuration accepted by ``validate_config``."""
    return {
        "project_id": "test",
        "data": {"mteb_data_dir": str(tmp_path)},
        "embeddings": {
            "huggingface": {
                "mini": "sentence-transformers/all-MiniLM-L6-v2",
            },
            "device": "cpu",
        },
        "search": {
            "alpha": [0.0, 0.5, 1.0],
            "metrics": {"mrr_k": [1, 10], "hit_rate_k": [5]},
        },
        "output": {"results_dir": str(tmp_path / "results")},
        "visualization": {},
        "model": {"embedding_batch_size": 8, "max_document_tokens": 512},
    }


@pytest.fixture
def evaluation_runtime(
    monkeypatch: pytest.MonkeyPatch, valid_config: dict[str, Any], tmp_path: Path
) -> SimpleNamespace:
    """Run the evaluation/cache flow with deterministic offline backend boundaries."""
    import numpy as np
    import pandas as pd

    import generate_evals
    from _core.utils import MTEBRetrievalData

    config = valid_config
    config["embeddings"]["cache_dir"] = str(tmp_path / "embeddings")
    config["evals"] = {"cache_dir": str(tmp_path / "evals")}
    config["search"]["include_bm25_baseline"] = False
    dataset = MTEBRetrievalData(
        pd.DataFrame({"id": ["d1", "d2"], "text": ["first", "second"]}),
        pd.DataFrame({"id": ["q1", "q2"], "text": ["one", "two"]}),
        pd.DataFrame({"query-id": ["q1", "q2"], "corpus-id": ["d1", "d2"]}),
    )
    encoded: list[list[str]] = []
    searched: list[dict[str, Any]] = []
    closed: list[bool] = []

    def encode(texts: list[str], **_kwargs: Any) -> Any:
        encoded.append(texts)
        if config["embeddings"].get("colbert"):
            return [np.ones((2, 3)) for _text in texts]
        return np.ones((len(texts), 3))

    def search(**kwargs: Any) -> SimpleNamespace:
        searched.append(kwargs)
        return SimpleNamespace(
            objects=[SimpleNamespace(properties={"document_id": "d1"})]
        )

    model = SimpleNamespace(encode=encode, to=lambda _device: None)
    collection = SimpleNamespace(query=SimpleNamespace(hybrid=search, bm25=search))
    collections = SimpleNamespace(
        exists=lambda _name: False,
        create=lambda **_kwargs: collection,
        delete=lambda _name: None,
    )
    client = SimpleNamespace(collections=collections, close=lambda: closed.append(True))
    rank = SimpleNamespace(
        rerank=lambda **kwargs: [
            [{"id": "d1", "score": 1.0}, {"id": "d2", "score": 0.0}]
            for _query in kwargs["queries_embeddings"]
        ]
    )
    monkeypatch.setattr(sys, "argv", ["generate_evals.py"])
    monkeypatch.setattr(generate_evals, "register_cleanup_handlers", lambda: None)
    monkeypatch.setattr(generate_evals, "version", lambda _package: "test-version")
    monkeypatch.setattr(generate_evals, "load_config", lambda _path: config)
    monkeypatch.setattr(
        generate_evals, "load_mteb_retrieval_data_from_dir", lambda _path: dataset
    )
    monkeypatch.setattr(
        generate_evals.weaviate, "connect_to_embedded", lambda **_kwargs: client
    )
    monkeypatch.setattr(
        generate_evals,
        "get_openrouter_embeddings",
        lambda texts, **kwargs: model.encode(texts, **kwargs),
    )
    monkeypatch.setattr(
        generate_evals, "SentenceTransformer", lambda *_args, **_kwargs: model
    )
    monkeypatch.setattr(
        generate_evals,
        "load_pylate",
        lambda: (SimpleNamespace(ColBERT=lambda **_kwargs: model), rank),
    )
    monkeypatch.setattr(generate_evals, "index_weaviate_documents", lambda *_args: None)
    for name in (
        "create_results_visualization",
        "create_memory_visualization",
        "create_tradeoff_visualization",
        "create_html_dashboard",
    ):
        monkeypatch.setattr(generate_evals, name, lambda *_args: None)
    return SimpleNamespace(
        config=config,
        dataset=dataset,
        encoded=encoded,
        searched=searched,
        model=model,
        rank=rank,
        collection=collection,
        client=client,
        closed=closed,
    )
