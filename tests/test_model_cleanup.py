import weakref
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import generate_evals


@pytest.mark.parametrize("backend", ["huggingface", "colbert"])
@pytest.mark.parametrize("failure", [None, "documents", "queries", "retrieval"])
def test_models_are_released_on_success_and_failure(
    evaluation_runtime: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    failure: str | None,
) -> None:
    runtime = evaluation_runtime
    runtime.config["embeddings"]["device"] = "cuda"
    runtime.config["search"]["alpha"] = [1.0]
    if backend == "colbert":
        runtime.config["embeddings"][backend] = runtime.config["embeddings"].pop(
            "huggingface"
        )
    references: list[weakref.ReferenceType[Any]] = []
    events: list[str] = []

    class Model:
        def encode(self, texts: list[str], **_kwargs: Any) -> Any:
            stage = "documents" if texts[0] == "first" else "queries"
            if stage == failure:
                raise ValueError(f"failed {stage}")
            if backend == "colbert":
                return [np.ones((2, 3)) for _text in texts]
            return np.ones((len(texts), 3))

    def create_model(*_args: Any, **_kwargs: Any) -> Model:
        model = Model()
        references.append(weakref.ref(model))
        return model

    collect = generate_evals.gc.collect

    def collect_released_model() -> int:
        assert references[-1]() is None
        events.append("gc")
        return collect()

    def empty_cache() -> None:
        assert references[-1]() is None
        events.append("cache")

    def fail_retrieval(**_kwargs: Any) -> None:
        raise RuntimeError("retrieval failed")

    monkeypatch.setattr(generate_evals, "SentenceTransformer", create_model)
    monkeypatch.setattr(
        generate_evals,
        "load_pylate",
        lambda: (SimpleNamespace(ColBERT=create_model), runtime.rank),
    )
    monkeypatch.setattr(generate_evals.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        generate_evals.torch.backends.mps, "is_available", lambda: False
    )
    monkeypatch.setattr(generate_evals.gc, "collect", collect_released_model)
    monkeypatch.setattr(generate_evals.torch.cuda, "empty_cache", empty_cache)
    if failure == "retrieval":
        monkeypatch.setattr(runtime.collection.query, "hybrid", fail_retrieval)
        monkeypatch.setattr(runtime.rank, "rerank", fail_retrieval)
        with pytest.raises(RuntimeError, match="retrieval failed"):
            generate_evals.main()
    else:
        assert generate_evals.main() == 0

    assert references[0]() is None
    assert events == ["gc", "cache"]
    assert runtime.closed == [True]


def test_failed_model_is_released_before_loading_next_model(
    evaluation_runtime: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = evaluation_runtime
    runtime.config["embeddings"]["huggingface"] = {"first": "first", "next": "next"}
    references: list[weakref.ReferenceType[Any]] = []
    collected: list[bool] = []

    class Model:
        def encode(self, _texts: list[str], **_kwargs: Any) -> Any:
            raise ValueError("document encoding failed")

    def create_model(*_args: Any, **_kwargs: Any) -> Model:
        assert all(reference() is None for reference in references)
        assert len(collected) == len(references)
        model = Model()
        references.append(weakref.ref(model))
        return model

    collect = generate_evals.gc.collect

    def collect_released_model() -> int:
        assert references[-1]() is None
        collected.append(True)
        return collect()

    monkeypatch.setattr(generate_evals, "SentenceTransformer", create_model)
    monkeypatch.setattr(generate_evals.gc, "collect", collect_released_model)
    assert generate_evals.main() == 0
    assert len(references) == 2
    assert all(reference() is None for reference in references)
