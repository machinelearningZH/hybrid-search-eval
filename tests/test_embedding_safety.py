from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import requests

import generate_evals
from _core import utils


@pytest.mark.parametrize(
    "items",
    [
        [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}],
        [{"index": 0, "embedding": [1.0]}, {"index": 2, "embedding": [2.0]}],
        [{"index": False, "embedding": [1.0]}, {"index": 1, "embedding": [2.0]}],
        [{"index": 0, "embedding": [1.0]}, {"index": 1, "embedding": []}],
        [{"index": 0, "embedding": [1.0]}, {"index": 1, "embedding": [float("nan")]}],
        [{"index": 0, "embedding": ["1"]}, {"index": 1, "embedding": ["2"]}],
        [{"index": 0, "embedding": [[1.0]]}, {"index": 1, "embedding": [[2.0]]}],
    ],
)
def test_openrouter_rejects_malformed_embeddings(
    monkeypatch: pytest.MonkeyPatch, items: list[dict[str, Any]]
) -> None:
    response = SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: {"data": items}
    )
    monkeypatch.setattr(utils.requests, "post", lambda *_args, **_kwargs: response)

    with pytest.raises(ValueError, match="embedding"):
        utils.get_openrouter_embeddings(["first", "second"], api_key="test-key")


def test_openrouter_passes_bounded_timeouts_and_propagates_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def post(_url: str, **kwargs: Any) -> None:
        assert kwargs["timeout"] == (2.0, 7.0)
        raise requests.Timeout("read timed out")

    monkeypatch.setattr(utils.requests, "post", post)
    with pytest.raises(requests.Timeout, match="read timed out"):
        utils.get_openrouter_embeddings(
            ["text"], api_key="test-key", connect_timeout=2.0, read_timeout=7.0
        )


@pytest.mark.parametrize("timeout", [0, -1, True, "5", float("inf"), float("nan")])
def test_openrouter_config_rejects_invalid_timeout(
    valid_config: dict[str, Any], timeout: object
) -> None:
    valid_config["embeddings"]["openrouter"] = {
        "models": {},
        "settings": {"connect_timeout": timeout, "read_timeout": timeout},
    }
    errors = utils.validate_config(valid_config)
    assert any("connect_timeout" in error for error in errors)
    assert any("read_timeout" in error for error in errors)


def test_openrouter_rejects_dimension_changes_between_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replies = iter([[[1.0]], [[1.0, 2.0]]])

    def post(_url: str, **_kwargs: Any) -> object:
        data = [{"index": 0, "embedding": next(replies)[0]}]
        return SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: {"data": data}
        )

    monkeypatch.setattr(utils.requests, "post", post)
    with pytest.raises(ValueError, match="embedding"):
        utils.get_openrouter_embeddings(
            ["first", "second"], batch_size=1, api_key="test-key"
        )


@pytest.mark.parametrize("is_colbert", [False, True])
def test_embedding_validation_rejects_incomplete_backend_output(
    is_colbert: bool,
) -> None:
    embeddings = [np.ones((2, 3))] if is_colbert else np.ones((1, 3))
    with pytest.raises(ValueError, match="embedding count mismatch"):
        utils.validate_embeddings(embeddings, 2, is_colbert=is_colbert)


def test_colbert_embedding_validation_accepts_variable_tokens_and_checks_dimensions() -> (
    None
):
    embeddings = [np.ones((2, 3)), np.ones((4, 3))]
    assert utils.validate_embeddings(embeddings, 2, is_colbert=True) == 3
    with pytest.raises(ValueError, match="embedding dimension"):
        utils.validate_embeddings(embeddings, 2, is_colbert=True, expected_dimension=4)


@pytest.mark.parametrize("backend", ["openrouter", "huggingface", "colbert"])
@pytest.mark.parametrize("invalid_stage", ["documents", "queries"])
def test_evaluation_rejects_short_backend_output_before_retrieval_or_cache_writes(
    monkeypatch: pytest.MonkeyPatch,
    evaluation_runtime: SimpleNamespace,
    backend: str,
    invalid_stage: str,
) -> None:
    runtime = evaluation_runtime
    config = runtime.config
    config["embeddings"].pop("huggingface")
    if backend == "openrouter":
        config["embeddings"][backend] = {
            "models": {"test": "test/model"},
            "settings": {"connect_timeout": 2.0, "read_timeout": 7.0},
        }
    else:
        config["embeddings"][backend] = {"test": "test/model"}

    def encode(texts: list[str], **kwargs: Any) -> Any:
        runtime.encoded.append(texts)
        if backend == "openrouter":
            assert kwargs["connect_timeout"] == 2.0
            assert kwargs["read_timeout"] == 7.0
        stage = "documents" if len(runtime.encoded) == 1 else "queries"
        count = 1 if stage == invalid_stage else 2
        return (
            [np.ones((2, 3)) for _ in range(count)]
            if backend == "colbert"
            else np.ones((count, 3))
        )

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("invalid embeddings reached retrieval or cache writes")

    runtime.model.encode = encode
    runtime.collection.query.hybrid = forbidden
    runtime.rank.rerank = forbidden
    for name in ("save_embeddings", "save_colbert_embeddings", "save_eval_results"):
        monkeypatch.setattr(generate_evals, name, forbidden)
    generate_evals.main()

    assert len(runtime.encoded) == (1 if invalid_stage == "documents" else 2)
    assert runtime.closed == [True]
    assert not runtime.searched
