import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import generate_evals
from _core import utils


def dataset() -> utils.MTEBRetrievalData:
    return utils.MTEBRetrievalData(
        pd.DataFrame({"id": ["d1", "d2"], "text": ["first", "second"]}),
        pd.DataFrame({"id": ["q1", "q2"], "text": ["one", "two"]}),
        pd.DataFrame(
            {"query-id": ["q1", "q2"], "corpus-id": ["d1", "d2"], "score": [1, 1]}
        ),
    )


@pytest.mark.parametrize(
    "change",
    [
        "corpus_text",
        "corpus_order",
        "query_text",
        "query_order",
        "qrels",
        "tokens",
        "candidates",
    ],
)
def test_dataset_cache_identity_changes_with_evaluation_inputs(change: str) -> None:
    original = dataset()
    changed = dataset()
    tokens, candidates = 512, 1000
    if change == "corpus_text":
        changed.corpus.loc[0, "text"] = "replacement"
    elif change == "corpus_order":
        changed.corpus = changed.corpus.iloc[::-1]
    elif change == "query_text":
        changed.queries.loc[0, "text"] = "replacement"
    elif change == "query_order":
        changed.queries = changed.queries.iloc[::-1]
    elif change == "qrels":
        changed.qrels.loc[0, "score"] = 2
    elif change == "tokens":
        tokens = 1024
    else:
        candidates = 50

    assert utils.generate_dataset_cache_identity(
        original, 512, 1000
    ) != utils.generate_dataset_cache_identity(changed, tokens, candidates)


def test_dataset_cache_identity_canonicalizes_qrels_order() -> None:
    original, reordered = dataset(), dataset()
    reordered.qrels = reordered.qrels.iloc[::-1]

    assert utils.generate_dataset_cache_identity(
        original, 512, None
    ) == utils.generate_dataset_cache_identity(reordered, 512, None)


def test_colbert_cache_rejects_object_arrays_without_pickle(tmp_path: Path) -> None:
    np.savez(
        tmp_path / "unsafe_colbert.npz",
        _count=np.array([1]),
        emb_0=np.array([["benign"]], dtype=object),
    )

    with pytest.raises(ValueError, match="Object arrays|numeric"):
        utils.load_colbert_embeddings("unsafe", tmp_path)


@pytest.mark.parametrize("colbert", [False, True])
def test_embedding_cache_probe_rejects_wrong_count_and_metadata(
    tmp_path: Path, colbert: bool
) -> None:
    save = utils.save_colbert_embeddings if colbert else utils.save_embeddings
    exists = utils.colbert_embeddings_exist if colbert else utils.embeddings_exist
    arrays = [np.ones((2, 3))] if colbert else np.ones((1, 3))
    save(arrays, "cache", tmp_path, {"num_documents": 1})

    assert exists(
        "cache", tmp_path, expected_count=1, expected_metadata={"num_documents": 1}
    )
    assert not exists("cache", tmp_path, expected_count=2)
    assert not exists("cache", tmp_path, expected_metadata={"num_documents": 2})


def test_eval_cache_rejects_renamed_and_legacy_payloads(tmp_path: Path) -> None:
    result = {"num_documents": 2, "num_queries": 2, "mrr@1": 1.0}
    utils.save_eval_results(result, "original", tmp_path)
    (tmp_path / "renamed.json").write_bytes((tmp_path / "original.json").read_bytes())
    (tmp_path / "legacy.json").write_text(json.dumps(result))

    assert not utils.eval_results_exist("renamed", tmp_path)
    assert not utils.eval_results_exist("legacy", tmp_path)
    assert not utils.eval_results_exist(
        "original", tmp_path, expected_metadata={"num_queries": 3}
    )


def test_colbert_cache_probe_treats_corrupt_archive_as_miss(tmp_path: Path) -> None:
    (tmp_path / "broken_colbert.npz").write_bytes(b"PK\x03\x04broken archive")

    assert not utils.colbert_embeddings_exist("broken", tmp_path)


@pytest.mark.parametrize(
    "array", [np.ones(3), np.array([[np.nan]]), np.array([["text"]])]
)
def test_colbert_cache_rejects_invalid_vectors(
    tmp_path: Path, array: np.ndarray
) -> None:
    np.savez(tmp_path / "invalid_colbert.npz", _count=np.array([1]), emb_0=array)

    with pytest.raises(ValueError, match="embedding|numeric|finite"):
        utils.load_colbert_embeddings("invalid", tmp_path)


@pytest.mark.parametrize(
    ("change", "new_embedding_batches"),
    [
        ("corpus_text", 1),
        ("corpus_order", 1),
        ("query_text", 1),
        ("query_order", 1),
        ("qrels", 0),
        ("tokens", 1),
        ("candidates", 0),
        ("backend", 2),
    ],
)
def test_evaluation_reuses_only_matching_dataset_and_backend_caches(
    evaluation_runtime: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    new_embedding_batches: int,
) -> None:
    runtime = evaluation_runtime
    runtime.config["search"]["include_bm25_baseline"] = True
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    assert generate_evals.main() == 0
    first_encoded = len(runtime.encoded)
    first_searched = len(runtime.searched)
    assert first_encoded == 2
    assert first_searched > 0

    assert generate_evals.main() == 0
    assert len(runtime.encoded) == first_encoded
    assert len(runtime.searched) == first_searched

    data = runtime.dataset
    if change == "corpus_text":
        data.corpus.loc[0, "text"] = "replacement document"
    elif change == "corpus_order":
        data.corpus = data.corpus.iloc[::-1]
    elif change == "query_text":
        data.queries.loc[0, "text"] = "replacement query"
    elif change == "query_order":
        data.queries = data.queries.iloc[::-1]
    elif change == "qrels":
        data.qrels.loc[0, "score"] = 2
    elif change == "tokens":
        runtime.config["model"]["max_document_tokens"] = 1024
    elif change == "candidates":
        runtime.config["search"]["bm25_candidate_limit"] = 50
    else:
        models = runtime.config["embeddings"].pop("huggingface")
        runtime.config["embeddings"]["openrouter"] = {"models": models}
    changed_data = utils.MTEBRetrievalData(data.corpus, data.queries, data.qrels)
    monkeypatch.setattr(
        generate_evals, "load_mteb_retrieval_data_from_dir", lambda _path: changed_data
    )

    assert generate_evals.main() == 0
    assert len(runtime.encoded) == first_encoded + new_embedding_batches
    assert len(runtime.searched) > first_searched
    assert runtime.closed == [True, True, True]


@pytest.mark.parametrize("backend", ["huggingface", "colbert"])
@pytest.mark.parametrize("invalid", ["count", "dimension"])
def test_evaluation_recomputes_incompatible_cached_query_embeddings(
    evaluation_runtime: SimpleNamespace, backend: str, invalid: str
) -> None:
    runtime = evaluation_runtime
    if backend == "colbert":
        runtime.config["embeddings"]["colbert"] = runtime.config["embeddings"].pop(
            "huggingface"
        )
    runtime.config["search"]["alpha"] = [1.0]
    assert generate_evals.main() == 0
    assert len(runtime.encoded) == 2

    cache_dir = Path(runtime.config["embeddings"]["cache_dir"])
    count, dimension = (1, 3) if invalid == "count" else (2, 2)
    if backend == "colbert":
        query_path = next(cache_dir.glob("*queries*_colbert.npz"))
        np.savez_compressed(
            query_path,
            _count=np.array([count]),
            **{f"emb_{index}": np.ones((2, dimension)) for index in range(count)},
        )
    else:
        query_path = next(cache_dir.glob("*queries*.npy"))
        np.save(query_path, np.ones((count, dimension)))

    assert generate_evals.main() == 0
    assert len(runtime.encoded) == 3
    assert runtime.encoded[-1] == ["one", "two"]
    loader = (
        utils.load_colbert_embeddings if backend == "colbert" else utils.load_embeddings
    )
    cache_key = query_path.name.removesuffix("_colbert.npz").removesuffix(".npy")
    loader(cache_key, cache_dir, expected_count=2, expected_dimension=3)
