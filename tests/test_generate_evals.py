import importlib
import signal
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import generate_evals
from generate_evals import (
    WeaviateRunResources,
    compute_metrics,
    get_max_k,
    get_metric_k_values,
    index_weaviate_documents,
)


def test_get_metric_k_values_supports_independent_and_legacy_configuration() -> None:
    assert get_metric_k_values(
        {
            "search": {
                "top_k": [20],
                "metrics": {"mrr_k": [1, 5], "hit_rate_k": [10]},
            }
        }
    ) == {"mrr": [1, 5], "hit_rate": [10]}
    assert get_metric_k_values({"search": {"top_k": 7}}) == {
        "mrr": [7],
        "hit_rate": [7],
    }
    assert get_metric_k_values({}) == {"mrr": [10], "hit_rate": [10]}


def test_get_max_k_finds_maximum() -> None:
    assert get_max_k({"mrr": [1, 5], "hit_rate": [10]}) == 10


@pytest.mark.parametrize(
    "cutoffs",
    [
        {},
        {"mrr": [10]},
        {"mrr": [10], "hit_rate": []},
        {"mrr": "10", "hit_rate": [10]},
        {"mrr": [0], "hit_rate": [10]},
        {"mrr": [True], "hit_rate": [10]},
        {"mrr": [1.5], "hit_rate": [10]},
        {"mrr": [10], "hit_rate": [10], "ndcg": [10]},
    ],
)
def test_metric_helpers_reject_malformed_cutoff_dictionaries(
    cutoffs: object,
) -> None:
    with pytest.raises(ValueError, match="metric cutoffs"):
        get_max_k(cutoffs)  # type: ignore[arg-type]


def test_get_metric_k_values_rejects_malformed_config() -> None:
    with pytest.raises(ValueError, match="metric cutoffs"):
        get_metric_k_values({"search": {"metrics": {"mrr_k": [], "hit_rate_k": [10]}}})


def test_compute_metrics_respects_each_cutoff() -> None:
    retrieved = [
        (["d1", "d2", "d3"], ["d2"]),
        (["d4", "d5", "d6"], ["d6"]),
    ]

    metrics = compute_metrics(retrieved, {"mrr": [1, 3], "hit_rate": [2, 3]})

    assert metrics == pytest.approx(
        {
            "mrr@1": 0.0,
            "mrr@3": (1 / 2 + 1 / 3) / 2,
            "hit_rate@2": 0.5,
            "hit_rate@3": 1.0,
        }
    )


def test_compute_metrics_rejects_empty_query_results() -> None:
    with pytest.raises(ValueError, match="query result collection.*empty"):
        compute_metrics([], {"mrr": [10], "hit_rate": [10]})


class FakeCollections:
    def __init__(self, existing: set[str] | None = None):
        self.names = set(existing or set())
        self.deleted: list[str] = []

    def exists(self, name: str) -> bool:
        return name in self.names

    def create(self, name: str, **_kwargs):
        self.names.add(name)
        return SimpleNamespace(name=name)

    def delete(self, name: str) -> None:
        self.deleted.append(name)
        self.names.discard(name)


class FakeClient:
    def __init__(self, existing: set[str] | None = None):
        self.collections = FakeCollections(existing)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_weaviate_collection_names_are_unique_valid_and_collision_safe() -> None:
    first_client = FakeClient()
    first = WeaviateRunResources(first_client, run_id="run-1")
    first_name = first.create_collection("model/org name").name
    second_name = first.create_collection("model/org name").name

    assert first_name == "HybridEval_model_org_name_run_1_1"
    assert second_name == "HybridEval_model_org_name_run_1_2"
    assert first_name != second_name

    colliding_client = FakeClient({first_name, "Documents"})
    colliding = WeaviateRunResources(colliding_client, run_id="run-1")
    with pytest.raises(RuntimeError, match="collision"):
        colliding.create_collection("model/org name")
    assert colliding_client.collections.deleted == []


def test_weaviate_cleanup_deletes_only_owned_collections_and_closes_client() -> None:
    client = FakeClient({"Documents", "unrelated"})
    resources = WeaviateRunResources(client, run_id="owned")
    owned_name = resources.create_collection("BM25").name

    with pytest.raises(ValueError, match="unowned"):
        resources.delete_collection("Documents")

    resources.cleanup()

    assert client.collections.deleted == [owned_name]
    assert client.collections.names == {"Documents", "unrelated"}
    assert client.closed


class FakeBatch:
    def __init__(self, number_errors: int):
        self.number_errors = number_errors
        self.added: list[dict] = []

    def add_object(self, **kwargs) -> None:
        self.added.append(kwargs)


class FakeCollection:
    def __init__(self, number_errors: int = 0, total_count: int = 1):
        self.dynamic_batch = FakeBatch(number_errors)
        self.batch = SimpleNamespace(
            dynamic=lambda: nullcontext(self.dynamic_batch),
            failed_objects=["failed"] if number_errors else [],
        )
        self.aggregate = SimpleNamespace(
            over_all=lambda total_count: SimpleNamespace(total_count=total_count_value)
        )
        total_count_value = total_count


def test_index_weaviate_documents_rejects_batch_failures() -> None:
    collection = FakeCollection(number_errors=1, total_count=0)

    with pytest.raises(RuntimeError, match="batch import failed.*1"):
        index_weaviate_documents(collection, [{"id": "d1", "text": "document"}])


def test_index_weaviate_documents_verifies_indexed_count() -> None:
    collection = FakeCollection(number_errors=0, total_count=0)

    with pytest.raises(RuntimeError, match="object count mismatch.*expected 1.*got 0"):
        index_weaviate_documents(collection, [{"id": "d1", "text": "document"}])


def test_signal_handler_cleans_owned_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient({"Documents"})
    resources = WeaviateRunResources(client, run_id="signal")
    owned_name = resources.create_collection("model").name
    monkeypatch.setattr(generate_evals, "_weaviate_resources", resources)

    with pytest.raises(SystemExit):
        generate_evals.signal_handler(signal.SIGTERM, None)

    assert client.collections.deleted == [owned_name]
    assert client.collections.names == {"Documents"}
    assert client.closed


def test_importing_generate_evals_does_not_register_signal_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[signal.Signals, object]] = []
    monkeypatch.setattr(
        signal, "signal", lambda signum, handler: calls.append((signum, handler))
    )
    sys.modules.pop("generate_evals", None)

    importlib.import_module("generate_evals")

    assert calls == []
