from _core.utils import (
    generate_cache_key,
    generate_eval_cache_key,
    parse_model_configs,
    repair_snowflake_position_ids,
    validate_config,
)

import torch


def _minimal_config(model_spec: object) -> dict:
    return {
        "project_id": "test",
        "data": {"mteb_data_dir": "_data/mteb_user"},
        "embeddings": {
            "huggingface": {
                "harrier": model_spec,
            },
            "device": "cpu",
        },
    }


def test_parse_model_configs_uses_explicit_prompt_names() -> None:
    config = _minimal_config(
        {
            "model": "microsoft/harrier-oss-v1-270m",
            "query_prompt_name": "web_search_query",
            "passage_prompt_name": "sts_query",
        }
    )

    model_config = parse_model_configs(config)[0]

    assert model_config["query_encode_kwargs"] == {
        "prompt_name": "web_search_query",
    }
    assert model_config["passage_encode_kwargs"] == {
        "prompt_name": "sts_query",
    }
    assert model_config["query_cache_identity"] == {
        "query_encode_kwargs": {"prompt_name": "web_search_query"},
    }
    assert model_config["passage_cache_identity"] == {
        "passage_encode_kwargs": {"prompt_name": "sts_query"},
    }
    assert model_config["cache_identity"] == {
        "query_encode_kwargs": {"prompt_name": "web_search_query"},
        "passage_encode_kwargs": {"prompt_name": "sts_query"},
    }


def test_parse_model_configs_omits_default_cache_identity() -> None:
    config = _minimal_config("sentence-transformers/all-MiniLM-L6-v2")

    model_config = parse_model_configs(config)[0]

    assert model_config["query_cache_identity"] == {}
    assert model_config["passage_cache_identity"] == {}
    assert model_config["cache_identity"] == {}


def test_validate_config_rejects_non_string_prompt_names() -> None:
    config = _minimal_config(
        {
            "model": "microsoft/harrier-oss-v1-270m",
            "query_prompt_name": 123,
        }
    )

    errors = validate_config(config)

    assert any("harrier.query_prompt_name" in error for error in errors)


def test_cache_keys_include_prompt_identity() -> None:
    unprompted = generate_cache_key(
        "project_dataset",
        "microsoft/harrier-oss-v1-270m",
        "queries",
        cache_identity={},
    )
    prompted = generate_cache_key(
        "project_dataset",
        "microsoft/harrier-oss-v1-270m",
        "queries",
        cache_identity={"query_encode_kwargs": {"prompt_name": "web_search_query"}},
    )

    assert unprompted != prompted


def test_eval_cache_keys_include_prompt_identity() -> None:
    unprompted = generate_eval_cache_key(
        "project_dataset",
        "microsoft/harrier-oss-v1-270m",
        0.5,
        10,
        "harrier",
        {"mrr": [10], "hit_rate": [10]},
        cache_identity={},
    )
    prompted = generate_eval_cache_key(
        "project_dataset",
        "microsoft/harrier-oss-v1-270m",
        0.5,
        10,
        "harrier",
        {"mrr": [10], "hit_rate": [10]},
        cache_identity={"query_encode_kwargs": {"prompt_name": "web_search_query"}},
    )

    assert unprompted != prompted


class _FakeEmbeddings(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "position_ids",
            torch.tensor([0, 4320601048, -1], dtype=torch.long),
            persistent=False,
        )


class _FakeAutoModel:
    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddings()


class _FakeTransformerModule:
    def __init__(self) -> None:
        self.auto_model = _FakeAutoModel()


class _FakeSentenceTransformer:
    def __init__(self) -> None:
        self.module = _FakeTransformerModule()

    def _first_module(self) -> _FakeTransformerModule:
        return self.module


def test_repair_snowflake_position_ids_resets_corrupted_buffer() -> None:
    model = _FakeSentenceTransformer()

    repaired = repair_snowflake_position_ids(
        model,
        "Snowflake/snowflake-arctic-embed-m-v2.0",
    )

    assert repaired is True
    assert model.module.auto_model.embeddings.position_ids.tolist() == [0, 1, 2]


def test_repair_snowflake_position_ids_ignores_other_models() -> None:
    model = _FakeSentenceTransformer()

    repaired = repair_snowflake_position_ids(
        model, "sentence-transformers/all-MiniLM-L6-v2"
    )

    assert repaired is False
    assert model.module.auto_model.embeddings.position_ids.tolist() == [
        0,
        4320601048,
        -1,
    ]
