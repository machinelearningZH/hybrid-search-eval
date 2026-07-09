from _core.utils import (
    generate_cache_key,
    generate_eval_cache_key,
    parse_model_configs,
    validate_config,
)


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
