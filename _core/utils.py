import yaml
import json
import hashlib
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
import tiktoken
import os
import requests
from dotenv import load_dotenv

console = Console()

# Load environment variables
load_dotenv()


# Console print helpers for standardized output
def print_loading_cached(item_type: str, extra_info: str = "") -> None:
    """Print message for loading cached items.

    Args:
        item_type: Type of items being loaded (e.g., 'documents', 'queries')
        extra_info: Additional information to append to message
    """
    msg = f"   📦 Loading cached {item_type}..."
    if extra_info:
        msg += f" {extra_info}"
    console.print(msg, style="cyan")


def print_loaded_cached(item_type: str, count: int, extra_info: str = "") -> None:
    """Print success message for loaded cached items.

    Args:
        item_type: Type of items that were loaded (e.g., 'documents', 'queries')
        count: Number of items loaded
        extra_info: Additional information to append to message
    """
    msg = f"   ✓ Loaded [green]{count:,}[/green] cached {item_type}"
    if extra_info:
        msg += f" {extra_info}"
    console.print(msg)


def print_generating(item_type: str, count: int, extra_info: str = "") -> None:
    """Print message for generating items.

    Args:
        item_type: Type of items being generated (e.g., 'documents', 'queries')
        count: Number of items to generate
        extra_info: Additional information to append to message
    """
    msg = f"   Generating embeddings for [yellow]{count}[/yellow] {item_type}..."
    if extra_info:
        msg += f" {extra_info}"
    console.print(msg, style="cyan")


def print_generated(
    item_type: str, count: int, time_ms: float, time_per_item_ms: float | None = None
) -> None:
    """Print success message for generated items.

    Args:
        item_type: Type of items that were generated (e.g., 'documents', 'queries')
        count: Number of items generated
        time_ms: Total time taken in milliseconds
        time_per_item_ms: Optional time per individual item in milliseconds
    """
    msg = f"   ✓ Embedded [green]{count}[/green] {item_type} in [yellow]{time_ms:.1f}ms[/yellow]"
    if time_per_item_ms is not None:
        msg += f" ([dim]{time_per_item_ms:.2f}ms per {item_type.rstrip('s')}[/dim])"
    console.print(msg)


def print_saved_to_cache(file_path: Path, show_full_path: bool = False) -> None:
    """Print message for saved cache files.

    Args:
        file_path: Path to the saved cache file
        show_full_path: If True, display full path; otherwise show only filename
    """
    display_path = str(file_path) if show_full_path else file_path.name
    console.print(f"   💾 Saved embeddings to: [dim]{display_path}[/dim]")


def print_saved_eval_to_cache(file_path: Path) -> None:
    """Print message for saved eval results.

    Args:
        file_path: Path to the saved evaluation results file
    """
    console.print(f"   💾 Saved eval results to: [dim]{file_path.name}[/dim]")


def print_indexing(item_type: str, count: int) -> None:
    """Print message for indexing items.

    Args:
        item_type: Type of items being indexed (e.g., 'documents', 'queries')
        count: Number of items to index
    """
    console.print(f"   Indexing [yellow]{count}[/yellow] {item_type}...", style="cyan")


def print_indexed(item_type: str, count: int) -> None:
    """Print success message for indexed items.

    Args:
        item_type: Type of items that were indexed (e.g., 'documents', 'queries')
        count: Number of items indexed
    """
    console.print(f"   ✓ Indexed [green]{count}[/green] {item_type}")


def print_generated_count(item_type: str, count: int) -> None:
    """Print success message for generated items (count only, no timing).

    Args:
        item_type: Type of items that were generated (e.g., 'documents', 'queries')
        count: Number of items generated
    """
    console.print(f"   ✓ Generated [green]{count}[/green] {item_type}")


def truncate_text_to_tokens(
    text: str, max_tokens: int, encoding_name: str = "cl100k_base"
) -> str:
    """
    Truncate text to a maximum number of tokens using tiktoken.

    Args:
        text: The text to truncate
        max_tokens: Maximum number of tokens to keep
        encoding_name: The tiktoken encoding to use (default: cl100k_base)

    Returns:
        Truncated text
    """
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)

    if len(tokens) <= max_tokens:
        return text

    # Truncate to max_tokens and decode back to text
    truncated_tokens = tokens[:max_tokens]
    return enc.decode(truncated_tokens)


def validate_config(
    config: dict[str, Any], config_path: str = "config.yaml"
) -> list[str]:
    """Validate configuration and return list of error messages.

    Args:
        config: The configuration dictionary to validate
        config_path: Path to config file (for error messages)

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Validate project_id
    if "project_id" not in config:
        errors.append("❌ Missing required field: 'project_id'")
    elif not isinstance(config["project_id"], str):
        errors.append(
            f"❌ 'project_id' must be a string, got {type(config['project_id']).__name__}"
        )
    elif not config["project_id"].strip():
        errors.append("❌ 'project_id' cannot be empty")

    # Validate data section
    if "data" not in config:
        errors.append("❌ Missing required section: 'data'")
    else:
        data = config["data"]
        if not isinstance(data, dict):
            errors.append(f"❌ 'data' must be a dictionary, got {type(data).__name__}")
        else:
            # Check for data directory
            if "mteb_data_dir" not in data:
                errors.append(
                    "❌ Missing required field: 'data.mteb_data_dir'\n"
                    "   💡 Specify the path to your MTEB format data directory"
                )
            elif not isinstance(data["mteb_data_dir"], str):
                errors.append(
                    f"❌ 'data.mteb_data_dir' must be a string, got {type(data['mteb_data_dir']).__name__}"
                )
            else:
                # Check if directory exists
                data_dir = Path(data["mteb_data_dir"])
                if not data_dir.exists():
                    errors.append(
                        f"❌ Data directory does not exist: '{data['mteb_data_dir']}'\n"
                        f"   💡 Create the directory or update the path in {config_path}"
                    )
                elif not data_dir.is_dir():
                    errors.append(
                        f"❌ 'data.mteb_data_dir' is not a directory: '{data['mteb_data_dir']}'"
                    )

    # Validate embeddings section
    if "embeddings" not in config:
        errors.append("❌ Missing required section: 'embeddings'")
    else:
        embeddings = config["embeddings"]
        if not isinstance(embeddings, dict):
            errors.append(
                f"❌ 'embeddings' must be a dictionary, got {type(embeddings).__name__}"
            )
        else:
            # Validate device
            if "device" in embeddings:
                valid_devices = ["cpu", "cuda", "mps", "auto"]
                if embeddings["device"] not in valid_devices:
                    errors.append(
                        f"❌ 'embeddings.device' must be one of {valid_devices}, got '{embeddings['device']}'"
                    )

            # Validate cache_dir
            if "cache_dir" in embeddings:
                if not isinstance(embeddings["cache_dir"], str):
                    errors.append(
                        f"❌ 'embeddings.cache_dir' must be a string, got {type(embeddings['cache_dir']).__name__}"
                    )

            # Check that at least one model source is configured
            has_huggingface = "huggingface" in embeddings and isinstance(
                embeddings["huggingface"], dict
            )
            has_colbert = "colbert" in embeddings and isinstance(
                embeddings["colbert"], dict
            )
            has_openrouter = (
                "openrouter" in embeddings
                and isinstance(embeddings["openrouter"], dict)
                and "models" in embeddings["openrouter"]
                and isinstance(embeddings["openrouter"]["models"], dict)
            )

            if not has_huggingface and not has_colbert and not has_openrouter:
                errors.append(
                    "❌ No embedding models configured\n"
                    "   💡 Add models under 'embeddings.huggingface', 'embeddings.colbert', or 'embeddings.openrouter.models'"
                )

            # Validate HuggingFace models if present
            if "huggingface" in embeddings:
                hf = embeddings["huggingface"]
                if not isinstance(hf, dict):
                    errors.append(
                        f"❌ 'embeddings.huggingface' must be a dictionary, got {type(hf).__name__}"
                    )
                elif len(hf) == 0:
                    pass  # Empty is ok if OpenRouter has models
                else:
                    # Validate each model config
                    for model_name, model_config in hf.items():
                        if isinstance(model_config, str):
                            # Simple string format is valid
                            pass
                        elif isinstance(model_config, dict):
                            # Dict format must have 'model' key
                            if "model" not in model_config:
                                errors.append(
                                    f"❌ HuggingFace model '{model_name}' is a dict but missing required 'model' field\n"
                                    f"   💡 Add 'model: org/model-id' or use simple format: '{model_name}: org/model-id'"
                                )
                            # Validate boolean flags
                            bool_flags = [
                                "use_query_prefix",
                                "use_passage_prefix",
                                "use_query_prompt",
                                "use_passage_prompt",
                            ]
                            for flag in bool_flags:
                                if flag in model_config and not isinstance(
                                    model_config[flag], bool
                                ):
                                    errors.append(
                                        f"❌ '{model_name}.{flag}' must be a boolean (true/false), got {type(model_config[flag]).__name__}"
                                    )
                        else:
                            errors.append(
                                f"❌ HuggingFace model '{model_name}' must be a string or dict, got {type(model_config).__name__}"
                            )

            # Validate ColBERT models if present
            if "colbert" in embeddings:
                colbert = embeddings["colbert"]
                if not isinstance(colbert, dict):
                    errors.append(
                        f"❌ 'embeddings.colbert' must be a dictionary, got {type(colbert).__name__}"
                    )
                elif len(colbert) == 0:
                    pass  # Empty is ok if other model sources have models
                else:
                    # Validate each ColBERT model config
                    for model_name, model_config in colbert.items():
                        if isinstance(model_config, str):
                            # Simple string format is valid
                            pass
                        elif isinstance(model_config, dict):
                            # Dict format must have 'model' key
                            if "model" not in model_config:
                                errors.append(
                                    f"❌ ColBERT model '{model_name}' is a dict but missing required 'model' field\n"
                                    f"   💡 Add 'model: org/model-id' or use simple format: '{model_name}: org/model-id'"
                                )
                        else:
                            errors.append(
                                f"❌ ColBERT model '{model_name}' must be a string or dict, got {type(model_config).__name__}"
                            )

            # Validate OpenRouter models if present
            if "openrouter" in embeddings:
                openrouter = embeddings["openrouter"]
                if not isinstance(openrouter, dict):
                    errors.append(
                        f"❌ 'embeddings.openrouter' must be a dictionary, got {type(openrouter).__name__}"
                    )
                else:
                    if "models" in openrouter:
                        models = openrouter["models"]
                        if not isinstance(models, dict):
                            errors.append(
                                f"❌ 'embeddings.openrouter.models' must be a dictionary, got {type(models).__name__}"
                            )
                        else:
                            # Check that models dict has entries (if configured)
                            for model_name, model_id in models.items():
                                if not isinstance(model_id, str):
                                    errors.append(
                                        f"❌ OpenRouter model '{model_name}' must be a string (model ID), got {type(model_id).__name__}"
                                    )

                    if "settings" in openrouter:
                        settings = openrouter["settings"]
                        if not isinstance(settings, dict):
                            errors.append(
                                f"❌ 'embeddings.openrouter.settings' must be a dictionary, got {type(settings).__name__}"
                            )
                        else:
                            if "api_batch_size" in settings:
                                batch_size = settings["api_batch_size"]
                                if not isinstance(batch_size, int):
                                    errors.append(
                                        f"❌ 'embeddings.openrouter.settings.api_batch_size' must be an integer, got {type(batch_size).__name__}"
                                    )
                                elif batch_size <= 0:
                                    errors.append(
                                        f"❌ 'embeddings.openrouter.settings.api_batch_size' must be positive, got {batch_size}"
                                    )

    # Validate search section
    if "search" not in config:
        errors.append("❌ Missing required section: 'search'")
    else:
        search = config["search"]
        if not isinstance(search, dict):
            errors.append(
                f"❌ 'search' must be a dictionary, got {type(search).__name__}"
            )
        else:
            # Validate alpha
            if "alpha" not in search:
                errors.append(
                    "❌ Missing required field: 'search.alpha'\n"
                    "   💡 Add a list of alpha values, e.g., alpha: [0.0, 0.5, 1.0]"
                )
            elif not isinstance(search["alpha"], list):
                errors.append(
                    f"❌ 'search.alpha' must be a list, got {type(search['alpha']).__name__}\n"
                    "   💡 Use list format: alpha: [0.0, 0.5, 1.0]"
                )
            elif len(search["alpha"]) == 0:
                errors.append("❌ 'search.alpha' cannot be empty")
            else:
                for i, alpha in enumerate(search["alpha"]):
                    if not isinstance(alpha, (int, float)):
                        errors.append(
                            f"❌ 'search.alpha[{i}]' must be a number, got {type(alpha).__name__}"
                        )
                    elif not (0.0 <= alpha <= 1.0):
                        errors.append(
                            f"❌ 'search.alpha[{i}]' must be between 0.0 and 1.0, got {alpha}\n"
                            "   💡 0.0 = pure lexical (BM25), 1.0 = pure semantic, 0.5 = balanced hybrid"
                        )

            # Validate metrics section (new format) or top_k (legacy format)
            has_metrics = "metrics" in search
            has_top_k = "top_k" in search

            if not has_metrics and not has_top_k:
                errors.append(
                    "❌ Missing required field: 'search.metrics'\n"
                    "   💡 Add metrics with K values, e.g.:\n"
                    "      metrics:\n"
                    "        mrr_k: [10]\n"
                    "        hit_rate_k: [10]"
                )
            elif has_metrics:
                metrics = search["metrics"]
                if not isinstance(metrics, dict):
                    errors.append(
                        f"❌ 'search.metrics' must be a dictionary, got {type(metrics).__name__}"
                    )
                else:
                    # Validate each metric's K values
                    for metric_name in ["mrr_k", "hit_rate_k"]:
                        if metric_name in metrics:
                            k_list = metrics[metric_name]
                            if not isinstance(k_list, list):
                                errors.append(
                                    f"❌ 'search.metrics.{metric_name}' must be a list, got {type(k_list).__name__}"
                                )
                            elif len(k_list) == 0:
                                errors.append(
                                    f"❌ 'search.metrics.{metric_name}' cannot be empty"
                                )
                            else:
                                for i, k in enumerate(k_list):
                                    if not isinstance(k, int):
                                        errors.append(
                                            f"❌ 'search.metrics.{metric_name}[{i}]' must be an integer, got {type(k).__name__}"
                                        )
                                    elif k <= 0:
                                        errors.append(
                                            f"❌ 'search.metrics.{metric_name}[{i}]' must be positive, got {k}"
                                        )
            elif has_top_k:
                # Legacy format validation
                if not isinstance(search["top_k"], list):
                    errors.append(
                        f"❌ 'search.top_k' must be a list, got {type(search['top_k']).__name__}\n"
                        "   💡 Use list format: top_k: [10, 20, 50]"
                    )
                elif len(search["top_k"]) == 0:
                    errors.append("❌ 'search.top_k' cannot be empty")
                else:
                    for i, k in enumerate(search["top_k"]):
                        if not isinstance(k, int):
                            errors.append(
                                f"❌ 'search.top_k[{i}]' must be an integer, got {type(k).__name__}"
                            )
                        elif k <= 0:
                            errors.append(
                                f"❌ 'search.top_k[{i}]' must be positive, got {k}"
                            )

            # Validate include_bm25_baseline (optional)
            if "include_bm25_baseline" in search:
                if not isinstance(search["include_bm25_baseline"], bool):
                    errors.append(
                        f"❌ 'search.include_bm25_baseline' must be a boolean (true/false), got {type(search['include_bm25_baseline']).__name__}"
                    )

    # Validate output section
    if "output" not in config:
        errors.append("❌ Missing required section: 'output'")
    else:
        output = config["output"]
        if not isinstance(output, dict):
            errors.append(
                f"❌ 'output' must be a dictionary, got {type(output).__name__}"
            )
        else:
            if "results_dir" not in output:
                errors.append(
                    "❌ Missing required field: 'output.results_dir'\n"
                    "   💡 Add results_dir: './_results'"
                )
            elif not isinstance(output["results_dir"], str):
                errors.append(
                    f"❌ 'output.results_dir' must be a string, got {type(output['results_dir']).__name__}"
                )

    # Validate visualization section
    if "visualization" not in config:
        errors.append("❌ Missing required section: 'visualization'")
    else:
        viz = config["visualization"]
        if not isinstance(viz, dict):
            errors.append(
                f"❌ 'visualization' must be a dictionary, got {type(viz).__name__}"
            )
        # Visualization fields are optional, so we don't validate individual fields

    # Validate model section
    if "model" not in config:
        errors.append("❌ Missing required section: 'model'")
    else:
        model = config["model"]
        if not isinstance(model, dict):
            errors.append(
                f"❌ 'model' must be a dictionary, got {type(model).__name__}"
            )
        else:
            # Validate embedding_batch_size (optional)
            if "embedding_batch_size" in model:
                batch_size = model["embedding_batch_size"]
                if not isinstance(batch_size, int):
                    errors.append(
                        f"❌ 'model.embedding_batch_size' must be an integer, got {type(batch_size).__name__}"
                    )
                elif batch_size <= 0:
                    errors.append(
                        f"❌ 'model.embedding_batch_size' must be positive, got {batch_size}"
                    )

            # Validate max_document_tokens (optional)
            if "max_document_tokens" in model:
                max_tokens = model["max_document_tokens"]
                if not isinstance(max_tokens, int):
                    errors.append(
                        f"❌ 'model.max_document_tokens' must be an integer, got {type(max_tokens).__name__}"
                    )
                elif max_tokens <= 0:
                    errors.append(
                        f"❌ 'model.max_document_tokens' must be positive, got {max_tokens}"
                    )

    # Validate query_generation section (optional but if present, validate)
    if "query_generation" in config:
        qg = config["query_generation"]
        if not isinstance(qg, dict):
            errors.append(
                f"❌ 'query_generation' must be a dictionary, got {type(qg).__name__}"
            )

    # Validate evals section (optional)
    if "evals" in config:
        evals = config["evals"]
        if not isinstance(evals, dict):
            errors.append(
                f"❌ 'evals' must be a dictionary, got {type(evals).__name__}"
            )
        elif "cache_dir" in evals and not isinstance(evals["cache_dir"], str):
            errors.append(
                f"❌ 'evals.cache_dir' must be a string, got {type(evals['cache_dir']).__name__}"
            )

    return errors


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing the configuration settings

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML config file: {e}")

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return config


def load_parquet(file_path: str) -> pd.DataFrame:
    """Load parquet file.

    Args:
        file_path: Path to the parquet file to load

    Returns:
        DataFrame containing the parquet file contents

    Raises:
        FileNotFoundError: If parquet file doesn't exist
    """
    parquet_file = Path(file_path)
    if not parquet_file.exists():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")
    return pd.read_parquet(file_path)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV or Parquet file.

    Automatically detects file format based on extension.
    Supports .csv, .parquet, and .pq extensions.

    Args:
        file_path: Path to the data file to load

    Returns:
        DataFrame containing the file contents

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported
    """
    data_file = Path(file_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Detect file format from extension
    suffix = data_file.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(file_path)
    elif suffix in [".parquet", ".pq", ".parq"]:
        return pd.read_parquet(file_path)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. Supported formats: .csv, .parquet, .parq, .pq"
        )


class MTEBRetrievalData:
    """
    Container for MTEB 2.x retrieval format data.

    MTEB 2.x Retrieval format:
    - corpus: DataFrame with columns ['id', 'text'] (optionally 'title')
    - queries: DataFrame with columns ['id', 'text'] (optionally 'instruction')
    - qrels: DataFrame with columns ['query-id', 'corpus-id'] (optionally 'score')
      If 'score' is missing, a default score of 1 is assumed (binary relevance).

    Attributes:
        corpus: DataFrame containing corpus documents
        queries: DataFrame containing queries
        qrels: DataFrame containing relevance judgments
        corpus_dict: Dict mapping corpus id to document dict
        queries_dict: Dict mapping query id to query dict
        qrels_dict: Dict mapping query id to dict of corpus id -> score
    """

    def __init__(
        self,
        corpus: pd.DataFrame,
        queries: pd.DataFrame,
        qrels: pd.DataFrame,
    ):
        """Initialize MTEB retrieval data container.

        Args:
            corpus: DataFrame with columns ['id', 'text'] (optionally 'title')
            queries: DataFrame with columns ['id', 'text']
            qrels: DataFrame with columns ['query-id', 'corpus-id'] (optionally 'score')
                   If 'score' is missing, a default score of 1 is assumed.
        """
        self.corpus = corpus
        self.queries = queries
        self.qrels = qrels

        # Validate required columns
        self._validate()

        # Build dictionaries for fast lookup
        self._build_dicts()

    def _validate(self) -> None:
        """Validate that required columns exist."""
        # Corpus validation
        if "id" not in self.corpus.columns or "text" not in self.corpus.columns:
            raise ValueError("Corpus must have 'id' and 'text' columns")

        # Queries validation
        if "id" not in self.queries.columns or "text" not in self.queries.columns:
            raise ValueError("Queries must have 'id' and 'text' columns")

        # Qrels validation
        required_qrels_cols = {"query-id", "corpus-id"}
        if not required_qrels_cols.issubset(self.qrels.columns):
            raise ValueError(f"Qrels must have columns: {required_qrels_cols}")
        
        # Add default score of 1 if score column is missing (binary relevance)
        if "score" not in self.qrels.columns:
            self.qrels["score"] = 1

    def _build_dicts(self) -> None:
        """Build dictionary representations for fast lookup."""
        # Corpus dict: id -> {"id": id, "text": text, "title": title}
        self.corpus_dict: dict[str, dict[str, Any]] = {}
        for _, row in self.corpus.iterrows():
            doc_id = str(row["id"])
            self.corpus_dict[doc_id] = {
                "id": doc_id,
                "text": row["text"],
                "title": row.get("title", ""),
            }

        # Queries dict: id -> {"id": id, "text": text}
        self.queries_dict: dict[str, dict[str, Any]] = {}
        for _, row in self.queries.iterrows():
            query_id = str(row["id"])
            self.queries_dict[query_id] = {
                "id": query_id,
                "text": row["text"],
            }

        # Qrels dict: query_id -> {corpus_id: score}
        self.qrels_dict: dict[str, dict[str, int]] = {}
        for _, row in self.qrels.iterrows():
            query_id = str(row["query-id"])
            corpus_id = str(row["corpus-id"])
            score = int(row["score"])

            if query_id not in self.qrels_dict:
                self.qrels_dict[query_id] = {}
            self.qrels_dict[query_id][corpus_id] = score

    def get_documents_list(self, max_tokens: int | None = None) -> list[dict[str, str]]:
        """Get list of documents in format compatible with existing code.

        Args:
            max_tokens: If provided, truncate text to this many tokens

        Returns:
            List of dicts with 'id' and 'text' keys
        """
        docs = []
        for doc_id, doc in self.corpus_dict.items():
            text = doc["text"]
            if max_tokens is not None:
                text = truncate_text_to_tokens(text, max_tokens)
            docs.append({"id": doc_id, "text": text})
        return docs

    def get_queries_list(self) -> list[dict[str, Any]]:
        """Get list of queries with relevant document IDs.

        Only documents with relevance score > 0 are considered relevant.
        In MTEB format, score=0 indicates non-relevant (negative examples),
        while score>=1 indicates relevant documents.

        Returns:
            List of dicts with 'id', 'query', and 'relevant_ids' keys
        """
        queries = []
        for query_id, query in self.queries_dict.items():
            qrels_for_query = self.qrels_dict.get(query_id, {})
            # Only include documents with score > 0 (score=0 means not relevant)
            relevant_ids = [
                corpus_id for corpus_id, score in qrels_for_query.items() if score > 0
            ]
            queries.append(
                {
                    "id": query_id,
                    "query": query["text"],
                    "relevant_ids": relevant_ids,
                }
            )
        return queries

    @property
    def num_documents(self) -> int:
        """Return number of documents in corpus."""
        return len(self.corpus)

    @property
    def num_queries(self) -> int:
        """Return number of queries."""
        return len(self.queries)

    @property
    def num_qrels(self) -> int:
        """Return number of relevance judgments."""
        return len(self.qrels)


def load_mteb_retrieval_data(
    corpus_path: str,
    queries_path: str,
    qrels_path: str,
) -> MTEBRetrievalData:
    """Load MTEB 2.x retrieval format data from files.

    Supports datasets with or without a 'score' column in qrels.
    For datasets without scores (e.g., NanoBEIR), assumes binary relevance (score=1).

    Args:
        corpus_path: Path to corpus file (parquet or csv)
        queries_path: Path to queries file (parquet or csv)
        qrels_path: Path to qrels file (parquet or csv with optional 'score' column)

    Returns:
        MTEBRetrievalData container with loaded data
    """
    corpus = load_data(corpus_path)
    queries = load_data(queries_path)
    qrels = load_data(qrels_path)

    return MTEBRetrievalData(corpus=corpus, queries=queries, qrels=qrels)


def load_mteb_retrieval_data_from_dir(
    data_dir: str,
    corpus_filename: str = "corpus.parquet",
    queries_filename: str = "queries.parquet",
    qrels_filename: str = "qrels.parquet",
) -> MTEBRetrievalData:
    """Load MTEB 2.x retrieval format data from a directory.

    Args:
        data_dir: Directory containing the data files
        corpus_filename: Name of corpus file
        queries_filename: Name of queries file
        qrels_filename: Name of qrels file

    Returns:
        MTEBRetrievalData container with loaded data
    """
    data_dir = Path(data_dir)

    return load_mteb_retrieval_data(
        corpus_path=str(data_dir / corpus_filename),
        queries_path=str(data_dir / queries_filename),
        qrels_path=str(data_dir / qrels_filename),
    )


def get_openrouter_embeddings(
    texts: list[str],
    model: str = "openai/text-embedding-3-small",
    batch_size: int = 100,
    api_key: str | None = None,
) -> np.ndarray:
    """
    Generate embeddings using OpenRouter API.

    Args:
        texts: List of texts to embed
        model: OpenRouter embedding model name (e.g., "openai/text-embedding-3-small")
        batch_size: Number of texts to process in each API call
        api_key: OpenRouter API key (if None, will use OPENROUTER_API_KEY env var)

    Returns:
        NumPy array of embeddings
    """
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY in .env file or pass api_key parameter."
            )

    all_embeddings = []
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = "https://openrouter.ai/api/v1/embeddings"

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        payload = {
            "input": batch,
            "model": model,
        }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        batch_embeddings = [item["embedding"] for item in data["data"]]
        all_embeddings.extend(batch_embeddings)

    return np.array(all_embeddings, dtype=np.float32)


def is_openrouter_model(model_name: str) -> bool:
    """Check if a model name refers to an OpenRouter embedding model.

    OpenRouter models typically use the format "provider/model-name".

    Args:
        model_name: Name of the embedding model to check

    Returns:
        True if the model is an OpenRouter model, False otherwise
    """
    # OpenRouter models use provider/model format
    return "/" in model_name


def parse_model_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse model configurations from the new YAML structure.

    Converts the nested embeddings.huggingface, embeddings.colbert, and
    embeddings.openrouter dictionaries into a flat list of model configurations
    with their parameters.

    Handles duplicate model names across providers by appending suffixes
    (_hf, _colbert, or _or) to disambiguate them.

    Args:
        config: The full configuration dictionary

    Returns:
        List of model config dictionaries, each containing:
        - model_id: The full model identifier (e.g., "intfloat/multilingual-e5-small")
        - model_name: Short name for display (e.g., "e5-small")
        - query_prefix: String to prepend to queries (e.g., "query: ")
        - passage_prefix: String to prepend to passages (e.g., "passage: ")
        - query_encode_kwargs: Dict of kwargs for query encoding
        - passage_encode_kwargs: Dict of kwargs for passage encoding
        - is_openrouter: Boolean indicating if this is an OpenRouter model
        - is_colbert: Boolean indicating if this is a ColBERT late-interaction model
    """
    model_configs = []
    embeddings_config = config.get("embeddings", {})

    # Collect all model names first to detect duplicates
    hf_models = embeddings_config.get("huggingface", {})
    colbert_models = embeddings_config.get("colbert", {})
    openrouter_config = embeddings_config.get("openrouter", {})
    openrouter_models = openrouter_config.get("models", {})

    hf_names = set(hf_models.keys()) if hf_models else set()
    colbert_names = set(colbert_models.keys()) if colbert_models else set()
    or_names = set(openrouter_models.keys()) if openrouter_models else set()

    # Known ColBERT model identifiers (substrings to check in model IDs)
    # These models use late-interaction and MaxSim scoring, requiring pylate
    known_colbert_patterns = [
        "colbert",
        "ColBERT",
        "answerai-colbert",
        "GTE-ModernColBERT",
    ]

    # Check if any HuggingFace models appear to be ColBERT models
    for model_name, model_spec in hf_models.items():
        model_id = model_spec if isinstance(model_spec, str) else model_spec.get("model", "")
        is_likely_colbert = any(pattern.lower() in model_id.lower() for pattern in known_colbert_patterns)
        if is_likely_colbert:
            console.print(
                f"\n⚠️  [bold yellow]WARNING: Potential ColBERT model in wrong section![/bold yellow]"
            )
            console.print(
                f"   Model [cyan]'{model_name}'[/cyan] ([dim]{model_id}[/dim]) appears to be a ColBERT model."
            )
            console.print(
                "   ColBERT models use late-interaction (MaxSim) scoring and must be loaded via PyLate."
            )
            console.print(
                "   \n   [bold]Please move this model to the 'embeddings.colbert' section in your config:[/bold]"
            )
            console.print(
                f"   [green]embeddings:\n     colbert:\n       {model_name}: {model_id}[/green]\n"
            )
            console.print(
                "   [dim]If loaded under 'huggingface', it will fail with: 'MaxSim' is not a valid SimilarityFunction[/dim]\n"
            )

    # Find names that appear in multiple providers
    all_names = [hf_names, colbert_names, or_names]
    duplicate_names: set[str] = set()
    for i, names_i in enumerate(all_names):
        for j, names_j in enumerate(all_names):
            if i < j:
                duplicate_names |= names_i & names_j

    if duplicate_names:
        console.print(
            f"   ⚠️  [yellow]Duplicate model names found across providers: {sorted(duplicate_names)}[/yellow]"
        )
        console.print("      [dim]Adding provider suffixes to disambiguate[/dim]")

    # Parse HuggingFace models
    for model_name, model_spec in hf_models.items():
        # Handle both simple string and dict formats
        if isinstance(model_spec, str):
            # Simple format: "model-name: org/model-id"
            model_id = model_spec
            use_query_prefix = False
            use_passage_prefix = False
            use_query_prompt = False
            use_passage_prompt = False
        else:
            # Dict format with options
            model_id = model_spec.get("model")
            if not model_id:
                console.print(
                    f"⚠️  [yellow]Skipping {model_name}: no 'model' field specified[/yellow]"
                )
                continue
            use_query_prefix = model_spec.get("use_query_prefix", False)
            use_passage_prefix = model_spec.get("use_passage_prefix", False)
            use_query_prompt = model_spec.get("use_query_prompt", False)
            use_passage_prompt = model_spec.get("use_passage_prompt", False)

        # Build prefixes
        query_prefix = "query: " if use_query_prefix else ""
        passage_prefix = "passage: " if use_passage_prefix else ""

        # Build encode kwargs
        query_encode_kwargs = {}
        passage_encode_kwargs = {}
        if use_query_prompt:
            query_encode_kwargs["prompt_name"] = "query"
        if use_passage_prompt:
            passage_encode_kwargs["prompt_name"] = "passage"

        # Disambiguate display name if it appears in multiple providers
        display_name = (
            f"{model_name}_hf" if model_name in duplicate_names else model_name
        )

        model_configs.append(
            {
                "model_id": model_id,
                "model_name": display_name,
                "query_prefix": query_prefix,
                "passage_prefix": passage_prefix,
                "query_encode_kwargs": query_encode_kwargs,
                "passage_encode_kwargs": passage_encode_kwargs,
                "is_openrouter": False,
                "is_colbert": False,
            }
        )

    # Parse ColBERT models
    for model_name, model_spec in colbert_models.items():
        # Handle both simple string and dict formats
        if isinstance(model_spec, str):
            # Simple format: "model-name: org/model-id"
            model_id = model_spec
        else:
            # Dict format with options
            model_id = model_spec.get("model")
            if not model_id:
                console.print(
                    f"⚠️  [yellow]Skipping ColBERT {model_name}: no 'model' field specified[/yellow]"
                )
                continue

        # Disambiguate display name if it appears in multiple providers
        display_name = (
            f"{model_name}_colbert" if model_name in duplicate_names else model_name
        )

        model_configs.append(
            {
                "model_id": model_id,
                "model_name": display_name,
                "query_prefix": "",
                "passage_prefix": "",
                "query_encode_kwargs": {},
                "passage_encode_kwargs": {},
                "is_openrouter": False,
                "is_colbert": True,
            }
        )

    # Parse OpenRouter models
    if openrouter_models:  # Only process if models are defined
        for model_name, model_spec in openrouter_models.items():
            # OpenRouter models are always simple strings (provider/model format)
            if isinstance(model_spec, str):
                model_id = model_spec
            else:
                model_id = model_spec.get("model")
                if not model_id:
                    console.print(
                        f"⚠️  [yellow]Skipping {model_name}: no 'model' field specified[/yellow]"
                    )
                    continue

            # Disambiguate display name if it appears in multiple providers
            display_name = (
                f"{model_name}_or" if model_name in duplicate_names else model_name
            )

            model_configs.append(
                {
                    "model_id": model_id,
                    "model_name": display_name,
                    "query_prefix": "",
                    "passage_prefix": "",
                    "query_encode_kwargs": {},
                    "passage_encode_kwargs": {},
                    "is_openrouter": True,
                    "is_colbert": False,
                }
            )

    # Check for duplicate display names after disambiguation (shouldn't happen, but safety check)
    final_names = [cfg["model_name"] for cfg in model_configs]
    seen_names: dict[str, int] = {}
    for i, name in enumerate(final_names):
        if name in seen_names:
            # This shouldn't happen with proper config, but handle it gracefully
            counter = 2
            new_name = f"{name}_{counter}"
            while new_name in seen_names or new_name in final_names:
                counter += 1
                new_name = f"{name}_{counter}"
            model_configs[i]["model_name"] = new_name
            console.print(
                f"   ⚠️  [yellow]Duplicate name '{name}' renamed to '{new_name}'[/yellow]"
            )
        seen_names[name] = i

    return model_configs


def generate_cache_key(project_id: str, model_name: str, data_type: str) -> str:
    """Generate a unique cache key for embeddings.

    Args:
        project_id: The project identifier
        model_name: The embedding model name
        data_type: Either 'documents' or 'queries'

    Returns:
        A hash-based cache key
    """
    # Create a deterministic key based on project_id, model, and data type
    key_string = f"{project_id}_{model_name}_{data_type}"
    # Use hash for cleaner filenames
    hash_suffix = hashlib.md5(key_string.encode()).hexdigest()[:8]
    # Use model short name for readability
    model_short = model_name.split("/")[-1]
    return f"{project_id}_{model_short}_{data_type}_{hash_suffix}"


def save_embeddings(
    embeddings: np.ndarray,
    cache_key: str,
    embeddings_dir: Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save embeddings to disk with metadata.

    Args:
        embeddings: The embeddings array to save
        cache_key: Unique identifier for this embedding set
        embeddings_dir: Directory to save embeddings
        metadata: Optional metadata to save alongside embeddings

    Returns:
        Path to the saved embeddings file
    """
    embeddings_dir.mkdir(exist_ok=True, parents=True)

    # Save embeddings as numpy array
    embeddings_file = embeddings_dir / f"{cache_key}.npy"
    np.save(embeddings_file, embeddings)

    # Save metadata as JSON
    if metadata:
        metadata_file = embeddings_dir / f"{cache_key}.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    return embeddings_file


def load_embeddings(
    cache_key: str, embeddings_dir: Path
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load embeddings and metadata from disk.

    Args:
        cache_key: Unique identifier for this embedding set
        embeddings_dir: Directory where embeddings are stored

    Returns:
        Tuple of (embeddings array, metadata dict)
    """
    embeddings_file = embeddings_dir / f"{cache_key}.npy"
    metadata_file = embeddings_dir / f"{cache_key}.json"

    if not embeddings_file.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")

    embeddings = np.load(embeddings_file)

    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

    return embeddings, metadata


def embeddings_exist(cache_key: str, embeddings_dir: Path) -> bool:
    """Check if cached embeddings exist.

    Args:
        cache_key: Unique identifier for this embedding set
        embeddings_dir: Directory where embeddings are stored

    Returns:
        True if embeddings exist, False otherwise
    """
    embeddings_file = embeddings_dir / f"{cache_key}.npy"
    return embeddings_file.exists()


def save_colbert_embeddings(
    embeddings: list[np.ndarray],
    cache_key: str,
    embeddings_dir: Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save ColBERT multi-vector embeddings to disk with metadata.

    ColBERT produces variable-length token embeddings per document/query,
    so we store them as a list of 2D arrays using numpy's object dtype.

    Args:
        embeddings: List of 2D arrays, each of shape (num_tokens, embedding_dim)
        cache_key: Unique identifier for this embedding set
        embeddings_dir: Directory to save embeddings
        metadata: Optional metadata to save alongside embeddings

    Returns:
        Path to the saved embeddings file
    """
    embeddings_dir.mkdir(exist_ok=True, parents=True)

    # Save embeddings as numpy array with object dtype to handle variable lengths
    embeddings_file = embeddings_dir / f"{cache_key}_colbert.npz"
    # Convert to object array for variable-length storage
    np.savez_compressed(
        embeddings_file,
        **{f"emb_{i}": emb for i, emb in enumerate(embeddings)},
        _count=np.array([len(embeddings)]),
    )

    # Save metadata as JSON
    if metadata:
        metadata["is_colbert"] = True
        metadata_file = embeddings_dir / f"{cache_key}_colbert.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    return embeddings_file


def load_colbert_embeddings(
    cache_key: str, embeddings_dir: Path
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Load ColBERT multi-vector embeddings and metadata from disk.

    Args:
        cache_key: Unique identifier for this embedding set
        embeddings_dir: Directory where embeddings are stored

    Returns:
        Tuple of (list of embedding arrays, metadata dict)
    """
    embeddings_file = embeddings_dir / f"{cache_key}_colbert.npz"
    metadata_file = embeddings_dir / f"{cache_key}_colbert.json"

    if not embeddings_file.exists():
        raise FileNotFoundError(f"ColBERT embeddings file not found: {embeddings_file}")

    # Load embeddings from npz
    data = np.load(embeddings_file, allow_pickle=True)
    count = int(data["_count"][0])
    embeddings = [data[f"emb_{i}"] for i in range(count)]

    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

    return embeddings, metadata


def colbert_embeddings_exist(cache_key: str, embeddings_dir: Path) -> bool:
    """Check if cached ColBERT embeddings exist.

    Args:
        cache_key: Unique identifier for this embedding set
        embeddings_dir: Directory where embeddings are stored

    Returns:
        True if ColBERT embeddings exist, False otherwise
    """
    embeddings_file = embeddings_dir / f"{cache_key}_colbert.npz"
    return embeddings_file.exists()


def generate_eval_cache_key(
    project_id: str, model_id: str, alpha: float, k: int, display_name: str = ""
) -> str:
    """Generate a unique cache key for evaluation results.

    Args:
        project_id: The project identifier
        model_id: The embedding model identifier (e.g., "intfloat/e5-large-v2")
        alpha: The alpha parameter for hybrid search
        k: The top-k parameter
        display_name: Optional display name to disambiguate models with same model_id
                      (e.g., when same model is run via HuggingFace and OpenRouter)

    Returns:
        A hash-based cache key
    """
    # Create a deterministic key based on all eval parameters
    # Include display_name in hash to disambiguate same model_id used via different providers
    key_string = f"{project_id}_{model_id}_{display_name}_alpha{alpha}_k{k}"
    # Use hash for cleaner filenames
    hash_suffix = hashlib.md5(key_string.encode()).hexdigest()[:8]
    # Use display name if provided, otherwise extract from model_id
    model_short = display_name if display_name else model_id.split("/")[-1]
    return f"{project_id}_{model_short}_a{alpha}_k{k}_{hash_suffix}"


def save_eval_results(
    results: dict[str, Any],
    cache_key: str,
    evals_dir: Path,
) -> Path:
    """Save evaluation results to disk.

    Args:
        results: The evaluation results dictionary to save
        cache_key: Unique identifier for this eval set
        evals_dir: Directory to save eval results

    Returns:
        Path to the saved results file
    """
    evals_dir.mkdir(exist_ok=True, parents=True)

    # Save results as JSON
    results_file = evals_dir / f"{cache_key}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    return results_file


def load_eval_results(cache_key: str, evals_dir: Path) -> dict[str, Any]:
    """Load evaluation results from disk.

    Args:
        cache_key: Unique identifier for this eval set
        evals_dir: Directory where eval results are stored

    Returns:
        Dictionary containing the evaluation results
    """
    results_file = evals_dir / f"{cache_key}.json"

    if not results_file.exists():
        raise FileNotFoundError(f"Eval results file not found: {results_file}")

    with open(results_file, "r") as f:
        results = json.load(f)

    return results


def eval_results_exist(cache_key: str, evals_dir: Path) -> bool:
    """Check if cached evaluation results exist.

    Args:
        cache_key: Unique identifier for this eval set
        evals_dir: Directory where eval results are stored

    Returns:
        True if eval results exist, False otherwise
    """
    results_file = evals_dir / f"{cache_key}.json"
    return results_file.exists()


def calculate_reciprocal_rank(
    retrieved_ids: list[str], relevant_ids: list[str]
) -> float:
    """Calculate Reciprocal Rank (RR) for a single query.

    RR is calculated as 1/rank where rank is the position of the first
    relevant document in the retrieved results. Returns 0.0 if no relevant
    documents are found.

    Note: MRR (Mean Reciprocal Rank) is the average of RR scores across
    all queries in the evaluation set.

    Args:
        retrieved_ids: List of retrieved document IDs in ranked order
        relevant_ids: List of relevant document IDs for the query

    Returns:
        RR score (1/rank of first relevant doc, or 0.0 if none found)
    """
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)

    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_set:
            return 1.0 / rank

    return 0.0


def calculate_hit(retrieved_ids: list[str], relevant_ids: list[str]) -> int:
    """Calculate Hit (binary) for a single query.

    Hit is 1 if any relevant document appears in the retrieved results,
    otherwise 0. Hit@k (averaged over queries) gives the "success rate".

    Args:
        retrieved_ids: List of retrieved document IDs in ranked order
        relevant_ids: List of relevant document IDs for the query

    Returns:
        1 if any relevant doc is found in retrieved_ids, else 0
    """
    if not relevant_ids:
        return 0

    relevant_set = set(relevant_ids)
    for doc_id in retrieved_ids:
        if doc_id in relevant_set:
            return 1
    return 0


def create_results_visualization(
    results: list[dict[str, Any]],
    output_dir: Path,
    timestamp: str,
    config: dict[str, Any],
) -> None:
    """Create and save seaborn bar charts of evaluation results.

    Generates visualization charts for each metric at each configured K value:
    - MRR@K (Mean Reciprocal Rank)
    - Hit Rate@K (Success Rate)
    - Embedding time comparison across models

    Charts are saved as PNG files with configurable styling based on
    visualization settings in the config.

    Args:
        results: List of evaluation result dictionaries containing metric scores,
                 model names, alpha values, and timing information
        output_dir: Directory where visualization PNG files will be saved
        timestamp: Timestamp string to include in output filenames
        config: Configuration dictionary containing visualization settings
    """
    if not results:
        console.print(
            "\n   ⚠️  [yellow]No evaluation results available; skipping visualization[/yellow]",
            style="dim",
        )
        return

    console.print("\n📊 Creating visualization...", style="bold cyan")

    # Load visualization settings from config
    viz = config.get("visualization", {})
    FIG_SIZE_X = viz.get("fig_size_x", 16)
    TITLE_SIZE = viz.get("title_size", 24)
    AXIS_LABEL_SIZE = viz.get("axis_label_size", 9)
    ROW_LABEL_SIZE = viz.get("row_label_size", 9)
    BAR_LABEL_OFFSET = viz.get("bar_label_offset", 0.02)
    MRR_DYNAMIC_XLIM = viz.get("mrr_dynamic_xlim", True)

    df = pd.DataFrame(results)

    # Create a fixed color palette for consistent colors across all charts
    unique_models = df["model_short"].unique()
    # Use a colorblind-friendly palette
    colors = sns.color_palette("tab10", n_colors=len(unique_models))
    model_colors = {model: colors[i] for i, model in enumerate(unique_models)}

    # Get number of documents and queries
    num_docs = df["num_documents"].iloc[0]
    num_queries = df["num_queries"].iloc[0]

    # Find all metric columns (format: metric@k)
    metric_cols = [col for col in df.columns if "@" in col]
    mrr_cols = sorted([col for col in metric_cols if col.startswith("mrr@")])
    hit_rate_cols = sorted([col for col in metric_cols if col.startswith("hit_rate@")])

    # Helper function to create a metric visualization
    def create_metric_chart(
        metric_col: str,
        title: str,
        xlabel: str,
        output_filename: str,
    ) -> None:
        if metric_col not in df.columns:
            return

        # Create config label for each row
        df_plot = df.copy()
        df_plot["config"] = df_plot.apply(
            lambda row: f"{row['model_short']}\n(α={row['alpha']:.1f})",
            axis=1,
        )

        plt.figure(figsize=(FIG_SIZE_X, max(6, len(df_plot) * 0.8)))

        # Sort by metric value in descending order
        df_sorted = df_plot.sort_values(metric_col, ascending=False)

        ax = sns.barplot(
            data=df_sorted,
            y="config",
            x=metric_col,
            hue="model_short",
            palette=model_colors,
            dodge=False,
            orient="h",
            order=df_sorted["config"].tolist(),
            legend=False,
        )

        plt.title(title, fontsize=TITLE_SIZE, pad=20)
        plt.ylabel("")
        plt.xlabel(xlabel, fontsize=AXIS_LABEL_SIZE)

        # Add document and query counts below the graph
        ax.text(
            0,
            -0.08,
            f"{num_docs} documents, {num_queries} queries",
            transform=ax.transAxes,
            fontsize=AXIS_LABEL_SIZE,
            va="top",
            ha="left",
        )

        # Set x-axis limits
        bar_label_offset = BAR_LABEL_OFFSET
        if MRR_DYNAMIC_XLIM:
            min_val = df_sorted[metric_col].min()
            max_val = df_sorted[metric_col].max()
            lower_limit = max(0, min_val - 0.1 * min_val)
            upper_limit = min(1.0, max_val + 0.1 * max_val)
            plt.xlim(lower_limit, upper_limit)
            bar_label_offset = 0.005
        else:
            plt.xlim(0, 1.1)

        # Style BM25 baseline bars
        for i, row in enumerate(df_sorted.itertuples()):
            if row.model == "BM25" or row.model_short == "Baseline_BM25":
                for bar in ax.patches:
                    bar_y_center = bar.get_y() + bar.get_height() / 2
                    if abs(bar_y_center - i) < 0.1:
                        bar.set_facecolor("#D3D3D3")
                        bar.set_edgecolor("#444444")
                        bar.set_linewidth(1.0)
                        bar.set_hatch("//")
                        break

        # Add value labels at the end of bars
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            ax.text(
                row[metric_col] + bar_label_offset,
                i,
                f"{row[metric_col]:.3f}",
                ha="left",
                va="center",
                fontsize=ROW_LABEL_SIZE,
            )

        plt.tight_layout()
        output_path = output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        console.print(f"   ✓ {title} saved to: [green]{output_path}[/green]")
        plt.close()

    # Create charts for each MRR@K
    for col in mrr_cols:
        k = col.split("@")[1]
        create_metric_chart(
            col,
            f"Mean Reciprocal Rank (MRR) @{k}",
            "MRR",
            f"metric_mean_rr_at{k}_{timestamp}.png",
        )

    # Create charts for each Hit Rate@K
    for col in hit_rate_cols:
        k = col.split("@")[1]
        create_metric_chart(
            col,
            f"Hit Rate (Success Rate) @{k}",
            "Hit Rate",
            f"metric_hit_rate_at{k}_{timestamp}.png",
        )

    # ============= Embedding Time Chart =============
    # Get unique models (embedding time is measured once per model, not per config)
    # Filter out BM25 baseline since it has no embedding time (always 0)
    df_embed = (
        df[df["model"] != "BM25"]
        .groupby("model")
        .agg(
            {
                "total_embed_time_ms": "first",
                "num_documents": "first",
                "model_short": "first",
            }
        )
        .reset_index()
    )

    if not df_embed.empty:
        df_embed = df_embed.sort_values(by="total_embed_time_ms", ascending=False)
        plt.figure(figsize=(FIG_SIZE_X, max(6, len(df_embed) * 0.8)))
        ax = sns.barplot(
            data=df_embed,
            y="model_short",
            x="total_embed_time_ms",
            hue="model_short",
            palette=model_colors,
            dodge=False,
            orient="h",
            legend=False,
        )

        plt.title("Embedding Latency", fontsize=TITLE_SIZE, pad=20)
        plt.ylabel("")
        plt.xlabel("Total Embedding Time (ms)", fontsize=AXIS_LABEL_SIZE)

        # Add document count below the graph
        ax.text(
            0,
            -0.08,
            f"{num_docs} documents",
            transform=ax.transAxes,
            fontsize=AXIS_LABEL_SIZE,
            va="top",
            ha="left",
        )

        # Add value labels at the end of bars
        max_embed_time = df_embed["total_embed_time_ms"].max()
        for i, (_, row) in enumerate(df_embed.iterrows()):
            index_percentage = (row["total_embed_time_ms"] / max_embed_time) * 100
            ax.text(
                row["total_embed_time_ms"] + (max_embed_time * BAR_LABEL_OFFSET),
                i,
                f"{row['total_embed_time_ms']:.1f}ms\n({index_percentage:.0f}%)",
                ha="left",
                va="center",
                fontsize=ROW_LABEL_SIZE,
            )

        plt.tight_layout()
        embed_time_output_path = output_dir / f"embed_time_{timestamp}.png"
        plt.savefig(embed_time_output_path, dpi=300, bbox_inches="tight")
        console.print(
            f"   ✓ Embedding time visualization saved to: [green]{embed_time_output_path}[/green]"
        )
        plt.close()


def create_memory_visualization(
    memory_data: dict[str, dict[str, float]],
    output_dir: Path,
    timestamp: str,
    config: dict[str, Any],
) -> None:
    """Create and save a bar chart of memory consumption per model.

    Generates a horizontal bar chart showing the memory footprint of each
    embedding model, including model loading memory and peak memory during
    embedding generation.

    Args:
        memory_data: Dictionary mapping model names to memory stats dicts
                     containing 'model_memory_mb' and 'peak_memory_mb'
        output_dir: Directory where visualization PNG file will be saved
        timestamp: Timestamp string to include in output filename
        config: Configuration dictionary containing visualization settings
    """
    if not memory_data:
        console.print(
            "\n   ⚠️  [yellow]No memory data available; skipping memory visualization[/yellow]",
            style="dim",
        )
        return

    console.print(
        "\n📊 Creating memory consumption visualization...", style="bold cyan"
    )

    # Load visualization settings from config
    viz = config.get("visualization", {})
    FIG_SIZE_X = viz.get("fig_size_x", 16)
    TITLE_SIZE = viz.get("title_size", 24)
    AXIS_LABEL_SIZE = viz.get("axis_label_size", 9)
    ROW_LABEL_SIZE = viz.get("row_label_size", 9)
    BAR_LABEL_OFFSET = viz.get("bar_label_offset", 0.02)

    # Prepare data for plotting (cap at 0 to handle any negative values from GC)
    models = list(memory_data.keys())
    peak_memory = [max(0.0, memory_data[m]["peak_memory_mb"]) for m in models]
    model_memory = [max(0.0, memory_data[m]["model_memory_mb"]) for m in models]

    # Create DataFrame for easier manipulation
    df_memory = pd.DataFrame(
        {
            "model": models,
            "peak_memory_mb": peak_memory,
            "model_memory_mb": model_memory,
        }
    )

    # Sort by peak memory in descending order
    df_memory = df_memory.sort_values("peak_memory_mb", ascending=False)

    # Create figure
    plt.figure(figsize=(FIG_SIZE_X, max(6, len(df_memory) * 0.8)))

    # Create color palette
    colors = sns.color_palette("Blues_r", n_colors=len(df_memory))

    ax = sns.barplot(
        data=df_memory,
        y="model",
        x="peak_memory_mb",
        hue="model",
        palette=colors,
        dodge=False,
        orient="h",
        legend=False,
    )

    plt.title(
        "Memory Consumption",
        fontsize=TITLE_SIZE,
        pad=20,
    )
    plt.ylabel("")
    plt.xlabel("Peak Memory (MB)", fontsize=AXIS_LABEL_SIZE)

    # Add value labels at the end of bars
    max_memory = df_memory["peak_memory_mb"].max()
    # Ensure max_memory is positive for proper scaling
    max_memory = max(1.0, max_memory)  # Minimum 1 MB for proper label placement
    
    for i, (_, row) in enumerate(df_memory.iterrows()):
        # Show both peak and model-only memory
        label_text = f"{row['peak_memory_mb']:.1f} MB"
        if row["model_memory_mb"] > 0:
            label_text += f"\n(model: {row['model_memory_mb']:.1f} MB)"

        ax.text(
            row["peak_memory_mb"] + (max_memory * BAR_LABEL_OFFSET),
            i,
            label_text,
            ha="left",
            va="center",
            fontsize=ROW_LABEL_SIZE,
        )

    # Extend x-axis to make room for labels
    plt.xlim(0, max_memory * 1.25)

    plt.tight_layout()
    memory_output_path = output_dir / f"memory_{timestamp}.png"
    plt.savefig(memory_output_path, dpi=300, bbox_inches="tight")
    console.print(
        f"   ✓ Memory consumption visualization saved to: [green]{memory_output_path}[/green]"
    )
    plt.close()


def create_tradeoff_visualization(
    results: list[dict[str, Any]],
    memory_data: dict[str, dict[str, float]],
    output_dir: Path,
    timestamp: str,
    config: dict[str, Any],
) -> None:
    """Create bubble chart showing quality vs latency vs memory tradeoffs.

    Generates a scatter plot where:
    - X-axis: Embedding latency (ms per document)
    - Y-axis: Retrieval quality (best MRR@k or configurable metric)
    - Bubble size: Memory consumption (squares for BM25 baseline)
    - Color: Model name (consistent with other charts)
    - Pareto frontier: Highlighted optimal models

    Each model+alpha configuration is shown as a separate point.
    BM25 baseline is shown as a square at x=0 (no embedding latency).

    Args:
        results: List of evaluation result dictionaries
        memory_data: Dictionary mapping model names to memory stats
        output_dir: Directory where visualization will be saved
        timestamp: Timestamp string for output filename
        config: Configuration dictionary containing visualization settings
    """
    if not results:
        console.print(
            "\n   ⚠️  [yellow]Insufficient data for tradeoff visualization[/yellow]",
            style="dim",
        )
        return

    console.print(
        "\n📊 Creating quality-latency-memory tradeoff visualization...",
        style="bold cyan",
    )

    # Load visualization settings from config
    viz = config.get("visualization", {})
    FIG_SIZE_X = viz.get("fig_size_x", 16)
    TITLE_SIZE = viz.get("title_size", 24)
    AXIS_LABEL_SIZE = viz.get("axis_label_size", 9)
    ROW_LABEL_SIZE = viz.get("row_label_size", 9)

    df_full = pd.DataFrame(results)

    # Create color palette from FULL results (same as other charts) for consistency
    unique_models_all = df_full["model_short"].unique()
    colors = sns.color_palette("tab10", n_colors=len(unique_models_all))
    model_colors = {model: colors[i] for i, model in enumerate(unique_models_all)}

    # Separate BM25 baseline and embedding models
    df_bm25 = df_full[df_full["model"] == "BM25"].copy()
    df = df_full[df_full["model"] != "BM25"].copy()

    # Find the best quality metric column (prefer highest k MRR)
    metric_cols = [col for col in df_full.columns if col.startswith("mrr@")]
    if not metric_cols:
        metric_cols = [col for col in df_full.columns if col.startswith("hit_rate@")]
    if not metric_cols:
        console.print(
            "\n   ⚠️  [yellow]No quality metrics found for tradeoff visualization[/yellow]",
            style="dim",
        )
        return

    # Use the metric with highest k value
    quality_metric = sorted(metric_cols, key=lambda x: int(x.split("@")[1]))[-1]

    # Check if we have any embedding models with memory data
    if df.empty and df_bm25.empty:
        console.print(
            "\n   ⚠️  [yellow]No models to visualize[/yellow]",
            style="dim",
        )
        return

    # Include ALL alpha configurations (not just best per model)
    df_plot = df.copy()

    # Add memory data for embedding models
    if not df_plot.empty:
        df_plot["memory_mb"] = df_plot["model_short"].map(
            lambda m: memory_data.get(m, {}).get("peak_memory_mb", 0.0)
        )
        # Separate models with and without memory data
        df_with_memory = df_plot[df_plot["memory_mb"] > 0].copy()
        df_no_memory = df_plot[df_plot["memory_mb"] == 0].copy()  # OpenRouter models
    else:
        df_with_memory = pd.DataFrame()
        df_no_memory = pd.DataFrame()

    # Create figure with extra space for legend on the right
    fig, ax = plt.subplots(figsize=(FIG_SIZE_X + 4, 10))

    # Track all points for Pareto calculation (including BM25 and OpenRouter)
    all_points = []

    # Process embedding models WITH memory data (will be bubbles)
    if not df_with_memory.empty:
        # Scale bubble sizes (normalize memory to reasonable visual size)
        min_memory = df_with_memory["memory_mb"].min()
        max_memory = df_with_memory["memory_mb"].max()
        # Scale to range [100, 2000] for bubble area
        if max_memory > min_memory:
            bubble_sizes = 100 + (df_with_memory["memory_mb"] - min_memory) / (
                max_memory - min_memory
            ) * 1900
        else:
            bubble_sizes = pd.Series([500] * len(df_with_memory), index=df_with_memory.index)

        # Collect points for Pareto calculation
        for idx, row in df_with_memory.iterrows():
            all_points.append({
                "idx": idx,
                "quality": row[quality_metric],
                "latency": row["avg_embed_time_ms"],
                "memory": row["memory_mb"],
                "model_short": row["model_short"],
                "alpha": row["alpha"],
                "is_bm25": False,
                "is_api": False,  # Has memory data, not an API model
                "size": bubble_sizes.loc[idx],
            })
    else:
        min_memory = 0
        max_memory = 0

    # Process OpenRouter/API models WITHOUT memory data (will be colored squares)
    for _, row in df_no_memory.iterrows():
        all_points.append({
            "idx": None,
            "quality": row[quality_metric],
            "latency": row["avg_embed_time_ms"],
            "memory": 0.0,  # API models have no local memory footprint
            "model_short": row["model_short"],
            "alpha": row["alpha"],
            "is_bm25": False,
            "is_api": True,  # OpenRouter/API model
            "size": 300,  # Fixed size for API model squares
        })

    # Add BM25 points (latency=0, memory=0)
    for _, row in df_bm25.iterrows():
        all_points.append({
            "idx": None,
            "quality": row[quality_metric],
            "latency": 0.0,  # BM25 has no embedding latency
            "memory": 0.0,   # BM25 has no memory footprint
            "model_short": row["model_short"],
            "alpha": row["alpha"],
            "is_bm25": True,
            "is_api": False,
            "size": 300,  # Fixed size for BM25 squares
        })

    if not all_points:
        console.print(
            "\n   ⚠️  [yellow]No models with data for tradeoff visualization[/yellow]",
            style="dim",
        )
        return

    # Calculate Pareto frontier (configurations not dominated by any other)
    # For Pareto: higher quality is better, lower latency is better, lower memory is better
    def is_pareto_optimal(point: dict, all_pts: list[dict]) -> bool:
        """Check if a point is Pareto optimal."""
        for other in all_pts:
            if other is point:
                continue
            # Check if 'other' dominates 'point'
            # Dominates means: at least as good in all dimensions, strictly better in at least one
            at_least_as_good = (
                other["quality"] >= point["quality"]
                and other["latency"] <= point["latency"]
                and other["memory"] <= point["memory"]
            )
            strictly_better = (
                other["quality"] > point["quality"]
                or other["latency"] < point["latency"]
                or other["memory"] < point["memory"]
            )
            if at_least_as_good and strictly_better:
                return False
        return True

    for point in all_points:
        point["is_pareto"] = is_pareto_optimal(point, all_points)

    # Plot BM25 baseline as squares (matching style from other charts)
    bm25_points = [p for p in all_points if p["is_bm25"]]
    for point in bm25_points:
        ax.scatter(
            point["latency"],
            point["quality"],
            s=point["size"],
            c="#D3D3D3",  # Light gray matching other charts
            marker="s",  # Square marker for BM25
            alpha=1.0,
            edgecolors="#444444",  # Grey outline matching other charts
            linewidths=1.0,
            zorder=3,
            hatch="//",  # Hatched pattern matching other charts
        )

    # Plot OpenRouter/API model points as colored squares (no memory data)
    api_points = [p for p in all_points if p.get("is_api", False)]
    for point in api_points:
        ax.scatter(
            point["latency"],
            point["quality"],
            s=point["size"],
            c=[model_colors[point["model_short"]]],
            marker="s",  # Square marker for API models (no memory)
            alpha=0.9 if point["is_pareto"] else 0.6,
            edgecolors="gold" if point["is_pareto"] else "gray",
            linewidths=3 if point["is_pareto"] else 1.0,
            zorder=5 if point["is_pareto"] else 3,
        )

    # Plot local embedding model points with memory data (bubbles)
    # Non-BM25, non-API models (have memory data)
    local_points = [p for p in all_points if not p["is_bm25"] and not p.get("is_api", False)]
    
    # Non-Pareto local embedding points
    for point in [p for p in local_points if not p["is_pareto"]]:
        ax.scatter(
            point["latency"],
            point["quality"],
            s=point["size"],
            c=[model_colors[point["model_short"]]],
            alpha=0.4,
            edgecolors="gray",
            linewidths=1,
        )

    # Pareto local embedding points
    for point in [p for p in local_points if p["is_pareto"]]:
        ax.scatter(
            point["latency"],
            point["quality"],
            s=point["size"],
            c=[model_colors[point["model_short"]]],
            alpha=0.9,
            edgecolors="gold",
            linewidths=3,
            zorder=5,
        )

    # Add labels for all points (below the markers)
    for point in all_points:
        if point["is_bm25"]:
            label = f"{point['model_short']}"
        else:
            label = f"{point['model_short']}\n(α={point['alpha']:.1f})"
        if point["is_pareto"]:
            if point["is_bm25"]:
                label = f"★ {point['model_short']}"
            else:
                label = f"★ {point['model_short']}\n(α={point['alpha']:.1f})"
        
        # Calculate offset based on marker size (labels go below)
        y_offset = -12 - (point["size"] / 300)  # Negative for below, closer to bubble
        
        ax.annotate(
            label,
            (point["latency"], point["quality"]),
            textcoords="offset points",
            xytext=(0, y_offset),
            ha="center",
            va="top",
            fontsize=ROW_LABEL_SIZE + 2,  # Bigger font for model labels
            fontweight="bold" if point["is_pareto"] else "normal",
            zorder=10,  # Always on top of bubbles and lines
        )

    # Draw Pareto frontier line (connect Pareto-optimal points)
    pareto_points = [p for p in all_points if p["is_pareto"]]
    if len(pareto_points) > 1:
        pareto_sorted = sorted(pareto_points, key=lambda p: p["latency"])
        ax.plot(
            [p["latency"] for p in pareto_sorted],
            [p["quality"] for p in pareto_sorted],
            "--",
            color="gold",
            alpha=0.7,
            linewidth=2,
            zorder=4,
        )

    # Add ideal zone indicator (top-left corner)
    ax.annotate(
        "← Better (faster)",
        xy=(0.02, 0.5),
        xycoords="axes fraction",
        fontsize=AXIS_LABEL_SIZE,
        alpha=0.5,
        ha="left",
    )
    ax.annotate(
        "↑ Better (higher quality)",
        xy=(0.5, 0.98),
        xycoords="axes fraction",
        fontsize=AXIS_LABEL_SIZE,
        alpha=0.5,
        ha="center",
    )

    # Build legend elements
    legend_elements = []

    # Determine which models are API models (no memory) vs local models (have memory)
    api_model_names = {p["model_short"] for p in all_points if p.get("is_api", False)}

    # Add model color legend (only models present in the plot)
    unique_models_in_plot = list(dict.fromkeys([p["model_short"] for p in all_points]))
    for model in unique_models_in_plot:
        # Use square for BM25 and API models, circle for local models with memory
        is_bm25_model = model == "Baseline_BM25"
        is_api_model = model in api_model_names
        legend_elements.append(
            plt.scatter(
                [],
                [],
                s=150,
                c="#D3D3D3" if is_bm25_model else [model_colors[model]],
                marker="s" if (is_bm25_model or is_api_model) else "o",
                alpha=0.8,
                label=model,
            )
        )

    # Add separator via empty entry
    legend_elements.append(
        plt.scatter([], [], s=0, c="white", label=" ")  # Empty spacer
    )

    # Add memory size legend bubbles (only if we have embedding models)
    if max_memory > 0:
        size_legend_values = [min_memory, (min_memory + max_memory) / 2, max_memory]
        size_legend_sizes = [
            100 + (v - min_memory) / (max_memory - min_memory) * 1900
            if max_memory > min_memory
            else 500
            for v in size_legend_values
        ]

        for size, val in zip(size_legend_sizes, size_legend_values):
            legend_elements.append(
                plt.scatter(
                    [],
                    [],
                    s=size,
                    c="gray",
                    alpha=0.5,
                    label=f"{val:.0f} MB",
                )
            )

        # Add separator via empty entry
        legend_elements.append(
            plt.scatter([], [], s=0, c="white", label=" ")  # Empty spacer
        )

    # Add BM25 indicator to legend
    if bm25_points:
        legend_elements.append(
            plt.scatter(
                [],
                [],
                s=200,
                c="#D3D3D3",  # Light gray matching other charts
                marker="s",
                alpha=1.0,
                edgecolors="#444444",
                linewidths=1.0,
                hatch="//",
                label="■ BM25 (no memory)",
            )
        )

    # Add API model indicator to legend (if any API models present)
    if api_points:
        legend_elements.append(
            plt.scatter(
                [],
                [],
                s=200,
                c="gray",
                marker="s",
                alpha=0.6,
                edgecolors="gray",
                linewidths=1.0,
                label="■ API (no memory)",
            )
        )

    # Add Pareto indicator to legend
    legend_elements.append(
        plt.scatter(
            [],
            [],
            s=200,
            c="white",
            edgecolors="gold",
            linewidths=3,
            label="★ Pareto Optimal",
        )
    )

    # Place legend outside the plot on the right
    ax.legend(
        handles=legend_elements,
        title="Models & Memory",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        framealpha=0.9,
        fontsize=ROW_LABEL_SIZE + 1,
        title_fontsize=ROW_LABEL_SIZE + 2,
        borderpad=1,
        labelspacing=1.2,
    )

    # Labels and title
    metric_label = quality_metric.upper().replace("@", " @")
    ax.set_xlabel("Embedding Latency (ms per document)", fontsize=AXIS_LABEL_SIZE + 2)
    ax.set_ylabel(metric_label, fontsize=AXIS_LABEL_SIZE + 2)
    ax.set_title(
        "Model Tradeoffs: Quality vs Latency vs Memory",
        fontsize=TITLE_SIZE,
        pad=20,
    )

    # Add document and query counts below the graph (left-aligned)
    num_docs = df_full["num_documents"].iloc[0]
    num_queries = df_full["num_queries"].iloc[0]
    ax.text(
        0,
        -0.08,
        f"{num_docs} documents, {num_queries} queries",
        transform=ax.transAxes,
        fontsize=AXIS_LABEL_SIZE,
        va="top",
        ha="left",
    )

    # Style adjustments
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    tradeoff_output_path = output_dir / f"tradeoff_{timestamp}.png"
    plt.savefig(tradeoff_output_path, dpi=300, bbox_inches="tight")
    console.print(
        f"   ✓ Tradeoff visualization saved to: [green]{tradeoff_output_path}[/green]"
    )
    plt.close()
