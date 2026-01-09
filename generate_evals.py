from pathlib import Path
from datetime import datetime
from typing import cast
import time
import csv
import argparse
import signal
import sys
import torch
import gc
import psutil
import os
import atexit
import weaviate
from weaviate.exceptions import WeaviateStartUpError
from weaviate.classes.config import Property, DataType
from sentence_transformers import SentenceTransformer
from pylate import models as pylate_models
from pylate import rank as pylate_rank
import matplotlib.pyplot as plt
from rich.console import Console
from rich.panel import Panel

from _core.utils import (
    load_config,
    validate_config,
    load_mteb_retrieval_data_from_dir,
    generate_cache_key,
    save_embeddings,
    load_embeddings,
    embeddings_exist,
    save_colbert_embeddings,
    load_colbert_embeddings,
    colbert_embeddings_exist,
    generate_eval_cache_key,
    save_eval_results,
    load_eval_results,
    eval_results_exist,
    calculate_reciprocal_rank,
    calculate_hit,
    create_results_visualization,
    create_memory_visualization,
    create_tradeoff_visualization,
    truncate_text_to_tokens,
    get_openrouter_embeddings,
    parse_model_configs,
    print_loading_cached,
    print_loaded_cached,
    print_generating,
    print_generated,
    print_generated_count,
    print_saved_to_cache,
    print_saved_eval_to_cache,
    print_indexing,
    print_indexed,
)

console = Console()

# Global reference for cleanup
_weaviate_client: weaviate.WeaviateClient | None = None


def cleanup_weaviate() -> None:
    """Cleanup Weaviate client on exit."""
    global _weaviate_client
    if _weaviate_client is not None:
        try:
            _weaviate_client.close()
            console.print("🧹 Weaviate instance closed.", style="dim")
        except Exception:
            pass  # Ignore errors during cleanup
        _weaviate_client = None


def signal_handler(signum: int, frame: object) -> None:
    """Handle termination signals gracefully."""
    sig_name = signal.Signals(signum).name
    console.print(
        f"\n⚠️  Received {sig_name}, shutting down gracefully...", style="yellow"
    )
    cleanup_weaviate()
    sys.exit(1)


# Register cleanup handlers
atexit.register(cleanup_weaviate)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Matplotlib styling
plt.style.use("ggplot")
params = {
    "text.color": (0.25, 0.25, 0.25),
    "font.family": "sans-serif",
    "axes.titlesize": 16,
    "axes.labelsize": 12,
}
plt.rcParams.update(params)


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_metric_k_values(config: dict) -> dict[str, list[int]]:
    """Extract per-metric K values from config.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with metric names as keys and lists of K values
    """
    search_config = config.get("search", {})
    metrics_config = search_config.get("metrics", {})

    # Default K values if not specified
    default_k = [10]

    # Support legacy top_k config for backward compatibility
    legacy_k = search_config.get("top_k", default_k)
    if isinstance(legacy_k, int):
        legacy_k = [legacy_k]

    return {
        "mrr": metrics_config.get("mrr_k", legacy_k),
        "hit_rate": metrics_config.get("hit_rate_k", legacy_k),
    }


def get_max_k(metric_k_values: dict[str, list[int]]) -> int:
    """Get the maximum K value across all metrics.

    Args:
        metric_k_values: Dictionary of metric names to K value lists

    Returns:
        Maximum K value needed for retrieval
    """
    all_k = []
    for k_list in metric_k_values.values():
        all_k.extend(k_list)
    return max(all_k) if all_k else 10


def compute_metrics(
    all_retrieved_ids: list[tuple[list[str], list[str]]],
    metric_k_values: dict[str, list[int]],
) -> dict[str, float]:
    """Compute all retrieval metrics at their configured K values.

    Args:
        all_retrieved_ids: List of (retrieved_ids, relevant_ids) tuples per query
        metric_k_values: Dictionary mapping metric names to K value lists

    Returns:
        Dictionary of metric results (e.g., {"mrr@10": 0.85, "hit_rate@10": 0.92})
    """
    results: dict[str, float] = {}

    # Compute MRR at each configured K
    for k in metric_k_values["mrr"]:
        rr_scores = []
        for retrieved_ids, relevant_ids in all_retrieved_ids:
            rr = calculate_reciprocal_rank(retrieved_ids[:k], relevant_ids)
            rr_scores.append(rr)
        results[f"mrr@{k}"] = sum(rr_scores) / len(rr_scores)

    # Compute Hit Rate at each configured K
    for k in metric_k_values["hit_rate"]:
        hit_scores = []
        for retrieved_ids, relevant_ids in all_retrieved_ids:
            hit = calculate_hit(retrieved_ids[:k], relevant_ids)
            hit_scores.append(hit)
        results[f"hit_rate@{k}"] = sum(hit_scores) / len(hit_scores)

    return results


def main() -> None:
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Hybrid Search Evaluation with Embedding Caching"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="_configs/config.yaml",
        help="Path to the configuration YAML file (default: _configs/config.yaml)",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of embeddings even if cached versions exist",
    )
    args = parser.parse_args()

    console.print(Panel("🚀 Starting Hybrid Search Evaluation", style="bold magenta"))

    config = load_config(args.config)

    # Validate configuration
    console.print("\n🔍 Validating configuration...", style="bold cyan")
    validation_errors = validate_config(config, args.config)

    if validation_errors:
        console.print("\n❌ [bold red]Configuration validation failed![/bold red]\n")
        console.print("The following issues were found:\n")
        for error in validation_errors:
            console.print(f"  {error}\n")
        console.print(
            f"📝 Please fix the errors in [yellow]{args.config}[/yellow] and run again.\n",
            style="bold",
        )
        return

    console.print("   ✅ Configuration is valid", style="green")
    console.print(f"📋 Project ID: [yellow]{config['project_id']}[/yellow]")

    # Setup embeddings directory
    embeddings_dir = Path(
        config.get("embeddings", {}).get("cache_dir", "_cache_embeddings")
    )
    embeddings_dir.mkdir(exist_ok=True, parents=True)

    # Setup evals directory
    evals_dir = Path(config.get("evals", {}).get("cache_dir", "_cache_evals"))
    evals_dir.mkdir(exist_ok=True, parents=True)

    if args.force_recompute:
        console.print(
            "⚠️  [yellow]Force recompute mode enabled - will regenerate all embeddings and evals[/yellow]"
        )
    else:
        console.print(f"💾 Embeddings cache directory: [cyan]{embeddings_dir}[/cyan]")
        console.print(f"💾 Evals cache directory: [cyan]{evals_dir}[/cyan]")

    console.print("\n📂 Loading data...", style="bold cyan")

    # Check for MTEB format data (required)
    mteb_data_dir = config.get("data", {}).get("mteb_data_dir")

    if not mteb_data_dir:
        console.print("\n❌ [bold red]No data source configured![/bold red]\n")
        console.print("   Please set 'data.mteb_data_dir' in your config file.\n")
        return

    # Extract dataset identifier for cache keys (use directory name)
    dataset_id = Path(mteb_data_dir).name

    console.print(f"   📁 Using MTEB format from: [cyan]{mteb_data_dir}[/cyan]")
    mteb_data = load_mteb_retrieval_data_from_dir(mteb_data_dir)

    # Get max document tokens from config
    max_document_tokens = config.get("model", {}).get("max_document_tokens", 512)
    console.print(
        f"   📏 Truncating documents to [yellow]{max_document_tokens}[/yellow] tokens"
    )

    # Convert to expected format with truncation
    documents = []
    for doc in mteb_data.get_documents_list():
        documents.append(
            {
                "id": doc["id"],
                "text": truncate_text_to_tokens(doc["text"], max_document_tokens),
            }
        )

    queries = mteb_data.get_queries_list()

    console.print(f"   ✓ Loaded [green]{len(documents):,}[/green] documents")
    console.print(f"   ✓ Loaded [green]{len(queries):,}[/green] queries")

    device = config["embeddings"]["device"]
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        console.print(
            "   ⚠️  [yellow]CUDA requested but not available; falling back to CPU[/yellow]",
            style="dim",
        )
        device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        fallback_device = "cuda" if torch.cuda.is_available() else "cpu"
        console.print(
            f"   ⚠️  [yellow]MPS requested but not available; falling back to {fallback_device.upper()}[/yellow]",
            style="dim",
        )
        device = fallback_device

    console.print(f"\n💻 Using device: [yellow]{device}[/yellow]")

    # Initialize Weaviate embedded
    console.print("\n🔧 Starting Weaviate embedded instance...", style="bold cyan")

    global _weaviate_client
    try:
        client = weaviate.connect_to_embedded(
            environment_variables={"LOG_LEVEL": "error"}
        )
        console.print("   ✓ Started new Weaviate embedded instance", style="green")
    except WeaviateStartUpError as e:
        error_msg = str(e)
        if "already listening on ports" in error_msg:
            console.print(
                "   ⚠️  [yellow]Weaviate embedded instance already running, connecting to existing instance...[/yellow]"
            )
            try:
                client = weaviate.connect_to_local(port=8079, grpc_port=50050)
                console.print(
                    "   ✓ Connected to existing Weaviate instance", style="green"
                )
            except Exception as connect_error:
                console.print(
                    f"\n❌ [red]Failed to connect to existing Weaviate instance:[/red] {connect_error}"
                )
                console.print(
                    "\n   [yellow]Try manually killing the Weaviate process:[/yellow]"
                )
                console.print("   [cyan]lsof -ti:8079 | xargs kill -9[/cyan]")
                console.print("   [cyan]lsof -ti:50050 | xargs kill -9[/cyan]\n")
                return
        else:
            console.print(f"\n❌ [red]Failed to start Weaviate:[/red] {e}")
            return

    _weaviate_client = client  # Store for cleanup handlers

    try:
        all_results = []
        memory_data = {}  # Track memory consumption per model

        # Check if BM25 baseline should be included
        include_bm25 = config.get("search", {}).get("include_bm25_baseline", True)

        # Process BM25 baseline if enabled
        if include_bm25:
            console.print(f"\n{'=' * 100}", style="bold blue")
            console.print("📊 Evaluating BM25 Baseline (Pure Lexical Search)")
            console.print(f"{'=' * 100}", style="bold blue")

            # For BM25, we use alpha=0.0 and don't need embeddings
            # We'll use a dummy model name for caching
            bm25_model_name = "BM25"
            model_short = "Baseline_BM25"

            # Create Weaviate collection for BM25 (delete if exists)
            collection_name = "Documents"
            if client.collections.exists(collection_name):
                client.collections.delete(collection_name)

            console.print("   Creating Weaviate collection...", style="cyan")
            collection = client.collections.create(
                name=collection_name,
                properties=[
                    Property(name="document_id", data_type=DataType.TEXT),
                    Property(name="text", data_type=DataType.TEXT),
                ],
                vectorizer_config=None,
            )

            # Index documents without vectors (BM25 only uses text)
            print_indexing("documents (text only)", len(documents))
            with collection.batch.dynamic() as batch:
                for document in documents:
                    batch.add_object(
                        properties={
                            "document_id": document["id"],
                            "text": document["text"],
                        },
                    )

            print_indexed("documents", len(documents))

            # Get per-metric K values and determine max K for retrieval
            metric_k_values = get_metric_k_values(config)
            max_k = get_max_k(metric_k_values)

            # Run BM25 experiments (alpha is always 0.0 for BM25)
            query_texts = [q["query"] for q in queries]
            alpha = 0.0  # Pure lexical search

            # Generate cache key for BM25 eval configuration
            # Include dataset_id to ensure different datasets use different caches
            eval_project_id = (
                f"{config['project_id']}_{dataset_id}"
                if dataset_id
                else config["project_id"]
            )
            eval_cache_key = generate_eval_cache_key(
                eval_project_id, bm25_model_name, alpha, max_k, model_short
            )

            # Check if eval results already exist in cache
            if not args.force_recompute and eval_results_exist(
                eval_cache_key, evals_dir
            ):
                print_loading_cached("BM25 eval results")
                cached_result = load_eval_results(eval_cache_key, evals_dir)
                all_results.append(cached_result)
            else:
                console.print(
                    f"   Testing BM25: [yellow]alpha={alpha}, max_k={max_k}[/yellow]"
                )

                # Collect per-query results at max_k
                all_retrieved_ids = []

                # Evaluate on all queries (pure lexical search, no embeddings)
                for query_data, query_text in zip(queries, query_texts):
                    relevant_ids = query_data["relevant_ids"]

                    # Perform pure lexical search (alpha=0.0) with max_k
                    response = collection.query.hybrid(
                        query=query_text,
                        alpha=alpha,
                        limit=max_k,
                    )

                    # Extract retrieved document IDs
                    retrieved_ids = [
                        cast(str, obj.properties["document_id"])
                        for obj in response.objects
                    ]
                    all_retrieved_ids.append((retrieved_ids, relevant_ids))

                # Build result with metrics at their respective K values
                result: dict = {
                    "model": bm25_model_name,
                    "model_short": model_short,
                    "alpha": alpha,
                    "avg_embed_time_ms": 0.0,  # No embedding time for BM25
                    "total_embed_time_ms": 0.0,
                    "num_queries": len(queries),
                    "num_documents": len(documents),
                }

                # Compute all metrics using helper function
                result.update(compute_metrics(all_retrieved_ids, metric_k_values))

                all_results.append(result)

                # Save eval results to cache
                saved_path = save_eval_results(result, eval_cache_key, evals_dir)
                print_saved_eval_to_cache(saved_path)

        # Parse model configurations from the new YAML structure
        model_configs = parse_model_configs(config)

        if not model_configs:
            console.print(
                "⚠️  [yellow]No models configured - skipping embedding evaluation[/yellow]"
            )

        # Iterate over each embedding model
        for model_config in model_configs:
            model_id = model_config["model_id"]
            model_name = model_config["model_name"]
            query_prefix = model_config["query_prefix"]
            passage_prefix = model_config["passage_prefix"]
            query_encode_kwargs = model_config["query_encode_kwargs"]
            passage_encode_kwargs = model_config["passage_encode_kwargs"]
            use_openrouter = model_config["is_openrouter"]
            is_colbert = model_config.get("is_colbert", False)

            model_type_label = "ColBERT" if is_colbert else "Embedding"
            console.print(f"\n{'=' * 100}", style="bold blue")
            console.print(
                f"📊 Evaluating {model_type_label} model: [bold yellow]{model_name}[/bold yellow] ([dim]{model_id}[/dim])"
            )
            console.print(f"{'=' * 100}", style="bold blue")

            # ColBERT models only support pure semantic search (alpha=1.0)
            if is_colbert:
                console.print(
                    "   🔷 ColBERT late-interaction model - using MaxSim scoring (alpha=1.0 only)",
                    style="cyan",
                )

            # Get batch size from config
            batch_size = config.get("model", {}).get("embedding_batch_size", 32)

            if query_prefix or passage_prefix:
                console.print(
                    f"   📝 Using instruction prefixes: query=[yellow]'{query_prefix}'[/yellow], passage=[yellow]'{passage_prefix}'[/yellow]",
                    style="cyan",
                )

            if query_encode_kwargs or passage_encode_kwargs:
                console.print(
                    f"   🔧 Using encode kwargs: query=[yellow]{query_encode_kwargs}[/yellow], passage=[yellow]{passage_encode_kwargs}[/yellow]",
                    style="cyan",
                )

            # Generate cache keys for embeddings (use model_id for cache key generation)
            # Include dataset_id to ensure different datasets use different caches
            cache_project_id = (
                f"{config['project_id']}_{dataset_id}"
                if dataset_id
                else config["project_id"]
            )
            doc_cache_key = generate_cache_key(cache_project_id, model_id, "documents")
            query_cache_key = generate_cache_key(cache_project_id, model_id, "queries")

            # Check if all embeddings and evals are cached
            # ColBERT models use different storage format
            if is_colbert:
                embeddings_cached = (
                    not args.force_recompute
                    and colbert_embeddings_exist(doc_cache_key, embeddings_dir)
                    and colbert_embeddings_exist(query_cache_key, embeddings_dir)
                )
            else:
                embeddings_cached = (
                    not args.force_recompute
                    and embeddings_exist(doc_cache_key, embeddings_dir)
                    and embeddings_exist(query_cache_key, embeddings_dir)
                )

            # Check if all evals are cached
            # Get per-metric K values to determine max K for cache key
            metric_k_values = get_metric_k_values(config)
            max_k = get_max_k(metric_k_values)

            # ColBERT models support hybrid search too (MaxSim + BM25)
            alpha_values = config["search"]["alpha"]

            all_evals_cached = True
            if not args.force_recompute and embeddings_cached:
                for alpha in alpha_values:
                    eval_cache_key = generate_eval_cache_key(
                        cache_project_id, model_id, alpha, max_k, model_name
                    )
                    if not eval_results_exist(eval_cache_key, evals_dir):
                        all_evals_cached = False
                        break

            # Skip model loading if everything is cached
            if embeddings_cached and all_evals_cached:
                console.print(
                    "   ⚡ [green]All embeddings and evals cached - skipping model load and searches[/green]",
                    style="cyan",
                )

                # Load cached embeddings metadata for timing info
                if is_colbert:
                    _, doc_metadata = load_colbert_embeddings(
                        doc_cache_key, embeddings_dir
                    )
                else:
                    _, doc_metadata = load_embeddings(doc_cache_key, embeddings_dir)
                total_embed_time_ms = doc_metadata.get("total_embed_time_ms", 0)
                avg_embed_time_per_doc_ms = doc_metadata.get(
                    "avg_embed_time_per_doc_ms", 0
                )

                # Load memory data from cache for visualization
                if (
                    "peak_memory_mb" in doc_metadata
                    and "model_memory_mb" in doc_metadata
                ):
                    memory_data[model_name] = {
                        "model_memory_mb": doc_metadata.get("model_memory_mb", 0.0),
                        "peak_memory_mb": doc_metadata.get("peak_memory_mb", 0.0),
                    }

                # Load all cached eval results
                for alpha in alpha_values:
                    eval_cache_key = generate_eval_cache_key(
                        cache_project_id, model_id, alpha, max_k, model_name
                    )
                    print_loading_cached(f"eval results for alpha={alpha}")
                    cached_result = load_eval_results(eval_cache_key, evals_dir)
                    # Ensure model_short matches current config (in case of stale cache)
                    cached_result["model_short"] = model_name
                    all_results.append(cached_result)

                # Skip to next model (no model cleanup needed since we didn't load it)
                continue

            # If we're here, we need to load the model (or prepare for OpenRouter API)
            model = None
            memory_before_mb = get_memory_usage_mb()
            memory_after_model_mb = memory_before_mb  # Default for API models

            if use_openrouter:
                console.print(
                    f"   Using OpenRouter API for model: [yellow]{model_id}[/yellow]",
                    style="cyan",
                )
                # Get OpenRouter batch size from config
                openrouter_batch_size = (
                    config.get("embeddings", {})
                    .get("openrouter", {})
                    .get("settings", {})
                    .get("api_batch_size", 100)
                )
                # Get API cost warning threshold from config
                api_cost_warning_threshold = (
                    config.get("embeddings", {})
                    .get("openrouter", {})
                    .get("settings", {})
                    .get("api_cost_warning_threshold", 5000)
                )

                # Check if total samples exceed threshold and warn user
                total_samples = len(documents) + len(queries)
                if total_samples > api_cost_warning_threshold:
                    console.print(
                        "\n⚠️  [bold yellow]WARNING: HIGH API USAGE DETECTED[/bold yellow]",
                        style="bold",
                    )
                    console.print(
                        f"   You are about to send [bold red]{total_samples:,}[/bold red] samples ([yellow]{len(documents):,}[/yellow] documents + [yellow]{len(queries):,}[/yellow] queries)",
                    )
                    console.print(
                        f"   to the OpenRouter API for model [cyan]{model_id}[/cyan].",
                    )
                    console.print(
                        f"   This exceeds the configured threshold of [yellow]{api_cost_warning_threshold:,}[/yellow] samples.",
                    )
                    console.print(
                        "\n   [bold]This may result in substantial API costs![/bold]",
                        style="red",
                    )
                    console.print(
                        "   (You can adjust this threshold in config.yaml under embeddings.openrouter.settings.api_cost_warning_threshold)\n",
                        style="dim",
                    )

                    user_input = (
                        input("   Do you want to proceed? (yes/no): ").strip().lower()
                    )
                    if user_input not in ["yes", "y"]:
                        console.print(
                            f"\n   [yellow]Skipping model '{model_name}' and continuing with remaining models...[/yellow]\n"
                        )
                        continue
                    console.print("   [green]Proceeding with API calls...[/green]\n")

                # OpenRouter API doesn't load model locally, so no memory impact
                model_memory_mb = 0.0
            elif is_colbert:
                # Load ColBERT model using pylate
                console.print("   Loading ColBERT model...", style="cyan")
                try:
                    model = pylate_models.ColBERT(
                        model_name_or_path=model_id,
                        device=device,
                        trust_remote_code=True,
                    )
                    memory_after_model_mb = get_memory_usage_mb()
                    model_memory_mb = max(0.0, memory_after_model_mb - memory_before_mb)
                    console.print(
                        f"   ✓ ColBERT model loaded (memory: [yellow]{model_memory_mb:.1f} MB[/yellow])",
                        style="cyan",
                    )
                except Exception as e:
                    error_type = type(e).__name__
                    error_msg = str(e)
                    console.print(
                        f"\n⚠️  [red bold]Error loading ColBERT model '{model_id}' ({error_type}):[/red bold]"
                    )
                    console.print(f"   [yellow]{error_msg[:500]}[/yellow]")
                    console.print(
                        f"\n   [cyan]Skipping model '{model_name}' and continuing with remaining models...[/cyan]\n"
                    )
                    continue
            else:
                # Check if this is a minishlab model - they don't work with MPS
                model_device = device
                if "minishlab" in model_id.lower():
                    model_device = "cpu"
                    if device == "mps":
                        console.print(
                            "   ⚠️  [yellow]Minishlab models don't support MPS, using CPU instead[/yellow]",
                            style="dim",
                        )

                console.print("   Loading model...", style="cyan")
                try:
                    model = SentenceTransformer(
                        model_id, device=model_device, trust_remote_code=True
                    )
                    memory_after_model_mb = get_memory_usage_mb()
                    model_memory_mb = max(0.0, memory_after_model_mb - memory_before_mb)
                    console.print(
                        f"   ✓ Model loaded (memory: [yellow]{model_memory_mb:.1f} MB[/yellow])",
                        style="cyan",
                    )
                except FileNotFoundError as e:
                    if "huggingface" in str(e) and "cache" in str(e):
                        console.print(
                            f"\n❌ [red]Error loading model {model_id}:[/red]",
                            style="bold",
                        )
                        console.print(
                            "   [yellow]HuggingFace cache is corrupted for this model.[/yellow]"
                        )
                        console.print("\n   To fix this, run:")
                        console.print(
                            f"   [cyan]rm -rf ~/.cache/huggingface/modules/transformers_modules/{model_id.replace('/', '/')}*[/cyan]"
                        )
                        console.print(
                            "   [cyan]rm -rf ~/.cache/huggingface/hub/models--{model_id.replace('/', '--')}*[/cyan]"
                        )
                        console.print("\n   Then run the script again.\n")
                        raise
                    else:
                        raise
                except OSError as e:
                    error_msg = str(e)
                    console.print(
                        f"\n⚠️  [red bold]Error loading model '{model_id}':[/red bold]"
                    )
                    if (
                        "not a valid model identifier" in error_msg
                        or "Repository Not Found" in error_msg
                    ):
                        console.print(
                            "   [yellow]Model not found on HuggingFace Hub.[/yellow]"
                        )
                        console.print(
                            "   [dim]Please verify the model ID is correct and accessible.[/dim]"
                        )
                        console.print(
                            "   [dim]If this is a private/gated model, ensure you are logged in via 'huggingface-cli login'.[/dim]"
                        )
                    else:
                        console.print(f"   [yellow]{error_msg}[/yellow]")
                    console.print(
                        f"\n   [cyan]Skipping model '{model_name}' and continuing with remaining models...[/cyan]\n"
                    )
                    continue
                except Exception as e:
                    # Catch any other unexpected errors during model loading
                    error_type = type(e).__name__
                    error_msg = str(e)
                    console.print(
                        f"\n⚠️  [red bold]Unexpected error loading model '{model_id}' ({error_type}):[/red bold]"
                    )
                    console.print(f"   [yellow]{error_msg[:500]}[/yellow]")
                    console.print(
                        f"\n   [cyan]Skipping model '{model_name}' and continuing with remaining models...[/cyan]\n"
                    )
                    continue

            document_texts = [d["text"] for d in documents]

            # Apply passage prefix if configured for this model (not used for ColBERT)
            if passage_prefix and not is_colbert:
                document_texts = [passage_prefix + text for text in document_texts]

            # Initialize memory tracking variables (may be updated below)
            total_memory_mb = model_memory_mb if not use_openrouter else 0.0

            # Check if document embeddings exist in cache
            # ColBERT uses different storage format
            doc_embeddings_cached = (
                colbert_embeddings_exist(doc_cache_key, embeddings_dir)
                if is_colbert
                else embeddings_exist(doc_cache_key, embeddings_dir)
            )

            if not args.force_recompute and doc_embeddings_cached:
                print_loading_cached("document embeddings")
                if is_colbert:
                    document_embeddings, doc_metadata = load_colbert_embeddings(
                        doc_cache_key, embeddings_dir
                    )
                else:
                    document_embeddings, doc_metadata = load_embeddings(
                        doc_cache_key, embeddings_dir
                    )
                total_embed_time_ms = doc_metadata.get("total_embed_time_ms", 0)
                avg_embed_time_per_doc_ms = doc_metadata.get(
                    "avg_embed_time_per_doc_ms", 0
                )
                # Load cached memory data if available
                if "peak_memory_mb" in doc_metadata:
                    total_memory_mb = doc_metadata.get(
                        "peak_memory_mb", model_memory_mb
                    )
                if "model_memory_mb" in doc_metadata:
                    model_memory_mb = doc_metadata.get(
                        "model_memory_mb", model_memory_mb
                    )
                # Store memory data for visualization (from cache)
                if (
                    "peak_memory_mb" in doc_metadata
                    and "model_memory_mb" in doc_metadata
                    and not use_openrouter
                ):
                    memory_data[model_name] = {
                        "model_memory_mb": doc_metadata.get("model_memory_mb", 0.0),
                        "peak_memory_mb": doc_metadata.get("peak_memory_mb", 0.0),
                    }
                print_loaded_cached(
                    "document embeddings",
                    len(document_embeddings),
                    f"(originally: [yellow]{total_embed_time_ms:.1f}ms[/yellow], [dim]{avg_embed_time_per_doc_ms:.2f}ms per doc[/dim])",
                )
                # Document embeddings loaded from cache, no need to save
                pending_doc_embeddings = None
            else:
                print_generating("documents", len(documents))

                # Measure embedding time for documents (use perf_counter for higher precision)
                start_time = time.perf_counter()

                try:
                    if use_openrouter:
                        # Use OpenRouter API
                        document_embeddings = get_openrouter_embeddings(
                            document_texts,
                            model=model_id,
                            batch_size=openrouter_batch_size,
                        )
                    elif is_colbert:
                        # Use ColBERT model (returns list of 2D arrays)
                        assert model is not None, "ColBERT model should be loaded"
                        document_embeddings = model.encode(
                            document_texts,
                            batch_size=batch_size,
                            is_query=False,  # Encoding documents
                            show_progress_bar=True,
                        )
                    else:
                        # Use SentenceTransformer
                        assert model is not None, (
                            "Model should be loaded for non-OpenRouter models"
                        )
                        document_embeddings = model.encode(
                            document_texts,
                            batch_size=batch_size,
                            show_progress_bar=False,  # Disable progress bar for accurate timing
                            normalize_embeddings=True,
                            convert_to_numpy=True,
                            **passage_encode_kwargs,  # Pass model-specific encode parameters
                        )
                except Exception as e:
                    error_type = type(e).__name__
                    error_msg = str(e)
                    console.print(
                        f"\n⚠️  [red bold]Error generating document embeddings for '{model_id}' ({error_type}):[/red bold]"
                    )
                    if "HTTPError" in error_type or "requests" in str(
                        type(e).__module__
                    ):
                        console.print(
                            "   [yellow]API request failed. Please check the model ID and your API credentials.[/yellow]"
                        )
                    console.print(f"   [dim]{error_msg[:500]}[/dim]")
                    console.print(
                        f"\n   [cyan]Skipping model '{model_name}' and continuing with remaining models...[/cyan]\n"
                    )
                    continue

                end_time = time.perf_counter()
                total_embed_time_ms = (end_time - start_time) * 1000
                avg_embed_time_per_doc_ms = total_embed_time_ms / len(documents)

                # Track peak memory after embedding (cap at 0 to handle GC fluctuations)
                peak_memory_mb = get_memory_usage_mb()
                total_memory_mb = max(0.0, peak_memory_mb - memory_before_mb)

                print_generated(
                    "documents",
                    len(documents),
                    total_embed_time_ms,
                    avg_embed_time_per_doc_ms,
                )

                # Prepare document metadata for caching (will be saved after all model processing succeeds)
                doc_metadata = {
                    "project_id": cache_project_id,
                    "dataset_id": dataset_id,
                    "model_name": model_id,
                    "data_type": "documents",
                    "num_documents": len(documents),
                    "total_embed_time_ms": total_embed_time_ms,
                    "avg_embed_time_per_doc_ms": avg_embed_time_per_doc_ms,
                    "model_memory_mb": model_memory_mb,
                    "peak_memory_mb": total_memory_mb,
                    "created_at": datetime.now().isoformat(),
                    "batch_size": batch_size,
                    "is_colbert": is_colbert,
                }
                # Track pending document embeddings to save after successful completion
                pending_doc_embeddings = (
                    document_embeddings,
                    doc_cache_key,
                    doc_metadata,
                    is_colbert,
                )

            # ColBERT models use direct MaxSim scoring, skip Weaviate indexing
            if is_colbert:
                # For ColBERT, we need to generate query embeddings and use pylate.rank.rerank
                query_texts = [q["query"] for q in queries]

                # Check if query embeddings exist in cache
                query_embeddings_cached = colbert_embeddings_exist(
                    query_cache_key, embeddings_dir
                )

                if not args.force_recompute and query_embeddings_cached:
                    print_loading_cached("query embeddings")
                    query_embeddings, query_metadata = load_colbert_embeddings(
                        query_cache_key, embeddings_dir
                    )
                    print_loaded_cached("query embeddings", len(query_embeddings))
                    # Query embeddings loaded from cache, no need to save
                    pending_query_embeddings = None
                else:
                    print_generating("queries", len(queries))

                    try:
                        assert model is not None, "ColBERT model should be loaded"
                        query_embeddings = model.encode(
                            query_texts,
                            batch_size=batch_size,
                            is_query=True,  # Encoding queries
                            show_progress_bar=True,
                        )
                    except Exception as e:
                        error_type = type(e).__name__
                        error_msg = str(e)
                        console.print(
                            f"\n⚠️  [red bold]Error generating query embeddings for '{model_id}' ({error_type}):[/red bold]"
                        )
                        console.print(f"   [dim]{error_msg[:500]}[/dim]")
                        console.print(
                            f"\n   [cyan]Skipping model '{model_name}' and continuing with remaining models...[/cyan]\n"
                        )
                        continue

                    print_generated_count("query embeddings", len(query_embeddings))

                    # Prepare query metadata for caching (will be saved after all model processing succeeds)
                    query_metadata = {
                        "project_id": cache_project_id,
                        "dataset_id": dataset_id,
                        "model_name": model_id,
                        "data_type": "queries",
                        "num_queries": len(queries),
                        "created_at": datetime.now().isoformat(),
                        "batch_size": batch_size,
                        "is_colbert": True,
                    }
                    # Track pending query embeddings to save after successful completion
                    pending_query_embeddings = (
                        query_embeddings,
                        query_cache_key,
                        query_metadata,
                        True,
                    )

                # Create Weaviate collection for BM25 scoring (text only, no vectors)
                collection_name = "Documents"
                if client.collections.exists(collection_name):
                    client.collections.delete(collection_name)

                console.print(
                    "   Creating Weaviate collection for BM25...", style="cyan"
                )
                collection = client.collections.create(
                    name=collection_name,
                    properties=[
                        Property(name="document_id", data_type=DataType.TEXT),
                        Property(name="text", data_type=DataType.TEXT),
                    ],
                    vectorizer_config=None,
                )

                # Index documents for BM25 (text only)
                print_indexing("documents (text only for BM25)", len(documents))
                with collection.batch.dynamic() as batch:
                    for document in documents:
                        batch.add_object(
                            properties={
                                "document_id": document["id"],
                                "text": document["text"],
                            },
                        )
                print_indexed("documents", len(documents))

                # Pre-compute all ColBERT MaxSim scores for efficiency
                console.print(
                    "   Computing ColBERT MaxSim scores for all queries...",
                    style="cyan",
                )

                # Build document IDs list
                all_doc_ids = [doc["id"] for doc in documents]

                # Rerank all documents for each query using MaxSim
                # This gives us scores for all doc-query pairs
                reranked_results = pylate_rank.rerank(
                    documents_ids=[all_doc_ids] * len(queries),
                    queries_embeddings=query_embeddings,
                    documents_embeddings=[document_embeddings] * len(queries),
                )

                # Convert reranked results to score dictionaries per query
                colbert_scores_per_query: list[dict[str, float]] = []
                for reranked in reranked_results:
                    scores = {r["id"]: r["score"] for r in reranked}
                    colbert_scores_per_query.append(scores)

                console.print(
                    f"   ✓ Computed MaxSim scores for [green]{len(queries)}[/green] queries"
                )

                # Run experiments for different alpha values (hybrid search)
                for alpha in alpha_values:
                    # Generate cache key for this specific eval configuration
                    eval_cache_key = generate_eval_cache_key(
                        cache_project_id, model_id, alpha, max_k, model_name
                    )

                    # Check if eval results already exist in cache
                    if not args.force_recompute and eval_results_exist(
                        eval_cache_key, evals_dir
                    ):
                        print_loading_cached(f"eval results for alpha={alpha}")
                        cached_result = load_eval_results(eval_cache_key, evals_dir)
                        cached_result["model_short"] = model_name
                        all_results.append(cached_result)
                        continue

                    console.print(
                        f"   Testing ColBERT hybrid: [yellow]alpha={alpha}, max_k={max_k}[/yellow]"
                    )

                    # Collect per-query results
                    all_retrieved_ids = []

                    for i, query_data in enumerate(queries):
                        query_text = query_data["query"]
                        relevant_ids = query_data["relevant_ids"]

                        # Get ColBERT scores for this query
                        colbert_scores = colbert_scores_per_query[i]

                        if alpha == 1.0:
                            # Pure ColBERT (no BM25 needed)
                            combined_scores = colbert_scores
                        elif alpha == 0.0:
                            # Pure BM25
                            response = collection.query.bm25(
                                query=query_text,
                                limit=max_k,
                                return_metadata=weaviate.classes.query.MetadataQuery(
                                    score=True
                                ),
                            )
                            combined_scores = {
                                cast(str, obj.properties["document_id"]): (
                                    obj.metadata.score if obj.metadata else 0.0
                                )
                                for obj in response.objects
                            }
                        else:
                            # Hybrid: combine ColBERT MaxSim + BM25
                            # Get BM25 scores from Weaviate
                            # Use configurable candidate limit to avoid performance issues with large corpora
                            bm25_candidate_limit = config.get("search", {}).get(
                                "bm25_candidate_limit", 1000
                            )
                            # If limit is None or exceeds corpus size, use corpus size
                            if bm25_candidate_limit is None:
                                bm25_limit = len(documents)
                            else:
                                bm25_limit = min(bm25_candidate_limit, len(documents))
                            response = collection.query.bm25(
                                query=query_text,
                                limit=bm25_limit,
                                return_metadata=weaviate.classes.query.MetadataQuery(
                                    score=True
                                ),
                            )
                            bm25_scores = {
                                cast(str, obj.properties["document_id"]): (
                                    obj.metadata.score if obj.metadata else 0.0
                                )
                                for obj in response.objects
                            }

                            # Normalize scores to 0-1 range for fair combination
                            def normalize_scores(
                                scores: dict[str, float],
                            ) -> dict[str, float]:
                                if not scores:
                                    return {}
                                max_score = max(scores.values())
                                min_score = min(scores.values())
                                if max_score == min_score:
                                    return {k: 1.0 for k in scores}
                                return {
                                    k: (v - min_score) / (max_score - min_score)
                                    for k, v in scores.items()
                                }

                            colbert_norm = normalize_scores(colbert_scores)
                            bm25_norm = normalize_scores(bm25_scores)

                            # Combine: alpha * semantic + (1 - alpha) * keyword
                            all_doc_ids_set = set(colbert_norm.keys()) | set(
                                bm25_norm.keys()
                            )
                            combined_scores = {}
                            for doc_id in all_doc_ids_set:
                                semantic = colbert_norm.get(doc_id, 0.0)
                                keyword = bm25_norm.get(doc_id, 0.0)
                                combined_scores[doc_id] = (
                                    alpha * semantic + (1 - alpha) * keyword
                                )

                        # Sort by score and get top-k
                        sorted_docs = sorted(
                            combined_scores.items(), key=lambda x: x[1], reverse=True
                        )[:max_k]
                        retrieved_ids = [doc_id for doc_id, _ in sorted_docs]
                        all_retrieved_ids.append((retrieved_ids, relevant_ids))

                    # Build result with metrics
                    result: dict = {
                        "model": model_id,
                        "model_short": model_name,
                        "alpha": alpha,
                        "avg_embed_time_ms": avg_embed_time_per_doc_ms,
                        "total_embed_time_ms": total_embed_time_ms,
                        "num_queries": len(queries),
                        "num_documents": len(documents),
                    }

                    # Compute all metrics using helper function
                    result.update(compute_metrics(all_retrieved_ids, metric_k_values))

                    all_results.append(result)

                    # Save eval results to cache
                    saved_path = save_eval_results(result, eval_cache_key, evals_dir)
                    print_saved_eval_to_cache(saved_path)

                # Store memory data for this model
                if model_name not in memory_data:
                    memory_data[model_name] = {
                        "model_memory_mb": model_memory_mb,
                        "peak_memory_mb": total_memory_mb,
                    }

                # All ColBERT processing succeeded - now save pending embeddings to cache
                if pending_doc_embeddings is not None:
                    emb, cache_key, metadata, is_cb = pending_doc_embeddings
                    saved_path = save_colbert_embeddings(
                        emb, cache_key, embeddings_dir, metadata
                    )
                    print_saved_to_cache(saved_path)

                if pending_query_embeddings is not None:
                    emb, cache_key, metadata, is_cb = pending_query_embeddings
                    saved_path = save_colbert_embeddings(
                        emb, cache_key, embeddings_dir, metadata
                    )
                    print_saved_to_cache(saved_path)

                # Cleanup ColBERT model
                if model is not None:
                    del model
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()

                continue  # Skip the standard Weaviate flow for ColBERT

            # Create Weaviate collection (delete if exists)
            collection_name = "Documents"
            if client.collections.exists(collection_name):
                client.collections.delete(collection_name)

            console.print("   Creating Weaviate collection...", style="cyan")
            collection = client.collections.create(
                name=collection_name,
                properties=[
                    Property(name="document_id", data_type=DataType.TEXT),
                    Property(name="text", data_type=DataType.TEXT),
                ],
                vectorizer_config=None,  # We provide vectors manually
            )

            # Index documents (only once per model)
            print_indexing("documents", len(documents))
            with collection.batch.dynamic() as batch:
                for document, embedding in zip(documents, document_embeddings):
                    batch.add_object(
                        properties={
                            "document_id": document["id"],
                            "text": document["text"],
                        },
                        vector=embedding.tolist(),
                    )

            print_indexed("documents", len(documents))

            # Generate or load query embeddings
            query_texts = [q["query"] for q in queries]

            # Apply query prefix if configured for this model
            if query_prefix:
                query_texts_prefixed = [query_prefix + text for text in query_texts]
            else:
                query_texts_prefixed = query_texts

            if not args.force_recompute and embeddings_exist(
                query_cache_key, embeddings_dir
            ):
                print_loading_cached("query embeddings")
                query_embeddings, query_metadata = load_embeddings(
                    query_cache_key, embeddings_dir
                )
                print_loaded_cached("query embeddings", len(query_embeddings))
                # Query embeddings loaded from cache, no need to save
                pending_query_embeddings = None
            else:
                print_generating("queries", len(queries))

                try:
                    if use_openrouter:
                        # Use OpenRouter API
                        query_embeddings = get_openrouter_embeddings(
                            query_texts_prefixed,
                            model=model_id,
                            batch_size=openrouter_batch_size,
                        )
                    else:
                        # Use SentenceTransformer
                        assert model is not None, (
                            "Model should be loaded for non-OpenRouter models"
                        )
                        query_embeddings = model.encode(
                            query_texts_prefixed,
                            batch_size=batch_size,
                            show_progress_bar=True,
                            normalize_embeddings=True,
                            convert_to_numpy=True,
                            **query_encode_kwargs,  # Pass model-specific encode parameters
                        )
                except Exception as e:
                    error_type = type(e).__name__
                    error_msg = str(e)
                    console.print(
                        f"\n⚠️  [red bold]Error generating query embeddings for '{model_id}' ({error_type}):[/red bold]"
                    )
                    if "HTTPError" in error_type or "requests" in str(
                        type(e).__module__
                    ):
                        console.print(
                            "   [yellow]API request failed. Please check the model ID and your API credentials.[/yellow]"
                        )
                    console.print(f"   [dim]{error_msg[:500]}[/dim]")
                    console.print(
                        f"\n   [cyan]Skipping model '{model_name}' and continuing with remaining models...[/cyan]\n"
                    )
                    continue

                print_generated_count("query embeddings", len(query_embeddings))

                # Prepare query metadata for caching (will be saved after all model processing succeeds)
                query_metadata = {
                    "project_id": cache_project_id,
                    "dataset_id": dataset_id,
                    "model_name": model_id,
                    "data_type": "queries",
                    "num_queries": len(queries),
                    "created_at": datetime.now().isoformat(),
                    "batch_size": openrouter_batch_size
                    if use_openrouter
                    else batch_size,
                }
                # Track pending query embeddings to save after successful completion
                pending_query_embeddings = (
                    query_embeddings,
                    query_cache_key,
                    query_metadata,
                    False,
                )

            # Get per-metric K values and determine max K for retrieval
            metric_k_values = get_metric_k_values(config)
            max_k = get_max_k(metric_k_values)

            # Run experiments for different alpha values
            for alpha in config["search"]["alpha"]:
                # Generate cache key for this specific eval configuration
                eval_cache_key = generate_eval_cache_key(
                    cache_project_id, model_id, alpha, max_k, model_name
                )

                # Check if eval results already exist in cache
                if not args.force_recompute and eval_results_exist(
                    eval_cache_key, evals_dir
                ):
                    print_loading_cached(f"eval results for alpha={alpha}")
                    cached_result = load_eval_results(eval_cache_key, evals_dir)
                    # Ensure model_short matches current config (in case of stale cache)
                    cached_result["model_short"] = model_name
                    all_results.append(cached_result)
                    continue

                console.print(
                    f"   Testing: [yellow]alpha={alpha}, max_k={max_k}[/yellow]"
                )

                # Collect per-query results at max_k
                all_retrieved_ids = []

                # Evaluate on all queries (using pre-computed embeddings)
                for query_data, query_embedding in zip(queries, query_embeddings):
                    query_text = query_data["query"]
                    relevant_ids = query_data["relevant_ids"]

                    # Perform hybrid search with max_k
                    response = collection.query.hybrid(
                        query=query_text,
                        vector=query_embedding.tolist(),
                        alpha=alpha,
                        limit=max_k,
                    )

                    # Extract retrieved document IDs
                    retrieved_ids = [
                        cast(str, obj.properties["document_id"])
                        for obj in response.objects
                    ]
                    all_retrieved_ids.append((retrieved_ids, relevant_ids))

                # Build result with metrics at their respective K values
                result: dict = {
                    "model": model_id,  # Full model ID for reference
                    "model_short": model_name,  # Short friendly name for display
                    "alpha": alpha,
                    "avg_embed_time_ms": avg_embed_time_per_doc_ms,
                    "total_embed_time_ms": total_embed_time_ms,
                    "num_queries": len(queries),
                    "num_documents": len(documents),
                }

                # Compute all metrics using helper function
                result.update(compute_metrics(all_retrieved_ids, metric_k_values))

                all_results.append(result)

                # Save eval results to cache
                saved_path = save_eval_results(result, eval_cache_key, evals_dir)
                print_saved_eval_to_cache(saved_path)

            # Store memory data for this model (only once per model, not per config)
            if model_name not in memory_data and not use_openrouter:
                memory_data[model_name] = {
                    "model_memory_mb": model_memory_mb,
                    "peak_memory_mb": total_memory_mb,
                }

            # All model processing succeeded - now save pending embeddings to cache
            if pending_doc_embeddings is not None:
                emb, cache_key, metadata, is_colbert_emb = pending_doc_embeddings
                saved_path = save_embeddings(emb, cache_key, embeddings_dir, metadata)
                print_saved_to_cache(saved_path)

            if pending_query_embeddings is not None:
                emb, cache_key, metadata, is_colbert_emb = pending_query_embeddings
                saved_path = save_embeddings(emb, cache_key, embeddings_dir, metadata)
                print_saved_to_cache(saved_path)

            # Cleanup model if it was loaded
            if model is not None:
                model.to("cpu")
                del model
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

        output_dir = Path(config["output"]["results_dir"])
        output_dir.mkdir(exist_ok=True, parents=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"results_{timestamp}.csv"
        with open(results_file, "w", newline="") as f:
            if all_results:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                writer.writeheader()
                writer.writerows(all_results)

        console.print(f"💾 Results saved to: [green]{results_file}[/green]")

        # Create and save visualization
        create_results_visualization(all_results, output_dir, timestamp, config)

        # Create memory consumption visualization (only if we have memory data)
        if memory_data:
            create_memory_visualization(memory_data, output_dir, timestamp, config)

        # Create tradeoff visualization (quality vs latency vs memory)
        if memory_data and all_results:
            create_tradeoff_visualization(
                all_results, memory_data, output_dir, timestamp, config
            )

    finally:
        # Cleanup
        console.print("\n🧹 Shutting down Weaviate...", style="bold cyan")
        client.close()
        _weaviate_client = None  # Clear global reference

    console.print(Panel("✅ Evaluation complete!", style="bold green"))


if __name__ == "__main__":
    main()
