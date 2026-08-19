import argparse
import json
import random
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import get_dataset_config_names, load_dataset
from datasets.exceptions import DatasetNotFoundError
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

console = Console()


@dataclass(frozen=True)
class DatasetConfigs:
    """The three repository configurations forming one retrieval task."""

    corpus: str
    queries: str
    qrels: str
    language: str | None


@dataclass(frozen=True)
class SelectedDatasets:
    """Coherent corpus, query, and qrels datasets and their selected splits."""

    corpus: Any
    queries: Any
    qrels: Any
    corpus_split: str
    query_split: str
    qrels_split: str
    evaluation_split: str


@dataclass(frozen=True)
class SampledRetrievalFrames:
    """A validated retrieval sample and its summary statistics."""

    corpus: pd.DataFrame
    queries: pd.DataFrame
    qrels: pd.DataFrame
    stats: dict[str, int | float | None]


def get_id_column(dataset) -> str:
    """Get the ID column name from a dataset (either '_id' or 'id')."""
    if "_id" in dataset.column_names:
        return "_id"
    elif "id" in dataset.column_names:
        return "id"
    else:
        raise ValueError(f"No ID column found. Available: {dataset.column_names}")


def detect_dataset_structure(
    dataset_name: str,
) -> tuple[bool, list[str]]:
    """
    Detect whether dataset uses language-prefixed configs (e.g., 'de-corpus').

    Returns:
        Tuple of (is_language_prefixed, available_languages)
    """
    try:
        configs = get_dataset_config_names(dataset_name)
    except Exception:  # noqa: BLE001 - discovery is intentionally best-effort
        return False, []

    # Check for standard configs
    if "corpus" in configs and "queries" in configs:
        return False, []

    # Check for language-prefixed configs (e.g., 'de-corpus', 'en-corpus')
    language_prefixes = set()
    for config in configs:
        if "-corpus" in config:
            lang = config.replace("-corpus", "")
            # Verify this language has all required configs
            if f"{lang}-queries" in configs and f"{lang}-qrels" in configs:
                language_prefixes.add(lang)

    if language_prefixes:
        return True, sorted(language_prefixes)

    return False, []


def resolve_dataset_configs(
    config_names: list[str], language: str | None
) -> DatasetConfigs:
    """Resolve one complete retrieval task from repository configuration metadata."""
    configs = set(config_names)
    if {"corpus", "queries"} <= configs:
        if language is not None:
            raise ValueError(
                f"Language '{language}' was requested, but this dataset uses standard "
                "non-language-prefixed configurations"
            )
        if "qrels" in configs:
            qrels_config = "qrels"
        elif "default" in configs:
            qrels_config = "default"
        else:
            raise ValueError(
                "Dataset has corpus and queries configurations but no qrels "
                "configuration ('qrels' or legacy 'default')"
            )
        return DatasetConfigs("corpus", "queries", qrels_config, None)

    languages = sorted(
        config.removesuffix("-corpus")
        for config in configs
        if config.endswith("-corpus")
        and f"{config.removesuffix('-corpus')}-queries" in configs
        and f"{config.removesuffix('-corpus')}-qrels" in configs
    )
    if not languages:
        raise ValueError(
            "Dataset does not expose a complete corpus, queries, and qrels "
            "configuration set"
        )
    if language is None:
        if len(languages) > 1:
            raise ValueError(
                "Dataset has multiple languages; select one explicitly with "
                f"--language ({', '.join(languages)})"
            )
        language = languages[0]
    elif language not in languages:
        raise ValueError(
            f"Language '{language}' is unavailable; choose from: {', '.join(languages)}"
        )

    return DatasetConfigs(
        corpus=f"{language}-corpus",
        queries=f"{language}-queries",
        qrels=f"{language}-qrels",
        language=language,
    )


def _as_split_mapping(datasets: Any, role: str) -> dict[str, Any]:
    """Expose a loaded dataset or DatasetDict as a named split mapping."""
    if isinstance(datasets, Mapping):
        if not datasets:
            raise ValueError(f"{role} configuration contains no splits")
        return {str(name): dataset for name, dataset in datasets.items()}

    split = getattr(datasets, "split", None)
    split_name = str(split) if split is not None else role
    return {split_name: datasets}


def select_dataset_splits(
    corpus_datasets: Any,
    queries_datasets: Any,
    qrels_datasets: Any,
    evaluation_split: str | None,
) -> SelectedDatasets:
    """Select corpus, queries, and qrels as one coherent retrieval evaluation."""
    corpus_splits = _as_split_mapping(corpus_datasets, "corpus")
    query_splits = _as_split_mapping(queries_datasets, "queries")
    qrels_splits = _as_split_mapping(qrels_datasets, "qrels")

    if evaluation_split is not None:
        if evaluation_split not in qrels_splits:
            raise ValueError(
                f"Requested evaluation split '{evaluation_split}' is not available "
                f"in qrels; choose from: {', '.join(sorted(qrels_splits))}"
            )
        qrels_split = evaluation_split
    elif len(qrels_splits) == 1:
        qrels_split = next(iter(qrels_splits))
    else:
        raise ValueError(
            "Qrels has multiple evaluation splits; select one explicitly with "
            f"--split ({', '.join(sorted(qrels_splits))})"
        )

    if qrels_split in query_splits:
        query_split = qrels_split
    elif "queries" in query_splits:
        query_split = "queries"
    elif len(query_splits) == 1:
        only_query_split = next(iter(query_splits))
        raise ValueError(
            f"Qrels split '{qrels_split}' cannot be matched to query split "
            f"'{only_query_split}'; expected a matching split or shared 'queries' split"
        )
    else:
        raise ValueError(
            f"Qrels split '{qrels_split}' cannot be matched to query splits: "
            f"{', '.join(sorted(query_splits))}"
        )

    if "corpus" in corpus_splits:
        corpus_split = "corpus"
    elif qrels_split in corpus_splits:
        corpus_split = qrels_split
    elif len(corpus_splits) == 1:
        corpus_split = next(iter(corpus_splits))
    else:
        raise ValueError(
            f"Corpus split is ambiguous for evaluation split '{qrels_split}'; "
            f"available splits: {', '.join(sorted(corpus_splits))}"
        )

    return SelectedDatasets(
        corpus=corpus_splits[corpus_split],
        queries=query_splits[query_split],
        qrels=qrels_splits[qrels_split],
        corpus_split=corpus_split,
        query_split=query_split,
        qrels_split=qrels_split,
        evaluation_split=qrels_split,
    )


def _prepare_retrieval_frames(
    corpus: pd.DataFrame, queries: pd.DataFrame, qrels: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Normalize selected frames and verify qrels/query split correspondence."""
    corpus = corpus.rename(columns={"_id": "id"}).copy()
    queries = queries.rename(columns={"_id": "id"}).copy()
    qrels = qrels.copy()

    for frame, table_name in ((corpus, "corpus"), (queries, "queries")):
        required_columns = {"id", "text"}
        if not required_columns <= set(frame.columns):
            raise ValueError(
                f"Selected {table_name} split must contain an 'id' or '_id' "
                "field and a 'text' field"
            )
    required_qrel_columns = {"query-id", "corpus-id"}
    if not required_qrel_columns <= set(qrels.columns):
        raise ValueError(
            "Selected qrels split must contain 'query-id' and 'corpus-id' fields"
        )

    for frame, table_name in (
        (corpus, "corpus"),
        (queries, "queries"),
        (qrels, "qrels"),
    ):
        if frame.empty:
            raise ValueError(f"Selected {table_name} split is empty")

    for frame, table_name in ((corpus, "corpus"), (queries, "queries")):
        invalid_text = ~frame["text"].map(lambda value: isinstance(value, str))
        if invalid_text.any():
            raise ValueError(
                f"Selected {table_name}.text contains {int(invalid_text.sum())} "
                "null or non-string value(s)"
            )
        if frame["id"].isna().any():
            raise ValueError(f"Selected {table_name}.id contains null values")
        frame["id"] = frame["id"].map(str)
        duplicate_ids = frame.loc[frame["id"].duplicated(keep=False), "id"]
        if not duplicate_ids.empty:
            raise ValueError(
                f"Selected {table_name}.id contains duplicate IDs after string "
                f"normalization; sample: {duplicate_ids.iloc[0]}"
            )

    if qrels[["query-id", "corpus-id"]].isna().any().any():
        raise ValueError("Selected qrels split contains null query or corpus IDs")
    qrels["query-id"] = qrels["query-id"].map(str)
    qrels["corpus-id"] = qrels["corpus-id"].map(str)
    if "score" not in qrels.columns:
        qrels["score"] = 1
    numeric_scores = pd.to_numeric(qrels["score"], errors="coerce")
    invalid_scores = numeric_scores.isna() | ~np.isfinite(numeric_scores)
    if invalid_scores.any():
        raise ValueError(
            f"Selected qrels.score contains {int(invalid_scores.sum())} "
            "non-numeric or non-finite value(s)"
        )
    qrels["score"] = numeric_scores

    qrel_query_ids = set(qrels["query-id"])
    missing_query_ids = qrel_query_ids - set(queries["id"])
    if missing_query_ids:
        raise ValueError(
            "Selected qrels do not correspond to the selected query split; "
            f"{len(missing_query_ids)} query ID(s) are missing, sample: "
            f"{min(missing_query_ids)}"
        )
    missing_corpus_ids = set(qrels["corpus-id"]) - set(corpus["id"])
    if missing_corpus_ids:
        raise ValueError(
            f"Selected qrels contain {len(missing_corpus_ids)} corpus ID(s) missing "
            f"from the selected corpus split, sample: {min(missing_corpus_ids)}"
        )

    # Shared query banks can contain queries for several evaluation splits.
    queries = queries.loc[queries["id"].isin(qrel_query_ids)].reset_index(drop=True)
    positive_query_ids = set(qrels.loc[qrels["score"] > 0, "query-id"])
    queries_without_positives = set(queries["id"]) - positive_query_ids
    if queries_without_positives:
        raise ValueError(
            f"Selected queries contain {len(queries_without_positives)} query ID(s) "
            "without positive qrels; sample: "
            f"{min(queries_without_positives)}"
        )
    return corpus, queries, qrels


def sample_retrieval_frames(
    corpus: pd.DataFrame,
    queries: pd.DataFrame,
    qrels: pd.DataFrame,
    corpus_size: int,
    seed: int,
    query_sample_size: int | None = None,
) -> SampledRetrievalFrames:
    """Create a reproducible query-led sample while retaining all positive docs."""
    if corpus_size <= 0:
        raise ValueError(f"corpus sample size must be positive, got {corpus_size}")
    if query_sample_size is not None and query_sample_size <= 0:
        raise ValueError(f"query sample size must be positive, got {query_sample_size}")

    corpus, queries, qrels = _prepare_retrieval_frames(corpus, queries, qrels)
    randomizer = random.Random(seed)
    query_ids = sorted(queries["id"].tolist())
    randomizer.shuffle(query_ids)

    positive_qrels = qrels.loc[qrels["score"] > 0]
    positives_by_query = {
        query_id: set(group["corpus-id"])
        for query_id, group in positive_qrels.groupby("query-id", sort=False)
    }

    if query_sample_size is not None:
        selected_query_ids = query_ids[: min(query_sample_size, len(query_ids))]
    elif corpus_size >= len(corpus):
        selected_query_ids = query_ids
    else:
        selected_query_ids = []
        selected_positive_ids: set[str] = set()
        for query_id in query_ids:
            prospective_ids = selected_positive_ids | positives_by_query[query_id]
            if not selected_query_ids or len(prospective_ids) <= corpus_size:
                selected_query_ids.append(query_id)
                selected_positive_ids = prospective_ids

    if not selected_query_ids:
        raise ValueError("query-led sampling produced no evaluation queries")

    selected_query_id_set = set(selected_query_ids)
    selected_positive_ids = set(
        positive_qrels.loc[
            positive_qrels["query-id"].isin(selected_query_id_set), "corpus-id"
        ]
    )
    distractor_ids = sorted(set(corpus["id"]) - selected_positive_ids)
    randomizer.shuffle(distractor_ids)
    distractors_needed = max(0, corpus_size - len(selected_positive_ids))
    retained_corpus_ids = selected_positive_ids | set(
        distractor_ids[:distractors_needed]
    )

    sampled_corpus = corpus.loc[corpus["id"].isin(retained_corpus_ids)].reset_index(
        drop=True
    )
    sampled_queries = queries.loc[
        queries["id"].isin(selected_query_id_set)
    ].reset_index(drop=True)
    sampled_qrels = qrels.loc[
        qrels["query-id"].isin(selected_query_id_set)
        & qrels["corpus-id"].isin(retained_corpus_ids)
    ].reset_index(drop=True)

    retained_positive_ids = set(
        sampled_qrels.loc[sampled_qrels["score"] > 0, "corpus-id"]
    )
    if retained_positive_ids != selected_positive_ids:
        raise RuntimeError(
            "query-led sampling failed to retain every positive document"
        )
    _prepare_retrieval_frames(sampled_corpus, sampled_queries, sampled_qrels)

    original_positive_ids = set(positive_qrels["corpus-id"])
    positive_coverage = len(retained_positive_ids) / len(original_positive_ids)
    stats: dict[str, int | float | None] = {
        "seed": seed,
        "requested_corpus_size": corpus_size,
        "requested_query_size": query_sample_size,
        "original_corpus_size": len(corpus),
        "original_query_size": len(queries),
        "original_qrels_size": len(qrels),
        "sampled_corpus_size": len(sampled_corpus),
        "sampled_query_size": len(sampled_queries),
        "sampled_qrels_size": len(sampled_qrels),
        "selected_positive_documents": len(selected_positive_ids),
        "positive_document_coverage": positive_coverage,
        "selected_query_positive_document_coverage": 1.0,
        "judgments_per_query": len(sampled_qrels) / len(sampled_queries),
    }
    return SampledRetrievalFrames(
        corpus=sampled_corpus,
        queries=sampled_queries,
        qrels=sampled_qrels,
        stats=stats,
    )


def download_mteb_dataset(
    dataset_name: str,
    output_dir: Path,
    sample_size: int | None = None,
    language: str | None = None,
    evaluation_split: str | None = None,
    seed: int = 42,
    query_sample_size: int | None = None,
    revision: str = "main",
) -> Path:
    """
    Download an MTEB v2 retrieval dataset from HuggingFace.

    Args:
        dataset_name: Name of the MTEB dataset on HuggingFace (e.g., "mteb/scifact")
        output_dir: Directory to save the dataset files
        sample_size: Target corpus size. Positive documents can make the result larger.
        language: Explicit language for a multilingual configuration set.
        evaluation_split: Qrels/query evaluation split, required when ambiguous.
        seed: Random seed for query and distractor sampling.
        query_sample_size: Optional explicit number of evaluation queries to sample.
        revision: Hugging Face dataset repository revision.

    Returns:
        Directory containing the parquet files and dataset manifest.
    """
    if sample_size is not None and sample_size <= 0:
        raise ValueError(f"sample size must be positive, got {sample_size}")
    if query_sample_size is not None and query_sample_size <= 0:
        raise ValueError(f"query sample size must be positive, got {query_sample_size}")
    if query_sample_size is not None and sample_size is None:
        raise ValueError("query sample size requires a corpus sample size")

    console.print(f"\n📥 Downloading dataset: [yellow]{dataset_name}[/yellow]")
    if sample_size is not None:
        console.print(f"   Sample size: [cyan]{sample_size}[/cyan] documents")
        console.print(f"   Sampling seed: [cyan]{seed}[/cyan]")

    config_names = get_dataset_config_names(dataset_name, revision=revision)
    configs = resolve_dataset_configs(config_names, language)
    console.print(
        f"   📦 Using configs: {configs.corpus}, {configs.queries}, {configs.qrels}"
    )

    # Create output directory (include language suffix for multilingual datasets)
    base_name = dataset_name.split("/")[-1]
    if configs.language:
        dataset_dir = output_dir / f"{base_name}_{configs.language}"
    else:
        dataset_dir = output_dir / base_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            corpus_task = progress.add_task("Downloading corpus...", total=None)
            corpus_datasets = load_dataset(
                dataset_name, name=configs.corpus, revision=revision
            )
            progress.update(corpus_task, completed=True, total=1)

            queries_task = progress.add_task("Downloading queries...", total=None)
            query_datasets = load_dataset(
                dataset_name, name=configs.queries, revision=revision
            )
            progress.update(queries_task, completed=True, total=1)

            qrels_task = progress.add_task("Downloading qrels...", total=None)
            qrels_datasets = load_dataset(
                dataset_name, name=configs.qrels, revision=revision
            )
            progress.update(qrels_task, completed=True, total=1)

        selected = select_dataset_splits(
            corpus_datasets,
            query_datasets,
            qrels_datasets,
            evaluation_split,
        )
        console.print(
            "   🧩 Using splits: "
            f"corpus={selected.corpus_split}, queries={selected.query_split}, "
            f"qrels={selected.qrels_split}"
        )

        corpus_df, queries_df, qrels_df = _prepare_retrieval_frames(
            selected.corpus.to_pandas(),
            selected.queries.to_pandas(),
            selected.qrels.to_pandas(),
        )
        original_stats = {
            "original_corpus_size": len(corpus_df),
            "original_query_size": len(queries_df),
            "original_qrels_size": len(qrels_df),
        }

        if sample_size is not None:
            console.print(f"\n✂️  Query-led sampling to {sample_size} documents...")
            sampled = sample_retrieval_frames(
                corpus_df,
                queries_df,
                qrels_df,
                corpus_size=sample_size,
                seed=seed,
                query_sample_size=query_sample_size,
            )
            corpus_df, queries_df, qrels_df = (
                sampled.corpus,
                sampled.queries,
                sampled.qrels,
            )
            sampling_stats = sampled.stats
            console.print(
                f"   • Corpus: {original_stats['original_corpus_size']} → "
                f"{len(corpus_df)} documents"
            )
            console.print(
                f"   • Queries: {original_stats['original_query_size']} → "
                f"{len(queries_df)} queries"
            )
            console.print(
                f"   • Qrels: {original_stats['original_qrels_size']} → "
                f"{len(qrels_df)} judgments"
            )
        else:
            sampling_stats = {
                "seed": seed,
                "requested_corpus_size": None,
                "requested_query_size": query_sample_size,
                **original_stats,
                "sampled_corpus_size": len(corpus_df),
                "sampled_query_size": len(queries_df),
                "sampled_qrels_size": len(qrels_df),
                "selected_positive_documents": int(
                    qrels_df.loc[qrels_df["score"] > 0, "corpus-id"].nunique()
                ),
                "positive_document_coverage": 1.0,
                "selected_query_positive_document_coverage": 1.0,
                "judgments_per_query": len(qrels_df) / len(queries_df),
            }

        console.print("\n💾 Saving to parquet files...")

        # Save files
        corpus_path = dataset_dir / "corpus.parquet"
        queries_path = dataset_dir / "queries.parquet"
        qrels_path = dataset_dir / "qrels.parquet"

        corpus_df.to_parquet(corpus_path, index=False)
        queries_df.to_parquet(queries_path, index=False)
        qrels_df.to_parquet(qrels_path, index=False)

        manifest = {
            "schema_version": 1,
            "repository": dataset_name,
            "revision": revision,
            "language": configs.language,
            "configurations": {
                "corpus": configs.corpus,
                "queries": configs.queries,
                "qrels": configs.qrels,
            },
            "splits": {
                "corpus": selected.corpus_split,
                "queries": selected.query_split,
                "qrels": selected.qrels_split,
                "evaluation": selected.evaluation_split,
            },
            "fingerprints": {
                "corpus": getattr(selected.corpus, "_fingerprint", None),
                "queries": getattr(selected.queries, "_fingerprint", None),
                "qrels": getattr(selected.qrels, "_fingerprint", None),
            },
            "sampling": sampling_stats,
        }
        manifest_path = dataset_dir / "dataset_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        console.print(
            f"   ✓ Corpus: [green]{corpus_path}[/green] ({len(corpus_df)} documents)"
        )
        console.print(
            f"   ✓ Queries: [green]{queries_path}[/green] ({len(queries_df)} queries)"
        )
        console.print(
            f"   ✓ Qrels: [green]{qrels_path}[/green] ({len(qrels_df)} relevance judgments)"
        )
        console.print(f"   ✓ Manifest: [green]{manifest_path}[/green]")

        # Print summary statistics
        console.print("\n📊 Dataset statistics:")
        console.print(f"   • Corpus size: {len(corpus_df)} documents")
        console.print(f"   • Queries: {len(queries_df)}")
        console.print(f"   • Relevance judgments: {len(qrels_df)}")

        # Calculate average relevance judgments per query
        avg_rels_per_query = (
            len(qrels_df) / len(queries_df) if len(queries_df) > 0 else 0
        )
        console.print(f"   • Avg relevance per query: {avg_rels_per_query:.2f}")
        console.print(
            "   • Positive document coverage: "
            f"{float(sampling_stats['positive_document_coverage']):.1%}"
        )
        return dataset_dir

    except DatasetNotFoundError:
        console.print(
            f"\n⚠️  [yellow]Dataset '{dataset_name}' not found on HuggingFace.[/yellow]"
        )
        console.print("   Please check the dataset name and try again.")
        console.print(
            "\n   [dim]Example: 'mteb/scifact', 'mteb/nfcorpus', 'mteb/fiqa'[/dim]"
        )
        raise
    except Exception as error:
        console.print(f"\n❌ [red]Error downloading dataset: {error}[/red]")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download MTEB v2 retrieval datasets from HuggingFace"
    )
    parser.add_argument(
        "dataset_name",
        type=str,
        help="HuggingFace dataset identifier (e.g., 'mteb/scifact', 'mteb/nfcorpus')",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./_data/mteb"),
        help="Output directory for downloaded datasets (default: ./_data/mteb)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Target N documents using seeded query-led sampling",
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Language code for multilingual datasets (required when ambiguous)",
    )
    parser.add_argument(
        "--split",
        type=str,
        help="Evaluation split (required when qrels has multiple splits)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed (default: 42)",
    )
    parser.add_argument(
        "--query-sample",
        type=int,
        help="Explicit number of evaluation queries to sample",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Hugging Face dataset revision (default: main)",
    )

    args = parser.parse_args()

    # Print header
    console.print(Panel("📦 MTEB Dataset Downloader", style="bold magenta"))

    try:
        download_mteb_dataset(
            dataset_name=args.dataset_name,
            output_dir=args.output_dir,
            sample_size=args.sample,
            language=args.language,
            evaluation_split=args.split,
            seed=args.seed,
            query_sample_size=args.query_sample,
            revision=args.revision,
        )

        console.print(Panel("✅ Download complete!", style="bold green"))
        return 0

    except Exception as error:  # noqa: BLE001 - CLI boundary reports failures
        console.print(f"\n❌ [red]Failed to download dataset: {error}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
