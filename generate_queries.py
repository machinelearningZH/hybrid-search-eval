import argparse
import os
from pathlib import Path
from typing import cast
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from rich.panel import Panel
import time

from _core.utils_prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from _core.utils import (
    load_config,
    truncate_text_to_tokens,
    load_mteb_retrieval_data_from_dir,
)

console = Console()

# Default output directory for user data
DEFAULT_USER_DATA_DIR = Path("_data/mteb_user")


def load_documents_from_file(file_path: Path) -> pd.DataFrame:
    """
    Load documents from various file formats (Excel, CSV, Parquet).

    Args:
        file_path: Path to the input file

    Returns:
        DataFrame with 'id' and 'text' columns

    Raises:
        ValueError: If file format is unsupported or required columns are missing
    """
    extension = file_path.suffix.lower()

    if extension == ".csv":
        df = pd.read_csv(file_path)
    elif extension in [".parquet", ".pq"]:
        df = pd.read_parquet(file_path)
    elif extension in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(
            f"Unsupported file format: {extension}. "
            "Supported formats: .csv, .parquet, .pq, .xlsx, .xls"
        )

    # Validate required columns
    if "text" not in df.columns:
        raise ValueError(
            f"Input file must have a 'text' column. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Generate IDs if not present
    if "id" not in df.columns:
        console.print("ℹ️  No 'id' column found, generating sequential IDs")
        df["id"] = [f"doc_{i}" for i in range(len(df))]

    # Ensure id column is string type
    df["id"] = df["id"].astype(str)

    return df[["id", "text"]]


def save_mteb_dataset(
    corpus_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Save data in MTEB 2.x format.

    Args:
        corpus_df: DataFrame with 'id' and 'text' columns for corpus
        queries_df: DataFrame with 'id' and 'text' columns for queries
        qrels_df: DataFrame with 'query-id', 'corpus-id', 'score' columns
        output_dir: Directory to save the MTEB dataset
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = output_dir / "corpus.parquet"
    queries_path = output_dir / "queries.parquet"
    qrels_path = output_dir / "qrels.parquet"

    corpus_df.to_parquet(corpus_path, index=False)
    queries_df.to_parquet(queries_path, index=False)
    qrels_df.to_parquet(qrels_path, index=False)

    console.print(f"\n💾 Saved MTEB dataset to: {output_dir}")
    console.print(f"   • corpus.parquet: {len(corpus_df)} documents")
    console.print(f"   • queries.parquet: {len(queries_df)} queries")
    console.print(f"   • qrels.parquet: {len(qrels_df)} relevance judgments")


def generate_queries_for_document(
    client: OpenAI,
    document_text: str,
    num_queries: int,
    model: str,
    max_retries: int,
    max_document_tokens: int,
    temperature: float,
    max_output_tokens: int,
) -> list[str]:
    """
    Generate search queries for a single document using OpenRouter or Ollama.

    Args:
        client: OpenAI client configured for OpenRouter or Ollama
        document_text: The document text to generate queries for
        num_queries: Number of queries to generate per document
        model: Model identifier
        max_retries: Maximum number of retry attempts
        max_document_tokens: Maximum number of tokens to send to LLM
        temperature: Temperature for LLM generation (higher = more diverse)
        max_output_tokens: Maximum number of tokens in LLM response

    Returns:
        List of generated queries
    """
    # Truncate document text to max_document_tokens
    truncated_text = truncate_text_to_tokens(document_text, max_document_tokens)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        document_text=truncated_text, num_queries=num_queries
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_output_tokens,
            )

            # Parse response - queries should be one per line
            content = response.choices[0].message.content
            if content is None:
                return []
            queries = [q.strip() for q in content.strip().split("\n") if q.strip()]

            # Filter out any numbered lists (e.g., "1. query")
            queries = [q.lstrip("0123456789.- ") for q in queries]

            return queries[:num_queries]  # Ensure we don't exceed requested number

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt  # Exponential backoff
                console.print(f"⚠️  Error (attempt {attempt + 1}/{max_retries}): {e}")
                console.print(f"   Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                console.print(f"❌ Failed after {max_retries} attempts: {e}")
                return []

    return []


def process_document_worker(
    idx: int,
    doc_id: str,
    document_text: str,
    num_queries: int,
    api_key: str,
    model: str,
    max_retries: int,
    max_document_tokens: int,
    temperature: float,
    max_output_tokens: int,
    provider: str = "openrouter",
    base_url: str = "https://openrouter.ai/api/v1",
) -> tuple[int, str, list[str]]:
    """
    Worker function to process a single document in parallel.

    Args:
        idx: Document index (for progress tracking)
        doc_id: Document ID
        document_text: The document text to generate queries for
        num_queries: Number of queries to generate per document
        api_key: API key (for OpenRouter) or "ollama" for local Ollama
        model: Model identifier
        max_retries: Maximum number of retry attempts
        max_document_tokens: Maximum number of tokens to send to LLM
        temperature: Temperature for LLM generation (higher = more diverse)
        max_output_tokens: Maximum number of tokens in LLM response
        provider: "openrouter" or "ollama"
        base_url: Base URL for the API

    Returns:
        Tuple of (document_index, document_id, list_of_queries)
    """
    # Each worker creates its own client instance (thread-safe)
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    queries = generate_queries_for_document(
        client=client,
        document_text=document_text,
        num_queries=num_queries,
        model=model,
        max_retries=max_retries,
        max_document_tokens=max_document_tokens,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    return idx, doc_id, queries


def generate_all_queries(
    documents_df: pd.DataFrame,
    num_queries_per_doc: int,
    api_key: str,
    model: str,
    max_workers: int,
    max_retries: int,
    max_document_tokens: int,
    temperature: float,
    max_output_tokens: int,
    provider: str = "openrouter",
    base_url: str = "https://openrouter.ai/api/v1",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate queries for all documents using parallel processing.

    Args:
        documents_df: DataFrame with 'id' and 'text' columns
        num_queries_per_doc: Number of queries to generate per document
        api_key: API key (for OpenRouter) or "ollama" for local Ollama
        model: Model identifier
        max_workers: Maximum number of parallel workers
        max_retries: Maximum number of retry attempts
        max_document_tokens: Maximum number of tokens to send to LLM
        temperature: Temperature for LLM generation (higher = more diverse)
        max_output_tokens: Maximum number of tokens in LLM response
        provider: "openrouter" or "ollama"
        base_url: Base URL for the API

    Returns:
        Tuple of (queries_df, qrels_df) in MTEB format
    """
    all_queries: list[dict[str, str | int]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Generating {num_queries_per_doc} queries per document "
            f"(using {max_workers} workers)...",
            total=len(documents_df),
        )

        # Submit all tasks to the thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create futures for all documents
            futures = {
                executor.submit(
                    process_document_worker,
                    cast(int, idx),
                    row["id"],
                    row["text"],
                    num_queries_per_doc,
                    api_key,
                    model,
                    max_retries,
                    max_document_tokens,
                    temperature,
                    max_output_tokens,
                    provider,
                    base_url,
                ): cast(int, idx)
                for idx, row in documents_df.iterrows()
            }

            # Process completed futures as they finish
            for future in as_completed(futures):
                try:
                    idx, doc_id, queries = future.result()

                    # Add queries with document ID
                    for query in queries:
                        all_queries.append(
                            {
                                "query_text": query,
                                "doc_id": doc_id,
                            }
                        )

                except Exception as e:
                    doc_idx = futures[future]
                    console.print(f"⚠️  Error processing document {doc_idx}: {e}")

                # Update progress
                progress.update(task, advance=1)

    # Create MTEB format DataFrames
    queries_df = pd.DataFrame(
        {
            "id": [f"query_{i}" for i in range(len(all_queries))],
            "text": [q["query_text"] for q in all_queries],
        }
    )

    qrels_df = pd.DataFrame(
        {
            "query-id": [f"query_{i}" for i in range(len(all_queries))],
            "corpus-id": [q["doc_id"] for q in all_queries],
            "score": [1] * len(all_queries),  # Binary relevance
        }
    )

    console.print(f"\n✅ Generated {len(queries_df)} total queries")
    console.print(
        f"   Average: {len(queries_df) / len(documents_df):.1f} queries per document"
    )

    return queries_df, qrels_df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate search queries from documents using LLM. "
        "Creates an MTEB-format dataset from your documents."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Input file with documents (Excel, CSV, or Parquet). "
        "Must have 'text' column, 'id' column is optional.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_USER_DATA_DIR,
        help=f"Output directory for MTEB dataset (default: {DEFAULT_USER_DATA_DIR})",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=10,
        help="Number of queries to generate per document (default: 10)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default="_configs/config.yaml",
        help="Path to config file (default: _configs/config.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="LLM model to use (overrides config)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of parallel workers (overrides config)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openrouter", "ollama"],
        default="openrouter",
        help="LLM provider to use: 'openrouter' (default) or 'ollama' for local",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434/v1",
        help="Ollama API URL (default: http://localhost:11434/v1)",
    )
    parser.add_argument(
        "--mteb-input-dir",
        type=Path,
        help="Load corpus from existing MTEB format directory instead of input_file",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(str(args.config))
        query_gen_config = config.get("query_generation", {})
    except Exception as e:
        console.print(f"❌ [red]Error loading config: {e}[/red]")
        return 1

    # Use config values, but allow CLI arguments to override
    if args.provider == "ollama":
        ollama_config = query_gen_config.get("ollama", {})
        default_model = ollama_config.get("model", "llama3.2:latest")
        default_max_workers = ollama_config.get("max_workers", 5)
    else:
        default_model = query_gen_config.get(
            "model", "google/gemini-2.5-flash-preview-09-2025"
        )
        default_max_workers = query_gen_config.get("max_workers", 25)

    model = args.model if args.model else default_model
    max_workers = args.max_workers if args.max_workers else default_max_workers
    max_retries = query_gen_config.get("max_retries", 3)
    max_document_tokens = config.get("model", {}).get("max_document_tokens", 512)
    temperature = query_gen_config.get("temperature", 1.0)
    max_output_tokens = query_gen_config.get("max_output_tokens", 500)

    # Print header
    console.print(
        Panel("🤖 Query Generator - Create MTEB Dataset", style="bold magenta")
    )

    # Load environment variables
    load_dotenv()

    # Handle API key based on provider
    api_key: str
    if args.provider == "ollama":
        api_key = "ollama"
        base_url = args.ollama_url
        console.print(f"🦙 Using Ollama locally at: {base_url}")
    else:
        api_key_env = os.getenv("OPENROUTER_API_KEY")
        base_url = "https://openrouter.ai/api/v1"

        if not api_key_env:
            console.print(
                "❌ [red]Error: OPENROUTER_API_KEY not found in environment[/red]"
            )
            console.print("   Please create a .env file with your OpenRouter API key")
            console.print("   Example: OPENROUTER_API_KEY=your_key_here")
            console.print("\n   Or use --provider ollama to run locally with Ollama")
            return 1
        api_key = api_key_env

    # Load documents
    if args.mteb_input_dir:
        # Load from existing MTEB format directory
        if not args.mteb_input_dir.exists():
            console.print(
                f"❌ [red]Error: MTEB input directory not found: {args.mteb_input_dir}[/red]"
            )
            return 1

        try:
            console.print(
                f"📂 Loading corpus from MTEB directory: {args.mteb_input_dir}"
            )
            mteb_data = load_mteb_retrieval_data_from_dir(str(args.mteb_input_dir))
            documents_df = mteb_data.corpus[["id", "text"]]
            console.print(f"✅ Loaded {len(documents_df)} documents (MTEB format)")
        except Exception as e:
            console.print(f"❌ [red]Error loading MTEB corpus: {e}[/red]")
            return 1
    else:
        # Load from user file (Excel, CSV, Parquet)
        if not args.input_file.exists():
            console.print(
                f"❌ [red]Error: Input file not found: {args.input_file}[/red]"
            )
            return 1

        try:
            console.print(f"📂 Loading documents from: {args.input_file}")
            documents_df = load_documents_from_file(args.input_file)
            console.print(
                f"✅ Loaded {len(documents_df)} documents "
                f"({args.input_file.suffix.lower()} format)"
            )
        except ValueError as e:
            console.print(f"❌ [red]Error: {e}[/red]")
            return 1
        except Exception as e:
            console.print(f"❌ [red]Error loading documents: {e}[/red]")
            return 1

    # Show configuration
    console.print("\n🔧 Configuration:")
    console.print(f"   Provider: {args.provider}")
    console.print(f"   Model: {model}")
    console.print(f"   Queries per document: {args.num_queries}")
    console.print(f"   Max workers: {max_workers}")
    console.print(f"   Max retries: {max_retries}")
    console.print(f"   Max document tokens: {max_document_tokens}")
    console.print(f"   Temperature: {temperature}")
    console.print(f"   Max output tokens: {max_output_tokens}")
    console.print(f"   Total documents: {len(documents_df)}")
    console.print(f"   Output directory: {args.output_dir}\n")

    # Generate queries
    try:
        queries_df, qrels_df = generate_all_queries(
            documents_df=documents_df,
            num_queries_per_doc=args.num_queries,
            api_key=api_key,
            model=model,
            max_workers=max_workers,
            max_retries=max_retries,
            max_document_tokens=max_document_tokens,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            provider=args.provider,
            base_url=base_url,
        )
    except Exception as e:
        console.print(f"❌ [red]Error generating queries: {e}[/red]")
        return 1

    # Save MTEB dataset
    save_mteb_dataset(
        corpus_df=documents_df,
        queries_df=queries_df,
        qrels_df=qrels_df,
        output_dir=args.output_dir,
    )

    # Show sample queries
    console.print("\n📋 Sample queries:")
    sample_size = min(5, len(queries_df))
    for i in range(sample_size):
        query_text = queries_df.iloc[i]["text"]
        corpus_id = qrels_df.iloc[i]["corpus-id"]
        display_text = query_text[:60] + "..." if len(query_text) > 60 else query_text
        console.print(f"   • [cyan]{display_text}[/cyan] → {corpus_id}")

    console.print(f"\n✅ [green]Done! MTEB dataset saved to {args.output_dir}[/green]")
    console.print("   You can now run evaluation with: uv run generate_evals.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
