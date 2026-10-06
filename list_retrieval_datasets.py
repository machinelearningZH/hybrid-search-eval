from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from typing import Any

from rich.console import Console
from rich.table import Table

DEFAULT_RETRIEVAL_TYPES = (
    "Retrieval",
    "InstructionRetrieval",
    "Any2AnyRetrieval",
    "Any2AnyMultilingualRetrieval",
)

# Minimal ISO-639-1 -> ISO-639-3 normalization for common cases.
# Extend as needed.
_ISO639_1_TO_3 = {
    "en": "eng",
    "de": "deu",
    "fr": "fra",
    "es": "spa",
    "it": "ita",
    "pt": "por",
    "nl": "nld",
    "sv": "swe",
    "no": "nor",
    "da": "dan",
    "fi": "fin",
    "ru": "rus",
    "pl": "pol",
    "cs": "ces",
    "tr": "tur",
    "ar": "ara",
    "he": "heb",
    "hi": "hin",
    "ja": "jpn",
    "ko": "kor",
    # MTEB commonly uses cmn for Mandarin Chinese; "zh" is a practical shorthand.
    "zh": "cmn",
}

_MTEB_BENCH_RE = re.compile(
    r"^(?P<suite>[A-Za-z0-9_-]+)\(\s*(?P<lang>[A-Za-z]{2,10})\s*(?:,\s*(?P<ver>[^)]+?)\s*)?\)$"
)


def _normalize_lang_code(lang: str) -> str:
    lang = lang.strip().lower()
    if lang in ("ger",):  # common alias
        return "deu"
    if len(lang) == 2:
        return _ISO639_1_TO_3.get(lang, lang)
    return lang


def _version_rank(ver: str) -> int:
    """
    Higher is better when choosing fallbacks.
    Examples: v2 > v1 > classic > others.
    """
    v = ver.strip().lower()
    if v.startswith("v"):
        try:
            return 100 + int(v[1:])
        except ValueError:
            return 10
    if v == "classic":
        return 1
    return 0


def _resolve_benchmark_name(user_value: str, default_version: str = "v2") -> str:
    """
    Accept:
      - exact benchmark name (e.g. "BEIR", "MTEB(eng, v2)")
      - language shorthand: "de" / "deu" / "eng" (interpreted as "MTEB(<lang>, <default_version>)")
      - "MTEB(de, v2)" shorthand (normalize de->deu, then try / fallback)
    """
    import mteb

    raw = user_value.strip()

    # Build an alias/name lookup table from get_benchmarks()
    benchmarks = mteb.get_benchmarks()
    name_set = {b.name for b in benchmarks}
    alias_to_name = {}
    for b in benchmarks:
        alias_to_name[b.name] = b.name
        for a in getattr(b, "aliases", ()) or ():
            alias_to_name[a] = b.name

    def exists(name: str) -> bool:
        return name in alias_to_name or name in name_set

    # Case A: looks like "Suite(lang, ver)"
    m = _MTEB_BENCH_RE.match(raw)
    if m:
        suite = m.group("suite")
        lang = m.group("lang")
        ver_raw = m.group("ver")
        ver = ver_raw.strip() if ver_raw else default_version

        # Only normalize language codes for MTEB(...) style suites (safe, user-requested behavior)
        if suite == "MTEB":
            lang_norm = _normalize_lang_code(lang)
            candidate = f"{suite}({lang_norm}, {ver})"
        else:
            candidate = raw  # leave other suites untouched

        if exists(candidate):
            return alias_to_name.get(candidate, candidate)

        # If it's MTEB(lang, ver) and doesn't exist, try best fallback among MTEB(lang, *)
        if suite == "MTEB":
            lang_norm = _normalize_lang_code(lang)
            prefix = f"MTEB({lang_norm}, "
            candidates = [b.name for b in benchmarks if b.name.startswith(prefix)]
            if candidates:
                best = max(
                    candidates,
                    key=lambda n: _version_rank(n.split(",")[-1].rstrip(")")),
                )
                _warn(f'Benchmark "{candidate}" not found; falling back to "{best}".')
                return best

        # Otherwise, let mteb.get_benchmark raise a helpful KeyError downstream
        return candidate

    # Case B: language shorthand like "de" or "deu"
    if raw.isalpha() and 2 <= len(raw) <= 3:
        lang_norm = _normalize_lang_code(raw)
        candidate = f"MTEB({lang_norm}, {default_version})"
        if exists(candidate):
            return alias_to_name.get(candidate, candidate)

        # Fallback: any MTEB(lang_norm, *)
        prefix = f"MTEB({lang_norm}, "
        candidates = [b.name for b in benchmarks if b.name.startswith(prefix)]
        if candidates:
            best = max(
                candidates,
                key=lambda n: _version_rank(n.split(",")[-1].rstrip(")")),
            )
            _warn(f'Benchmark "{candidate}" not found; falling back to "{best}".')
            return best

        # No MTEB(...) benchmarks for that language; return candidate and let KeyError happen later
        return candidate

    # Case C: exact benchmark name like "BEIR"
    return raw


console = Console(stderr=True)


def _warn(msg: str) -> None:
    console.print(f"[bold yellow]⚠ {msg}[/bold yellow]")


def _get_mteb_version() -> str | None:
    try:
        return pkg_version("mteb")
    except PackageNotFoundError:
        return None


def _ensure_v2x(ver: str | None) -> None:
    if ver is None:
        _warn("mteb is not installed. Install with: pip install -U mteb")
        return
    # Soft check: user asked for v2.x compatibility
    if not ver.startswith("2."):
        _warn(
            f"Detected mteb=={ver}. This script targets MTEB v2.x; results may differ."
        )


def _metadata_dataset_dict(task: Any) -> dict[str, Any]:
    """
    Task metadata in MTEB v2.x is a TaskMetadata object. 'dataset' is typically a dict like:
      {"path": "mteb/SomeRetrievalTask", "revision": "...", ...}

    We try to be robust to:
      - dataclasses
      - plain objects
      - dict-like metadata
    """
    md = getattr(task, "metadata", None) or getattr(task, "metadata_dict", None) or task

    # If metadata itself is a dict
    if isinstance(md, dict):
        ds = md.get("dataset") or {}
        return ds if isinstance(ds, dict) else {}

    # If metadata is a dataclass, convert to dict
    if is_dataclass(md):
        md_dict = asdict(md)
        ds = md_dict.get("dataset") or {}
        return ds if isinstance(ds, dict) else {}

    # Otherwise attribute access
    ds = getattr(md, "dataset", None)
    return ds if isinstance(ds, dict) else {}


def _metadata_name(task: Any) -> str:
    md = getattr(task, "metadata", None) or getattr(task, "metadata_dict", None) or task
    if isinstance(md, dict):
        return str(
            md.get("name") or getattr(task, "__class__", type("X", (), {})).__name__
        )
    return str(
        getattr(md, "name", None)
        or getattr(task, "__class__", type("X", (), {})).__name__
    )


def _metadata_type(task: Any) -> str:
    md = getattr(task, "metadata", None) or getattr(task, "metadata_dict", None) or task
    if isinstance(md, dict):
        return str(md.get("type") or "")
    return str(getattr(md, "type", "") or "")


def _load_tasks(args: argparse.Namespace) -> list[Any]:
    import mteb

    if args.benchmark:
        bench_name = _resolve_benchmark_name(args.benchmark, default_version="v2")
        benchmark = mteb.get_benchmark(bench_name)
        tasks = list(getattr(benchmark, "tasks", benchmark))
    else:
        tasks = mteb.get_tasks(task_types=list(args.task_types))
    return tasks


def _filter_tasks_by_type(
    tasks: Iterable[Any], allowed_types: tuple[str, ...]
) -> list[Any]:
    out = []
    for t in tasks:
        t_type = _metadata_type(t)
        if t_type in allowed_types:
            out.append(t)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--benchmark",
        default=None,
        help=(
            'Benchmark selector. Accepts an exact benchmark name (e.g. "MTEB(eng, v2)", "BEIR") '
            'or language shorthand ("de"/"deu") which expands to "MTEB(<lang>, v2)". '
            'Also accepts "MTEB(<lang>)" which defaults to v2.'
        ),
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help='Only include tasks with metadata.type == "Retrieval" (exclude Instruction/Any2Any retrieval variants).',
    )
    ap.add_argument(
        "--format",
        choices=("text", "json", "csv"),
        default="text",
        help="Output format.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Write output to a file instead of stdout.",
    )
    ap.add_argument(
        "--include-task-names",
        action="store_true",
        help="Include the list of task names that reference each dataset repo.",
    )
    ap.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Do not deduplicate by (dataset_path, revision). Output one row per task.",
    )
    args = ap.parse_args()

    ver = _get_mteb_version()
    _ensure_v2x(ver)

    # Determine which task types to treat as "retrieval"
    args.task_types = ("Retrieval",) if args.strict else DEFAULT_RETRIEVAL_TYPES

    try:
        tasks = _load_tasks(args)
    except (KeyError, ValueError, OSError) as e:
        _warn(f"Failed to load tasks via mteb: {e}")
        return 2

    # If tasks came from a benchmark, we still enforce retrieval-type filtering here.
    tasks = _filter_tasks_by_type(tasks, tuple(args.task_types))

    # Collect dataset repos from metadata
    per_task_rows: list[dict[str, Any]] = []
    for t in tasks:
        ds = _metadata_dataset_dict(t)
        path = ds.get("path") or ds.get("name") or ds.get("repo_id") or ""
        revision = ds.get("revision") or ""
        name = _metadata_name(t)
        t_type = _metadata_type(t)

        per_task_rows.append(
            {
                "task_name": name,
                "task_type": t_type,
                "dataset_path": path,
                "dataset_revision": revision,
            }
        )

    # Deduplicate to "all HF datasets" (unique repos/revisions), optionally including task lists
    if not args.no_dedupe:
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        tasks_by_ds: dict[tuple[str, str], list[str]] = defaultdict(list)

        for r in per_task_rows:
            key = (r["dataset_path"], r["dataset_revision"])
            tasks_by_ds[key].append(r["task_name"])
            if key not in grouped:
                grouped[key] = {
                    "dataset_path": r["dataset_path"],
                    "dataset_revision": r["dataset_revision"],
                }

        rows: list[dict[str, Any]] = []
        for key, base in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
            row = dict(base)
            if args.include_task_names:
                row["tasks"] = sorted(set(tasks_by_ds[key]))
            rows.append(row)
    else:
        # One row per task
        rows = sorted(
            per_task_rows,
            key=lambda r: (r["dataset_path"], r["dataset_revision"], r["task_name"]),
        )

    # Emit output
    with (
        open(args.out, "w", newline="", encoding="utf-8")
        if args.out
        else nullcontext(sys.stdout)
    ) as out_f:
        if args.format == "json":
            if out_f is sys.stdout:
                from rich.json import JSON

                console_out = Console(file=out_f)
                console_out.print(JSON.from_data(rows))
            else:
                json.dump(rows, out_f, ensure_ascii=False, indent=2)
        elif args.format == "csv":
            if not rows:
                _warn("No rows to output.")
                return 0
            fieldnames = list(rows[0].keys())
            w = csv.DictWriter(out_f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                # CSV can't store lists cleanly; join if present
                r2 = dict(r)
                if isinstance(r2.get("tasks"), list):
                    r2["tasks"] = ";".join(r2["tasks"])
                w.writerow(r2)
        else:  # text
            console_out = Console(file=out_f)
            if out_f is sys.stdout and rows:
                # Use rich Table for stdout text output
                table = Table(show_header=True, header_style="bold magenta")

                if args.no_dedupe:
                    table.add_column("Dataset Path", style="cyan")
                    table.add_column("Revision", style="green")
                    table.add_column("Task Type", style="yellow")
                    table.add_column("Task Name", style="blue")
                    for r in rows:
                        table.add_row(
                            r["dataset_path"],
                            r["dataset_revision"],
                            r["task_type"],
                            r["task_name"],
                        )
                else:
                    table.add_column("Dataset Path", style="cyan")
                    table.add_column("Revision", style="green")
                    if args.include_task_names:
                        table.add_column("Tasks", style="yellow")
                        for r in rows:
                            tasks_str = ", ".join(r.get("tasks", []))
                            table.add_row(
                                r["dataset_path"], r["dataset_revision"], tasks_str
                            )
                    else:
                        for r in rows:
                            table.add_row(r["dataset_path"], r["dataset_revision"])
                console_out.print(table)
            else:
                # Plain text output for files or when no rich formatting
                if args.no_dedupe:
                    for r in rows:
                        console_out.print(
                            f"{r['dataset_path']}\t{r['dataset_revision']}\t{r['task_type']}\t{r['task_name']}"
                        )
                else:
                    for r in rows:
                        if args.include_task_names:
                            tasks_str = ", ".join(r.get("tasks", []))
                            console_out.print(
                                f"{r['dataset_path']}\t{r['dataset_revision']}\t{tasks_str}"
                            )
                        else:
                            console_out.print(
                                f"{r['dataset_path']}\t{r['dataset_revision']}"
                            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
