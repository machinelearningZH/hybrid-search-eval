# Notes

- EmbeddingGemma 2's processor imports Gemma 4 image-processing code, which
  requires torchvision even for text-only evaluation. Checking the model class
  import alone does not verify that the Sentence Transformers loading path works;
  verify processor loading and text encoding as well.
- EmbeddingGemma 2 needs Transformers 5.19.0; Sentence Transformers 6.x alone
  does not provide the architecture. PyLate 1.6.0 requires Transformers <=5.3.0,
  so the mutually exclusive profiles must resolve separate Transformers versions.
  The 5.19.0 pin and package-specific cutoff were approved on 2026-10-06;
  the global seven-day cooldown remains in place for other packages.
- Evaluation model-loading and encoding boundaries intentionally catch arbitrary
  third-party exceptions so a failed model does not stop other evaluations. Keep
  these local, documented Ruff exceptions; query generation retries only OpenAI
  API errors and allows programming errors to propagate.
- New embedding-cache metadata writes use UTC offsets; existing cache metadata
  is not migrated. Result filenames retain local-time names using an aware
  datetime.
- PyLate is imported when ColBERT models are configured, before data loading or
  cache checks. Even a fully cached ColBERT run requires that backend. Profile
  commands are documented in [README.md](README.md#install).
- Snowflake Arctic Embed v2's custom GTE code calls `get_extended_attention_mask`,
  removed in Transformers 5. Restore the bidirectional encoder mask helper on
  the loaded instance only when missing. Non-persistent position IDs and rotary
  buffers can also load uninitialized: invalid position IDs cause indexing
  errors, while corrupt rotary caches can silently produce non-finite embeddings.
  Rebuild position IDs and use GTE's native `_init_rope(config)` to preserve RoPE
  scaling without changing weights. Helpers/tests live in `_core/utils.py` and
  `tests/test_utils.py`. Offline CPU encoding with the cached Arctic Embed m-v2.0
  checkpoint verified finite document/query embeddings and padding invariance;
  other checkpoints and accelerator devices have not been verified.
- `WeaviateRunResources` registers uniquely named collections and cleanup targets
  only those collections. Name collisions fail without deletion; deletion
  failures are reported while cleanup continues. Embedded startup failure does
  not trigger a connection to another server. `index_weaviate_documents` checks
  document/vector counts, batch errors, and the final indexed object count.

## Review fixes — 2026-11-16

- Cache identities now include dataset contents and evaluation settings. Legacy
  or incompatible entries are recomputed without deleting files; unpinned model
  revisions still require explicit recomputation after upstream changes.
- Embeddings must preserve input counts and have consistent finite numeric
  vectors before indexing or retrieval. OpenRouter response indexes restore
  input order, and configurable connection/read timeouts bound stalled requests.
- ColBERT caches use numeric arrays with pickle disabled. Dashboard text and
  inline JSON use context-appropriate escaping; model names use DOM text APIs.
- Duplicate normalized qrel pairs are rejected rather than aggregated. Existing
  datasets containing duplicates must be cleaned by their owner.
- Invalid evaluation configuration and embedded startup failure return status 1.
  The existing policy of skipping failed models is unchanged; defining partial
  and total model-failure statuses remains deferred in `PLAN.md`.
- Lexical searches explicitly target `text`. Result-cache identities include
  this restriction and ColBERT's ascending document-ID tie rule, so old results
  are recomputed without deleting them or recomputing compatible embeddings.
  An isolated offline Weaviate 1.39.7 check confirmed unrestricted BM25/hybrid
  matched document IDs, while text-restricted searches excluded ID-only matches.
- Generated query IDs now use `<source ID>::query_<zero-based position>` and
  output follows source order. `generation_summary.json` records completeness;
  partial/failed generation exits with status 1. Total failure leaves existing
  Parquet files unchanged. Existing generated datasets are not migrated.
- Loaded models are released in a per-model `finally` block before garbage
  collection and device-cache cleanup. Moving a model to CPU before release is
  unnecessary and could mask the original failure if the transfer raises.

## Embedding selection — 2026-11-16

The active additions in `_configs/config.yaml` target compact, commercially
usable open weights. Existing baselines remain enabled. Parameter counts below
include vocabulary embeddings; Bekko's advertised 8M is its active transformer
count, not its total size. Model cards link to the weights and licence terms.

| Added/enabled model | Total parameters | Licence | Reason to evaluate |
| --- | ---: | --- | --- |
| [Nemotron 3 Embed 1B](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16) | 1.14B | OpenMDW-1.1 | Strong German retrieval; upper end of this size budget |
| [Harrier 270M](https://huggingface.co/microsoft/harrier-oss-v1-270m) | 268M | MIT | Compact multilingual decoder |
| [Harrier 0.6B](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) | 596M | MIT | Higher-quality sibling with complete German benchmark coverage |
| [Perplexity Embed v1 0.6B](https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b) | 596M | MIT | Bidirectional diffusion-pretrained encoder; no query instructions |
| [Granite multilingual R2 311M](https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2) | 312M | Apache-2.0 | Larger encoder counterpart to the existing 97M baseline |
| [F2LLM-v2 160M](https://huggingface.co/codefuse-ai/F2LLM-v2-160M) | 159M | Apache-2.0 | Small pruned decoder; an efficiency comparison, not a German quality leader |
| [Bekko a8m](https://huggingface.co/hotchpotch/bekko-embedding-v1-a8m) | 106M | MIT | Only 7.7M active parameters; multilingual CPU-oriented alternative |
| [Voyage 4 Nano](https://huggingface.co/MongoDB/voyage-4-nano) | 346M | Apache-2.0 | Compact multilingual model with 2048-dimensional embeddings |

Prompt settings follow the saved Sentence Transformers configurations: Harrier
uses `web_search_query`, F2LLM uses `query`, Nemotron uses `query: ` / `passage: `,
and Voyage uses distinct `query` / `document` prompts. The existing
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) entry also now
uses its saved `query` / `document` prompts; bare `encode()` selected neither.
Prompt-aware cache identities cause affected embeddings/results to be recomputed.
No backend or dependency changes were required.

### German benchmark evidence

These are calculated means of published per-task `main_score` values, multiplied
by 100, from the [official MTEB result repository](https://github.com/embeddings-benchmark/results).
They are not new local benchmark runs. MTEB German retrieval averages
GermanQuAD-Retrieval (MRR@5), GermanDPR, XMarket's German subset, and full GerDaLIR
(the latter three use nDCG@10). RTEB German averages LegalQuAD, German1Retrieval,
GermanHealthcare1Retrieval, and GermanLegal1Retrieval (all nDCG@10). These are
different task sets, so compare models within each column. Missing scores are
not zeros, and general multilingual MTEB means are not German retrieval scores.

| Model / published submission | MTEB German retrieval, 4 tasks | RTEB German, 4 tasks |
| --- | ---: | ---: |
| [Nemotron 3 1B](https://github.com/embeddings-benchmark/results/tree/main/results/nvidia__Nemotron-3-Embed-1B-BF16/f880174635613cff04033875fc6a69296cb72006) | 59.10 | 80.65 |
| [Harrier 0.6B](https://github.com/embeddings-benchmark/results/tree/main/results/microsoft__harrier-oss-v1-0.6b/f9b9dc8d367d443f2479d27aa5d8d2850c0774ee) | 55.53 | 73.61 |
| [Bekko a8m](https://github.com/embeddings-benchmark/results/tree/main/results/hotchpotch__bekko-embedding-v1-a8m/a8cedb6b46fad5df6f10e1c94750ff62e298fbd2) | 53.09 | 66.48 |
| [Harrier 270M](https://github.com/embeddings-benchmark/results/tree/main/results/microsoft__harrier-oss-v1-270m/31de22b673913c7d658c0f03f792d77c2dcf8ebd) | 51.99 | 69.11 |
| [F2LLM-v2 160M](https://github.com/embeddings-benchmark/results/tree/main/results/codefuse-ai__F2LLM-v2-160M/9eb35007d12e8624e6bf3e9e35d2255b5269597f) | 47.50 | Incomplete |
| [Granite R2 311M](https://github.com/embeddings-benchmark/results/tree/main/results/ibm-granite__granite-embedding-311m-multilingual-r2/dba7b0ee9d789f330fecfb85df57699f9e7d9c42) | Incomplete | 62.93 |

The inspected Perplexity submission lacked these complete German task sets;
Voyage was selected from its published multilingual retrieval documentation.
Published submissions can use earlier model revisions than today's Hub files.
For a first German evaluation, prioritize Nemotron, Harrier 0.6B, and Bekko at
different compute budgets; keep the other additions as comparisons on local data.

### Exclusions

- Removed the commented `most-embed-de` and Jina v3 examples: their released
  weights use CC BY-NC-4.0. Jina v5 text weights have the same noncommercial
  restriction, despite strong benchmark results.
- [Nemotron 3.5 1B](https://huggingface.co/nvidia/Nemotron-3.5-Embed-1B-BF16)
  adds vision parameters while retaining Nemotron 3's frozen text embedding
  space. The text-only runner gains no new text model from those extra weights.
- [Perplexity v2 late](https://huggingface.co/perplexity-ai/pplx-embed-v2-late-0.6b)
  needs `MultiVectorEncoder`; its model card explicitly warns that PyLate puts
  its markers in the wrong position. The v2 contextual model needs a different
  chunk/query API. Neither is a configuration-only addition to this runner.
- Blink 8M is English-only, Meta Encoder v1 is about 28B parameters, and the
  larger 4B–27B variants add substantial cost outside this compact shortlist.

### Runtime verification

All eight additions and the corrected EmbeddingGemma 2 entry passed real
Sentence Transformers loading and encoding on Apple MPS with Python 3.12.16,
PyTorch 2.15.0, Sentence Transformers 6.2.0, and Transformers 5.19.0. Checks used
the repository's parsed prefixes/prompts and embedding validator: finite vectors,
correct dimensions/counts, three German questions retrieving their expected
documents from four candidates, and single-query versus padded-batch cosine
above 0.999. Each also encoded a batch of 32 German documents truncated with
the runner's `cl100k_base` helper to the configured 512-token limit.

| Model | Tested Hub revision | Output dimensions |
| --- | --- | ---: |
| EmbeddingGemma 2 | `f9972ac22e696a105c482202646979e33e4b5a1e` | 768 |
| Nemotron 3 1B | `c0c9fea93ea424587517f2c59e20db9f1d6bf615` | 2048 |
| Harrier 270M | `31de22b673913c7d658c0f03f792d77c2dcf8ebd` | 640 |
| Harrier 0.6B | `f9b9dc8d367d443f2479d27aa5d8d2850c0774ee` | 1024 |
| Perplexity v1 0.6B | `2c4d510dd4a732063c31a0f70193e35067b51fd8` | 1024 |
| Granite R2 311M | `44399559930365213510b1ee2eb15ded83374f0e` | 768 |
| F2LLM-v2 160M | `4ffe22c31406fc321b65a05b15563d38275ee51f` | 640 |
| Bekko a8m | `c721113d59a1d91b447450324f51c4b3332c924a` | 384 |
| Voyage 4 Nano | `5fb34e870f49f19e79c94d5e118f5ee730133279` | 2048 |

Checkpoints select BF16 except Perplexity (FP32). BF16 normalization differs
from unit norm by a few thousandths, so the check used dtype epsilon rather
than FP32 tolerances and computed padding-invariance cosine in FP32. Nemotron
emits its documented `apply_yarn_scaling` configuration warning; encoding passed.
These checks establish runtime compatibility, not full-corpus retrieval quality
or CPU/CUDA compatibility. The 205 offline tests and Ruff checks also passed.
