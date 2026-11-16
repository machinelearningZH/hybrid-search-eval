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
