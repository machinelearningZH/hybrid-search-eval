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
- Snowflake Arctic Embed v2 models can load a corrupted non-persistent
  `embeddings.position_ids` buffer in their custom GTE module. The symptom is
  an out-of-bounds RoPE cache index during `SentenceTransformer.encode`, often
  showing a huge integer index with a small valid range. Resetting that buffer
  to `torch.arange(num_positions, device=..., dtype=...)` immediately after
  model load replaces that buffer without changing weights. The helper and
  synthetic-buffer tests are in `_core/utils.py` and `tests/test_utils.py`;
  those tests do not establish compatibility with every checkpoint or device.
- `WeaviateRunResources` registers uniquely named collections and cleanup targets
  only those collections. Name collisions fail without deletion; deletion
  failures are reported while cleanup continues. Embedded startup failure does
  not trigger a connection to another server. `index_weaviate_documents` checks
  document/vector counts, batch errors, and the final indexed object count.
