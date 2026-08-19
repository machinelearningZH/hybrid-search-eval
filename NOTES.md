# Notes

- Snowflake Arctic Embed v2 models can load a corrupted non-persistent
  `embeddings.position_ids` buffer in their custom GTE module. The symptom is
  an out-of-bounds RoPE cache index during `SentenceTransformer.encode`, often
  showing a huge integer index with a small valid range. Resetting that buffer
  to `torch.arange(num_positions, device=..., dtype=...)` immediately after
  model load fixes CPU and MPS encoding without changing model weights.
- MTEB downloads must select an explicit language when several complete
  language configuration groups exist and an explicit evaluation split when
  qrels has several splits. Query-led corpus sampling treats `--sample` as a
  target: all positive documents for selected queries take precedence, so the
  saved corpus may exceed that target. The dataset manifest records this case.
- Evaluation runs own only Weaviate collections whose names they generate and
  record. Names include a random per-run nonce and per-collection counter;
  collisions fail without deletion. Occupied embedded ports are treated as a
  startup failure rather than permission to connect to an arbitrary local
  server. Batch indexing is accepted only when it reports zero errors and the
  aggregate object count matches the requested document count.
