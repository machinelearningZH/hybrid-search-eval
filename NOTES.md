# Notes

- Snowflake Arctic Embed v2 models can load a corrupted non-persistent
  `embeddings.position_ids` buffer in their custom GTE module. The symptom is
  an out-of-bounds RoPE cache index during `SentenceTransformer.encode`, often
  showing a huge integer index with a small valid range. Resetting that buffer
  to `torch.arange(num_positions, device=..., dtype=...)` immediately after
  model load fixes CPU and MPS encoding without changing model weights.
