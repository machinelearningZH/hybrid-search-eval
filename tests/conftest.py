import sys
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def valid_config(tmp_path: Path) -> dict[str, Any]:
    """Return the smallest configuration accepted by ``validate_config``."""
    return {
        "project_id": "test",
        "data": {"mteb_data_dir": str(tmp_path)},
        "embeddings": {
            "huggingface": {
                "mini": "sentence-transformers/all-MiniLM-L6-v2",
            },
            "device": "cpu",
        },
        "search": {
            "alpha": [0.0, 0.5, 1.0],
            "metrics": {"mrr_k": [1, 10], "hit_rate_k": [5]},
        },
        "output": {"results_dir": str(tmp_path / "results")},
        "visualization": {},
        "model": {"embedding_batch_size": 8, "max_document_tokens": 512},
    }
