import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import generate_evals


def test_invalid_evaluation_config_exits_unsuccessfully(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("{}\n")

    result = subprocess.run(
        ["uv", "run", "--no-sync", "generate_evals.py", "--config", str(config_path)],
        cwd=Path(generate_evals.__file__).parent,
        env={**os.environ, "UV_OFFLINE": "1", "UV_PYTHON_DOWNLOADS": "never"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert "Configuration validation failed" in result.stdout
    assert result.returncode == 1


def test_weaviate_startup_failure_returns_failure_status(
    evaluation_runtime: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_startup(**_kwargs: Any) -> None:
        raise generate_evals.WeaviateStartUpError("startup failed")

    monkeypatch.setattr(generate_evals.weaviate, "connect_to_embedded", fail_startup)

    assert generate_evals.main() == 1
    assert evaluation_runtime.encoded == []
