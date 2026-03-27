"""File I/O utilities."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml


def create_run_dir(output_dir: str, experiment_name: str) -> Path:
    """Create a timestamped run directory."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(output_dir) / experiment_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "tensorboard").mkdir(exist_ok=True)
    (run_dir / "predictions").mkdir(exist_ok=True)
    return run_dir


def save_config_snapshot(config: dict[str, Any], run_dir: Path) -> None:
    """Save a complete config snapshot to the run directory."""
    snapshot_path = run_dir / "config_snapshot.yaml"
    with open(snapshot_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def save_environment_info(run_dir: Path) -> None:
    """Capture and save environment information for reproducibility."""
    info: dict[str, Any] = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    # Save environment info
    with open(run_dir / "environment.yaml", "w") as f:
        yaml.dump(info, f, default_flow_style=False)

    # Save pip freeze
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        (run_dir / "requirements_frozen.txt").write_text(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Save git hash if available
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            info["git_hash"] = result.stdout.strip()
            with open(run_dir / "environment.yaml", "w") as f:
                yaml.dump(info, f, default_flow_style=False)
    except FileNotFoundError:
        pass


def save_metrics(metrics: dict[str, Any], run_dir: Path) -> None:
    """Save metrics to JSON file."""
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(run_dir: Path) -> dict[str, Any]:
    """Load metrics from a run directory."""
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path) as f:
        return json.load(f)
