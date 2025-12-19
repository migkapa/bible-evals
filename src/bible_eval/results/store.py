from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_history(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Invalid history format (expected list): {path}")
    return data


def save_history(path: Path, runs: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")


def append_run(history_path: Path, run_summary: Dict[str, Any]) -> None:
    runs = load_history(history_path)
    runs.append(run_summary)
    save_history(history_path, runs)


def write_run_copy(runs_dir: Path, run_id: str, run_summary: Dict[str, Any]) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    out = runs_dir / f"{run_id}.json"
    out.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out

