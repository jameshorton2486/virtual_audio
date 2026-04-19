from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_TRAINING_DATA = {
    "rules": {},
    "examples": [],
}


class TrainingStore:
    def __init__(self, path: str = "data/training_data.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._save(dict(DEFAULT_TRAINING_DATA))

    def _load(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return dict(DEFAULT_TRAINING_DATA)

    def _save(self, data: dict[str, Any]) -> None:
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def add_rule(self, wrong: str, correct: str) -> None:
        data = self._load()
        data.setdefault("rules", {})[wrong.lower()] = correct
        self._save(data)

    def add_example(self, original: str, corrected: str) -> None:
        data = self._load()
        data.setdefault("examples", []).append(
            {
                "original": original,
                "corrected": corrected,
            }
        )
        self._save(data)

    def get_rules(self) -> dict[str, str]:
        data = self._load()
        return dict(data.get("rules", {}))

    def get_examples(self) -> list[dict[str, str]]:
        data = self._load()
        return list(data.get("examples", []))
