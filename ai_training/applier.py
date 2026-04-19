from __future__ import annotations

import re

from ai_training.store import TrainingStore


class TrainingApplier:
    def __init__(self, store: TrainingStore | None = None):
        self.store = store or TrainingStore()
        self.rules = self.store.get_rules()

    def apply(self, text: str) -> str:
        updated = text
        for wrong, correct in self.rules.items():
            pattern = re.compile(rf"(?<!\w){re.escape(wrong)}(?!\w)", flags=re.IGNORECASE)
            updated = pattern.sub(correct, updated)
        return updated
