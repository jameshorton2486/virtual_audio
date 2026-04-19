from __future__ import annotations

from ai_training.store import TrainingStore


def is_valid_rule(wrong: str, correct: str) -> bool:
    wrong = wrong.strip()
    correct = correct.strip()
    if len(wrong) < 3:
        return False
    if wrong.lower() == correct.lower():
        return False
    return True


class Trainer:
    def __init__(self, store: TrainingStore | None = None):
        self.store = store or TrainingStore()

    def learn_from_edit(self, original_text: str, corrected_text: str) -> None:
        if not is_valid_rule(original_text, corrected_text):
            return
        self.store.add_rule(original_text, corrected_text)
        self.store.add_example(original_text, corrected_text)
