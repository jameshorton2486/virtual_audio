from __future__ import annotations

from ai_training.prompt_builder import build_prompt
from ai_training.store import TrainingStore


def is_safe(original: str, corrected: str) -> bool:
    if not corrected:
        return False
    if abs(len(original) - len(corrected)) > len(original) * 0.4:
        return False
    orig_words = original.split()
    new_words = corrected.split()
    if orig_words and abs(len(orig_words) - len(new_words)) > len(orig_words) * 0.3:
        return False
    return True


class AICorrector:
    """Safe placeholder AI corrector.

    This repo does not currently include the OpenAI SDK in requirements or a
    review workflow for approving model output, so this class only prepares the
    prompt and returns the original text unchanged.
    """

    def __init__(self, store: TrainingStore | None = None):
        self.store = store or TrainingStore()

    def correct(self, text: str) -> str:
        _prompt = build_prompt(text, self.store.get_examples())
        return text
