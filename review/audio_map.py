from __future__ import annotations


class AudioMap:
    def __init__(self, words: list[dict]):
        self.words = words

    def find_timestamp(self, target_text: str):
        target_words = target_text.lower().split()
        if not target_words:
            return None

        for index in range(len(self.words)):
            match = True
            for offset, target_word in enumerate(target_words):
                if index + offset >= len(self.words):
                    match = False
                    break
                candidate = str(self.words[index + offset].get("word", "")).lower()
                if candidate != target_word:
                    match = False
                    break
            if match:
                return self.words[index].get("start")
        return None
