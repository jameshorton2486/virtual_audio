from __future__ import annotations

from difflib import ndiff


class DiffEngine:
    def compare(self, original: str, corrected: str) -> list[dict[str, list[str]]]:
        diff = list(ndiff(original.split(), corrected.split()))
        changes: list[dict[str, list[str]]] = []
        current = {"original": [], "corrected": []}

        for item in diff:
            if item.startswith("- "):
                current["original"].append(item[2:])
            elif item.startswith("+ "):
                current["corrected"].append(item[2:])
            else:
                if current["original"] or current["corrected"]:
                    changes.append(current)
                    current = {"original": [], "corrected": []}

        if current["original"] or current["corrected"]:
            changes.append(current)

        return changes
