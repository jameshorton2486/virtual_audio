from __future__ import annotations


class ReviewState:
    def __init__(self):
        self.decisions: list[dict[str, object]] = []

    def add(self, change: dict[str, object]) -> None:
        self.decisions.append(
            {
                "change": change,
                "status": "pending",
            }
        )

    def accept(self, index: int) -> None:
        self.decisions[index]["status"] = "accepted"

    def reject(self, index: int) -> None:
        self.decisions[index]["status"] = "rejected"

    def get_all(self) -> list[dict[str, object]]:
        return list(self.decisions)
