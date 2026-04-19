from __future__ import annotations

import customtkinter as ctk


class ReviewPanel(ctk.CTkFrame):
    def __init__(self, master, review_state, audio_map=None, audio_player=None):
        super().__init__(master)
        self.review_state = review_state
        self.audio_map = audio_map
        self.audio_player = audio_player
        self.rows = []

        self.title = ctk.CTkLabel(self, text="AI Review Dashboard", font=("Arial", 18))
        self.title.pack(pady=10)

        self.container = ctk.CTkScrollableFrame(self, height=400)
        self.container.pack(fill="both", expand=True)

    def load_changes(self) -> None:
        for widget in self.container.winfo_children():
            widget.destroy()

        self.rows = []
        for index, item in enumerate(self.review_state.get_all()):
            change = item["change"]

            frame = ctk.CTkFrame(self.container)
            frame.pack(fill="x", padx=5, pady=5)

            original = " ".join(change["original"])
            corrected = " ".join(change["corrected"])

            ctk.CTkLabel(frame, text=f"Original: {original}", anchor="w").pack(fill="x")
            ctk.CTkLabel(frame, text=f"Corrected: {corrected}", anchor="w").pack(fill="x")

            button_row = ctk.CTkFrame(frame)
            button_row.pack()

            ctk.CTkButton(button_row, text="Accept", command=lambda idx=index: self.accept(idx)).pack(side="left", padx=5)
            ctk.CTkButton(button_row, text="Reject", command=lambda idx=index: self.reject(idx)).pack(side="left", padx=5)
            ctk.CTkButton(button_row, text="Play", command=lambda text=corrected: self.play_audio(text)).pack(side="left", padx=5)

            self.rows.append(frame)

    def accept(self, index: int) -> None:
        self.review_state.accept(index)
        self.load_changes()

    def reject(self, index: int) -> None:
        self.review_state.reject(index)
        self.load_changes()

    def play_audio(self, text: str) -> None:
        if self.audio_map is None or self.audio_player is None:
            return
        timestamp = self.audio_map.find_timestamp(text)
        if timestamp is None:
            return
        self.audio_player.seek(timestamp)
        self.audio_player.play()
