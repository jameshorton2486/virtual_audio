from __future__ import annotations

import customtkinter as ctk

from meter_widget import AudioLevelMeter


class AudioPanel(ctk.CTkFrame):
    """Reusable audio controls panel for future UI modularization.

    The main app still owns the richer Monitor/Routing tabs; this component is
    a small extracted building block that can be adopted incrementally.
    """

    def __init__(self, master, *, on_run_callback=None):
        super().__init__(master)
        self.on_run_callback = on_run_callback

        self.run_button = ctk.CTkButton(
            self,
            text="Run (Auto Detect Audio)",
            command=self._handle_run,
        )
        self.run_button.pack(fill="x", padx=12, pady=(12, 8))

        self.status_label = ctk.CTkLabel(self, text="Idle", anchor="w")
        self.status_label.pack(fill="x", padx=12, pady=(0, 8))

        self.meter = AudioLevelMeter(self, width=420, height=64)
        self.meter.pack(fill="x", padx=12, pady=(0, 12))

    def _handle_run(self) -> None:
        self.status_label.configure(text="Detecting...")
        if callable(self.on_run_callback):
            self.on_run_callback()

    def set_status(self, text: str) -> None:
        self.status_label.configure(text=text)

    def set_meter_levels(self, rms_text: str, peak_text: str, status_text: str, color: str, progress: float) -> None:
        self.meter.set_levels(rms_text, peak_text, status_text, color, progress)
