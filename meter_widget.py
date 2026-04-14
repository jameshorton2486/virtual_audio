import tkinter as tk

import customtkinter as ctk


class AudioLevelMeter(ctk.CTkFrame):
    def __init__(self, parent, width=480, height=80, **kwargs):
        """Compact audio level meter for the main dashboard."""
        super().__init__(parent, width=width, height=height, **kwargs)

        self.meter_width = width - 20
        self.meter_height = 22
        self._progress = 0.0
        self._color = "#4CAF50"

        self.grid_columnconfigure(0, weight=1)
        self._build_ui()
        self._draw_meter()

    def _build_ui(self):
        """Build meter UI components."""
        self.title_label = ctk.CTkLabel(
            self,
            text="AUDIO LEVEL",
            font=("Arial", 12, "bold"),
        )
        self.title_label.pack(pady=(4, 2))

        self.meter_frame = ctk.CTkFrame(self, fg_color="#1a1a1a")
        self.meter_frame.pack(pady=4, padx=10, fill="x")

        self.canvas = tk.Canvas(
            self.meter_frame,
            width=self.meter_width,
            height=self.meter_height,
            bg="#1a1a1a",
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(pady=6, padx=10)

        label_frame = ctk.CTkFrame(self.meter_frame, fg_color="transparent")
        label_frame.pack(fill="x", padx=10, pady=(0, 6))

        self.rms_label = ctk.CTkLabel(
            label_frame,
            text="RMS: -∞ dB",
            font=("Courier New", 9),
            width=110,
            anchor="w",
        )
        self.rms_label.pack(side="left", padx=5)

        self.peak_label = ctk.CTkLabel(
            label_frame,
            text="Peak: -∞ dB",
            font=("Courier New", 9),
            width=110,
            anchor="w",
        )
        self.peak_label.pack(side="left", padx=5)

        self.status_label = ctk.CTkLabel(
            label_frame,
            text="Status: Monitoring",
            font=("Arial", 9, "bold"),
            text_color="#4CAF50",
        )
        self.status_label.pack(side="right", padx=5)

    def _draw_meter(self) -> None:
        self.canvas.delete("all")

        radius = 8
        width = self.meter_width
        height = self.meter_height
        filled = max(radius * 2, int(width * max(0.0, min(1.0, self._progress)))) if self._progress > 0 else 0

        self._draw_round_rect(0, 0, width, height, radius, "#2A2A2A")
        if filled > 0:
            self._draw_round_rect(0, 0, filled, height, radius, self._color)

    def _draw_round_rect(self, x1: int, y1: int, x2: int, y2: int, radius: int, color: str) -> None:
        if x2 - x1 <= 0:
            return

        radius = min(radius, max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
        points = [
            x1 + radius,
            y1,
            x2 - radius,
            y1,
            x2,
            y1,
            x2,
            y1 + radius,
            x2,
            y2 - radius,
            x2,
            y2,
            x2 - radius,
            y2,
            x1 + radius,
            y2,
            x1,
            y2,
            x1,
            y2 - radius,
            x1,
            y1 + radius,
            x1,
            y1,
        ]
        self.canvas.create_polygon(points, smooth=True, fill=color, outline=color)

    def set_levels(self, rms_text: str, peak_text: str, status_text: str, color: str, progress: float) -> None:
        self.rms_label.configure(text=rms_text)
        self.peak_label.configure(text=peak_text)
        self.status_label.configure(text=f"Status: {status_text}", text_color=color)
        self._color = color
        self._progress = max(0.0, min(1.0, progress))
        self._draw_meter()
