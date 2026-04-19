from __future__ import annotations


class AudioPlayer:
    """Small playback abstraction.

    This is intentionally conservative: the repo does not currently depend on
    VLC/python-vlc, so this class acts as a placeholder interface until an
    actual player backend is installed and wired in.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.position_seconds = 0.0

    def play(self) -> None:
        return None

    def seek(self, seconds: float) -> None:
        self.position_seconds = max(0.0, float(seconds))
