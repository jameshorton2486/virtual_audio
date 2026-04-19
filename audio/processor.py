from __future__ import annotations

import numpy as np


class AudioProcessor:
    def __init__(self, *, silence_threshold: float = 0.003, target_peak: float = 0.92):
        self.silence_threshold = silence_threshold
        self.target_peak = target_peak

    def to_mono(self, chunk) -> np.ndarray:
        samples = np.asarray(chunk, dtype=np.float32)
        if samples.ndim == 2:
            samples = np.mean(samples, axis=1)
        samples = np.squeeze(samples)
        if samples.ndim == 0:
            samples = np.asarray([float(samples)], dtype=np.float32)
        samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(samples, -1.0, 1.0)

    def attenuate_clipping(self, chunk: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(chunk))) if chunk.size else 0.0
        if peak <= 0.0 or peak <= self.target_peak:
            return chunk
        return chunk * (self.target_peak / peak)

    def remove_silence(self, chunk: np.ndarray) -> np.ndarray | None:
        if chunk.size == 0:
            return None
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        peak = float(np.max(np.abs(chunk)))
        if rms < self.silence_threshold and peak < self.silence_threshold * 2.0:
            return None
        return chunk

    def process(self, chunk) -> np.ndarray | None:
        samples = self.to_mono(chunk)
        samples = self.attenuate_clipping(samples)
        return self.remove_silence(samples)
