from __future__ import annotations

from typing import Any, Callable

from audio.detector import AudioDeviceEntry, AutoModeName, DeviceDetector


class RoutingManager:
    def __init__(self, detector: DeviceDetector, signal_sampler: Callable[..., dict[str, Any] | None], logger=None):
        self.detector = detector
        self.signal_sampler = signal_sampler
        self.logger = logger

    def has_signal(
        self,
        device: AudioDeviceEntry,
        sample_rate_hz: int,
        duration_seconds: float = 0.35,
    ) -> tuple[bool, dict[str, Any] | None]:
        signal = self.signal_sampler(
            device.index,
            sample_rate_hz,
            device.name,
            duration_seconds=duration_seconds,
            device_info=device.info,
        )
        if signal is None:
            return False, None
        rms_db = float(signal.get("rms_db", -100.0))
        peak_db = float(signal.get("peak_db", -100.0))
        if self.logger is not None:
            self.logger.info(
                "[Routing] Tested device=%s rms_db=%.1f peak_db=%.1f state=%s",
                device.name,
                rms_db,
                peak_db,
                signal.get("state", "unknown"),
            )
        return rms_db > -55.0, signal

    def select_working_device(
        self,
        mode_order: list[AutoModeName],
        sample_rate_hz: int,
    ) -> tuple[AudioDeviceEntry | None, AutoModeName | None, dict[str, Any] | None]:
        self.detector.refresh_devices()
        for mode_name in mode_order:
            entry = self.detector.select_best_input_device(mode_name)
            if entry is None:
                continue
            has_signal, signal = self.has_signal(
                entry,
                sample_rate_hz,
                duration_seconds=0.5 if mode_name == "VAC" else 0.35,
            )
            if has_signal:
                if self.logger is not None:
                    self.logger.info("[Routing] Selected working mode=%s device=%s", mode_name, entry.name)
                return entry, mode_name, signal
        if self.logger is not None:
            self.logger.warning("[Routing] No working input device found for modes=%s", ",".join(mode_order))
        return None, None, None
