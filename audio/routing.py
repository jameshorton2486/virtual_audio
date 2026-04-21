from __future__ import annotations

from typing import Any, Callable, Mapping

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
        preferred_name: str = "",
        preferred_names: Mapping[AutoModeName, str] | None = None,
    ) -> tuple[AudioDeviceEntry | None, AutoModeName | None, dict[str, Any] | None]:
        self.detector.refresh_devices()
        first_candidate: tuple[AudioDeviceEntry, AutoModeName] | None = None
        seen_indexes: set[int] = set()
        for mode_name in mode_order:
            mode_preferred_name = preferred_names.get(mode_name, preferred_name) if preferred_names is not None else preferred_name
            for entry in self.detector.list_candidate_input_devices(
                preferred_name=mode_preferred_name,
                mode_name=mode_name,
            ):
                if entry.index in seen_indexes:
                    continue
                seen_indexes.add(entry.index)
                if first_candidate is None:
                    first_candidate = (entry, mode_name)
                has_signal, signal = self.has_signal(
                    entry,
                    sample_rate_hz,
                    duration_seconds=0.5 if mode_name == "VAC" else 0.35,
                )
                if has_signal:
                    if self.logger is not None:
                        self.logger.info("[Routing] Selected working mode=%s device=%s", mode_name, entry.name)
                    return entry, mode_name, signal
        if first_candidate is not None:
            entry, mode_name = first_candidate
            if self.logger is not None:
                self.logger.warning("[Routing] No input device had confirmed signal; falling back to device=%s mode=%s", entry.name, mode_name)
            return entry, mode_name, {"state": "silent", "detail": "No confirmed signal during routing probe."}
        if self.logger is not None:
            self.logger.warning("[Routing] No working input device found for modes=%s; using silence fallback", ",".join(mode_order))
        null_entry = self.detector.build_null_input_entry()
        return null_entry, mode_order[0] if mode_order else None, {"state": "silence", "detail": "No input devices detected."}
