from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import sounddevice as sd


AutoModeName = Literal["Microphone", "VAC", "Mixed"]


def normalize_device_name(name: str) -> str:
    return str(name or "").strip()


@dataclass(frozen=True)
class AudioDeviceEntry:
    index: int
    info: dict[str, Any]
    name: str


class DeviceDetector:
    def __init__(self, logger=None):
        self.logger = logger
        self.devices: list[dict[str, Any]] = []

    def refresh_devices(self) -> list[dict[str, Any]]:
        self.devices = list(sd.query_devices())
        return self.devices

    def get_devices(self) -> list[dict[str, Any]]:
        if not self.devices:
            self.refresh_devices()
        return self.devices

    def _input_entries(self) -> list[AudioDeviceEntry]:
        entries: list[AudioDeviceEntry] = []
        for index, info in enumerate(self.get_devices()):
            if int(info.get("max_input_channels", 0)) <= 0:
                continue
            name = normalize_device_name(info.get("name", ""))
            if not name:
                continue
            entries.append(AudioDeviceEntry(index=index, info=dict(info), name=name))
        return entries

    def _output_entries(self) -> list[AudioDeviceEntry]:
        entries: list[AudioDeviceEntry] = []
        for index, info in enumerate(self.get_devices()):
            if int(info.get("max_output_channels", 0)) <= 0:
                continue
            name = normalize_device_name(info.get("name", ""))
            if not name:
                continue
            entries.append(AudioDeviceEntry(index=index, info=dict(info), name=name))
        return entries

    def classify_input_devices(self) -> dict[str, list[AudioDeviceEntry]]:
        categories = {
            "mic": [],
            "vac": [],
            "voicemeeter": [],
            "other": [],
        }

        for entry in self._input_entries():
            lowered = entry.name.lower()
            if "voicemeeter" in lowered:
                categories["voicemeeter"].append(entry)
            elif "cable" in lowered or "vb-audio" in lowered:
                categories["vac"].append(entry)
            elif "mic" in lowered or "microphone" in lowered:
                categories["mic"].append(entry)
            else:
                categories["other"].append(entry)

        return categories

    def classify_output_devices(self) -> dict[str, list[AudioDeviceEntry]]:
        categories = {
            "speaker": [],
            "vac": [],
            "other": [],
        }

        for entry in self._output_entries():
            lowered = entry.name.lower()
            if "cable" in lowered or "vb-audio" in lowered:
                categories["vac"].append(entry)
            elif any(token in lowered for token in ("speaker", "headphone", "realtek", "tv", "hdmi", "nvidia")):
                categories["speaker"].append(entry)
            else:
                categories["other"].append(entry)

        return categories

    def list_input_names(self) -> list[str]:
        return [entry.name for entry in self._input_entries()]

    def list_output_names(self) -> list[str]:
        return [entry.name for entry in self._output_entries()]

    def _find_device_by_keywords(self, keywords: list[str]) -> tuple[int | None, str | None]:
        lowered_keywords = [keyword.lower() for keyword in keywords]
        for entry in self._input_entries():
            lowered_name = entry.name.lower()
            if any(keyword in lowered_name for keyword in lowered_keywords):
                return entry.index, entry.name
        return None, None

    def select_best_input_device(self, mode_name: AutoModeName | None = None) -> AudioDeviceEntry | None:
        categories = self.classify_input_devices()
        priorities = {
            "VAC": ("vac", "voicemeeter", "mic", "other"),
            "Mixed": ("voicemeeter", "vac", "mic", "other"),
            "Microphone": ("mic", "voicemeeter", "vac", "other"),
            None: ("vac", "voicemeeter", "mic", "other"),
        }
        for category in priorities.get(mode_name, priorities[None]):
            if categories[category]:
                entry = categories[category][0]
                if self.logger is not None:
                    self.logger.info("[DeviceDetector] Selected input: %s | category=%s | mode=%s", entry.name, category, mode_name or "Auto")
                return entry
        return None

    def select_best_output_device(self, mode_name: AutoModeName | None = None) -> AudioDeviceEntry | None:
        categories = self.classify_output_devices()
        priorities = {
            "VAC": ("vac", "speaker", "other"),
            "Mixed": ("speaker", "other", "vac"),
            "Microphone": ("speaker", "other", "vac"),
            None: ("speaker", "other", "vac"),
        }
        for category in priorities.get(mode_name, priorities[None]):
            if categories[category]:
                entry = categories[category][0]
                if self.logger is not None:
                    self.logger.info("[DeviceDetector] Selected output: %s | category=%s | mode=%s", entry.name, category, mode_name or "Auto")
                return entry
        return None
