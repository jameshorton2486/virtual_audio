from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Literal

import sounddevice as sd


AutoModeName = Literal["Microphone", "VAC", "Mixed"]
NULL_INPUT_DEVICE_INDEX = -1
NULL_INPUT_DEVICE_NAME = "Silence Generator (No Audio Device)"


def normalize_device_name(name: str) -> str:
    return str(name or "").strip()


def _is_microphone_name(name: str) -> bool:
    lowered = normalize_device_name(name).lower()
    if "microphone" in lowered:
        return True
    return bool(re.search(r"(?<![a-z])mic(?![a-z])", lowered))


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

    def get_default_input_entry(self) -> AudioDeviceEntry | None:
        try:
            device_info = sd.query_devices(kind="input")
        except Exception:
            return None
        if not isinstance(device_info, dict):
            return None
        try:
            index = int(device_info.get("index", NULL_INPUT_DEVICE_INDEX))
        except (TypeError, ValueError):
            return None
        if index < 0:
            return None
        name = normalize_device_name(device_info.get("name", ""))
        if not name or int(device_info.get("max_input_channels", 0)) <= 0:
            return None
        if self.devices:
            for entry in self._input_entries():
                if entry.index == index or entry.name == name:
                    return entry
            return None
        return AudioDeviceEntry(index=index, info=dict(device_info), name=name)

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
            elif _is_microphone_name(entry.name):
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

    def build_null_input_entry(self) -> AudioDeviceEntry:
        return AudioDeviceEntry(
            index=NULL_INPUT_DEVICE_INDEX,
            name=NULL_INPUT_DEVICE_NAME,
            info={
                "name": NULL_INPUT_DEVICE_NAME,
                "max_input_channels": 1,
                "default_samplerate": 16000,
                "is_synthetic": True,
            },
        )

    def _ordered_input_entries_for_mode(self, mode_name: AutoModeName | None = None) -> list[AudioDeviceEntry]:
        entries = self._input_entries()
        if mode_name is None:
            return entries

        categories = self.classify_input_devices()
        priorities: dict[AutoModeName, tuple[str, ...]] = {
            "Microphone": ("mic", "other", "vac", "voicemeeter"),
            "VAC": ("vac", "voicemeeter", "other", "mic"),
            "Mixed": ("voicemeeter", "vac", "other", "mic"),
        }
        ordered: list[AudioDeviceEntry] = []
        seen: set[int] = set()
        for category in priorities.get(mode_name, ()):
            for entry in categories[category]:
                if entry.index in seen:
                    continue
                ordered.append(entry)
                seen.add(entry.index)
        for entry in entries:
            if entry.index in seen:
                continue
            ordered.append(entry)
            seen.add(entry.index)
        return ordered

    def list_candidate_input_devices(
        self,
        preferred_name: str = "",
        mode_name: AutoModeName | None = None,
    ) -> list[AudioDeviceEntry]:
        entries = self._ordered_input_entries_for_mode(mode_name)
        ordered: list[AudioDeviceEntry] = []
        seen: set[int] = set()

        normalized_preferred = normalize_device_name(preferred_name).lower()
        if normalized_preferred:
            for entry in entries:
                if entry.name.lower() == normalized_preferred:
                    ordered.append(entry)
                    seen.add(entry.index)
                    break
            if not ordered:
                for entry in entries:
                    entry_lower = entry.name.lower()
                    if normalized_preferred in entry_lower or entry_lower in normalized_preferred:
                        ordered.append(entry)
                        seen.add(entry.index)
                        break

        default_entry = self.get_default_input_entry()
        if mode_name is None and default_entry is not None and default_entry.index not in seen:
            ordered.append(default_entry)
            seen.add(default_entry.index)

        remaining = [entry for entry in entries if entry.index not in seen]
        ordered.extend(remaining)
        return ordered

    def _find_device_by_keywords(self, keywords: list[str]) -> tuple[int | None, str | None]:
        lowered_keywords = [keyword.lower() for keyword in keywords]
        for entry in self._input_entries():
            lowered_name = entry.name.lower()
            if any(keyword in lowered_name for keyword in lowered_keywords):
                return entry.index, entry.name
        return None, None

    def select_best_input_device(self, mode_name: AutoModeName | None = None) -> AudioDeviceEntry | None:
        candidates = self.list_candidate_input_devices(mode_name=mode_name)
        if candidates:
            entry = candidates[0]
            if self.logger is not None:
                self.logger.info("[DeviceDetector] Selected input: %s | mode=%s", entry.name, mode_name or "Auto")
            return entry
        if self.logger is not None:
            self.logger.warning("[DeviceDetector] No physical input devices detected; using silence fallback")
        return self.build_null_input_entry()

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
