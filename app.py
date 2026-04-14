import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from tkinter import messagebox
from typing import Any


APP_DIR = Path(__file__).resolve().parent
VENV_PYTHON = APP_DIR / ".venv" / "Scripts" / "python.exe"


def ensure_local_venv() -> None:
    if sys.platform != "win32":
        return

    if not VENV_PYTHON.exists():
        return

    current_python = Path(sys.executable).resolve()
    target_python = VENV_PYTHON.resolve()
    if current_python == target_python:
        return

    # Restart inside the project venv so local dependencies are always used.
    os.execv(str(target_python), [str(target_python), str(APP_DIR / "app.py"), *sys.argv[1:]])


ensure_local_venv()

import customtkinter as ctk
import numpy as np
import sounddevice as sd

from meter_widget import AudioLevelMeter


CONFIG_PATH = APP_DIR / "config.json"
NIRCMD_PATH = APP_DIR / "nircmd.exe"

DEFAULT_CONFIG = {
    "mic_device": "Microphone (Realtek Audio)",
    "vac_device": "CABLE Output (VB-Audio Virtual Cable)",
    "speaker_device": "Speakers (Realtek Audio)",
    "vac_playback_device": "CABLE Input (VB-Audio Virtual Cable)",
    "voicemeeter_device": "VoiceMeeter Output (VB-Audio VoiceMeeter VAIO)",
    "wer_mode_enabled": True,
    "quality_check_interval_seconds": 2.0,
    "sample_rate_hz": 24000,
    "last_mode": "Microphone",
}

QUALITY_COLORS = {
    "excellent": "#4CAF50",
    "good": "#8BC34A",
    "low": "#F9A825",
    "too_quiet": "#F57C00",
    "clipping": "#D32F2F",
    "error": "#9E9E9E",
}

QUALITY_PROGRESS = {
    "excellent": 0.85,
    "good": 0.70,
    "low": 0.35,
    "too_quiet": 0.12,
    "clipping": 1.0,
    "error": 0.0,
}

MODE_TEXT = {
    "Microphone": "Microphone only",
    "VAC": "Virtual Audio Cable only",
    "Mixed": "Voicemeeter / mixed audio",
}

MODE_STATUS = {
    "Microphone": "Sets the Windows default recording device to your microphone and restores playback to your speakers.",
    "VAC": "Sets recording to CABLE Output and playback to CABLE Input so system audio is routed through the virtual cable.",
    "Mixed": "Sets the Windows default recording device to the Voicemeeter output and restores playback to your speakers.",
}


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def sanitize_config(config: dict[str, Any]) -> dict[str, Any]:
    sanitized = DEFAULT_CONFIG.copy()
    sanitized.update({k: v for k, v in config.items() if k in DEFAULT_CONFIG})

    for key in ("mic_device", "vac_device", "speaker_device", "vac_playback_device", "voicemeeter_device"):
        value = sanitized.get(key, DEFAULT_CONFIG[key])
        sanitized[key] = value.strip() if isinstance(value, str) and value.strip() else DEFAULT_CONFIG[key]

    sanitized["wer_mode_enabled"] = _coerce_bool(
        sanitized.get("wer_mode_enabled"),
        bool(DEFAULT_CONFIG["wer_mode_enabled"]),
    )

    try:
        interval = float(sanitized.get("quality_check_interval_seconds", DEFAULT_CONFIG["quality_check_interval_seconds"]))
        if interval <= 0:
            raise ValueError
        sanitized["quality_check_interval_seconds"] = interval
    except (TypeError, ValueError):
        sanitized["quality_check_interval_seconds"] = float(DEFAULT_CONFIG["quality_check_interval_seconds"])

    try:
        sample_rate = int(sanitized.get("sample_rate_hz", DEFAULT_CONFIG["sample_rate_hz"]))
        if sample_rate <= 0:
            raise ValueError
        sanitized["sample_rate_hz"] = sample_rate
    except (TypeError, ValueError):
        sanitized["sample_rate_hz"] = int(DEFAULT_CONFIG["sample_rate_hz"])

    last_mode = sanitized.get("last_mode", DEFAULT_CONFIG["last_mode"])
    sanitized["last_mode"] = last_mode if last_mode in MODE_TEXT else DEFAULT_CONFIG["last_mode"]

    return sanitized


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG.copy())
        return DEFAULT_CONFIG.copy()

    try:
        stored = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        stored = {}

    merged = DEFAULT_CONFIG.copy()
    merged.update({k: v for k, v in stored.items() if k in DEFAULT_CONFIG})
    return sanitize_config(merged)


def save_config(config: dict[str, Any]) -> None:
    CONFIG_PATH.write_text(json.dumps(sanitize_config(config), indent=2), encoding="utf-8")


def _list_devices(channel_key: str) -> list[str]:
    devices: list[str] = []
    seen: set[str] = set()

    try:
        query = sd.query_devices()
    except Exception:
        return devices

    for device in query:
        max_channels = int(device.get(channel_key, 0))
        name = str(device.get("name", "")).strip()
        if max_channels <= 0 or not name or name in seen:
            continue
        normalized = re.sub(r",\s*(MME|Windows .+|DirectSound|WDM-KS)$", "", name, flags=re.IGNORECASE).strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        devices.append(normalized)

    return devices


def list_input_devices() -> list[str]:
    return _list_devices("max_input_channels")


def list_output_devices() -> list[str]:
    return _list_devices("max_output_channels")


def infer_device(preferred_name: str, devices: list[str], keywords: list[str]) -> str:
    if preferred_name in devices:
        return preferred_name

    lowered = [keyword.lower() for keyword in keywords]
    matches: list[str] = []
    partial_matches: list[str] = []

    for device in devices:
        device_lower = device.lower()
        if all(keyword in device_lower for keyword in lowered):
            matches.append(device)
        elif any(keyword in device_lower for keyword in lowered):
            partial_matches.append(device)

    ranked = matches or partial_matches
    if ranked:
        ranked.sort(key=_device_rank_key)
        return ranked[0]

    return preferred_name


def infer_device_by_patterns(
    preferred_name: str,
    devices: list[str],
    required_patterns: list[tuple[str, ...]],
    optional_patterns: list[tuple[str, ...]] | None = None,
    excluded_terms: tuple[str, ...] = (),
) -> str:
    if preferred_name in devices:
        return preferred_name

    optional_patterns = optional_patterns or []
    excluded = tuple(term.lower() for term in excluded_terms)

    def is_excluded(device_lower: str) -> bool:
        return any(term in device_lower for term in excluded)

    def matching_score(device: str) -> tuple[int, tuple[int, int, str]] | None:
        device_lower = device.lower()
        if is_excluded(device_lower):
            return None

        for index, pattern in enumerate(required_patterns):
            if all(term in device_lower for term in pattern):
                return (200 - index, _device_rank_key(device))

        for index, pattern in enumerate(optional_patterns):
            if all(term in device_lower for term in pattern):
                return (100 - index, _device_rank_key(device))

        return None

    ranked: list[tuple[int, tuple[int, int, str], str]] = []
    for device in devices:
        score = matching_score(device)
        if score is None:
            continue
        ranked.append((score[0], score[1], device))

    if ranked:
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return ranked[0][2]

    return preferred_name


def infer_vac_recording_device(preferred_name: str, devices: list[str]) -> str:
    return infer_device_by_patterns(
        preferred_name,
        devices,
        required_patterns=[("cable", "output"), ("vb-audio", "output")],
        optional_patterns=[("virtual", "cable"), ("cable",)],
    )


def infer_vac_playback_device(preferred_name: str, devices: list[str]) -> str:
    return infer_device_by_patterns(
        preferred_name,
        devices,
        required_patterns=[("cable", "input"), ("vb-audio", "input")],
        optional_patterns=[("virtual", "cable"), ("cable",)],
    )


def infer_speaker_output_device(preferred_name: str, devices: list[str]) -> str:
    return infer_device_by_patterns(
        preferred_name,
        devices,
        required_patterns=[("speaker",), ("headphone",), ("realtek",)],
        optional_patterns=[("output",)],
        excluded_terms=("cable", "vb-audio", "voicemeeter"),
    )


def _device_rank_key(device: str) -> tuple[int, int, str]:
    lowered = device.lower()
    penalty = 0

    if "mapper" in lowered or "primary sound capture driver" in lowered:
        penalty += 50
    if "pc speaker" in lowered or "speaker" in lowered:
        penalty += 20
    if "stereo mix" in lowered or "line in" in lowered or "internal aux" in lowered:
        penalty += 15
    if lowered.endswith("microph"):
        penalty += 10
    if "microphone" in lowered:
        penalty -= 8
    if "yeti" in lowered or "vb-audio" in lowered or "voicemeeter" in lowered or "cable" in lowered:
        penalty -= 10

    return (penalty, -len(device), device)


class AudioDeviceManager:
    @staticmethod
    def _set_default_device(device_name: str) -> None:
        if not device_name.strip():
            raise ValueError("Device name is empty.")

        if not NIRCMD_PATH.exists():
            raise FileNotFoundError(f"Missing {NIRCMD_PATH.name} in {APP_DIR}.")

        for role in ("0", "1", "2"):
            subprocess.run(
                [str(NIRCMD_PATH), "setdefaultsounddevice", device_name, role],
                check=True,
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )

    @classmethod
    def set_default_recording_device(cls, device_name: str) -> tuple[bool, str]:
        try:
            cls._set_default_device(device_name)
            return True, f"Switched recording device to {device_name} for all Windows audio roles."
        except subprocess.CalledProcessError as exc:
            error_text = exc.stderr.decode("utf-8", errors="ignore").strip() if exc.stderr else ""
            return False, error_text or f"Failed to switch to {device_name}."
        except Exception as exc:
            return False, str(exc)

    @classmethod
    def set_default_playback_device(cls, device_name: str) -> tuple[bool, str]:
        try:
            cls._set_default_device(device_name)
            return True, f"Switched playback device to {device_name} for all Windows audio roles."
        except subprocess.CalledProcessError as exc:
            error_text = exc.stderr.decode("utf-8", errors="ignore").strip() if exc.stderr else ""
            return False, error_text or f"Failed to switch to {device_name}."
        except Exception as exc:
            return False, str(exc)

    @staticmethod
    def toggle_mute() -> tuple[bool, str]:
        if not NIRCMD_PATH.exists():
            return False, f"Missing {NIRCMD_PATH.name} in {APP_DIR}."

        try:
            subprocess.run(
                [str(NIRCMD_PATH), "mutesysvolume", "2", "default_record"],
                check=True,
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            return True, "Toggled mute on the default recording device."
        except subprocess.CalledProcessError as exc:
            error_text = exc.stderr.decode("utf-8", errors="ignore").strip() if exc.stderr else ""
            return False, error_text or "Failed to toggle mute."
        except Exception as exc:
            return False, str(exc)


class AudioQualityMonitor:
    def __init__(self, sample_rate_hz: int, interval_seconds: float, callback):
        self.sample_rate_hz = sample_rate_hz
        self.interval_seconds = interval_seconds
        self.callback = callback
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=max(0.5, self.interval_seconds + 0.5))
        self._thread = None

    def _loop(self) -> None:
        while self._running:
            result = self._sample_quality()
            self.callback(result)
            if self._stop_event.wait(self.interval_seconds):
                break

    def _sample_quality(self) -> dict[str, Any]:
        duration_seconds = 0.4
        frames = max(1, int(self.sample_rate_hz * duration_seconds))

        try:
            recording = sd.rec(frames, samplerate=self.sample_rate_hz, channels=1, dtype="float32")
            sd.wait()
            samples = np.asarray(np.squeeze(recording), dtype=np.float64)
            if samples.size == 0:
                raise ValueError("Audio monitor returned an empty sample buffer.")

            if not np.all(np.isfinite(samples)):
                samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)

            samples = np.clip(samples, -1.0, 1.0)
            rms = float(np.sqrt(np.mean(samples * samples)))
            peak = float(np.max(np.abs(samples)))
        except Exception as exc:
            return {
                "quality": "error",
                "level_text": "Unavailable",
                "status_text": "Audio monitor unavailable",
                "detail_text": str(exc),
            }

        if rms < 0.005:
            quality = "too_quiet"
            status = "No usable input"
            detail = "Input is effectively silent. Check the selected recording device and gain."
        elif rms < 0.02:
            quality = "low"
            status = "Low signal"
            detail = "Speech may be too quiet for reliable transcription."
        elif peak > 0.90:
            quality = "clipping"
            status = "Clipping risk"
            detail = "Input is peaking too high. Lower mic gain or mixer output."
        elif peak > 0.75:
            quality = "good"
            status = "Strong signal"
            detail = "Signal is usable but getting close to clipping."
        else:
            quality = "excellent"
            status = "Optimal"
            detail = "Signal level looks healthy for speech capture."

        return {
            "quality": quality,
            "level_text": f"RMS {rms:.3f} | Peak {peak:.3f}",
            "status_text": status,
            "detail_text": detail,
        }


class App:
    def __init__(self) -> None:
        self.config = load_config()
        self.detected_input_devices = list_input_devices()
        self.detected_output_devices = list_output_devices()
        self._hydrate_config_from_detected_devices()
        self.device_manager = AudioDeviceManager()
        self.monitor = AudioQualityMonitor(
            sample_rate_hz=int(self.config["sample_rate_hz"]),
            interval_seconds=float(self.config["quality_check_interval_seconds"]),
            callback=self._queue_quality_update,
        )
        self.current_mode = self.config.get("last_mode", DEFAULT_CONFIG["last_mode"])
        self.is_muted = False
        self.settings_window: ctk.CTkToplevel | None = None
        self.setup_notes_label = None
        self.setup_notes_visible = False
        self._closing = False
        self._vac_test_running = False

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Virtual Audio Control")
        self.root.geometry("520x680")
        self.root.minsize(480, 600)
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.status_var = ctk.StringVar(value="Ready - Monitoring active" if self.config["wer_mode_enabled"] else "Ready")
        self.mode_var = ctk.StringVar(value=self.current_mode)
        self.monitor_status_var = ctk.StringVar(value="Excellent")
        self.monitor_level_var = ctk.StringVar(value="RMS: -∞ dB | Peak: -∞ dB")
        self.monitor_detail_var = ctk.StringVar(value="No issues detected")
        self.monitor_summary_var = ctk.StringVar(value="Monitoring active")
        self.monitor_recommendation_var = ctk.StringVar(value="No issues detected\nAll systems optimal")
        self.footer_var = ctk.StringVar(value="v3.0 Pro | Real-Time WER Optimization")

        self.mic_var = ctk.StringVar(value=self.config["mic_device"])
        self.vac_var = ctk.StringVar(value=self.config["vac_device"])
        self.speaker_var = ctk.StringVar(value=self.config["speaker_device"])
        self.vac_playback_var = ctk.StringVar(value=self.config["vac_playback_device"])
        self.mix_var = ctk.StringVar(value=self.config["voicemeeter_device"])
        self.wer_enabled_var = ctk.BooleanVar(value=bool(self.config["wer_mode_enabled"]))

        self._build_ui()
        self._refresh_mode_hint()
        self._refresh_detection_summary()

        if self.wer_enabled_var.get():
            self.monitor.start()

    def _build_ui(self) -> None:
        self.root.grid_columnconfigure(0, weight=1)

        header_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        header_frame.pack(pady=8, padx=20, fill="x")

        ctk.CTkLabel(
            header_frame,
            text="Audio Control Panel Pro",
            font=("Arial", 20, "bold"),
        ).pack()

        ctk.CTkLabel(
            header_frame,
            text="Real-Time WER Optimization for Deepgram",
            font=("Arial", 10),
            text_color="#888888",
        ).pack(pady=(2, 0))

        self.meter = AudioLevelMeter(self.root, width=480, height=80)
        self.meter.pack(pady=6, padx=20, fill="x")

        self.wer_status_frame = ctk.CTkFrame(self.root, fg_color="#1a1a1a")
        self.wer_status_frame.pack(pady=6, padx=20, fill="x")

        status_header = ctk.CTkFrame(self.wer_status_frame, fg_color="transparent")
        status_header.pack(pady=6, padx=10, fill="x")

        ctk.CTkLabel(
            status_header,
            text="WER OPTIMIZATION",
            font=("Arial", 12, "bold"),
        ).pack(side="left")

        self.monitoring_toggle = ctk.CTkSwitch(
            status_header,
            text="Monitor",
            variable=self.wer_enabled_var,
            command=self.toggle_wer_monitoring,
            onvalue=True,
            offvalue=False,
        )
        self.monitoring_toggle.pack(side="right")
        if self.wer_enabled_var.get():
            self.monitoring_toggle.select()

        status_content = ctk.CTkFrame(self.wer_status_frame, fg_color="#252525")
        status_content.pack(pady=6, padx=10, fill="x")

        status_row = ctk.CTkFrame(status_content, fg_color="transparent")
        status_row.pack(pady=4, padx=10, fill="x")

        ctk.CTkLabel(status_row, text="Quality:", font=("Arial", 10), width=60, anchor="w").pack(side="left")
        self.monitor_status_label = ctk.CTkLabel(
            status_row,
            text="Excellent",
            font=("Arial", 10, "bold"),
            text_color="#66BB6A",
        )
        self.monitor_status_label.pack(side="left", padx=(0, 10))

        ctk.CTkLabel(status_row, text="|", text_color="#444444").pack(side="left", padx=5)
        ctk.CTkLabel(status_row, text="Stability:", font=("Arial", 10), width=60, anchor="w").pack(side="left")
        self.monitor_stability_label = ctk.CTkLabel(
            status_row,
            text="Stable",
            font=("Arial", 10, "bold"),
            text_color="#66BB6A",
        )
        self.monitor_stability_label.pack(side="left", padx=(0, 10))

        ctk.CTkLabel(status_row, text="|", text_color="#444444").pack(side="left", padx=5)
        ctk.CTkLabel(status_row, text="Est. WER:", font=("Arial", 10), width=60, anchor="w").pack(side="left")
        self.monitor_wer_label = ctk.CTkLabel(
            status_row,
            text="3-7%",
            font=("Arial", 10, "bold"),
            text_color="#66BB6A",
        )
        self.monitor_wer_label.pack(side="left")

        self.warnings_box = ctk.CTkTextbox(
            status_content,
            height=60,
            font=("Arial", 9),
            fg_color="#1a1a1a",
            wrap="word",
        )
        self.warnings_box.pack(pady=6, padx=10, fill="x")
        self._set_warnings_text(self.monitor_recommendation_var.get())

        mode_frame = ctk.CTkFrame(self.root)
        mode_frame.pack(pady=6, padx=20, fill="x")

        ctk.CTkLabel(
            mode_frame,
            text="Active Mode:",
            font=("Arial", 11),
        ).pack(pady=(5, 2))

        self.mode_display = ctk.CTkLabel(
            mode_frame,
            text=f"{self.current_mode}",
            font=("Arial", 15, "bold"),
            text_color="#4CAF50",
        )
        self.mode_display.pack(pady=(0, 2))

        self.mode_hint_label = ctk.CTkLabel(
            mode_frame,
            text="",
            text_color="#8AB4F8",
            font=("Arial", 9),
            wraplength=460,
            justify="center",
        )
        self.mode_hint_label.pack(pady=(0, 5), padx=12)

        self.mode_status_label = ctk.CTkLabel(
            mode_frame,
            text="",
            text_color="#C6C6C6",
            font=("Arial", 9),
            wraplength=460,
            justify="center",
        )
        self.mode_status_label.pack(pady=(0, 6), padx=12)

        button_frame = ctk.CTkFrame(self.root)
        button_frame.pack(pady=6, padx=20, fill="x")

        self.btn_mic = ctk.CTkButton(
            button_frame,
            text="Microphone\nWER: ~10-15%",
            command=lambda: self.switch_mode("Microphone", self.mic_var.get()),
            height=42,
            font=("Arial", 11, "bold"),
        )
        self.btn_mic.pack(pady=4, padx=15, fill="x")

        self.btn_vac = ctk.CTkButton(
            button_frame,
            text="Virtual Audio Cable\nWER: ~3-7% BEST",
            command=lambda: self.switch_mode("VAC", self.vac_var.get()),
            height=42,
            font=("Arial", 11, "bold"),
            fg_color="#2E7D32",
            hover_color="#1B5E20",
        )
        self.btn_vac.pack(pady=4, padx=15, fill="x")

        self.btn_vac_test = ctk.CTkButton(
            button_frame,
            text="Test VAC Routing",
            command=self.test_vac_routing,
            height=34,
            font=("Arial", 10, "bold"),
            fg_color="#1565C0",
            hover_color="#0D47A1",
        )
        self.btn_vac_test.pack(pady=(0, 6), padx=15, fill="x")

        self.btn_mix = ctk.CTkButton(
            button_frame,
            text="Mixed Mode\nWER: ~5-10%",
            command=lambda: self.switch_mode("Mixed", self.mix_var.get()),
            height=42,
            font=("Arial", 11, "bold"),
        )
        self.btn_mix.pack(pady=4, padx=15, fill="x")

        self.mute_button = ctk.CTkButton(
            button_frame,
            text="Mute Toggle",
            command=self.toggle_mute,
            height=35,
            font=("Arial", 10),
            fg_color="#d32f2f",
            hover_color="#b71c1c",
        )
        self.mute_button.pack(pady=6, padx=15, fill="x")

        util_frame = ctk.CTkFrame(self.root)
        util_frame.pack(pady=6, padx=20, fill="x")

        utils_row = ctk.CTkFrame(util_frame, fg_color="transparent")
        utils_row.pack(fill="x", padx=5, pady=4)

        ctk.CTkButton(
            utils_row,
            text="Settings",
            command=self.open_config,
            height=28,
            font=("Arial", 9),
            width=150,
        ).pack(side="left", padx=3, expand=True, fill="x")

        ctk.CTkButton(
            utils_row,
            text="Help",
            command=self.show_help,
            height=28,
            font=("Arial", 9),
            width=150,
        ).pack(side="left", padx=3, expand=True, fill="x")

        self.status_label = ctk.CTkLabel(
            self.root,
            textvariable=self.status_var,
            font=("Arial", 9),
            text_color="#4CAF50",
        )
        self.status_label.pack(pady=6)

        ctk.CTkLabel(
            self.root,
            textvariable=self.footer_var,
            font=("Arial", 8),
            text_color="#555555",
        ).pack(pady=(0, 6))

    def _add_device_selector(self, parent, label: str, variable: ctk.StringVar, devices: list[str]):
        ctk.CTkLabel(parent, text=label, font=ctk.CTkFont(size=10)).pack(anchor="w", padx=10)
        menu = ctk.CTkComboBox(
            parent,
            variable=variable,
            values=self._menu_values_for(variable.get(), devices),
            state="normal",
            height=30,
            font=ctk.CTkFont(size=10),
            dropdown_font=ctk.CTkFont(size=10),
        )
        menu.pack(fill="x", padx=10, pady=(2, 6))
        return menu

    def _set_warnings_text(self, text: str) -> None:
        self.warnings_box.configure(state="normal")
        self.warnings_box.delete("1.0", "end")
        self.warnings_box.insert("1.0", text)
        self.warnings_box.configure(state="disabled")

    def _menu_values_for(self, current_value: str, devices: list[str]) -> list[str]:
        values = list(devices)
        if current_value and current_value not in values:
            values.insert(0, current_value)
        if not values:
            values = [current_value or "No devices detected"]
        return values

    def _hydrate_config_from_detected_devices(self) -> None:
        devices = self.detected_input_devices
        current_mic = self.config.get("mic_device", "")
        current_vac = self.config.get("vac_device", "")
        current_speaker = self.config.get("speaker_device", "")
        current_vac_playback = self.config.get("vac_playback_device", "")
        current_mix = self.config.get("voicemeeter_device", "")
        output_devices = self.detected_output_devices

        if current_mic not in devices or "microphone" not in current_mic.lower():
            self.config["mic_device"] = infer_device(current_mic, devices, ["microphone"])

        if current_vac not in devices or not all(token in current_vac.lower() for token in ("cable", "output")):
            self.config["vac_device"] = infer_vac_recording_device(DEFAULT_CONFIG["vac_device"], devices)

        if current_speaker not in output_devices or not any(token in current_speaker.lower() for token in ("speaker", "headphone", "realtek")):
            self.config["speaker_device"] = infer_speaker_output_device(current_speaker, output_devices)

        if current_vac_playback not in output_devices or not all(token in current_vac_playback.lower() for token in ("cable", "input")):
            self.config["vac_playback_device"] = infer_vac_playback_device(DEFAULT_CONFIG["vac_playback_device"], output_devices)

        if current_mix not in devices or "voicemeeter" not in current_mix.lower():
            self.config["voicemeeter_device"] = infer_device(DEFAULT_CONFIG["voicemeeter_device"], devices, ["voicemeeter"])

        save_config(self.config)

    def refresh_detected_devices(self) -> None:
        self.detected_input_devices = list_input_devices()
        self.detected_output_devices = list_output_devices()
        self._hydrate_config_from_detected_devices()
        self.mic_var.set(self.config["mic_device"])
        self.vac_var.set(self.config["vac_device"])
        self.speaker_var.set(self.config["speaker_device"])
        self.vac_playback_var.set(self.config["vac_playback_device"])
        self.mix_var.set(self.config["voicemeeter_device"])
        self.mic_menu.configure(values=self._menu_values_for(self.mic_var.get(), self.detected_input_devices))
        self.vac_menu.configure(values=self._menu_values_for(self.vac_var.get(), self.detected_input_devices))
        self.speaker_menu.configure(values=self._menu_values_for(self.speaker_var.get(), self.detected_output_devices))
        self.vac_playback_menu.configure(values=self._menu_values_for(self.vac_playback_var.get(), self.detected_output_devices))
        self.mix_menu.configure(values=self._menu_values_for(self.mix_var.get(), self.detected_input_devices))
        self._refresh_detection_summary()
        self.status_var.set("Refreshed detected Windows recording and playback devices.")

    def _refresh_detection_summary(self) -> None:
        if self.detected_input_devices:
            joined = ", ".join(self.detected_input_devices[:8])
            if len(self.detected_input_devices) > 8:
                joined += ", ..."
            text = f"Input devices: {joined}"
        else:
            text = "Input devices: none. Check Windows audio drivers or device connection."
        if self.detected_output_devices:
            joined_output = ", ".join(self.detected_output_devices[:6])
            if len(self.detected_output_devices) > 6:
                joined_output += ", ..."
            text = f"{text}\nOutput devices: {joined_output}"
        else:
            text = f"{text}\nOutput devices: none detected."
        if hasattr(self, "detected_devices_label") and self.detected_devices_label.winfo_exists():
            self.detected_devices_label.configure(text=text)

    def save_form_config(self) -> None:
        self.config["mic_device"] = self.mic_var.get().strip()
        self.config["vac_device"] = self.vac_var.get().strip()
        self.config["speaker_device"] = self.speaker_var.get().strip()
        self.config["vac_playback_device"] = self.vac_playback_var.get().strip()
        self.config["voicemeeter_device"] = self.mix_var.get().strip()
        self.config["wer_mode_enabled"] = bool(self.wer_enabled_var.get())
        save_config(self.config)
        self.status_var.set(f"Saved configuration to {CONFIG_PATH.name}.")

    def save_settings(self) -> None:
        self.save_form_config()
        self.refresh_detected_devices()
        self._close_settings_window()

    def _close_settings_window(self) -> None:
        if not self.settings_window:
            return
        if self.settings_window.winfo_exists():
            self.settings_window.grab_release()
            self.settings_window.destroy()
        self.settings_window = None

    def open_config(self) -> None:
        if self.settings_window and self.settings_window.winfo_exists():
            self.settings_window.focus()
            return

        self.settings_window = ctk.CTkToplevel(self.root)
        self.settings_window.title("Settings")
        self.settings_window.geometry("520x470")
        self.settings_window.resizable(True, True)
        self.settings_window.transient(self.root)
        self.settings_window.grab_set()
        self.settings_window.protocol("WM_DELETE_WINDOW", self._close_settings_window)

        body = ctk.CTkFrame(self.settings_window)
        body.pack(fill="both", expand=True, padx=12, pady=12)

        ctk.CTkLabel(body, text="Audio Device Settings", font=("Arial", 16, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
        self.detected_devices_label = ctk.CTkLabel(
            body,
            text="Detected input devices: checking...",
            font=("Arial", 10),
            wraplength=460,
            justify="left",
            text_color="#C6C6C6",
        )
        self.detected_devices_label.pack(anchor="w", padx=10, pady=(0, 6))
        self._refresh_detection_summary()

        self.mic_menu = self._add_device_selector(body, "Microphone device", self.mic_var, self.detected_input_devices)
        self.vac_menu = self._add_device_selector(body, "VAC recording device", self.vac_var, self.detected_input_devices)
        self.speaker_menu = self._add_device_selector(body, "Speaker playback device", self.speaker_var, self.detected_output_devices)
        self.vac_playback_menu = self._add_device_selector(body, "VAC playback target", self.vac_playback_var, self.detected_output_devices)
        self.mix_menu = self._add_device_selector(body, "Voicemeeter device", self.mix_var, self.detected_input_devices)

        actions = ctk.CTkFrame(body, fg_color="transparent")
        actions.pack(fill="x", padx=10, pady=(8, 6))
        ctk.CTkButton(actions, text="Refresh Devices", command=self.refresh_detected_devices).pack(side="left", expand=True, fill="x")
        ctk.CTkButton(actions, text="Open config.json", command=self.open_config_file).pack(side="left", expand=True, fill="x", padx=8)
        ctk.CTkButton(actions, text="Save", command=self.save_settings).pack(side="left", expand=True, fill="x")

    def show_help(self) -> None:
        messagebox.showinfo(
            "Help",
            "1. Set Zoom microphone to 'Same as System'.\n"
            "2. Mode buttons switch the Windows default recording device for all audio roles.\n"
            "3. VAC mode also switches Windows playback to CABLE Input.\n"
            "4. Microphone and Mixed modes restore playback to the configured speaker device.\n"
            "5. Microphone mode is best for live speech.\n"
            "6. VAC is best for playback-only transcription.\n"
            "7. CABLE Output is the recording side and CABLE Input is the playback side.\n"
            "8. Use Test VAC Routing to send a short tone through the cable.\n"
            "9. Mixed mode needs Voicemeeter routing.\n"
            "10. Use Settings to refresh devices or update device names.",
        )

    def open_config_file(self) -> None:
        if sys.platform == "win32":
            os.startfile(CONFIG_PATH)  # type: ignore[attr-defined]
            return
        self.status_var.set(f"Open {CONFIG_PATH} manually on this platform.")

    def open_readme(self) -> None:
        readme_path = APP_DIR / "README.md"
        if sys.platform == "win32":
            os.startfile(readme_path)  # type: ignore[attr-defined]
            return
        self.status_var.set(f"Open {readme_path} manually on this platform.")

    def open_app_folder(self) -> None:
        if sys.platform == "win32":
            os.startfile(APP_DIR)  # type: ignore[attr-defined]
            return
        self.status_var.set(f"Open {APP_DIR} manually on this platform.")

    def toggle_setup_notes(self) -> None:
        if self.setup_notes_label is None:
            self.status_var.set("Setup notes are not available in this view.")
            return

        if self.setup_notes_visible:
            self.setup_notes_label.pack_forget()
            self.setup_notes_visible = False
            self.status_var.set("Setup notes hidden.")
        else:
            self.setup_notes_label.pack(anchor="w", padx=10, pady=(0, 8))
            self.setup_notes_visible = True
            self.status_var.set("Setup notes shown.")

    def switch_mode(self, mode_name: str, device_name: str) -> None:
        self.save_form_config()
        ok_record, record_message = self.device_manager.set_default_recording_device(device_name)
        if not ok_record:
            self.status_var.set(record_message)
            return

        playback_target = None
        if mode_name == "VAC":
            playback_target = self.vac_playback_var.get().strip()
        elif mode_name in {"Microphone", "Mixed"}:
            playback_target = self.speaker_var.get().strip()

        playback_message = ""
        if playback_target:
            ok_playback, playback_message = self.device_manager.set_default_playback_device(playback_target)
            if not ok_playback:
                self.status_var.set(f"{record_message} Playback switch failed: {playback_message}")
                return

        if ok_record:
            self.current_mode = mode_name
            self.config["last_mode"] = mode_name
            save_config(self.config)
            self.mode_var.set(mode_name)
            self._refresh_mode_hint()
            if mode_name == "VAC":
                self.status_var.set(f"{record_message} {playback_message}")
            else:
                self.status_var.set(f"{record_message} {playback_message}".strip())

    def test_vac_routing(self) -> None:
        if self._vac_test_running:
            self.status_var.set("VAC routing test is already running.")
            return

        self.save_form_config()
        ok_record, record_message = self.device_manager.set_default_recording_device(self.vac_var.get().strip())
        if not ok_record:
            self.status_var.set(record_message)
            return

        ok_playback, playback_message = self.device_manager.set_default_playback_device(self.vac_playback_var.get().strip())
        if not ok_playback:
            self.status_var.set(f"{record_message} Playback switch failed: {playback_message}")
            return

        self.current_mode = "VAC"
        self.config["last_mode"] = "VAC"
        save_config(self.config)
        self.mode_var.set("VAC")
        self._refresh_mode_hint()
        self._vac_test_running = True
        self.btn_vac_test.configure(state="disabled", text="Testing VAC...")
        self.status_var.set("VAC routing test started. A short tone is being sent through CABLE Input.")
        threading.Thread(target=self._run_vac_test_tone, daemon=True).start()

    def _run_vac_test_tone(self) -> None:
        sample_rate = 48000
        duration_seconds = 1.2
        amplitude = 0.18
        frequency_hz = 880.0

        try:
            timeline = np.linspace(0.0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
            envelope = np.minimum(1.0, timeline * 6.0) * np.minimum(1.0, (duration_seconds - timeline) * 6.0)
            tone = (np.sin(2 * np.pi * frequency_hz * timeline) * envelope * amplitude).astype(np.float32)
            sd.play(tone, samplerate=sample_rate, blocking=True)
            result_message = "VAC routing test completed. If the app meter moved, the route is working."
        except Exception as exc:
            result_message = f"VAC routing test failed: {exc}"
        finally:
            try:
                sd.stop()
            except Exception:
                pass
            if self._closing:
                return
            try:
                self.root.after(0, lambda: self._finish_vac_test(result_message))
            except Exception:
                return

    def _finish_vac_test(self, message: str) -> None:
        self._vac_test_running = False
        self.btn_vac_test.configure(state="normal", text="Test VAC Routing")
        self.status_var.set(message)

    def toggle_mute(self) -> None:
        ok, message = self.device_manager.toggle_mute()
        if not ok:
            self.status_var.set(message)
            return

        self.is_muted = not self.is_muted
        if self.is_muted:
            self.mute_button.configure(text="Muted", fg_color="#5F2120", hover_color="#471816")
        else:
            self.mute_button.configure(text="Mute Toggle", fg_color="#d32f2f", hover_color="#b71c1c")
        self.status_var.set(message)

    def toggle_wer_monitoring(self) -> None:
        enabled = bool(self.wer_enabled_var.get())
        self.config["wer_mode_enabled"] = enabled
        save_config(self.config)

        if enabled:
            self.monitor.start()
            self.monitor_status_var.set("Excellent")
            self.monitor_level_var.set("Listening for input...")
            self.monitor_detail_var.set("Sampling the current default recording device.")
            self.monitor_recommendation_var.set("Monitoring enabled\nListening for healthy speech levels")
            self._set_warnings_text(self.monitor_recommendation_var.get())
            self.monitor_status_label.configure(text="Starting", text_color="#8AB4F8")
            self.monitor_stability_label.configure(text="Sampling", text_color="#8AB4F8")
            self.monitor_wer_label.configure(text="...", text_color="#8AB4F8")
            self.meter.set_levels("RMS: sampling", "Peak: sampling", "Starting", "#8AB4F8", 0.0)
            self.status_var.set("WER monitoring enabled.")
        else:
            self.monitor.stop()
            self.monitor_status_var.set("Monitoring disabled")
            self.monitor_level_var.set("RMS: -∞ dB | Peak: -∞ dB")
            self.monitor_detail_var.set("Enable WER mode to monitor input quality.")
            self.monitor_recommendation_var.set("Monitoring disabled\nEnable Monitor to resume live analysis")
            self.monitor_status_label.configure(text="Disabled", text_color="#9E9E9E")
            self.monitor_stability_label.configure(text="Paused", text_color="#9E9E9E")
            self.monitor_wer_label.configure(text="--", text_color="#9E9E9E")
            self.meter.set_levels("RMS: -∞ dB", "Peak: -∞ dB", "Paused", "#9E9E9E", 0.0)
            self._set_warnings_text(self.monitor_recommendation_var.get())
            self.status_var.set("WER monitoring disabled.")

    def _queue_quality_update(self, result: dict[str, Any]) -> None:
        if self._closing:
            return
        try:
            if not self.root.winfo_exists():
                return
            self.root.after(0, lambda: self._apply_quality_update(result))
        except Exception:
            return

    def _apply_quality_update(self, result: dict[str, Any]) -> None:
        if self._closing or not self.wer_enabled_var.get():
            return
        quality = result["quality"]
        self.monitor_status_var.set(result["status_text"])
        self.monitor_level_var.set(result["level_text"])
        self.monitor_detail_var.set(result["detail_text"])
        color = QUALITY_COLORS.get(quality, "#9E9E9E")
        self.monitor_status_label.configure(text=result["status_text"], text_color=color)
        stability_text = "Stable" if quality in {"excellent", "good"} else "Check"
        wer_text = {
            "excellent": "3-7%",
            "good": "5-9%",
            "low": "8-14%",
            "too_quiet": ">15%",
            "clipping": ">15%",
            "error": "N/A",
        }.get(quality, "N/A")
        self.monitor_stability_label.configure(text=stability_text, text_color=color)
        self.monitor_wer_label.configure(text=wer_text, text_color=color)
        self.monitor_summary_var.set(f"Quality: {result['status_text']} | WER mode: {'On' if self.wer_enabled_var.get() else 'Off'}")
        progress = QUALITY_PROGRESS.get(quality, 0.0)
        rms_text, peak_text = self._split_levels(result["level_text"])
        status_text = "Monitoring" if quality != "error" else "Unavailable"
        self.meter.set_levels(rms_text, peak_text, status_text, color, progress)
        self.monitor_recommendation_var.set(self._recommendation_text(quality, result["detail_text"]))
        self._set_warnings_text(self.monitor_recommendation_var.get())

    def _split_levels(self, level_text: str) -> tuple[str, str]:
        parts = [segment.strip() for segment in level_text.split("|")]
        if len(parts) == 2:
            return parts[0], parts[1]
        return level_text, "Peak: n/a"

    def _recommendation_text(self, quality: str, detail: str) -> str:
        recommendation = {
            "excellent": "No issues detected\nAll systems optimal",
            "good": "Signal is strong\nLeave a little more headroom if possible",
            "low": "Signal is low\nIncrease mic gain or move closer to the mic",
            "too_quiet": "Input is too quiet\nCheck device selection and source levels",
            "clipping": "Clipping detected\nReduce mic gain or mixer output immediately",
            "error": "Monitor unavailable\nCheck audio device permissions and configuration",
        }.get(quality, detail)
        return recommendation

    def _refresh_mode_hint(self) -> None:
        hint = {
            "Microphone": "Best when you are speaking live. Expect more room noise than direct digital audio.",
            "VAC": "Best WER path for playback-only transcription because the signal stays fully digital.",
            "Mixed": "Use when you need narration over playback. Quality depends on Voicemeeter routing and gain staging.",
        }.get(self.current_mode, "")
        self.mode_hint_label.configure(text=hint)
        self.mode_status_label.configure(text=MODE_STATUS.get(self.current_mode, ""))
        self.mode_display.configure(text=self.current_mode)

    def on_close(self) -> None:
        self._closing = True
        self._close_settings_window()
        self.monitor.stop()
        if self.root.winfo_exists():
            self.save_form_config()
            self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
