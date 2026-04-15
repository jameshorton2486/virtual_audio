import json
import logging
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any


APP_DIR = Path(__file__).resolve().parent
VENV_PYTHON = APP_DIR / ".venv" / "Scripts" / "python.exe"
ENV_PATH = APP_DIR / ".env"


def load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


def ensure_local_venv() -> None:
    if sys.platform != "win32":
        return

    if not VENV_PYTHON.exists():
        return

    # PyCharm should use its configured interpreter directly instead of being
    # force-relaunched behind the IDE's back.
    if os.environ.get("PYCHARM_HOSTED"):
        return

    current_python = Path(sys.executable).resolve()
    target_python = VENV_PYTHON.resolve()
    if current_python == target_python:
        return

    # Restart inside the project venv so local dependencies are always used.
    print(f"[virtual_audio] Restarting in local venv: {target_python}")
    os.execv(str(target_python), [str(target_python), str(APP_DIR / "app.py"), *sys.argv[1:]])


load_dotenv_file(ENV_PATH)

import customtkinter as ctk
import numpy as np
import sounddevice as sd

from meter_widget import AudioLevelMeter


CONFIG_PATH = APP_DIR / "config.json"
NIRCMD_PATH = APP_DIR / "nircmd.exe"
TRANSCRIPTS_DIR = APP_DIR / "transcripts"
LOGS_DIR = APP_DIR / "logs"
LOG_PATH = LOGS_DIR / "virtual_audio.log"
DEEPGRAM_LISTEN_URL = "https://api.deepgram.com/v1/listen"

SUPPORTED_TRANSCRIPTION_SUFFIXES = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".ogg",
    ".wav",
    ".webm",
    ".wma",
}

DEFAULT_CONFIG = {
    "mic_device": "Microphone (Realtek Audio)",
    "vac_device": "CABLE Output (VB-Audio Virtual Cable)",
    "speaker_device": "Speakers (Realtek Audio)",
    "vac_playback_device": "CABLE Input (VB-Audio Virtual Cable)",
    "voicemeeter_device": "VoiceMeeter Output (VB-Audio VoiceMeeter VAIO)",
    "deepgram_smart_format": True,
    "deepgram_diarize": True,
    "deepgram_paragraphs": True,
    "deepgram_filler_words": True,
    "deepgram_numerals": True,
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

MODE_UI = {
    "Microphone": {
        "title": "Microphone Mode",
        "badge": "WER 10-15%",
        "accent": "#4CAF50",
        "route": "Voice -> Microphone -> Windows recording",
    },
    "VAC": {
        "title": "Virtual Audio Cable Mode",
        "badge": "WER 3-7% BEST",
        "accent": "#2E7D32",
        "route": "Playback -> CABLE Input -> CABLE Output -> Recording",
    },
    "Mixed": {
        "title": "Mixed Mode",
        "badge": "WER 5-10%",
        "accent": "#7B1FA2",
        "route": "Mic + system audio -> Voicemeeter -> Recording",
    },
}


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("virtual_audio")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


LOGGER = setup_logging()


def debug_log(message: str, level: str = "info") -> None:
    print(message)
    log_fn = getattr(LOGGER, level, LOGGER.info)
    log_fn(message)


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
    sanitized["deepgram_smart_format"] = _coerce_bool(
        sanitized.get("deepgram_smart_format"),
        bool(DEFAULT_CONFIG["deepgram_smart_format"]),
    )
    sanitized["deepgram_diarize"] = _coerce_bool(
        sanitized.get("deepgram_diarize"),
        bool(DEFAULT_CONFIG["deepgram_diarize"]),
    )
    sanitized["deepgram_paragraphs"] = _coerce_bool(
        sanitized.get("deepgram_paragraphs"),
        bool(DEFAULT_CONFIG["deepgram_paragraphs"]),
    )
    sanitized["deepgram_filler_words"] = _coerce_bool(
        sanitized.get("deepgram_filler_words"),
        bool(DEFAULT_CONFIG["deepgram_filler_words"]),
    )
    sanitized["deepgram_numerals"] = _coerce_bool(
        sanitized.get("deepgram_numerals"),
        bool(DEFAULT_CONFIG["deepgram_numerals"]),
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


def get_deepgram_api_key() -> str:
    return os.environ.get("DEEPGRAM_API_KEY", "").strip()


def extract_transcript_text(payload: dict[str, Any]) -> str:
    try:
        channels = payload["results"]["channels"]
        if not channels:
            return ""
        alternatives = channels[0].get("alternatives", [])
        if not alternatives:
            return ""
        return str(alternatives[0].get("transcript", "")).strip()
    except (KeyError, TypeError, IndexError):
        return ""


def build_transcript_output_paths(media_path: Path, output_dir: Path = TRANSCRIPTS_DIR) -> tuple[Path, Path]:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stem = media_path.stem or "transcript"
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "transcript"
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = output_dir / f"{safe_stem}_{timestamp}.txt"
    payload_path = output_dir / f"{safe_stem}_{timestamp}.json"
    return transcript_path, payload_path


def build_live_transcript_output_path(output_dir: Path = TRANSCRIPTS_DIR) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"live_transcript_{timestamp}.txt"


def build_live_transcript_metadata_path(transcript_path: Path) -> Path:
    return transcript_path.with_suffix(".json")


def normalize_audio_device_name(name: str) -> str:
    return re.sub(r",\s*(MME|Windows .+|DirectSound|WDM-KS)$", "", name, flags=re.IGNORECASE).strip()


def resolve_input_device(name: str) -> tuple[int | None, dict[str, Any] | None]:
    target = normalize_audio_device_name(name)
    try:
        devices = sd.query_devices()
    except Exception:
        return None, None

    exact_match: tuple[int, dict[str, Any]] | None = None
    partial_match: tuple[int, dict[str, Any]] | None = None

    for index, device in enumerate(devices):
        if int(device.get("max_input_channels", 0)) <= 0:
            continue
        raw_name = str(device.get("name", "")).strip()
        normalized = normalize_audio_device_name(raw_name)
        if normalized == target:
            exact_match = (index, device)
            break
        if target and target.lower() in normalized.lower() and partial_match is None:
            partial_match = (index, device)

    match = exact_match or partial_match
    if match is None:
        return None, None
    return match[0], match[1]


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
            debug_log(f"[AudioDeviceManager] Setting default device role={role} name={device_name}")
            subprocess.run(
                [str(NIRCMD_PATH), "setdefaultsounddevice", device_name, role],
                check=True,
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )

    @classmethod
    def set_default_recording_device(cls, device_name: str) -> tuple[bool, str]:
        try:
            debug_log(f"[AudioDeviceManager] Request to switch recording device -> {device_name}")
            cls._set_default_device(device_name)
            return True, f"Switched recording device to {device_name} for all Windows audio roles."
        except subprocess.CalledProcessError as exc:
            error_text = exc.stderr.decode("utf-8", errors="ignore").strip() if exc.stderr else ""
            debug_log(f"[AudioDeviceManager] Recording device switch failed: {error_text or exc}", level="error")
            return False, error_text or f"Failed to switch to {device_name}."
        except Exception as exc:
            debug_log(f"[AudioDeviceManager] Recording device switch exception: {exc}", level="error")
            return False, str(exc)

    @classmethod
    def set_default_playback_device(cls, device_name: str) -> tuple[bool, str]:
        try:
            debug_log(f"[AudioDeviceManager] Request to switch playback device -> {device_name}")
            cls._set_default_device(device_name)
            return True, f"Switched playback device to {device_name} for all Windows audio roles."
        except subprocess.CalledProcessError as exc:
            error_text = exc.stderr.decode("utf-8", errors="ignore").strip() if exc.stderr else ""
            debug_log(f"[AudioDeviceManager] Playback device switch failed: {error_text or exc}", level="error")
            return False, error_text or f"Failed to switch to {device_name}."
        except Exception as exc:
            debug_log(f"[AudioDeviceManager] Playback device switch exception: {exc}", level="error")
            return False, str(exc)

    @staticmethod
    def toggle_mute() -> tuple[bool, str]:
        if not NIRCMD_PATH.exists():
            return False, f"Missing {NIRCMD_PATH.name} in {APP_DIR}."

        try:
            debug_log("[AudioDeviceManager] Toggling mute on default recording device")
            subprocess.run(
                [str(NIRCMD_PATH), "mutesysvolume", "2", "default_record"],
                check=True,
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            return True, "Toggled mute on the default recording device."
        except subprocess.CalledProcessError as exc:
            error_text = exc.stderr.decode("utf-8", errors="ignore").strip() if exc.stderr else ""
            debug_log(f"[AudioDeviceManager] Toggle mute failed: {error_text or exc}", level="error")
            return False, error_text or "Failed to toggle mute."
        except Exception as exc:
            debug_log(f"[AudioDeviceManager] Toggle mute exception: {exc}", level="error")
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


class DeepgramFileTranscriber:
    def __init__(
        self,
        api_key: str,
        *,
        smart_format: bool = True,
        diarize: bool = True,
        paragraphs: bool = True,
        filler_words: bool = True,
        numerals: bool = True,
    ):
        self.api_key = api_key.strip()
        self.smart_format = bool(smart_format)
        self.diarize = bool(diarize)
        self.paragraphs = bool(paragraphs)
        self.filler_words = bool(filler_words)
        self.numerals = bool(numerals)

    def transcribe_file(self, media_path: Path, output_dir: Path = TRANSCRIPTS_DIR) -> tuple[bool, str]:
        debug_log(
            f"[DeepgramFileTranscriber] Starting file transcription for {media_path} "
            f"smart_format={self.smart_format} diarize={self.diarize} paragraphs={self.paragraphs} "
            f"filler_words={self.filler_words} numerals={self.numerals}"
        )
        if not self.api_key:
            return False, "Missing DEEPGRAM_API_KEY in .env."

        if not media_path.exists():
            return False, f"File not found: {media_path}"

        if media_path.suffix.lower() not in SUPPORTED_TRANSCRIPTION_SUFFIXES:
            supported = ", ".join(sorted(SUPPORTED_TRANSCRIPTION_SUFFIXES))
            return False, f"Unsupported media type. Supported extensions: {supported}"

        mime_type, _ = mimetypes.guess_type(str(media_path))
        content_type = mime_type or "application/octet-stream"
        query = urllib.parse.urlencode(
            {
                "model": "nova-3",
                "smart_format": "true" if self.smart_format else "false",
                "punctuate": "true",
                "paragraphs": "true" if self.paragraphs else "false",
                "diarize": "true" if self.diarize else "false",
                "filler_words": "true" if self.filler_words else "false",
                "numerals": "true" if self.numerals else "false",
                "detect_language": "true",
            }
        )

        try:
            media_bytes = media_path.read_bytes()
        except OSError as exc:
            return False, f"Failed to read media file: {exc}"

        request = urllib.request.Request(
            url=f"{DEEPGRAM_LISTEN_URL}?{query}",
            data=media_bytes,
            headers={
                "Authorization": f"Token {self.api_key}",
                "Content-Type": content_type,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore").strip()
            debug_log(f"[DeepgramFileTranscriber] HTTP error for {media_path.name}: {error_body or exc}", level="error")
            return False, error_body or f"Deepgram request failed with HTTP {exc.code}."
        except urllib.error.URLError as exc:
            debug_log(f"[DeepgramFileTranscriber] Network error for {media_path.name}: {exc.reason}", level="error")
            return False, f"Unable to reach Deepgram: {exc.reason}"
        except json.JSONDecodeError as exc:
            debug_log(f"[DeepgramFileTranscriber] Invalid JSON for {media_path.name}: {exc}", level="error")
            return False, f"Deepgram returned invalid JSON: {exc}"

        transcript_text = extract_transcript_text(payload)
        transcript_path, payload_path = build_transcript_output_paths(media_path, output_dir=output_dir)

        try:
            transcript_path.write_text(transcript_text or "[No transcript text returned]", encoding="utf-8")
            payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            debug_log(f"[DeepgramFileTranscriber] Failed to save outputs for {media_path.name}: {exc}", level="error")
            return False, f"Failed to save transcript output: {exc}"

        if not transcript_text:
            debug_log(f"[DeepgramFileTranscriber] No transcript text returned for {media_path.name}", level="warning")
            return False, f"Deepgram returned no transcript text. Saved raw response to {payload_path.name}."

        debug_log(f"[DeepgramFileTranscriber] Completed file transcription for {media_path.name} -> {transcript_path.name}")
        return True, f"Transcript saved to {transcript_path.name}\nRaw response saved to {payload_path.name}"


class LiveTranscriptionSession:
    def __init__(
        self,
        api_key: str,
        input_device_name: str,
        sample_rate_hz: int,
        mode_name: str,
        smart_format: bool,
        diarize: bool,
        paragraphs: bool,
        filler_words: bool,
        numerals: bool,
        on_transcript,
        on_status,
    ):
        self.api_key = api_key.strip()
        self.input_device_name = input_device_name.strip()
        self.sample_rate_hz = sample_rate_hz
        self.mode_name = mode_name.strip() or "Unknown"
        self.smart_format = bool(smart_format)
        self.diarize = bool(diarize)
        self.paragraphs = bool(paragraphs)
        self.filler_words = bool(filler_words)
        self.numerals = bool(numerals)
        self.on_transcript = on_transcript
        self.on_status = on_status
        self.connection = None
        self.stream = None
        self.transcript_path: Path | None = None
        self.metadata_path: Path | None = None
        self.final_lines: list[str] = []
        self.current_interim = ""
        self.running = False
        self.error_message = ""
        self.actual_device_name = self.input_device_name
        self.started_at = ""
        self.stopped_at = ""

    def start(self) -> tuple[bool, str]:
        debug_log(
            f"[LiveTranscriptionSession] Start requested mode={self.mode_name} "
            f"configured_device={self.input_device_name} smart_format={self.smart_format} "
            f"diarize={self.diarize} paragraphs={self.paragraphs} "
            f"filler_words={self.filler_words} numerals={self.numerals}"
        )
        if not self.api_key:
            return False, "Missing DEEPGRAM_API_KEY in .env."

        if not self.input_device_name:
            return False, "No recording device selected for live transcription."

        device_index, device_info = resolve_input_device(self.input_device_name)
        if device_index is None or device_info is None:
            debug_log(f"[LiveTranscriptionSession] Unable to resolve input device: {self.input_device_name}", level="error")
            return False, f"Unable to find recording device: {self.input_device_name}"

        try:
            from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents
        except Exception as exc:
            return False, f"Deepgram SDK is not available in this Python environment: {exc}"

        self.transcript_path = build_live_transcript_output_path()
        self.metadata_path = build_live_transcript_metadata_path(self.transcript_path)
        deepgram = DeepgramClient(self.api_key)
        self.connection = deepgram.listen.websocket.v("1")
        self.connection.on(LiveTranscriptionEvents.Open, self._on_open)
        self.connection.on(LiveTranscriptionEvents.Transcript, self._on_transcript)
        self.connection.on(LiveTranscriptionEvents.Error, self._on_error)
        self.connection.on(LiveTranscriptionEvents.Close, self._on_close)

        options = LiveOptions(
            model="nova-3",
            language="en-US",
            smart_format=self.smart_format,
            punctuate=True,
            interim_results=True,
            diarize=self.diarize,
            paragraphs=self.paragraphs,
            filler_words=self.filler_words,
            numerals=self.numerals,
            encoding="linear16",
            channels=1,
            sample_rate=self.sample_rate_hz,
        )

        if not self.connection.start(options):
            debug_log("[LiveTranscriptionSession] Deepgram websocket start returned false", level="error")
            return False, "Failed to start Deepgram live transcription connection."

        try:
            self.stream = sd.RawInputStream(
                samplerate=self.sample_rate_hz,
                blocksize=1024,
                device=device_index,
                channels=1,
                dtype="int16",
                callback=self._audio_callback,
            )
            self.stream.start()
        except Exception as exc:
            try:
                self.connection.finish()
            except Exception:
                pass
            self.connection = None
            debug_log(f"[LiveTranscriptionSession] Failed to open RawInputStream for {self.input_device_name}: {exc}", level="error")
            return False, f"Failed to open input stream on {self.input_device_name}: {exc}"

        self.running = True
        self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.actual_device_name = normalize_audio_device_name(str(device_info.get("name", self.input_device_name)))
        debug_log(
            f"[LiveTranscriptionSession] Running mode={self.mode_name} resolved_device={self.actual_device_name} "
            f"device_index={device_index} sample_rate={self.sample_rate_hz}"
        )
        self._write_metadata(status="running")
        return True, f"Live transcription started from {self.actual_device_name}."

    def stop(self) -> tuple[bool, str]:
        debug_log(f"[LiveTranscriptionSession] Stop requested mode={self.mode_name} resolved_device={self.actual_device_name}")
        self.running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        if self.connection is not None:
            try:
                self.connection.finish()
            except Exception:
                pass
            self.connection = None

        transcript_body = "\n".join(line for line in self.final_lines if line.strip()).strip()
        transcript_text = transcript_body or "[No final transcript captured]"
        output_path = self.transcript_path or build_live_transcript_output_path()
        self.stopped_at = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            output_path.write_text(transcript_text, encoding="utf-8")
        except OSError as exc:
            return False, f"Failed to save live transcript: {exc}"

        self._write_metadata(status="error" if self.error_message else "completed")

        if self.error_message:
            debug_log(f"[LiveTranscriptionSession] Stopped with error: {self.error_message}", level="warning")
            return False, f"{self.error_message}\nPartial transcript saved to {output_path.name}"
        debug_log(f"[LiveTranscriptionSession] Completed successfully -> {output_path.name}")
        return True, f"Live transcript saved to {output_path.name}"

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if not self.running or self.connection is None:
            return
        if status:
            debug_log(f"[LiveTranscriptionSession] Audio callback status: {status}", level="warning")
            self.on_status(f"Audio stream status: {status}")
        try:
            self.connection.send(bytes(indata))
        except Exception as exc:
            self.error_message = f"Audio send failed: {exc}"
            self.on_status(self.error_message)

    def _on_open(self, client, open=None, **kwargs) -> None:
        debug_log("[LiveTranscriptionSession] Deepgram websocket opened")
        self.on_status("Deepgram live connection opened.")

    def _on_close(self, client, close=None, **kwargs) -> None:
        debug_log("[LiveTranscriptionSession] Deepgram websocket closed")
        self.on_status("Deepgram live connection closed.")

    def _on_error(self, client, error=None, **kwargs) -> None:
        self.error_message = str(error) if error else "Deepgram live transcription error."
        debug_log(f"[LiveTranscriptionSession] Deepgram error: {self.error_message}", level="error")
        self.on_status(self.error_message)

    def _on_transcript(self, client, result=None, **kwargs) -> None:
        if result is None:
            return

        try:
            alternatives = result.channel.alternatives
            transcript = alternatives[0].transcript.strip() if alternatives else ""
        except Exception:
            transcript = ""

        if not transcript:
            return

        if getattr(result, "is_final", False):
            self.final_lines.append(transcript)
            self.current_interim = ""
            debug_log(f"[LiveTranscriptionSession] Final transcript segment #{len(self.final_lines)}: {transcript[:120]}")
            self._write_partial_transcript()
            combined = "\n".join(self.final_lines)
            self.on_transcript(combined, "")
        else:
            self.current_interim = transcript
            debug_log(f"[LiveTranscriptionSession] Interim transcript: {transcript[:120]}")
            combined = "\n".join(self.final_lines)
            self.on_transcript(combined, self.current_interim)

    def _write_partial_transcript(self) -> None:
        if self.transcript_path is None:
            return
        transcript_body = "\n".join(line for line in self.final_lines if line.strip()).strip()
        if not transcript_body:
            return
        try:
            self.transcript_path.write_text(transcript_body, encoding="utf-8")
        except OSError:
            pass

    def _write_metadata(self, status: str) -> None:
        if self.metadata_path is None:
            return
        payload = {
            "status": status,
            "mode": self.mode_name,
            "configured_input_device": self.input_device_name,
            "actual_input_device": self.actual_device_name,
            "sample_rate_hz": self.sample_rate_hz,
            "smart_format": self.smart_format,
            "diarize": self.diarize,
            "paragraphs": self.paragraphs,
            "filler_words": self.filler_words,
            "numerals": self.numerals,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "final_segment_count": len(self.final_lines),
            "error_message": self.error_message,
            "transcript_file": self.transcript_path.name if self.transcript_path else "",
        }
        try:
            self.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass
        else:
            debug_log(f"[LiveTranscriptionSession] Metadata updated -> {self.metadata_path.name}")


class App:
    def __init__(self) -> None:
        debug_log("[App] Initializing application")
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
        self._transcription_running = False
        self._live_transcription_running = False
        self._live_transcription_starting = False
        self.live_transcription_session: LiveTranscriptionSession | None = None
        self.live_transcript_final_text = ""
        self.live_transcript_interim_text = ""

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
        self.deepgram_smart_format_var = ctk.BooleanVar(value=bool(self.config["deepgram_smart_format"]))
        self.deepgram_diarize_var = ctk.BooleanVar(value=bool(self.config["deepgram_diarize"]))
        self.deepgram_paragraphs_var = ctk.BooleanVar(value=bool(self.config["deepgram_paragraphs"]))
        self.deepgram_filler_words_var = ctk.BooleanVar(value=bool(self.config["deepgram_filler_words"]))
        self.deepgram_numerals_var = ctk.BooleanVar(value=bool(self.config["deepgram_numerals"]))
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
        self.direct_recording_var = ctk.StringVar(value=self.config["mic_device"])
        self.direct_playback_var = ctk.StringVar(value=self.config["speaker_device"])
        self.wer_enabled_var = ctk.BooleanVar(value=bool(self.config["wer_mode_enabled"]))

        self._build_ui()
        self._refresh_mode_hint()
        self._refresh_detection_summary()

        if self.wer_enabled_var.get():
            self.monitor.start()

    def _build_ui(self) -> None:
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        self.main_scroll_frame = ctk.CTkScrollableFrame(self.root, fg_color="transparent")
        self.main_scroll_frame.pack(fill="both", expand=True)
        self.main_scroll_frame.grid_columnconfigure(0, weight=1)
        self.ui_parent = self.main_scroll_frame

        header_frame = ctk.CTkFrame(self.ui_parent, fg_color="transparent")
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

        self.meter = AudioLevelMeter(self.ui_parent, width=480, height=80)
        self.meter.pack(pady=6, padx=20, fill="x")

        self.wer_status_frame = ctk.CTkFrame(self.ui_parent, fg_color="#1a1a1a")
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

        mode_frame = ctk.CTkFrame(self.ui_parent)
        mode_frame.pack(pady=6, padx=20, fill="x")

        ctk.CTkLabel(
            mode_frame,
            text="Active Mode:",
            font=("Arial", 11),
        ).pack(pady=(5, 2))

        mode_summary_row = ctk.CTkFrame(mode_frame, fg_color="transparent")
        mode_summary_row.pack(fill="x", padx=12, pady=(0, 2))

        mode_text_frame = ctk.CTkFrame(mode_summary_row, fg_color="transparent")
        mode_text_frame.pack(side="left", fill="both", expand=True)

        self.mode_display = ctk.CTkLabel(
            mode_text_frame,
            text=f"{self.current_mode}",
            font=("Arial", 16, "bold"),
            text_color="#4CAF50",
            anchor="w",
        )
        self.mode_display.pack(anchor="w")

        self.mode_device_label = ctk.CTkLabel(
            mode_text_frame,
            text="",
            text_color="#C6C6C6",
            font=("Arial", 10),
            anchor="w",
            wraplength=330,
            justify="left",
        )
        self.mode_device_label.pack(anchor="w", pady=(1, 0))

        self.mode_badge_label = ctk.CTkLabel(
            mode_summary_row,
            text="",
            width=96,
            height=42,
            corner_radius=8,
            font=("Arial", 10, "bold"),
            fg_color="#2E7D32",
        )
        self.mode_badge_label.pack(side="right", padx=(10, 0))

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

        self.mode_route_label = ctk.CTkLabel(
            mode_frame,
            text="",
            text_color="#64B5F6",
            font=("Arial", 10, "bold"),
            wraplength=460,
            justify="center",
        )
        self.mode_route_label.pack(pady=(0, 8), padx=12)

        button_frame = ctk.CTkFrame(self.ui_parent)
        button_frame.pack(pady=6, padx=20, fill="x")

        self.btn_mic = ctk.CTkButton(
            button_frame,
            text="Microphone\nLive speaking | WER 10-15%",
            command=lambda: self.switch_mode("Microphone", self.mic_var.get()),
            height=42,
            font=("Arial", 11, "bold"),
        )
        self.btn_mic.pack(pady=4, padx=15, fill="x")

        self.btn_vac = ctk.CTkButton(
            button_frame,
            text="Virtual Audio Cable\nPlayback routing | WER 3-7% BEST",
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
            text="Mixed Mode\nMic + system audio | WER 5-10%",
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

        self._build_direct_device_controls()
        self._build_transcription_controls()
        self._build_live_transcription_controls()

        util_frame = ctk.CTkFrame(self.ui_parent)
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
            self.ui_parent,
            textvariable=self.status_var,
            font=("Arial", 9),
            text_color="#4CAF50",
        )
        self.status_label.pack(pady=6)

        ctk.CTkLabel(
            self.ui_parent,
            textvariable=self.footer_var,
            font=("Arial", 8),
            text_color="#555555",
        ).pack(pady=(0, 6))

    def _build_direct_device_controls(self) -> None:
        control_frame = ctk.CTkFrame(self.ui_parent)
        control_frame.pack(pady=6, padx=20, fill="x")

        ctk.CTkLabel(
            control_frame,
            text="Direct Audio Device Control",
            font=("Arial", 12, "bold"),
        ).pack(anchor="w", padx=12, pady=(8, 4))

        ctk.CTkLabel(
            control_frame,
            text="Change Windows recording and playback devices directly from here.",
            font=("Arial", 9),
            text_color="#C6C6C6",
            wraplength=460,
            justify="left",
        ).pack(anchor="w", padx=12, pady=(0, 8))

        self.direct_recording_menu = self._add_device_selector(
            control_frame,
            "Recording input",
            self.direct_recording_var,
            self.detected_input_devices,
        )

        direct_recording_actions = ctk.CTkFrame(control_frame, fg_color="transparent")
        direct_recording_actions.pack(fill="x", padx=10, pady=(0, 6))
        ctk.CTkButton(
            direct_recording_actions,
            text="Set Input",
            command=self.apply_selected_recording_device,
            height=28,
            font=("Arial", 9, "bold"),
        ).pack(side="left", expand=True, fill="x")

        self.direct_playback_menu = self._add_device_selector(
            control_frame,
            "Playback output",
            self.direct_playback_var,
            self.detected_output_devices,
        )

        direct_playback_actions = ctk.CTkFrame(control_frame, fg_color="transparent")
        direct_playback_actions.pack(fill="x", padx=10, pady=(0, 6))
        ctk.CTkButton(
            direct_playback_actions,
            text="Set Output",
            command=self.apply_selected_playback_device,
            height=28,
            font=("Arial", 9, "bold"),
        ).pack(side="left", expand=True, fill="x")

        preset_row = ctk.CTkFrame(control_frame, fg_color="transparent")
        preset_row.pack(fill="x", padx=10, pady=(0, 8))

        ctk.CTkButton(
            preset_row,
            text="Normal",
            command=lambda: self.apply_device_preset("normal"),
            height=30,
            font=("Arial", 9, "bold"),
        ).pack(side="left", padx=(0, 6), expand=True, fill="x")

        ctk.CTkButton(
            preset_row,
            text="VAC",
            command=lambda: self.apply_device_preset("vac"),
            height=30,
            font=("Arial", 9, "bold"),
            fg_color="#2E7D32",
            hover_color="#1B5E20",
        ).pack(side="left", padx=3, expand=True, fill="x")

        ctk.CTkButton(
            preset_row,
            text="Voicemeeter",
            command=lambda: self.apply_device_preset("mixed"),
            height=30,
            font=("Arial", 9, "bold"),
        ).pack(side="left", padx=(6, 0), expand=True, fill="x")

        utility_row = ctk.CTkFrame(control_frame, fg_color="transparent")
        utility_row.pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkButton(
            utility_row,
            text="Refresh Devices",
            command=self.refresh_detected_devices,
            height=28,
            font=("Arial", 9),
        ).pack(side="left", expand=True, fill="x")

    def _build_transcription_controls(self) -> None:
        transcription_frame = ctk.CTkFrame(self.ui_parent)
        transcription_frame.pack(pady=6, padx=20, fill="x")

        ctk.CTkLabel(
            transcription_frame,
            text="Deepgram File Transcription",
            font=("Arial", 12, "bold"),
        ).pack(anchor="w", padx=12, pady=(8, 4))

        ctk.CTkLabel(
            transcription_frame,
            text="Use saved Zoom recordings, YouTube downloads, and other media files.",
            font=("Arial", 9),
            text_color="#C6C6C6",
            wraplength=460,
            justify="left",
        ).pack(anchor="w", padx=12, pady=(0, 4))

        api_key_ready = bool(get_deepgram_api_key())
        self.transcription_status_label = ctk.CTkLabel(
            transcription_frame,
            text=(
                ("Deepgram API key detected" if api_key_ready else "Deepgram API key missing from .env")
                + "\n"
                + self._deepgram_options_summary()
            ),
            font=("Arial", 9, "bold"),
            text_color="#66BB6A" if api_key_ready else "#F9A825",
            wraplength=460,
            justify="left",
        )
        self.transcription_status_label.pack(anchor="w", padx=12, pady=(0, 8))

        options_frame = ctk.CTkFrame(transcription_frame, fg_color="transparent")
        options_frame.pack(fill="x", padx=10, pady=(0, 8))

        ctk.CTkLabel(
            options_frame,
            text="Deepgram Options",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", padx=2, pady=(0, 4))

        ctk.CTkSwitch(
            options_frame,
            text="Smart Format",
            variable=self.deepgram_smart_format_var,
            command=self._save_deepgram_options,
        ).pack(anchor="w", padx=2, pady=2)

        ctk.CTkSwitch(
            options_frame,
            text="Diarization",
            variable=self.deepgram_diarize_var,
            command=self._save_deepgram_options,
        ).pack(anchor="w", padx=2, pady=2)

        ctk.CTkSwitch(
            options_frame,
            text="Paragraphs",
            variable=self.deepgram_paragraphs_var,
            command=self._save_deepgram_options,
        ).pack(anchor="w", padx=2, pady=2)

        ctk.CTkSwitch(
            options_frame,
            text="Filler Words",
            variable=self.deepgram_filler_words_var,
            command=self._save_deepgram_options,
        ).pack(anchor="w", padx=2, pady=2)

        ctk.CTkSwitch(
            options_frame,
            text="Numerals",
            variable=self.deepgram_numerals_var,
            command=self._save_deepgram_options,
        ).pack(anchor="w", padx=2, pady=2)

        transcription_actions = ctk.CTkFrame(transcription_frame, fg_color="transparent")
        transcription_actions.pack(fill="x", padx=10, pady=(0, 10))

        self.btn_transcribe_file = ctk.CTkButton(
            transcription_actions,
            text="Transcribe File",
            command=self.transcribe_media_file,
            height=34,
            font=("Arial", 10, "bold"),
            fg_color="#1565C0",
            hover_color="#0D47A1",
        )
        self.btn_transcribe_file.pack(side="left", padx=(0, 6), expand=True, fill="x")

        ctk.CTkButton(
            transcription_actions,
            text="Open Transcripts Folder",
            command=self.open_transcripts_folder,
            height=34,
            font=("Arial", 10),
        ).pack(side="left", padx=(6, 0), expand=True, fill="x")

    def _build_live_transcription_controls(self) -> None:
        live_frame = ctk.CTkFrame(self.ui_parent)
        live_frame.pack(pady=6, padx=20, fill="x")

        ctk.CTkLabel(
            live_frame,
            text="Live Transcription",
            font=("Arial", 12, "bold"),
        ).pack(anchor="w", padx=12, pady=(8, 4))

        ctk.CTkLabel(
            live_frame,
            text="Streams the currently active recording device to Deepgram and saves the session transcript automatically.",
            font=("Arial", 9),
            text_color="#C6C6C6",
            wraplength=460,
            justify="left",
        ).pack(anchor="w", padx=12, pady=(0, 6))

        self.live_transcription_device_label = ctk.CTkLabel(
            live_frame,
            text="Live input source: " + self._current_live_input_device_name(),
            font=("Arial", 9, "bold"),
            text_color="#8AB4F8",
            wraplength=460,
            justify="left",
        )
        self.live_transcription_device_label.pack(anchor="w", padx=12, pady=(0, 6))

        api_key_ready = bool(get_deepgram_api_key())
        self.live_transcription_key_label = ctk.CTkLabel(
            live_frame,
            text=(
                ("Deepgram live key detected" if api_key_ready else "Deepgram live key missing from .env")
                + "\n"
                + self._deepgram_options_summary()
            ),
            font=("Arial", 9, "bold"),
            text_color="#66BB6A" if api_key_ready else "#F9A825",
            wraplength=460,
            justify="left",
        )
        self.live_transcription_key_label.pack(anchor="w", padx=12, pady=(0, 6))

        live_actions = ctk.CTkFrame(live_frame, fg_color="transparent")
        live_actions.pack(fill="x", padx=10, pady=(0, 8))

        self.btn_start_live = ctk.CTkButton(
            live_actions,
            text="Start Live Transcription",
            command=self.start_live_transcription,
            height=34,
            font=("Arial", 10, "bold"),
            fg_color="#2E7D32",
            hover_color="#1B5E20",
        )
        self.btn_start_live.pack(side="left", padx=(0, 6), expand=True, fill="x")

        self.btn_stop_live = ctk.CTkButton(
            live_actions,
            text="Stop Live Transcription",
            command=self.stop_live_transcription,
            height=34,
            font=("Arial", 10, "bold"),
            fg_color="#B71C1C",
            hover_color="#7F1010",
            state="disabled",
        )
        self.btn_stop_live.pack(side="left", padx=(6, 0), expand=True, fill="x")

        self.live_transcription_status_label = ctk.CTkLabel(
            live_frame,
            text="Idle",
            font=("Arial", 9, "bold"),
            text_color="#C6C6C6",
            wraplength=460,
            justify="left",
        )
        self.live_transcription_status_label.pack(anchor="w", padx=12, pady=(0, 6))

        self.live_transcript_box = ctk.CTkTextbox(
            live_frame,
            height=180,
            font=("Consolas", 10),
            wrap="word",
        )
        self.live_transcript_box.pack(fill="x", padx=12, pady=(0, 10))
        self.live_transcript_box.insert("1.0", "Live transcript will appear here.")
        self.live_transcript_box.configure(state="disabled")

        ctk.CTkButton(
            live_frame,
            text="Open Transcripts Folder",
            command=self.open_transcripts_folder,
            height=30,
            font=("Arial", 9),
        ).pack(fill="x", padx=12, pady=(0, 10))

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
        if self._live_transcription_running or self._live_transcription_starting:
            self.status_var.set("Stop live transcription before refreshing device lists.")
            return
        debug_log("[App] Refreshing detected input and output devices")
        self.detected_input_devices = list_input_devices()
        self.detected_output_devices = list_output_devices()
        self._hydrate_config_from_detected_devices()
        self.mic_var.set(self.config["mic_device"])
        self.vac_var.set(self.config["vac_device"])
        self.speaker_var.set(self.config["speaker_device"])
        self.vac_playback_var.set(self.config["vac_playback_device"])
        self.mix_var.set(self.config["voicemeeter_device"])
        self.direct_recording_var.set(self._normalize_direct_device_selection(self.direct_recording_var.get(), self.detected_input_devices))
        self.direct_playback_var.set(self._normalize_direct_device_selection(self.direct_playback_var.get(), self.detected_output_devices))
        self._refresh_device_menus()
        self._refresh_detection_summary()
        self._refresh_mode_hint()
        self.status_var.set("Refreshed detected Windows recording and playback devices.")

    def _normalize_direct_device_selection(self, current_value: str, devices: list[str]) -> str:
        if current_value in devices:
            return current_value
        if devices:
            return devices[0]
        return current_value

    def _refresh_device_menus(self) -> None:
        menu_specs = [
            ("mic_menu", self.mic_var, self.detected_input_devices),
            ("vac_menu", self.vac_var, self.detected_input_devices),
            ("speaker_menu", self.speaker_var, self.detected_output_devices),
            ("vac_playback_menu", self.vac_playback_var, self.detected_output_devices),
            ("mix_menu", self.mix_var, self.detected_input_devices),
            ("direct_recording_menu", self.direct_recording_var, self.detected_input_devices),
            ("direct_playback_menu", self.direct_playback_var, self.detected_output_devices),
        ]

        for menu_name, variable, devices in menu_specs:
            menu = getattr(self, menu_name, None)
            if menu is None:
                continue
            try:
                if menu.winfo_exists():
                    menu.configure(values=self._menu_values_for(variable.get(), devices))
            except Exception:
                continue

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
        self.config["deepgram_smart_format"] = bool(self.deepgram_smart_format_var.get())
        self.config["deepgram_diarize"] = bool(self.deepgram_diarize_var.get())
        self.config["deepgram_paragraphs"] = bool(self.deepgram_paragraphs_var.get())
        self.config["deepgram_filler_words"] = bool(self.deepgram_filler_words_var.get())
        self.config["deepgram_numerals"] = bool(self.deepgram_numerals_var.get())
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

    def apply_selected_recording_device(self) -> None:
        if self._live_transcription_running or self._live_transcription_starting:
            self.status_var.set("Stop live transcription before changing the recording device.")
            return
        device_name = self.direct_recording_var.get().strip()
        debug_log(f"[App] Applying selected recording device: {device_name}")
        ok, message = self.device_manager.set_default_recording_device(device_name)
        self.status_var.set(message)

    def apply_selected_playback_device(self) -> None:
        if self._live_transcription_running or self._live_transcription_starting:
            self.status_var.set("Stop live transcription before changing the playback device.")
            return
        device_name = self.direct_playback_var.get().strip()
        debug_log(f"[App] Applying selected playback device: {device_name}")
        ok, message = self.device_manager.set_default_playback_device(device_name)
        self.status_var.set(message)

    def apply_device_preset(self, preset_name: str) -> None:
        presets = {
            "normal": (self.mic_var.get().strip(), self.speaker_var.get().strip(), "Microphone"),
            "vac": (self.vac_var.get().strip(), self.vac_playback_var.get().strip(), "VAC"),
            "mixed": (self.mix_var.get().strip(), self.speaker_var.get().strip(), "Mixed"),
        }
        recording_device, playback_device, mode_name = presets[preset_name]

        self.direct_recording_var.set(recording_device)
        self.direct_playback_var.set(playback_device)
        self.switch_mode(mode_name, recording_device)

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

    def open_transcripts_folder(self) -> None:
        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(TRANSCRIPTS_DIR)  # type: ignore[attr-defined]
            return
        self.status_var.set(f"Open {TRANSCRIPTS_DIR} manually on this platform.")

    def _current_live_input_device_name(self) -> str:
        if self.current_mode == "VAC":
            return self.vac_var.get().strip() or "Not configured"
        if self.current_mode == "Mixed":
            return self.mix_var.get().strip() or "Not configured"
        return self.mic_var.get().strip() or "Not configured"

    def _deepgram_options_summary(self) -> str:
        smart_format = "On" if self.deepgram_smart_format_var.get() else "Off"
        diarize = "On" if self.deepgram_diarize_var.get() else "Off"
        paragraphs = "On" if self.deepgram_paragraphs_var.get() else "Off"
        filler_words = "On" if self.deepgram_filler_words_var.get() else "Off"
        numerals = "On" if self.deepgram_numerals_var.get() else "Off"
        return (
            f"Deepgram options: Smart Format {smart_format} | Diarization {diarize} | "
            f"Paragraphs {paragraphs} | Filler Words {filler_words} | Numerals {numerals}"
        )

    def _save_deepgram_options(self) -> None:
        self.config["deepgram_smart_format"] = bool(self.deepgram_smart_format_var.get())
        self.config["deepgram_diarize"] = bool(self.deepgram_diarize_var.get())
        self.config["deepgram_paragraphs"] = bool(self.deepgram_paragraphs_var.get())
        self.config["deepgram_filler_words"] = bool(self.deepgram_filler_words_var.get())
        self.config["deepgram_numerals"] = bool(self.deepgram_numerals_var.get())
        save_config(self.config)
        if hasattr(self, "transcription_status_label") and self.transcription_status_label.winfo_exists():
            api_key_ready = bool(get_deepgram_api_key())
            base_text = "Deepgram API key detected" if api_key_ready else "Deepgram API key missing from .env"
            self.transcription_status_label.configure(
                text=f"{base_text}\n{self._deepgram_options_summary()}",
                text_color="#66BB6A" if api_key_ready else "#F9A825",
            )
        self._refresh_live_transcription_labels()
        self.status_var.set(self._deepgram_options_summary())

    def _refresh_live_transcription_labels(self) -> None:
        label = getattr(self, "live_transcription_device_label", None)
        if label is not None:
            label.configure(text="Live input source: " + self._current_live_input_device_name())
        key_label = getattr(self, "live_transcription_key_label", None)
        if key_label is not None:
            api_key_ready = bool(get_deepgram_api_key())
            key_label.configure(
                text=(
                    ("Deepgram live key detected" if api_key_ready else "Deepgram live key missing from .env")
                    + "\n"
                    + self._deepgram_options_summary()
                ),
                text_color="#66BB6A" if api_key_ready else "#F9A825",
            )

    def _set_live_transcript_box_text(self, text: str) -> None:
        self.live_transcript_box.configure(state="normal")
        self.live_transcript_box.delete("1.0", "end")
        self.live_transcript_box.insert("1.0", text)
        self.live_transcript_box.configure(state="disabled")

    def _render_live_transcript(self) -> None:
        final_text = self.live_transcript_final_text.strip()
        interim_text = self.live_transcript_interim_text.strip()
        sections = []
        if final_text:
            sections.append(final_text)
        if interim_text:
            sections.append(f"[Listening] {interim_text}")
        rendered = "\n\n".join(sections) if sections else "Live transcript will appear here."
        self._set_live_transcript_box_text(rendered)
        try:
            self.live_transcript_box.see("end")
        except Exception:
            pass

    def _queue_live_transcript_update(self, final_text: str, interim_text: str) -> None:
        if self._closing:
            return
        try:
            self.root.after(0, lambda: self._apply_live_transcript_update(final_text, interim_text))
        except Exception:
            return

    def _apply_live_transcript_update(self, final_text: str, interim_text: str) -> None:
        self.live_transcript_final_text = final_text
        self.live_transcript_interim_text = interim_text
        self._render_live_transcript()

    def _queue_live_status_update(self, message: str) -> None:
        if self._closing:
            return
        try:
            self.root.after(0, lambda: self._apply_live_status_update(message))
        except Exception:
            return

    def _apply_live_status_update(self, message: str) -> None:
        lowered = message.lower()
        is_problem = any(token in lowered for token in ("error", "failed", "unable"))
        self.live_transcription_status_label.configure(
            text=message,
            text_color="#F57C00" if is_problem else ("#66BB6A" if self._live_transcription_running else ("#F9A825" if self._live_transcription_starting else "#C6C6C6")),
        )
        self.status_var.set(message)

    def _set_live_controls_state(self, *, running: bool = False, starting: bool = False) -> None:
        self._live_transcription_running = running
        self._live_transcription_starting = starting
        if hasattr(self, "btn_start_live") and self.btn_start_live.winfo_exists():
            self.btn_start_live.configure(
                state="disabled" if (running or starting) else "normal",
                text="Starting..." if starting else "Start Live Transcription",
            )
        if hasattr(self, "btn_stop_live") and self.btn_stop_live.winfo_exists():
            self.btn_stop_live.configure(state="normal" if running else "disabled")

    def start_live_transcription(self) -> None:
        if self._live_transcription_running or self._live_transcription_starting:
            self.status_var.set("Live transcription is already running.")
            return

        if self._vac_test_running:
            self.status_var.set("Wait for the VAC routing test to finish before starting live transcription.")
            return

        api_key = get_deepgram_api_key()
        if not api_key:
            messagebox.showerror(
                "Deepgram API Key Missing",
                "Add DEEPGRAM_API_KEY to .env before starting live transcription.",
            )
            return

        input_device_name = self._current_live_input_device_name()
        debug_log(f"[App] Starting live transcription mode={self.current_mode} input={input_device_name}")
        self._set_live_controls_state(starting=True)
        self.live_transcript_final_text = ""
        self.live_transcript_interim_text = ""
        self._render_live_transcript()
        self.live_transcription_status_label.configure(text="Connecting to Deepgram live transcription...", text_color="#F9A825")
        self.status_var.set(f"Starting live transcription from {input_device_name}...")
        threading.Thread(
            target=self._start_live_transcription_worker,
            args=(
                api_key,
                input_device_name,
                self.current_mode,
                bool(self.deepgram_smart_format_var.get()),
                bool(self.deepgram_diarize_var.get()),
                bool(self.deepgram_paragraphs_var.get()),
                bool(self.deepgram_filler_words_var.get()),
                bool(self.deepgram_numerals_var.get()),
            ),
            daemon=True,
        ).start()

    def _start_live_transcription_worker(
        self,
        api_key: str,
        input_device_name: str,
        mode_name: str,
        smart_format: bool,
        diarize: bool,
        paragraphs: bool,
        filler_words: bool,
        numerals: bool,
    ) -> None:
        session = LiveTranscriptionSession(
            api_key=api_key,
            input_device_name=input_device_name,
            sample_rate_hz=int(self.config["sample_rate_hz"]),
            mode_name=mode_name,
            smart_format=smart_format,
            diarize=diarize,
            paragraphs=paragraphs,
            filler_words=filler_words,
            numerals=numerals,
            on_transcript=self._queue_live_transcript_update,
            on_status=self._queue_live_status_update,
        )
        success, message = session.start()
        if self._closing:
            if success:
                session.stop()
            return
        try:
            self.root.after(0, lambda: self._finish_start_live_transcription(success, message, session))
        except Exception:
            if success:
                session.stop()
            return

    def _finish_start_live_transcription(self, success: bool, message: str, session: LiveTranscriptionSession) -> None:
        if not success:
            debug_log(f"[App] Live transcription failed to start: {message}", level="error")
            self.live_transcription_session = None
            self._set_live_controls_state(running=False, starting=False)
            self.live_transcription_status_label.configure(text=message, text_color="#F57C00")
            messagebox.showerror("Live Transcription Failed", message)
            self.status_var.set(message)
            return

        self.live_transcription_session = session
        debug_log(f"[App] Live transcription started successfully: {message}")
        self._set_live_controls_state(running=True, starting=False)
        self.live_transcription_status_label.configure(text=message, text_color="#66BB6A")
        self.status_var.set(message)

    def stop_live_transcription(self) -> None:
        if self._live_transcription_starting:
            self.status_var.set("Live transcription is still starting. Wait a moment and stop it again.")
            return

        if not self._live_transcription_running or self.live_transcription_session is None:
            self.status_var.set("Live transcription is not running.")
            return

        debug_log("[App] Stopping live transcription")
        success, message = self.live_transcription_session.stop()
        self._set_live_controls_state(running=False, starting=False)
        self.live_transcription_session = None
        self.live_transcription_status_label.configure(
            text=message,
            text_color="#66BB6A" if success else "#F57C00",
        )
        self.status_var.set(message)

    def transcribe_media_file(self) -> None:
        if self._transcription_running:
            self.status_var.set("A transcription job is already running.")
            return

        if not get_deepgram_api_key():
            messagebox.showerror(
                "Deepgram API Key Missing",
                "Add DEEPGRAM_API_KEY to .env before starting a transcription.",
            )
            return

        selected_path = filedialog.askopenfilename(
            title="Select Audio or Video File",
            filetypes=[
                ("Media files", "*.aac *.flac *.m4a *.mp3 *.mp4 *.mpeg *.mpga *.ogg *.wav *.webm *.wma"),
                ("All files", "*.*"),
            ],
        )
        if not selected_path:
            return

        self._transcription_running = True
        self.btn_transcribe_file.configure(state="disabled", text="Transcribing...")
        self.status_var.set(f"Uploading {Path(selected_path).name} to Deepgram for transcription...")
        threading.Thread(
            target=self._run_file_transcription,
            args=(Path(selected_path),),
            daemon=True,
        ).start()

    def _run_file_transcription(self, media_path: Path) -> None:
        transcriber = DeepgramFileTranscriber(
            get_deepgram_api_key(),
            smart_format=bool(self.deepgram_smart_format_var.get()),
            diarize=bool(self.deepgram_diarize_var.get()),
            paragraphs=bool(self.deepgram_paragraphs_var.get()),
            filler_words=bool(self.deepgram_filler_words_var.get()),
            numerals=bool(self.deepgram_numerals_var.get()),
        )
        success, message = transcriber.transcribe_file(media_path)
        if self._closing:
            return
        try:
            self.root.after(0, lambda: self._finish_file_transcription(success, message))
        except Exception:
            return

    def _finish_file_transcription(self, success: bool, message: str) -> None:
        self._transcription_running = False
        self.btn_transcribe_file.configure(state="normal", text="Transcribe File")
        self.status_var.set(message.splitlines()[0] if message else "Transcription finished.")
        if success:
            messagebox.showinfo("Transcription Complete", message)
        else:
            messagebox.showerror("Transcription Failed", message)

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
        if self._live_transcription_running or self._live_transcription_starting:
            self.status_var.set("Stop live transcription before switching modes.")
            return
        debug_log(f"[App] Switching mode -> {mode_name} using recording device {device_name}")
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
            self.direct_recording_var.set(device_name)
            if playback_target:
                self.direct_playback_var.set(playback_target)
            self._refresh_mode_hint()
            if mode_name == "VAC":
                self.status_var.set(f"{record_message} {playback_message}")
            else:
                self.status_var.set(f"{record_message} {playback_message}".strip())

    def test_vac_routing(self) -> None:
        if self._vac_test_running:
            self.status_var.set("VAC routing test is already running.")
            return

        debug_log("[App] Starting VAC routing test")

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
        debug_log(f"[App] VAC routing test finished: {message}")
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
        ui_config = MODE_UI.get(self.current_mode, MODE_UI["Microphone"])
        hint = {
            "Microphone": "Best when you are speaking live. Expect more room noise than direct digital audio.",
            "VAC": "Best WER path for playback-only transcription because the signal stays fully digital.",
            "Mixed": "Use when you need narration over playback. Quality depends on Voicemeeter routing and gain staging.",
        }.get(self.current_mode, "")
        self.mode_hint_label.configure(text=hint)
        self.mode_status_label.configure(text=MODE_STATUS.get(self.current_mode, ""))
        self.mode_display.configure(text=ui_config["title"], text_color=ui_config["accent"])
        self.mode_badge_label.configure(text=ui_config["badge"], fg_color=ui_config["accent"])
        self.mode_route_label.configure(text=ui_config["route"])
        self.mode_device_label.configure(text=self._current_mode_device_summary())
        self._refresh_live_transcription_labels()

    def _current_mode_device_summary(self) -> str:
        if self.current_mode == "VAC":
            recording_device = self.vac_var.get().strip() or "Not configured"
            playback_device = self.vac_playback_var.get().strip() or "Not configured"
        elif self.current_mode == "Mixed":
            recording_device = self.mix_var.get().strip() or "Not configured"
            playback_device = self.speaker_var.get().strip() or "Not configured"
        else:
            recording_device = self.mic_var.get().strip() or "Not configured"
            playback_device = self.speaker_var.get().strip() or "Not configured"

        return f"Recording: {recording_device}\nPlayback: {playback_device}"

    def on_close(self) -> None:
        debug_log("[App] Closing application")
        self._closing = True
        self._close_settings_window()
        if self.live_transcription_session is not None:
            self.live_transcription_session.stop()
            self.live_transcription_session = None
        self.monitor.stop()
        if self.root.winfo_exists():
            self.save_form_config()
            self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    ensure_local_venv()
    App().run()
