import json
import inspect
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
from typing import Any, Final, Literal, Mapping, TypedDict


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
LIVE_TRANSCRIPTION_SAMPLE_RATE_HZ = 16000

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

ModeName = Literal["Microphone", "VAC", "Mixed"]


class ConfigDict(TypedDict):
    mic_device: str
    vac_device: str
    speaker_device: str
    vac_playback_device: str
    voicemeeter_device: str
    deepgram_smart_format: bool
    deepgram_diarize: bool
    deepgram_paragraphs: bool
    deepgram_filler_words: bool
    deepgram_numerals: bool
    wer_mode_enabled: bool
    quality_check_interval_seconds: float
    sample_rate_hz: int
    last_mode: ModeName


class ActiveAudioDevice(TypedDict):
    name: str
    index: int
    info: dict[str, Any]
    sample_rate: int


DEFAULT_CONFIG: Final[ConfigDict] = {
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


def sanitize_config(config: dict[str, Any]) -> ConfigDict:
    sanitized: ConfigDict = {**DEFAULT_CONFIG}
    sanitized["mic_device"] = config.get("mic_device", sanitized["mic_device"])
    sanitized["vac_device"] = config.get("vac_device", sanitized["vac_device"])
    sanitized["speaker_device"] = config.get("speaker_device", sanitized["speaker_device"])
    sanitized["vac_playback_device"] = config.get("vac_playback_device", sanitized["vac_playback_device"])
    sanitized["voicemeeter_device"] = config.get("voicemeeter_device", sanitized["voicemeeter_device"])
    sanitized["wer_mode_enabled"] = config.get("wer_mode_enabled", sanitized["wer_mode_enabled"])
    sanitized["deepgram_smart_format"] = config.get("deepgram_smart_format", sanitized["deepgram_smart_format"])
    sanitized["deepgram_diarize"] = config.get("deepgram_diarize", sanitized["deepgram_diarize"])
    sanitized["deepgram_paragraphs"] = config.get("deepgram_paragraphs", sanitized["deepgram_paragraphs"])
    sanitized["deepgram_filler_words"] = config.get("deepgram_filler_words", sanitized["deepgram_filler_words"])
    sanitized["deepgram_numerals"] = config.get("deepgram_numerals", sanitized["deepgram_numerals"])
    sanitized["quality_check_interval_seconds"] = config.get(
        "quality_check_interval_seconds",
        sanitized["quality_check_interval_seconds"],
    )
    sanitized["sample_rate_hz"] = config.get("sample_rate_hz", sanitized["sample_rate_hz"])
    sanitized["last_mode"] = config.get("last_mode", sanitized["last_mode"])

    mic_device: Any = sanitized["mic_device"]
    sanitized["mic_device"] = mic_device.strip() if isinstance(mic_device, str) and mic_device.strip() else DEFAULT_CONFIG["mic_device"]
    vac_device: Any = sanitized["vac_device"]
    sanitized["vac_device"] = vac_device.strip() if isinstance(vac_device, str) and vac_device.strip() else DEFAULT_CONFIG["vac_device"]
    speaker_device: Any = sanitized["speaker_device"]
    sanitized["speaker_device"] = speaker_device.strip() if isinstance(speaker_device, str) and speaker_device.strip() else DEFAULT_CONFIG["speaker_device"]
    vac_playback_device: Any = sanitized["vac_playback_device"]
    sanitized["vac_playback_device"] = (
        vac_playback_device.strip()
        if isinstance(vac_playback_device, str) and vac_playback_device.strip()
        else DEFAULT_CONFIG["vac_playback_device"]
    )
    voicemeeter_device: Any = sanitized["voicemeeter_device"]
    sanitized["voicemeeter_device"] = (
        voicemeeter_device.strip()
        if isinstance(voicemeeter_device, str) and voicemeeter_device.strip()
        else DEFAULT_CONFIG["voicemeeter_device"]
    )

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
        raw_interval: Any = sanitized["quality_check_interval_seconds"]
        interval = float(raw_interval)
        if interval <= 0:
            raise ValueError
        sanitized["quality_check_interval_seconds"] = interval
    except (TypeError, ValueError):
        sanitized["quality_check_interval_seconds"] = float(DEFAULT_CONFIG["quality_check_interval_seconds"])

    try:
        raw_sample_rate: Any = sanitized["sample_rate_hz"]
        sample_rate = int(raw_sample_rate)
        if sample_rate <= 0:
            raise ValueError
        sanitized["sample_rate_hz"] = sample_rate
    except (TypeError, ValueError):
        sanitized["sample_rate_hz"] = int(DEFAULT_CONFIG["sample_rate_hz"])

    last_mode: Any = sanitized["last_mode"]
    sanitized["last_mode"] = last_mode if last_mode in MODE_TEXT else DEFAULT_CONFIG["last_mode"]

    return sanitized


def load_config() -> ConfigDict:
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG.copy())
        return DEFAULT_CONFIG.copy()

    try:
        stored = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        stored = {}

    merged: dict[str, Any] = {**DEFAULT_CONFIG}
    merged.update({k: v for k, v in stored.items() if k in DEFAULT_CONFIG})
    return sanitize_config(merged)


def save_config(config: Mapping[str, Any]) -> None:
    CONFIG_PATH.write_text(json.dumps(sanitize_config(dict(config)), indent=2), encoding="utf-8")


def get_deepgram_api_key() -> str:
    return os.environ.get("DEEPGRAM_API_KEY", "").strip()


def _deepgram_value(node: Any, key: str, default: Any = None) -> Any:
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _deepgram_words_to_speaker_lines(words: list[Any]) -> str:
    segments: list[tuple[str | None, list[str]]] = []

    for word_data in words:
        token = str(
            _deepgram_value(word_data, "punctuated_word")
            or _deepgram_value(word_data, "word")
            or ""
        ).strip()
        if not token:
            continue

        speaker_raw = _deepgram_value(word_data, "speaker")
        speaker = None if speaker_raw in (None, "") else str(speaker_raw)
        if segments and segments[-1][0] == speaker:
            segments[-1][1].append(token)
        else:
            segments.append((speaker, [token]))

    lines: list[str] = []
    for speaker, tokens in segments:
        text = " ".join(token for token in tokens if token).strip()
        if not text:
            continue
        if speaker is None:
            lines.append(text)
        else:
            lines.append(f"Speaker {speaker}: {text}")
    return "\n".join(lines).strip()


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


def format_deepgram_payload_text(payload: dict[str, Any]) -> str:
    try:
        channels = payload["results"]["channels"]
        if not channels:
            return ""
        alternatives = channels[0].get("alternatives", [])
        if not alternatives:
            return ""
        words = alternatives[0].get("words", [])
        speaker_text = _deepgram_words_to_speaker_lines(words)
        if speaker_text:
            return speaker_text
    except (KeyError, TypeError, IndexError, AttributeError):
        pass
    return extract_transcript_text(payload)


def format_live_result_text(result: Any) -> str:
    try:
        channel = _deepgram_value(result, "channel")
        alternatives = _deepgram_value(channel, "alternatives", [])
        if not alternatives:
            return ""
        primary = alternatives[0]
        words = _deepgram_value(primary, "words", [])
        speaker_text = _deepgram_words_to_speaker_lines(words)
        if speaker_text:
            return speaker_text
        return str(_deepgram_value(primary, "transcript", "")).strip()
    except Exception:
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


def analyze_live_input_signal(raw_bytes: bytes) -> dict[str, Any] | None:
    try:
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
    except Exception:
        return None
    if samples.size == 0:
        return None

    normalized = samples / 32768.0
    rms = float(np.sqrt(np.mean(normalized * normalized)))
    peak = float(np.max(np.abs(normalized)))

    if rms < 0.0015:
        state = "silent"
        color = "#F57C00"
        detail = "No meaningful audio is reaching the selected input."
    elif rms < 0.008:
        state = "low"
        color = "#F9A825"
        detail = "Audio is reaching the selected input, but the level is very low."
    elif peak > 0.95:
        state = "clipping"
        color = "#D32F2F"
        detail = "Audio is reaching the selected input, but the signal is clipping."
    else:
        state = "active"
        color = "#66BB6A"
        detail = "Audio is reaching the selected input before Deepgram."

    return {
        "state": state,
        "rms": rms,
        "peak": peak,
        "color": color,
        "detail": detail,
    }


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


def get_default_input_device() -> tuple[int | None, dict[str, Any] | None]:
    try:
        device_info = sd.query_devices(kind="input")
    except Exception:
        return None, None

    if not isinstance(device_info, dict):
        return None, None

    try:
        index = int(device_info.get("index", -1))
    except (TypeError, ValueError):
        index = -1

    if index < 0:
        return None, None
    return index, device_info


def get_default_output_device() -> tuple[int | None, dict[str, Any] | None]:
    try:
        device_info = sd.query_devices(kind="output")
    except Exception:
        return None, None

    if not isinstance(device_info, dict):
        return None, None

    try:
        index = int(device_info.get("index", -1))
    except (TypeError, ValueError):
        index = -1

    if index < 0:
        return None, None
    return index, device_info


def sample_input_signal(device_name: str, sample_rate_hz: int, duration_seconds: float = 0.35) -> dict[str, Any] | None:
    device_index, device_info = resolve_input_device(device_name)
    if device_index is None or device_info is None:
        return None

    actual_sample_rate = int(float(device_info.get("default_samplerate", sample_rate_hz)))
    if actual_sample_rate <= 0:
        actual_sample_rate = int(sample_rate_hz)
    frames = max(1, int(actual_sample_rate * duration_seconds))

    try:
        recording = sd.rec(
            frames,
            samplerate=actual_sample_rate,
            channels=1,
            dtype="float32",
            device=device_index,
        )
        sd.wait()
    except Exception:
        return None

    samples = np.asarray(np.squeeze(recording), dtype=np.float64)
    if samples.size == 0:
        return None

    samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
    samples = np.clip(samples, -1.0, 1.0)
    pcm16 = np.asarray(samples * 32767.0, dtype=np.int16).tobytes()
    signal = analyze_live_input_signal(pcm16)
    if signal is None:
        return None
    signal["device_name"] = normalize_audio_device_name(str(device_info.get("name", device_name)))
    return signal


def sample_resolved_input_signal(
    device_index: int,
    sample_rate_hz: int,
    device_name: str,
    duration_seconds: float = 0.35,
) -> dict[str, Any] | None:
    frames = max(1, int(sample_rate_hz * duration_seconds))

    try:
        recording = sd.rec(
            frames,
            samplerate=sample_rate_hz,
            channels=1,
            dtype="float32",
            device=device_index,
        )
        sd.wait()
    except Exception:
        return None

    samples = np.asarray(np.squeeze(recording), dtype=np.float64)
    if samples.size == 0:
        return None

    samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
    samples = np.clip(samples, -1.0, 1.0)
    pcm16 = np.asarray(samples * 32767.0, dtype=np.int16).tobytes()
    signal = analyze_live_input_signal(pcm16)
    if signal is None:
        return None
    signal["device_name"] = normalize_audio_device_name(device_name)
    return signal


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
    def __init__(self, sample_rate_hz: int, interval_seconds: float, callback, device_provider=None):
        self.sample_rate_hz = sample_rate_hz
        self.interval_seconds = interval_seconds
        self.callback = callback
        self.device_provider = device_provider
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
        requested_device_name = ""

        try:
            active_device = self.device_provider() if callable(self.device_provider) else None
            device_index: int | None = None
            device_info: dict[str, Any] | None = None

            if active_device:
                requested_device_name = str(active_device.get("name", "")).strip()
                device_index = int(active_device["index"])
                device_info = active_device["info"]
                actual_sample_rate = int(active_device["sample_rate"])
            else:
                device_index, device_info = get_default_input_device()
                if device_index is None or device_info is None:
                    raise ValueError("Unable to resolve an input device for monitoring.")
                requested_device_name = str(device_info.get("name", "")).strip()
                actual_sample_rate = int(float(device_info.get("default_samplerate", self.sample_rate_hz)))
            if actual_sample_rate <= 0:
                actual_sample_rate = int(self.sample_rate_hz)
            frames = max(1, int(actual_sample_rate * duration_seconds))

            recording = sd.rec(
                frames,
                samplerate=actual_sample_rate,
                channels=1,
                dtype="float32",
                device=device_index,
            )
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
                "detail_text": f"{requested_device_name or 'Default input'}: {exc}",
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
            "detail_text": (
                f"{detail} Input: {normalize_audio_device_name(str(device_info.get('name', requested_device_name or 'Default input')))}."
            ),
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

        transcript_text = format_deepgram_payload_text(payload)
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
        input_device: ActiveAudioDevice,
        mode_name: str,
        smart_format: bool,
        diarize: bool,
        paragraphs: bool,
        filler_words: bool,
        numerals: bool,
        on_transcript,
        on_status,
        on_signal,
    ):
        self.api_key = api_key.strip()
        self.input_device_name = input_device["name"].strip()
        self.input_device_index = int(input_device["index"])
        self.input_device_info = input_device["info"]
        self.sample_rate_hz = LIVE_TRANSCRIPTION_SAMPLE_RATE_HZ
        self.capture_channels = max(1, min(int(self.input_device_info.get("max_input_channels", 1)), 2))
        self.mode_name = mode_name.strip() or "Unknown"
        self.smart_format = bool(smart_format)
        self.diarize = bool(diarize)
        self.paragraphs = bool(paragraphs)
        self.filler_words = bool(filler_words)
        self.numerals = bool(numerals)
        self.on_transcript = on_transcript
        self.on_status = on_status
        self.on_signal = on_signal
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
        self.callback_count = 0
        self.last_signal_debug_at = 0.0
        self.last_signal_status = ""

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

        if self.input_device_index is None or self.input_device_info is None:
            debug_log(f"[LiveTranscriptionSession] Unable to resolve input device: {self.input_device_name}", level="error")
            return False, f"Unable to find recording device: {self.input_device_name}"

        try:
            from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents
        except Exception as exc:
            return False, f"Deepgram SDK is not available in this Python environment: {exc}"

        self.transcript_path = build_live_transcript_output_path()
        self.metadata_path = build_live_transcript_metadata_path(self.transcript_path)
        deepgram = DeepgramClient(self.api_key)
        connection = deepgram.listen.websocket.v("1")
        self.connection = connection
        connection.on(LiveTranscriptionEvents.Open, self._on_open)
        connection.on(LiveTranscriptionEvents.Transcript, self._on_transcript)
        connection.on(LiveTranscriptionEvents.Error, self._on_error)
        connection.on(LiveTranscriptionEvents.Close, self._on_close)

        requested_live_options = {
            "model": "nova-3",
            "language": "en-US",
            "smart_format": self.smart_format,
            "punctuate": True,
            "interim_results": True,
            "diarize": self.diarize,
            "paragraphs": self.paragraphs,
            "filler_words": self.filler_words,
            "numerals": self.numerals,
            "encoding": "linear16",
            "channels": 1,
            "sample_rate": self.sample_rate_hz,
        }
        supported_names = set(inspect.signature(LiveOptions.__init__).parameters.keys())
        live_options_payload = {
            key: value for key, value in requested_live_options.items() if key in supported_names
        }
        omitted_names = sorted(set(requested_live_options.keys()) - set(live_options_payload.keys()))
        if omitted_names:
            debug_log(
                f"[LiveTranscriptionSession] Live SDK does not support: {', '.join(omitted_names)}. "
                f"Continuing with supported options only.",
                level="warning",
            )

        options = LiveOptions(**live_options_payload)

        if not connection.start(options):
            debug_log("[LiveTranscriptionSession] Deepgram websocket start returned false", level="error")
            return False, "Failed to start Deepgram live transcription connection."

        try:
            sd.check_input_settings(
                device=self.input_device_index,
                samplerate=self.sample_rate_hz,
                channels=self.capture_channels,
            )
            if self.stream is not None:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None

            stream = sd.InputStream(
                samplerate=self.sample_rate_hz,
                blocksize=1024,
                device=self.input_device_index,
                channels=self.capture_channels,
                dtype="float32",
                callback=self._audio_callback,
            )
            self.stream = stream
            stream.start()
        except Exception as exc:
            try:
                connection.finish()
            except Exception:
                pass
            self.connection = None
            debug_log(f"[LiveTranscriptionSession] Failed to open InputStream for {self.input_device_name}: {exc}", level="error")
            return False, f"Failed to open input stream on {self.input_device_name}: {exc}"

        self.running = True
        self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.actual_device_name = normalize_audio_device_name(str(self.input_device_info.get("name", self.input_device_name)))
        debug_log(
            f"[LiveTranscriptionSession] Running mode={self.mode_name} resolved_device={self.actual_device_name} "
            f"device_index={self.input_device_index} sample_rate={self.sample_rate_hz}"
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
        self.callback_count += 1
        if status:
            debug_log(f"[LiveTranscriptionSession] Audio callback status: {status}", level="warning")
            self.on_status(f"Audio stream status: {status}")
        pcm_bytes = self._pcm16_bytes_from_input(indata)
        self._report_input_signal(pcm_bytes)
        try:
            self.connection.send(pcm_bytes)
        except Exception as exc:
            self.error_message = f"Audio send failed: {exc}"
            self.on_status(self.error_message)

    def _pcm16_bytes_from_input(self, indata: np.ndarray) -> bytes:
        samples = np.asarray(indata, dtype=np.float32)
        if samples.ndim == 2:
            samples = np.mean(samples, axis=1)
        samples = np.squeeze(samples)
        if samples.ndim == 0:
            samples = np.asarray([float(samples)], dtype=np.float32)
        samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
        samples = np.clip(samples, -1.0, 1.0)
        pcm16 = (samples * 32767.0).astype(np.int16, copy=False)
        return pcm16.tobytes()

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

        transcript = format_live_result_text(result)
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
            "capture_channels": self.capture_channels,
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

    def _report_input_signal(self, raw_bytes: bytes) -> None:
        now = time.time()
        if now - self.last_signal_debug_at < 2.0:
            return

        signal = analyze_live_input_signal(raw_bytes)
        if signal is None:
            return

        self.last_signal_debug_at = now
        self.on_signal(
            {
                **signal,
                "device_name": self.actual_device_name,
                "mode_name": self.mode_name,
            }
        )

        rms = float(signal["rms"])
        peak = float(signal["peak"])
        state = str(signal["state"])
        if state == "silent":
            status_text = f"Live input looks silent on {self.actual_device_name} (RMS {rms:.4f}, Peak {peak:.4f})"
        elif state == "low":
            status_text = f"Live input is very low on {self.actual_device_name} (RMS {rms:.4f}, Peak {peak:.4f})"
        elif state == "clipping":
            status_text = f"Live input is clipping on {self.actual_device_name} (RMS {rms:.4f}, Peak {peak:.4f})"
        else:
            status_text = f"Live input active on {self.actual_device_name} (RMS {rms:.4f}, Peak {peak:.4f})"

        if status_text != self.last_signal_status:
            self.last_signal_status = status_text
            debug_log(f"[LiveTranscriptionSession] {status_text}")
            self.on_status(status_text)


class App:
    def __init__(self) -> None:
        debug_log("[App] Initializing application")
        self.config = load_config()
        self.detected_input_devices = list_input_devices()
        self.detected_output_devices = list_output_devices()
        self._hydrate_config_from_detected_devices()
        self.device_manager = AudioDeviceManager()
        self.current_mode: ModeName = self.config["last_mode"]
        self.active_audio_device: ActiveAudioDevice | None = None
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
        self.live_signal_status_text = "Waiting to sample the selected input."
        self._vac_test_forced_monitoring = False
        self._pending_vac_test = False
        self._audio_switch_in_progress = False
        self._latest_signal_state = "Unknown"
        self._resume_monitor_after_live = False

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Virtual Audio Control")
        self.root.geometry("920x760")
        self.root.minsize(820, 640)
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
        self.runtime_audio_var = ctk.StringVar(value="Active Input: Detecting... | Active Output: Detecting... | Signal: Unknown")

        self.mic_var = ctk.StringVar(value=self.config["mic_device"])
        self.vac_var = ctk.StringVar(value=self.config["vac_device"])
        self.speaker_var = ctk.StringVar(value=self.config["speaker_device"])
        self.vac_playback_var = ctk.StringVar(value=self.config["vac_playback_device"])
        self.mix_var = ctk.StringVar(value=self.config["voicemeeter_device"])
        self.direct_recording_var = ctk.StringVar(value=self.config["mic_device"])
        self.direct_playback_var = ctk.StringVar(value=self.config["speaker_device"])
        self.wer_enabled_var = ctk.BooleanVar(value=bool(self.config["wer_mode_enabled"]))
        self.resolve_active_device(self._current_live_input_device_name())
        self.monitor = AudioQualityMonitor(
            sample_rate_hz=int(self.config["sample_rate_hz"]),
            interval_seconds=float(self.config["quality_check_interval_seconds"]),
            callback=self._queue_quality_update,
            device_provider=self.get_active_audio_device,
        )

        self._build_ui()
        self._refresh_mode_hint()
        self._refresh_detection_summary()
        self._refresh_runtime_audio_status()

        if self.wer_enabled_var.get():
            self.monitor.start()

    def _build_ui(self) -> None:
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        app_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        app_frame.pack(fill="both", expand=True)
        app_frame.grid_columnconfigure(0, weight=1)
        app_frame.grid_rowconfigure(1, weight=1)

        header_frame = ctk.CTkFrame(app_frame, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(12, 8))
        header_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header_frame,
            text="Audio Control Panel Pro",
            font=("Arial", 20, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            header_frame,
            text="Real-Time WER Optimization for Deepgram",
            font=("Arial", 10),
            text_color="#888888",
            anchor="w",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        self.header_mode_chip = ctk.CTkLabel(
            header_frame,
            text="",
            width=120,
            height=32,
            corner_radius=16,
            font=("Arial", 11, "bold"),
            fg_color="#1F6AA5",
        )
        self.header_mode_chip.grid(row=0, column=1, rowspan=2, sticky="e")

        self.tabview = ctk.CTkTabview(app_frame)
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 8))

        self.dashboard_tab = self._create_tab_scroll_frame("Dashboard")
        self.routing_tab = self._create_tab_scroll_frame("Routing")
        self.transcribe_tab = self._create_tab_scroll_frame("Transcribe")
        self.settings_tab = self._create_tab_scroll_frame("Settings")
        self.advanced_tab = self._create_tab_scroll_frame("Advanced")

        self._build_dashboard_tab()
        self._build_routing_tab()
        self._build_transcribe_tab()
        self._build_settings_tab()
        self._build_advanced_tab()

        status_bar = ctk.CTkFrame(app_frame, corner_radius=10)
        status_bar.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 12))
        status_bar.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(
            status_bar,
            textvariable=self.status_var,
            font=("Arial", 10),
            text_color="#4CAF50",
            anchor="w",
        )
        self.status_label.grid(row=0, column=0, sticky="w", padx=12, pady=(8, 2))

        ctk.CTkLabel(
            status_bar,
            text="Shortcuts: Ctrl+1 Dashboard | Ctrl+2 Routing | Ctrl+3 Transcribe | Ctrl+4 Settings | Ctrl+5 Advanced | F5 Refresh",
            font=("Arial", 9),
            text_color="#A0A0A0",
            anchor="w",
        ).grid(row=1, column=0, sticky="w", padx=12, pady=(0, 2))

        ctk.CTkLabel(
            status_bar,
            textvariable=self.runtime_audio_var,
            font=("Arial", 9),
            text_color="#C6C6C6",
            anchor="w",
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 8))

        ctk.CTkLabel(
            status_bar,
            textvariable=self.footer_var,
            font=("Arial", 8),
            text_color="#555555",
            anchor="e",
        ).grid(row=0, column=1, rowspan=2, sticky="e", padx=12)

        self._bind_shortcuts()

    def _create_tab_scroll_frame(self, tab_name: str) -> ctk.CTkScrollableFrame:
        self.tabview.add(tab_name)
        tab = self.tabview.tab(tab_name)
        tab.grid_columnconfigure(0, weight=1)
        scroll_frame = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll_frame.pack(fill="both", expand=True, padx=8, pady=8)
        scroll_frame.grid_columnconfigure(0, weight=1)
        return scroll_frame

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-1>", lambda event: self._select_tab("Dashboard"))
        self.root.bind("<Control-2>", lambda event: self._select_tab("Routing"))
        self.root.bind("<Control-3>", lambda event: self._select_tab("Transcribe"))
        self.root.bind("<Control-4>", lambda event: self._select_tab("Settings"))
        self.root.bind("<Control-5>", lambda event: self._select_tab("Advanced"))
        self.root.bind("<F5>", lambda event: self.refresh_detected_devices())

    def _select_tab(self, tab_name: str) -> str:
        self.tabview.set(tab_name)
        return "break"

    def _add_section_title(self, parent, title: str, subtitle: str | None = None) -> None:
        ctk.CTkLabel(
            parent,
            text=title,
            font=("Arial", 13, "bold"),
            anchor="w",
        ).pack(anchor="w", padx=12, pady=(10, 2))
        if subtitle:
            ctk.CTkLabel(
                parent,
                text=subtitle,
                font=("Arial", 9),
                text_color="#A9A9A9",
                anchor="w",
                wraplength=760,
                justify="left",
            ).pack(anchor="w", padx=12, pady=(0, 6))

    def _build_dashboard_tab(self) -> None:
        self.dashboard_tab.grid_columnconfigure(0, weight=1)

        mode_frame = ctk.CTkFrame(self.dashboard_tab)
        mode_frame.pack(fill="x", padx=6, pady=(0, 8))

        ctk.CTkLabel(
            mode_frame,
            text="Current Mode",
            font=("Arial", 10),
            text_color="#A9A9A9",
        ).pack(anchor="w", padx=12, pady=(10, 2))

        mode_summary_row = ctk.CTkFrame(mode_frame, fg_color="transparent")
        mode_summary_row.pack(fill="x", padx=12, pady=(0, 6))

        mode_text_frame = ctk.CTkFrame(mode_summary_row, fg_color="transparent")
        mode_text_frame.pack(side="left", fill="both", expand=True)

        self.mode_display = ctk.CTkLabel(
            mode_text_frame,
            text=f"{self.current_mode}",
            font=("Arial", 20, "bold"),
            text_color="#4CAF50",
            anchor="w",
        )
        self.mode_display.pack(anchor="w")

        self.mode_device_label = ctk.CTkLabel(
            mode_text_frame,
            text="",
            text_color="#C6C6C6",
            font=("Arial", 9),
            anchor="w",
            wraplength=500,
            justify="left",
        )
        self.mode_device_label.pack(anchor="w", pady=(1, 0))

        self.mode_badge_label = ctk.CTkLabel(
            mode_summary_row,
            text="",
            width=116,
            height=40,
            corner_radius=12,
            font=("Arial", 10, "bold"),
            fg_color="#2E7D32",
        )
        self.mode_badge_label.pack(side="right", padx=(10, 0))

        info_grid = ctk.CTkFrame(mode_frame, fg_color="transparent")
        info_grid.pack(fill="x", padx=12, pady=(0, 8))
        info_grid.grid_columnconfigure(0, weight=1)
        info_grid.grid_columnconfigure(1, weight=1)

        self.mode_hint_label = ctk.CTkLabel(
            info_grid,
            text="",
            text_color="#8AB4F8",
            font=("Arial", 9),
            wraplength=340,
            justify="left",
            anchor="w",
        )
        self.mode_hint_label.grid(row=0, column=0, sticky="w", padx=(0, 8))

        self.mode_status_label = ctk.CTkLabel(
            info_grid,
            text="",
            text_color="#C6C6C6",
            font=("Arial", 9),
            wraplength=340,
            justify="left",
            anchor="w",
        )
        self.mode_status_label.grid(row=0, column=1, sticky="w", padx=(8, 0))

        self.mode_route_label = ctk.CTkLabel(
            mode_frame,
            text="",
            text_color="#64B5F6",
            font=("Arial", 9, "bold"),
            wraplength=760,
            justify="left",
            anchor="w",
        )
        self.mode_route_label.pack(fill="x", padx=12, pady=(0, 8))

        self.meter = AudioLevelMeter(mode_frame, width=760, height=78)
        self.meter.pack(fill="x", padx=12, pady=(0, 10))

        content_grid = ctk.CTkFrame(self.dashboard_tab, fg_color="transparent")
        content_grid.pack(fill="x", padx=6, pady=(0, 6))
        content_grid.grid_columnconfigure(0, weight=1)
        content_grid.grid_columnconfigure(1, weight=1)

        self.wer_status_frame = ctk.CTkFrame(content_grid, fg_color="#1A1A1A")
        self.wer_status_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        status_header = ctk.CTkFrame(self.wer_status_frame, fg_color="transparent")
        status_header.pack(fill="x", padx=12, pady=(10, 6))

        ctk.CTkLabel(
            status_header,
            text="WER Optimization",
            font=("Arial", 11, "bold"),
        ).pack(side="left")

        self.monitoring_toggle = ctk.CTkSwitch(
            status_header,
            text="Monitor",
            variable=self.wer_enabled_var,
            command=self.toggle_wer_monitoring,
            onvalue=True,
            offvalue=False,
            width=78,
        )
        self.monitoring_toggle.pack(side="right")
        if self.wer_enabled_var.get():
            self.monitoring_toggle.select()

        status_row = ctk.CTkFrame(self.wer_status_frame, fg_color="transparent")
        status_row.pack(fill="x", padx=12, pady=(0, 4))

        ctk.CTkLabel(status_row, text="Quality", font=("Arial", 9), width=52, anchor="w").pack(side="left")
        self.monitor_status_label = ctk.CTkLabel(
            status_row,
            text="Excellent",
            font=("Arial", 9, "bold"),
            text_color="#66BB6A",
        )
        self.monitor_status_label.pack(side="left", padx=(0, 10))

        ctk.CTkLabel(status_row, text="Stability", font=("Arial", 9), width=54, anchor="w").pack(side="left")
        self.monitor_stability_label = ctk.CTkLabel(
            status_row,
            text="Stable",
            font=("Arial", 9, "bold"),
            text_color="#66BB6A",
        )
        self.monitor_stability_label.pack(side="left", padx=(0, 10))

        ctk.CTkLabel(status_row, text="WER", font=("Arial", 9), width=34, anchor="w").pack(side="left")
        self.monitor_wer_label = ctk.CTkLabel(
            status_row,
            text="3-7%",
            font=("Arial", 9, "bold"),
            text_color="#66BB6A",
        )
        self.monitor_wer_label.pack(side="left")

        self.warnings_box = ctk.CTkTextbox(
            self.wer_status_frame,
            height=52,
            font=("Arial", 9),
            fg_color="#121212",
            wrap="word",
        )
        self.warnings_box.pack(fill="x", padx=12, pady=(0, 10))
        self._set_warnings_text(self.monitor_recommendation_var.get())

        controls_frame = ctk.CTkFrame(content_grid)
        controls_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self._add_section_title(controls_frame, "Run Controls")

        mode_buttons = ctk.CTkFrame(controls_frame, fg_color="transparent")
        mode_buttons.pack(fill="x", padx=12, pady=(0, 8))
        for column in range(3):
            mode_buttons.grid_columnconfigure(column, weight=1)

        self.btn_mic = ctk.CTkButton(
            mode_buttons,
            text="Microphone",
            command=lambda: self.apply_audio_mode("Microphone"),
            height=40,
            font=("Arial", 10, "bold"),
        )
        self.btn_mic.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.btn_vac = ctk.CTkButton(
            mode_buttons,
            text="VAC",
            command=lambda: self.apply_audio_mode("VAC"),
            height=40,
            font=("Arial", 10, "bold"),
            fg_color="#2E7D32",
            hover_color="#1B5E20",
        )
        self.btn_vac.grid(row=0, column=1, sticky="ew", padx=4)

        self.btn_mix = ctk.CTkButton(
            mode_buttons,
            text="Mixed",
            command=lambda: self.apply_audio_mode("Mixed"),
            height=40,
            font=("Arial", 10, "bold"),
        )
        self.btn_mix.grid(row=0, column=2, sticky="ew", padx=(4, 0))

        self.dashboard_live_input_label = ctk.CTkLabel(
            controls_frame,
            text="Live input source: " + self._current_live_input_device_name(),
            font=("Arial", 9, "bold"),
            text_color="#8AB4F8",
            anchor="w",
            justify="left",
        )
        self.dashboard_live_input_label.pack(fill="x", padx=12, pady=(0, 8))

        dashboard_live_actions = ctk.CTkFrame(controls_frame, fg_color="transparent")
        dashboard_live_actions.pack(fill="x", padx=12, pady=(0, 10))
        for column in range(3):
            dashboard_live_actions.grid_columnconfigure(column, weight=1)

        self.btn_start_live = ctk.CTkButton(
            dashboard_live_actions,
            text="Start",
            command=self.start_live_transcription,
            height=36,
            font=("Arial", 10, "bold"),
            fg_color="#2E7D32",
            hover_color="#1B5E20",
        )
        self.btn_start_live.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.btn_stop_live = ctk.CTkButton(
            dashboard_live_actions,
            text="Stop",
            command=self.stop_live_transcription,
            height=36,
            font=("Arial", 10, "bold"),
            fg_color="#B71C1C",
            hover_color="#7F1010",
            state="disabled",
        )
        self.btn_stop_live.grid(row=0, column=1, sticky="ew", padx=4)

        self.mute_button = ctk.CTkButton(
            dashboard_live_actions,
            text="Mute",
            command=self.toggle_mute,
            height=36,
            font=("Arial", 10),
            fg_color="#D32F2F",
            hover_color="#B71C1C",
        )
        self.mute_button.grid(row=0, column=2, sticky="ew", padx=(4, 0))

    def _build_routing_tab(self) -> None:
        routing_frame = ctk.CTkFrame(self.routing_tab)
        routing_frame.pack(fill="x", padx=8, pady=(0, 12))
        self._add_section_title(routing_frame, "Windows Audio Routing", "Device setup plus inline VAC route verification.")

        self.direct_recording_menu = self._add_device_selector(
            routing_frame,
            "Recording input",
            self.direct_recording_var,
            self.detected_input_devices,
        )

        direct_recording_actions = ctk.CTkFrame(routing_frame, fg_color="transparent")
        direct_recording_actions.pack(fill="x", padx=10, pady=(0, 8))
        ctk.CTkButton(
            direct_recording_actions,
            text="Set Input",
            command=self.apply_selected_recording_device,
            height=32,
            font=("Arial", 10, "bold"),
        ).pack(side="left", expand=True, fill="x")

        self.direct_playback_menu = self._add_device_selector(
            routing_frame,
            "Playback output",
            self.direct_playback_var,
            self.detected_output_devices,
        )

        direct_playback_actions = ctk.CTkFrame(routing_frame, fg_color="transparent")
        direct_playback_actions.pack(fill="x", padx=10, pady=(0, 8))
        ctk.CTkButton(
            direct_playback_actions,
            text="Set Output",
            command=self.apply_selected_playback_device,
            height=32,
            font=("Arial", 10, "bold"),
        ).pack(side="left", expand=True, fill="x")

        self.route_vac_playback_menu = self._add_device_selector(
            routing_frame,
            "VAC playback target",
            self.vac_playback_var,
            self.detected_output_devices,
        )
        self.route_mix_menu = self._add_device_selector(
            routing_frame,
            "Voicemeeter input",
            self.mix_var,
            self.detected_input_devices,
        )

        preset_row = ctk.CTkFrame(routing_frame, fg_color="transparent")
        preset_row.pack(fill="x", padx=10, pady=(4, 8))

        ctk.CTkButton(
            preset_row,
            text="Normal Preset",
            command=lambda: self.apply_device_preset("normal"),
            height=34,
            font=("Arial", 9, "bold"),
        ).pack(side="left", padx=(0, 6), expand=True, fill="x")

        ctk.CTkButton(
            preset_row,
            text="VAC Preset",
            command=lambda: self.apply_device_preset("vac"),
            height=34,
            font=("Arial", 9, "bold"),
            fg_color="#2E7D32",
            hover_color="#1B5E20",
        ).pack(side="left", padx=6, expand=True, fill="x")

        ctk.CTkButton(
            preset_row,
            text="Voicemeeter Preset",
            command=lambda: self.apply_device_preset("mixed"),
            height=34,
            font=("Arial", 9, "bold"),
        ).pack(side="left", padx=(6, 0), expand=True, fill="x")

        utility_row = ctk.CTkFrame(routing_frame, fg_color="transparent")
        utility_row.pack(fill="x", padx=10, pady=(0, 12))

        self.btn_vac_test = ctk.CTkButton(
            utility_row,
            text="Test VAC Routing",
            command=self.test_vac_routing,
            height=34,
            font=("Arial", 10, "bold"),
            fg_color="#1565C0",
            hover_color="#0D47A1",
        )
        self.btn_vac_test.pack(side="left", padx=(0, 6), expand=True, fill="x")

        ctk.CTkButton(
            utility_row,
            text="Refresh Devices",
            command=self.refresh_detected_devices,
            height=34,
            font=("Arial", 10),
        ).pack(side="left", padx=(6, 0), expand=True, fill="x")

        self.routing_meter = AudioLevelMeter(
            routing_frame,
            width=720,
            height=62,
            title="ROUTING SIGNAL",
            compact=True,
        )
        self.routing_meter.pack(fill="x", padx=12, pady=(0, 10))

    def _build_transcribe_tab(self) -> None:
        api_key_ready = bool(get_deepgram_api_key())

        file_frame = ctk.CTkFrame(self.transcribe_tab)
        file_frame.pack(fill="x", padx=8, pady=(0, 12))
        self._add_section_title(file_frame, "File Transcription")

        self.transcription_status_label = ctk.CTkLabel(
            file_frame,
            text=(
                ("Deepgram API key detected" if api_key_ready else "Deepgram API key missing from .env")
                + "\n"
                + self._deepgram_options_summary()
            ),
            font=("Arial", 10, "bold"),
            text_color="#66BB6A" if api_key_ready else "#F9A825",
            wraplength=760,
            justify="left",
            anchor="w",
        )
        self.transcription_status_label.pack(fill="x", padx=14, pady=(0, 10))

        transcription_actions = ctk.CTkFrame(file_frame, fg_color="transparent")
        transcription_actions.pack(fill="x", padx=12, pady=(0, 12))
        for column in range(2):
            transcription_actions.grid_columnconfigure(column, weight=1)

        self.btn_transcribe_file = ctk.CTkButton(
            transcription_actions,
            text="Transcribe File",
            command=self.transcribe_media_file,
            height=36,
            font=("Arial", 10, "bold"),
            fg_color="#1565C0",
            hover_color="#0D47A1",
        )
        self.btn_transcribe_file.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        ctk.CTkButton(
            transcription_actions,
            text="Open Transcripts Folder",
            command=self.open_transcripts_folder,
            height=36,
            font=("Arial", 10),
        ).grid(row=0, column=1, sticky="ew", padx=(6, 0))

        live_frame = ctk.CTkFrame(self.transcribe_tab)
        live_frame.pack(fill="x", padx=8, pady=(0, 12))
        self._add_section_title(live_frame, "Live Transcription")

        self.live_transcription_device_label = ctk.CTkLabel(
            live_frame,
            text="Live input source: " + self._current_live_input_device_name(),
            font=("Arial", 10, "bold"),
            text_color="#8AB4F8",
            wraplength=760,
            justify="left",
            anchor="w",
        )
        self.live_transcription_device_label.pack(fill="x", padx=14, pady=(0, 6))

        self.live_transcription_key_label = ctk.CTkLabel(
            live_frame,
            text=(
                ("Deepgram live key detected" if api_key_ready else "Deepgram live key missing from .env")
                + "\n"
                + self._deepgram_options_summary()
            ),
            font=("Arial", 10, "bold"),
            text_color="#66BB6A" if api_key_ready else "#F9A825",
            wraplength=760,
            justify="left",
            anchor="w",
        )
        self.live_transcription_key_label.pack(fill="x", padx=14, pady=(0, 6))

        live_actions = ctk.CTkFrame(live_frame, fg_color="transparent")
        live_actions.pack(fill="x", padx=12, pady=(0, 8))
        for column in range(2):
            live_actions.grid_columnconfigure(column, weight=1)

        self.transcribe_btn_start_live = ctk.CTkButton(
            live_actions,
            text="Start Live Transcription",
            command=self.start_live_transcription,
            height=34,
            font=("Arial", 10, "bold"),
            fg_color="#2E7D32",
            hover_color="#1B5E20",
        )
        self.transcribe_btn_start_live.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.transcribe_btn_stop_live = ctk.CTkButton(
            live_actions,
            text="Stop Live Transcription",
            command=self.stop_live_transcription,
            height=34,
            font=("Arial", 10, "bold"),
            fg_color="#B71C1C",
            hover_color="#7F1010",
            state="disabled",
        )
        self.transcribe_btn_stop_live.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        self.live_transcription_status_label = ctk.CTkLabel(
            live_frame,
            text="Idle",
            font=("Arial", 10, "bold"),
            text_color="#C6C6C6",
            wraplength=760,
            justify="left",
            anchor="w",
        )
        self.live_transcription_status_label.pack(fill="x", padx=14, pady=(0, 6))

        self.live_signal_status_label = ctk.CTkLabel(
            live_frame,
            text="Input signal: " + self.live_signal_status_text,
            font=("Arial", 10, "bold"),
            text_color="#C6C6C6",
            wraplength=760,
            justify="left",
            anchor="w",
        )
        self.live_signal_status_label.pack(fill="x", padx=14, pady=(0, 10))

        output_frame = ctk.CTkFrame(self.transcribe_tab)
        output_frame.pack(fill="both", expand=True, padx=8, pady=(0, 12))
        self._add_section_title(output_frame, "Transcript Output")

        self.live_transcript_box = ctk.CTkTextbox(
            output_frame,
            height=260,
            font=("Consolas", 10),
            wrap="word",
        )
        self.live_transcript_box.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        self.live_transcript_box.insert("1.0", "Live transcript will appear here.")
        self.live_transcript_box.configure(state="disabled")

        advanced_frame = ctk.CTkFrame(self.transcribe_tab)
        advanced_frame.pack(fill="x", padx=8, pady=(0, 12))
        self._add_section_title(advanced_frame, "Advanced Options")

        self.deepgram_options_visible = False
        self.deepgram_options_toggle = ctk.CTkButton(
            advanced_frame,
            text="Show Deepgram Options",
            command=self.toggle_deepgram_options,
            height=32,
            font=("Arial", 10, "bold"),
        )
        self.deepgram_options_toggle.pack(fill="x", padx=14, pady=(0, 8))

        self.deepgram_options_frame = ctk.CTkFrame(advanced_frame, fg_color="transparent")
        for label, variable in (
            ("Smart Format", self.deepgram_smart_format_var),
            ("Diarization", self.deepgram_diarize_var),
            ("Paragraphs", self.deepgram_paragraphs_var),
            ("Filler Words", self.deepgram_filler_words_var),
            ("Numerals", self.deepgram_numerals_var),
        ):
            ctk.CTkSwitch(
                self.deepgram_options_frame,
                text=label,
                variable=variable,
                command=self._save_deepgram_options,
            ).pack(anchor="w", padx=2, pady=2)

    def _build_settings_tab(self) -> None:
        settings_frame = ctk.CTkFrame(self.settings_tab)
        settings_frame.pack(fill="x", padx=8, pady=(0, 12))
        self._add_section_title(settings_frame, "Device Defaults", "Saved values used by mode switching and live transcription.")

        self.detected_devices_label = ctk.CTkLabel(
            settings_frame,
            text="Detected input devices: checking...",
            font=("Arial", 10),
            wraplength=760,
            justify="left",
            text_color="#C6C6C6",
            anchor="w",
        )
        self.detected_devices_label.pack(fill="x", padx=14, pady=(0, 10))
        self._refresh_detection_summary()

        self.mic_menu = self._add_device_selector(settings_frame, "Microphone device", self.mic_var, self.detected_input_devices)
        self.vac_menu = self._add_device_selector(settings_frame, "VAC recording device", self.vac_var, self.detected_input_devices)
        self.speaker_menu = self._add_device_selector(settings_frame, "Speaker playback device", self.speaker_var, self.detected_output_devices)
        self.vac_playback_menu = self._add_device_selector(settings_frame, "VAC playback target", self.vac_playback_var, self.detected_output_devices)
        self.mix_menu = self._add_device_selector(settings_frame, "Voicemeeter device", self.mix_var, self.detected_input_devices)

        actions = ctk.CTkFrame(settings_frame, fg_color="transparent")
        actions.pack(fill="x", padx=12, pady=(10, 12))
        for column in range(3):
            actions.grid_columnconfigure(column, weight=1)

        ctk.CTkButton(actions, text="Refresh Devices", command=self.refresh_detected_devices, height=34).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkButton(actions, text="Open config.json", command=self.open_config_file, height=34).grid(row=0, column=1, sticky="ew", padx=6)
        ctk.CTkButton(actions, text="Save Settings", command=self.save_settings, height=34).grid(row=0, column=2, sticky="ew", padx=(6, 0))

        help_frame = ctk.CTkFrame(self.settings_tab)
        help_frame.pack(fill="x", padx=8, pady=(0, 12))
        self._add_section_title(help_frame, "Help")
        ctk.CTkLabel(
            help_frame,
            text=(
                "1. Set Zoom microphone to 'Same as System'.\n"
                "2. Mode buttons change the Windows default recording device for all audio roles.\n"
                "3. VAC mode also switches Windows playback to the configured CABLE Input target.\n"
                "4. Mixed mode expects Voicemeeter routing to be configured before use.\n"
                "5. Use the Advanced tab to run VAC routing diagnostics.\n"
                "6. Use the Dashboard for day-to-day operation and the Routing tab for setup."
            ),
            font=("Arial", 10),
            text_color="#C6C6C6",
            justify="left",
            anchor="w",
            wraplength=760,
        ).pack(fill="x", padx=14, pady=(0, 14))

    def _build_advanced_tab(self) -> None:
        advanced_frame = ctk.CTkFrame(self.advanced_tab)
        advanced_frame.pack(fill="x", padx=8, pady=(0, 12))
        self._add_section_title(advanced_frame, "Diagnostics", "Use this tab for routing checks and low-level troubleshooting.")

        ctk.CTkLabel(
            advanced_frame,
            text="VAC routing verification now lives in the Routing tab so the signal can be confirmed inline with a compact meter.",
            font=("Arial", 9),
            text_color="#C6C6C6",
            justify="left",
            anchor="w",
            wraplength=760,
        ).pack(fill="x", padx=12, pady=(0, 8))

        ctk.CTkButton(
            advanced_frame,
            text="Open App Folder",
            command=self.open_app_folder,
            height=34,
            font=("Arial", 10),
        ).pack(fill="x", padx=12, pady=(0, 8))

        ctk.CTkButton(
            advanced_frame,
            text="Open README",
            command=self.open_readme,
            height=34,
            font=("Arial", 10),
        ).pack(fill="x", padx=12, pady=(0, 12))

    def toggle_deepgram_options(self) -> None:
        self.deepgram_options_visible = not self.deepgram_options_visible
        if self.deepgram_options_visible:
            self.deepgram_options_frame.pack(fill="x", padx=14, pady=(0, 12))
            self.deepgram_options_toggle.configure(text="Hide Deepgram Options")
        else:
            self.deepgram_options_frame.pack_forget()
            self.deepgram_options_toggle.configure(text="Show Deepgram Options")

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

    def _set_meter_levels(self, rms_text: str, peak_text: str, status_text: str, color: str, progress: float) -> None:
        self.meter.set_levels(rms_text, peak_text, status_text, color, progress)
        routing_meter = getattr(self, "routing_meter", None)
        if routing_meter is not None and routing_meter.winfo_exists():
            routing_meter.set_levels(rms_text, peak_text, status_text, color, progress)

    def _refresh_runtime_audio_status(self, *, signal_state: str | None = None) -> None:
        if self.active_audio_device is not None:
            input_name = self.active_audio_device["name"]
        else:
            _input_index, input_info = get_default_input_device()
            input_name = normalize_audio_device_name(str(input_info.get("name", "Unavailable"))) if input_info else "Unavailable"
        _output_index, output_info = get_default_output_device()
        output_name = normalize_audio_device_name(str(output_info.get("name", "Unavailable"))) if output_info else "Unavailable"
        if signal_state is not None:
            self._latest_signal_state = signal_state
        self.runtime_audio_var.set(
            f"Active Input: {input_name} | Active Output: {output_name} | Signal: {self._latest_signal_state}"
        )

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
        self._refresh_runtime_audio_status()
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
            ("route_vac_playback_menu", self.vac_playback_var, self.detected_output_devices),
            ("mix_menu", self.mix_var, self.detected_input_devices),
            ("route_mix_menu", self.mix_var, self.detected_input_devices),
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
        self.tabview.set("Settings")

    def _close_settings_window(self) -> None:
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
        presets: dict[str, ModeName] = {
            "normal": "Microphone",
            "vac": "VAC",
            "mixed": "Mixed",
        }
        self.apply_audio_mode(presets[preset_name])

    def open_config(self) -> None:
        self.tabview.set("Settings")

    def show_help(self) -> None:
        self.tabview.set("Settings")

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

    def get_active_audio_device(self) -> ActiveAudioDevice | None:
        return self.active_audio_device

    def resolve_active_device(self, device_name: str) -> ActiveAudioDevice:
        device_index, device_info = resolve_input_device(device_name)
        if device_index is None or device_info is None:
            raise RuntimeError(f"Failed to resolve device: {device_name}")

        sample_rate = int(float(device_info.get("default_samplerate", self.config["sample_rate_hz"])))
        if sample_rate <= 0:
            sample_rate = int(self.config["sample_rate_hz"])

        self.active_audio_device = {
            "name": normalize_audio_device_name(str(device_info.get("name", device_name))),
            "index": device_index,
            "info": device_info,
            "sample_rate": sample_rate,
        }
        return self.active_audio_device

    def verify_active_device(self) -> bool:
        if self.active_audio_device is None:
            return False
        current_index, _ = resolve_input_device(self.active_audio_device["name"])
        return current_index == self.active_audio_device["index"]

    def _current_live_input_device_name(self) -> str:
        if self.active_audio_device is not None:
            return self.active_audio_device["name"]
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
        dashboard_label = getattr(self, "dashboard_live_input_label", None)
        if dashboard_label is not None:
            dashboard_label.configure(text="Live input source: " + self._current_live_input_device_name())
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

    def _queue_live_signal_update(self, payload: dict[str, Any]) -> None:
        if self._closing:
            return
        try:
            self.root.after(0, lambda: self._apply_live_signal_update(payload))
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

    def _apply_live_signal_update(self, payload: dict[str, Any]) -> None:
        device_name = str(payload.get("device_name", "selected input"))
        rms = float(payload.get("rms", 0.0))
        peak = float(payload.get("peak", 0.0))
        detail = str(payload.get("detail", "")).strip()
        state = str(payload.get("state", "unknown")).strip().lower()
        signal_text = f"Input signal: {detail} Device: {device_name}. RMS {rms:.4f}, Peak {peak:.4f}."
        self.live_signal_status_text = signal_text
        self.live_signal_status_label.configure(
            text=signal_text,
            text_color=str(payload.get("color", "#C6C6C6")),
        )
        signal_state = {
            "silent": "No signal",
            "low": "Low signal",
            "clipping": "Clipping",
            "active": "Active",
        }.get(state, "Unknown")
        self._refresh_runtime_audio_status(signal_state=signal_state)

    def _set_live_controls_state(self, *, running: bool = False, starting: bool = False) -> None:
        self._live_transcription_running = running
        self._live_transcription_starting = starting
        for button_name in ("btn_start_live", "transcribe_btn_start_live"):
            button = getattr(self, button_name, None)
            if button is not None and button.winfo_exists():
                default_text = "Start" if button_name == "btn_start_live" else "Start Live Transcription"
                button.configure(
                    state="disabled" if (running or starting) else "normal",
                    text="Starting..." if starting else default_text,
                )
        for button_name in ("btn_stop_live", "transcribe_btn_stop_live"):
            button = getattr(self, button_name, None)
            if button is not None and button.winfo_exists():
                button.configure(state="normal" if running else "disabled")

    def start_live_transcription(self) -> None:
        if self._audio_switch_in_progress:
            self.status_var.set("Wait for the audio device switch to finish before starting live transcription.")
            return
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

        active_device = self.active_audio_device
        if active_device is None:
            message = "No active audio device is bound. Switch modes or reselect the input device first."
            self.live_transcription_status_label.configure(text=message, text_color="#F57C00")
            self.status_var.set(message)
            messagebox.showerror("Live Input Unavailable", message)
            return

        input_device_name = active_device["name"]
        debug_log(
            f"[App] Starting live transcription mode={self.current_mode} input={input_device_name} "
            f"index={active_device['index']} sample_rate={active_device['sample_rate']}"
        )
        self._resume_monitor_after_live = False
        if self.wer_enabled_var.get():
            self.monitor.stop()
            self._resume_monitor_after_live = True
        self._set_live_controls_state(starting=True)
        self.live_transcript_final_text = ""
        self.live_transcript_interim_text = ""
        self.live_signal_status_text = f"Waiting for audio on {input_device_name} before sending to Deepgram."
        self.live_signal_status_label.configure(text="Input signal: " + self.live_signal_status_text, text_color="#F9A825")
        self._render_live_transcript()
        self.live_transcription_status_label.configure(text="Connecting to Deepgram live transcription...", text_color="#F9A825")
        self.status_var.set(f"Starting live transcription from {input_device_name}...")
        threading.Thread(
            target=self._start_live_transcription_worker,
            args=(
                api_key,
                active_device,
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
        active_device: ActiveAudioDevice,
        mode_name: str,
        smart_format: bool,
        diarize: bool,
        paragraphs: bool,
        filler_words: bool,
        numerals: bool,
    ) -> None:
        input_device_name = active_device["name"]
        if not self.verify_active_device():
            success = False
            message = f"Device mismatch detected before live transcription start: {input_device_name}"
            session = None
            try:
                self.root.after(0, lambda: self._finish_start_live_transcription(success, message, session))
            except Exception:
                return
            return

        signal = sample_resolved_input_signal(
            active_device["index"],
            active_device["sample_rate"],
            active_device["name"],
            duration_seconds=0.25,
        )
        if signal is None:
            success = False
            message = f"Unable to read audio from {input_device_name} before starting live transcription."
            session = None
            try:
                self.root.after(0, lambda: self._finish_start_live_transcription(success, message, session))
            except Exception:
                return
            return
        if str(signal.get("state", "")).lower() == "silent":
            success = False
            message = f"No audio signal detected on {input_device_name}. Check routing before starting live transcription."
            session = None
            try:
                self.root.after(0, lambda: self._finish_start_live_transcription(success, message, session))
            except Exception:
                return
            return

        session = LiveTranscriptionSession(
            api_key=api_key,
            input_device=active_device,
            mode_name=mode_name,
            smart_format=smart_format,
            diarize=diarize,
            paragraphs=paragraphs,
            filler_words=filler_words,
            numerals=numerals,
            on_transcript=self._queue_live_transcript_update,
            on_status=self._queue_live_status_update,
            on_signal=self._queue_live_signal_update,
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

    def _finish_start_live_transcription(
        self,
        success: bool,
        message: str,
        session: LiveTranscriptionSession | None,
    ) -> None:
        if not success:
            debug_log(f"[App] Live transcription failed to start: {message}", level="error")
            self.live_transcription_session = None
            if self._resume_monitor_after_live and self.wer_enabled_var.get():
                self.monitor.start()
            self._resume_monitor_after_live = False
            self._set_live_controls_state(running=False, starting=False)
            self.live_transcription_status_label.configure(text=message, text_color="#F57C00")
            self.live_signal_status_text = "No live input signal available because the session did not start."
            self.live_signal_status_label.configure(text="Input signal: " + self.live_signal_status_text, text_color="#F57C00")
            self._refresh_runtime_audio_status(signal_state="No signal")
            messagebox.showerror("Live Transcription Failed", message)
            self.status_var.set(message)
            return

        self.live_transcription_session = session
        debug_log(f"[App] Live transcription started successfully: {message}")
        self._set_live_controls_state(running=True, starting=False)
        self.live_transcription_status_label.configure(text=message, text_color="#66BB6A")
        self._refresh_runtime_audio_status(signal_state="Active")
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
        self.live_signal_status_text = "Stopped. Start live transcription to sample the selected input again."
        self.live_signal_status_label.configure(text="Input signal: " + self.live_signal_status_text, text_color="#C6C6C6")
        if self._resume_monitor_after_live and self.wer_enabled_var.get():
            self.monitor.start()
        self._resume_monitor_after_live = False
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

    def _resolve_mode_devices(self, mode_name: ModeName) -> tuple[str, str]:
        if mode_name == "Microphone":
            return self.mic_var.get().strip(), self.speaker_var.get().strip()
        if mode_name == "VAC":
            return self.vac_var.get().strip(), self.vac_playback_var.get().strip()
        return self.mix_var.get().strip(), self.speaker_var.get().strip()

    def apply_audio_mode(self, mode_name: ModeName) -> None:
        if self._audio_switch_in_progress:
            self.status_var.set("Audio device switch already in progress.")
            return
        if self._live_transcription_running or self._live_transcription_starting:
            self.status_var.set("Stop live transcription before switching modes.")
            return

        debug_log(f"[App] Applying full audio mode -> {mode_name}")
        self.save_form_config()
        input_device, playback_target = self._resolve_mode_devices(mode_name)
        if not input_device:
            self.status_var.set(f"No input device configured for {mode_name} mode.")
            return
        if not playback_target:
            self.status_var.set(f"No playback device configured for {mode_name} mode.")
            return
        self._audio_switch_in_progress = True
        self.status_var.set(f"Switching audio device... {mode_name} mode requested.")
        threading.Thread(
            target=self._apply_audio_mode_worker,
            args=(mode_name, input_device, playback_target),
            daemon=True,
        ).start()

    def _apply_audio_mode_worker(self, mode_name: ModeName, input_device: str, playback_target: str) -> None:
        resolved_device: ActiveAudioDevice | None = None
        try:
            resolved_device = self.resolve_active_device(input_device)
        except Exception as exc:
            success = False
            message = str(exc)
        else:
            ok_record, record_message = self.device_manager.set_default_recording_device(input_device)
            if not ok_record:
                success = False
                message = record_message
            else:
                ok_playback, playback_message = self.device_manager.set_default_playback_device(playback_target)
                if not ok_playback:
                    success = False
                    message = f"{record_message} Playback switch failed: {playback_message}"
                elif not self._wait_for_active_input_device(resolved_device["name"]):
                    success = False
                    message = f"Switched Windows defaults, but the active input did not become {resolved_device['name']} in time."
                else:
                    success = True
                    message = f"{mode_name} mode active | Input: {resolved_device['name']} | Output: {playback_target}"
        if success and resolved_device is None:
            success = False
            message = f"Unable to resolve an active audio device for {mode_name} mode."

        if self._closing:
            return
        try:
            self.root.after(
                0,
                lambda: self._finish_apply_audio_mode(success, mode_name, resolved_device, playback_target, message),
            )
        except Exception:
            return

    def _finish_apply_audio_mode(
        self,
        success: bool,
        mode_name: ModeName,
        resolved_device: ActiveAudioDevice | None,
        playback_target: str,
        message: str,
    ) -> None:
        self._audio_switch_in_progress = False
        if not success or resolved_device is None:
            self._pending_vac_test = False
            self._refresh_runtime_audio_status()
            self.status_var.set(message)
            return

        self.active_audio_device = resolved_device
        if not self.verify_active_device():
            self._pending_vac_test = False
            self._refresh_runtime_audio_status()
            self.status_var.set("Device mismatch detected after switching modes.")
            return

        self.current_mode = mode_name
        self.config["last_mode"] = mode_name
        save_config(self.config)
        self.mode_var.set(mode_name)
        self.direct_recording_var.set(resolved_device["name"])
        self.direct_playback_var.set(playback_target)
        self._refresh_mode_hint()
        self._refresh_runtime_audio_status()
        self.status_var.set(
            f"{mode_name} mode active | Input: {resolved_device['name']} | Output: {playback_target} | Signal: {self._latest_signal_state}"
        )
        debug_log(
            f"[App] Mode applied successfully -> {mode_name} | input={resolved_device['name']} | "
            f"index={resolved_device['index']} | sample_rate={resolved_device['sample_rate']} | output={playback_target}"
        )
        if self._pending_vac_test and mode_name == "VAC":
            self._pending_vac_test = False
            self._start_verified_vac_test()

    def _wait_for_active_input_device(self, expected_device_name: str, timeout_seconds: float = 1.5) -> bool:
        normalized_expected = normalize_audio_device_name(expected_device_name)
        deadline = time.time() + timeout_seconds

        while time.time() < deadline:
            _index, device_info = get_default_input_device()
            if device_info is not None:
                active_name = normalize_audio_device_name(str(device_info.get("name", "")))
                if active_name == normalized_expected:
                    return True
            time.sleep(0.1)
        return False

    def test_vac_routing(self) -> None:
        if self._vac_test_running:
            self.status_var.set("VAC routing test is already running.")
            return

        debug_log("[App] Starting VAC routing test")
        self._pending_vac_test = True
        self.apply_audio_mode("VAC")

    def _start_verified_vac_test(self) -> None:
        self.tabview.set("Routing")

        self._vac_test_forced_monitoring = False
        if not self.wer_enabled_var.get():
            self._vac_test_forced_monitoring = True
            self.wer_enabled_var.set(True)
            self.toggle_wer_monitoring()

        self._vac_test_running = True
        self.btn_vac_test.configure(state="disabled", text="Testing VAC...")
        self.status_var.set("VAC routing test started. Watch the inline Routing meter for movement while the tone plays through CABLE Input.")
        threading.Thread(target=self._run_vac_test_tone, daemon=True).start()

    def _run_vac_test_tone(self) -> None:
        sample_rate = 48000
        duration_seconds = 1.2
        amplitude = 0.18
        frequency_hz = 880.0
        active_device = self.active_audio_device

        try:
            timeline = np.linspace(0.0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
            envelope = np.minimum(1.0, timeline * 6.0) * np.minimum(1.0, (duration_seconds - timeline) * 6.0)
            tone = (np.sin(2 * np.pi * frequency_hz * timeline) * envelope * amplitude).astype(np.float32)
            sd.play(tone, samplerate=sample_rate, blocking=False)
            time.sleep(0.15)
            signal = None
            if active_device is not None:
                signal = sample_resolved_input_signal(
                    active_device["index"],
                    active_device["sample_rate"],
                    active_device["name"],
                    duration_seconds=0.35,
                )
            sd.wait()
            if signal is None:
                result_message = "VAC routing test completed, but the app could not verify input signal on the selected VAC device."
            else:
                state = str(signal.get("state", "")).lower()
                if state == "silent":
                    result_message = "VAC routing test finished, but no audio was detected on the selected input. Do not assume routing is working."
                elif state == "low":
                    result_message = "VAC routing test finished with only low signal on the selected input. Routing may be weak or misconfigured."
                else:
                    result_message = "VAC routing test completed with confirmed signal on the selected input."
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
        if self._vac_test_forced_monitoring:
            self._vac_test_forced_monitoring = False
            self.wer_enabled_var.set(False)
            self.toggle_wer_monitoring()
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
            self.mute_button.configure(text="Mute", fg_color="#d32f2f", hover_color="#b71c1c")
        self.status_var.set(message)

    def toggle_wer_monitoring(self) -> None:
        enabled = bool(self.wer_enabled_var.get())
        self.config["wer_mode_enabled"] = enabled
        save_config(self.config)

        if enabled:
            if self._live_transcription_running or self._live_transcription_starting:
                self._resume_monitor_after_live = True
            else:
                self.monitor.start()
            self.monitor_status_var.set("Excellent")
            self.monitor_level_var.set("Listening for input...")
            self.monitor_detail_var.set("Sampling the current default recording device.")
            self.monitor_recommendation_var.set("Monitoring enabled\nListening for healthy speech levels")
            self._set_warnings_text(self.monitor_recommendation_var.get())
            self.monitor_status_label.configure(text="Starting", text_color="#8AB4F8")
            self.monitor_stability_label.configure(text="Sampling", text_color="#8AB4F8")
            self.monitor_wer_label.configure(text="...", text_color="#8AB4F8")
            self.mode_badge_label.configure(fg_color="#8AB4F8")
            self._set_meter_levels("RMS: sampling", "Peak: sampling", "Starting", "#8AB4F8", 0.0)
            self._refresh_runtime_audio_status(signal_state="Sampling")
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
            self.mode_badge_label.configure(fg_color="#4A4A4A")
            self._set_meter_levels("RMS: -∞ dB", "Peak: -∞ dB", "Paused", "#9E9E9E", 0.0)
            self._refresh_runtime_audio_status(signal_state="Monitoring off")
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
        ui_config = MODE_UI.get(self.current_mode, MODE_UI["Microphone"])
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
        signal_state = {
            "excellent": "Active",
            "good": "Active",
            "low": "Low signal",
            "too_quiet": "No signal",
            "clipping": "Clipping",
            "error": "Unavailable",
        }.get(quality, "Unknown")
        self._set_meter_levels(rms_text, peak_text, status_text, color, progress)
        self.monitor_recommendation_var.set(self._recommendation_text(quality, result["detail_text"]))
        self.mode_badge_label.configure(text=ui_config["badge"], fg_color=color)
        self._refresh_runtime_audio_status(signal_state=signal_state)
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
        if hasattr(self, "header_mode_chip") and self.header_mode_chip.winfo_exists():
            self.header_mode_chip.configure(text=self.current_mode, fg_color=ui_config["accent"])
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
