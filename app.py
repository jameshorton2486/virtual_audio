from __future__ import annotations

import json
import inspect
import logging
import math
import mimetypes
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any, Callable, Final, Literal, Mapping, TypedDict


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

from audio.auto_audio_engine import AutoAudioEngine
from audio.detector import NULL_INPUT_DEVICE_INDEX, NULL_INPUT_DEVICE_NAME
from audio.engine import AudioEngine
from audio.processor import AudioProcessor
from audio.routing import RoutingManager
from meter_widget import AudioLevelMeter


CONFIG_PATH = APP_DIR / "config.json"
NIRCMD_PATH = APP_DIR / "nircmd.exe"
SOUNDVOLUMEVIEW_PATH = APP_DIR / "SoundVolumeView.exe"
TRANSCRIPTS_DIR = APP_DIR / "transcripts"
LOGS_DIR = APP_DIR / "logs"
LOG_PATH = LOGS_DIR / "virtual_audio.log"
ERROR_LOG_PATH = LOGS_DIR / "errors.log"
DEEPGRAM_LISTEN_URL = "https://api.deepgram.com/v1/listen"
LIVE_TRANSCRIPTION_SAMPLE_RATE_HZ = 16000
LIVE_PCM_QUEUE_MAX_BLOCKS = 60
DEEPGRAM_KEEPALIVE_SECONDS = 5.0
DEEPGRAM_KEEPALIVE_IDLE_THRESHOLD = 3.0
RECONNECT_BACKOFF_SECONDS = (0.5, 1.0, 2.0, 4.0, 8.0, 8.0)
AUDIO_CALLBACK_STALL_SECONDS = 5.0
DEEPGRAM_FINISH_TIMEOUT_SECONDS = 1.5
PARTIAL_TRANSCRIPT_DEBOUNCE_SECONDS = 0.5
DROPPED_BLOCKS_LOG_EVERY = 50
PREFLIGHT_SIGNAL_THRESHOLD_RMS = 0.001
DEVICE_VERIFY_INITIAL_TIMEOUT_SECONDS = 1.5
DEVICE_VERIFY_EXTENDED_TIMEOUT_SECONDS = 5.0
RMS_DB_SILENCE_FLOOR = -45.0
PREFLIGHT_MIN_SIGNAL_DB = -80.0
RMS_DB_TOO_QUIET = -25.0
RMS_DB_OPTIMAL_MAX = -12.0
PEAK_DB_CLIP_WARNING = -3.0
PEAK_DB_CLIP_HARD = -0.1

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
ModeKey = Literal["mic", "vac", "mixed"]


class ModeDeviceConfig(TypedDict):
    input_device: str
    output_device: str


class ConfigDict(TypedDict):
    mic_device: str
    vac_device: str
    speaker_device: str
    vac_playback_device: str
    voicemeeter_device: str
    mixed_playback_device: str
    restore_devices_on_exit: bool
    require_signal_check: bool
    wer_mode_enabled: bool
    quality_check_interval_seconds: float
    sample_rate_hz: int
    last_mode: ModeName
    active_mode: ModeKey
    modes: dict[ModeKey, ModeDeviceConfig]


class ActiveAudioDevice(TypedDict):
    name: str
    index: int
    info: dict[str, Any]
    sample_rate: int


class AudioState:
    def __init__(self) -> None:
        self.selected_device_name: str | None = None
        self.resolved_device_index: int | None = None
        self.locked = False


class DeviceVerificationResult(str, Enum):
    CONFIRMED = "confirmed"
    EVENTUALLY_CONFIRMED = "eventually_confirmed"
    TIMED_OUT = "timed_out"
    DEVICE_UNAVAILABLE = "device_unavailable"


class ModeSwitchOutcome(str, Enum):
    SUCCESS = "success"
    SUCCESS_WITH_WARNING = "success_with_warning"
    HARD_FAILURE = "hard_failure"


class LiveSessionState(str, Enum):
    CONNECTING = "CONNECTING"
    RUNNING = "RUNNING"
    RECONNECTING = "RECONNECTING"
    STOPPED = "STOPPED"


MODE_KEY_BY_NAME: Final[dict[ModeName, ModeKey]] = {
    "Microphone": "mic",
    "VAC": "vac",
    "Mixed": "mixed",
}

MODE_NAME_BY_KEY: Final[dict[ModeKey, ModeName]] = {
    "mic": "Microphone",
    "vac": "VAC",
    "mixed": "Mixed",
}

DEFAULT_MODE_CONFIGS: Final[dict[ModeKey, ModeDeviceConfig]] = {
    "mic": {
        "input_device": "Microphone (Realtek Audio)",
        "output_device": "Speakers (Realtek Audio)",
    },
    "vac": {
        "input_device": "CABLE Output (VB-Audio Virtual Cable)",
        "output_device": "CABLE Input (VB-Audio Virtual Cable)",
    },
    "mixed": {
        "input_device": "VoiceMeeter Output (VB-Audio VoiceMeeter VAIO)",
        "output_device": "",
    },
}

DEFAULT_CONFIG: Final[ConfigDict] = {
    "mic_device": "Microphone (Realtek Audio)",
    "vac_device": "CABLE Output (VB-Audio Virtual Cable)",
    "speaker_device": "Speakers (Realtek Audio)",
    "vac_playback_device": "CABLE Input (VB-Audio Virtual Cable)",
    "voicemeeter_device": "VoiceMeeter Output (VB-Audio VoiceMeeter VAIO)",
    "mixed_playback_device": "",
    "restore_devices_on_exit": True,
    "require_signal_check": True,
    "wer_mode_enabled": True,
    "quality_check_interval_seconds": 2.0,
    "sample_rate_hz": 24000,
    "last_mode": "Microphone",
    "active_mode": "mic",
    "modes": {
        "mic": dict(DEFAULT_MODE_CONFIGS["mic"]),
        "vac": dict(DEFAULT_MODE_CONFIGS["vac"]),
        "mixed": dict(DEFAULT_MODE_CONFIGS["mixed"]),
    },
}

DEEPGRAM_BASE_CONFIG: Final[dict[str, Any]] = {
    "model": "nova-3",
    "diarize": True,
    "punctuate": True,
    "smart_format": False,
    "paragraphs": False,
    "utterances": True,
    "filler_words": True,
    "numerals": False,
    "utt_split": 0.8,
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
    "Mixed": "Mixed loopback audio",
}

MODE_STATUS = {
    "Microphone": "Sets the Windows default recording device to your microphone and restores playback to your speakers.",
    "VAC": "Sets recording to CABLE Output and playback to CABLE Input so system audio is routed through the virtual cable.",
    "Mixed": "Uses a mixed-capable input such as VB Cable, Stereo Mix, or Voicemeeter and restores playback to your speakers.",
}

MODE_UI = {
    "Microphone": {
        "label": "Microphone",
        "title": "Microphone Mode",
        "badge": "WER 10-15%",
        "accent": "#1565C0",
        "hover": "#0D47A1",
        "text_color": "#42A5F5",
        "route": "Voice -> Microphone -> Windows recording",
    },
    "VAC": {
        "label": "VAC",
        "title": "Virtual Audio Cable Mode",
        "badge": "WER 3-7% BEST",
        "accent": "#2E7D32",
        "hover": "#1B5E20",
        "text_color": "#66BB6A",
        "route": "Playback -> CABLE Input -> CABLE Output -> Recording",
    },
    "Mixed": {
        "label": "Mixed",
        "title": "Mixed Mode",
        "badge": "WER 5-10%",
        "accent": "#8E24AA",
        "hover": "#6A1B9A",
        "text_color": "#BA68C8",
        "route": "Mic/system loopback -> mixed-capable input -> Recording",
    },
}


def _quote_log_value(value: Any) -> str:
    text = str(value)
    if not text:
        return "''"
    if any(char.isspace() for char in text):
        return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"
    return text


def _log_message(tag: str, **fields: Any) -> str:
    parts = [f"{key}={_quote_log_value(value)}" for key, value in fields.items()]
    return f"[{tag}]" + (f" {' '.join(parts)}" if parts else "")


def log_event(tag: str, *, level: str = "info", **fields: Any) -> None:
    debug_log(_log_message(tag, **fields), level=level)


def log_failure(code: str, *, level: str = "error", **fields: Any) -> None:
    debug_log(_log_message(f"Failure: {code}", **fields), level=level)


def reset_logs_on_startup(archive: bool = True) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for path in (LOG_PATH, ERROR_LOG_PATH):
        try:
            if not path.exists():
                continue
            if archive:
                archived = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
                path.replace(archived)
            else:
                path.unlink()
        except OSError as exc:
            print(f"[LogReset] Failed to reset {path}: {exc}")


def log_run_header(config: Mapping[str, Any] | None = None) -> None:
    try:
        separator = "=" * 60
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        LOGGER.info(separator)
        LOGGER.info(" NEW APPLICATION RUN")
        LOGGER.info("Time: %s", now)
        LOGGER.info("Version: %s", "3.0")
        LOGGER.info("Python: %s", sys.version.split()[0])
        LOGGER.info("Platform: %s", sys.platform)
        if config is not None:
            LOGGER.info("Last Mode: %s", config.get("last_mode"))
            LOGGER.info("Sample Rate: %s", config.get("sample_rate_hz"))
            LOGGER.info("WER Mode: %s", config.get("wer_mode_enabled"))
        LOGGER.info(separator)
    except Exception as exc:
        print(f"[LogHeader] Failed to write run header: {exc}")


def log_section(title: str) -> None:
    try:
        LOGGER.info("--- %s ---", str(title or "").strip().upper())
    except Exception as exc:
        print(f"[LogSection] Failed to write section {title!r}: {exc}")


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("virtual_audio")
    if logger.handlers:
        return logger

    debug_enabled = os.environ.get("VIRTUAL_AUDIO_DEBUG", "").strip() == "1"
    logger.setLevel(logging.DEBUG if debug_enabled else logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(threadName)s | %(message)s")

    file_handler = RotatingFileHandler(LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG if debug_enabled else logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    error_handler = RotatingFileHandler(ERROR_LOG_PATH, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8")
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG if debug_enabled else (logging.WARNING if getattr(sys, "frozen", False) else logging.INFO))
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


reset_logs_on_startup(archive=True)
LOGGER = setup_logging()


class UILogHandler(logging.Handler):
    def __init__(self, enqueue_fn, watchdog_fn: Callable[[str], None] | None = None, stalled_fn: Callable[[], None] | None = None):
        super().__init__()
        self._enqueue_fn = enqueue_fn
        self._watchdog_fn = watchdog_fn
        self._stalled_fn = stalled_fn

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            if self._watchdog_fn is not None and "audio_callback_stalled" in message:
                self._watchdog_fn("stalled")
                if self._stalled_fn is not None:
                    self._stalled_fn()
            self._enqueue_fn(message, record.levelno)
        except Exception:
            self.handleError(record)


class AutoFixEngine:
    def __init__(self, app: "App", cooldown_seconds: float = 8.0):
        self.app = app
        self.cooldown_seconds = cooldown_seconds
        self.safe_mode = True
        self.last_suggestion: tuple[str, Callable[[], None] | None] | None = None
        self._suggestion_time = 0.0
        self._suggestion_lock = threading.Lock()
        self._last_action_at: dict[str, float] = {}
        self._lock = threading.Lock()
        self._active_fix = False
        self._high_queue_count = 0

    def on_signal(self, speech_state: str, rms_db: float, peak_db: float) -> None:
        if speech_state == "no_signal":
            self._execute_fix(
                key="no_signal",
                description=f"No signal on {self.app.current_mode} -> switch to a fallback input mode",
                fix_fn=self._fix_no_signal,
                is_hard=False,
            )
        elif speech_state == "too_quiet":
            self._execute_fix(
                key="too_quiet",
                description=f"Signal is quiet (RMS {rms_db:.1f} dB) -> increase source volume or mic gain",
                fix_fn=lambda: self._log(f"AutoFix: signal is quiet (RMS {rms_db:.1f} dB). Increase source volume or mic gain.", "warning"),
                is_hard=False,
            )
        elif speech_state in {"clipping", "too_loud"}:
            self._execute_fix(
                key="too_loud",
                description=f"Signal is hot (Peak {peak_db:.1f} dB) -> reduce source level",
                fix_fn=lambda: self._log(f"AutoFix: signal is hot (Peak {peak_db:.1f} dB). Reduce source level to avoid clipping.", "warning"),
                is_hard=False,
            )

    def on_queue(self, size: int) -> None:
        queue_threshold = max(40, int(LIVE_PCM_QUEUE_MAX_BLOCKS * 0.67))
        if size >= queue_threshold:
            self._high_queue_count += 1
        else:
            self._high_queue_count = 0
        if self._high_queue_count >= 3:
            if not self._run_recovery_pipeline("queue_overflow"):
                self._execute_fix(
                    key="queue_overflow",
                    description=f"Queue pressure detected (size={size}) -> restart live transcription",
                    fix_fn=lambda: self._restart_live_session("queue pressure"),
                    is_hard=True,
                )
            self._high_queue_count = 0

    def on_watchdog_stalled(self) -> None:
        if not self._run_recovery_pipeline("stalled"):
            self._execute_fix(
                key="stalled",
                description="Audio callback stalled -> restart live transcription",
                fix_fn=lambda: self._restart_live_session("audio callback stalled"),
                is_hard=True,
            )

    def on_bad_routing(self, mode: str, requested: str, resolved: str) -> None:
        normalized = normalize_audio_device_name(resolved).lower()
        if mode == "Mixed" and resolved and is_valid_mixed_input_device(normalized):
            return
        if resolved and requested and normalize_audio_device_name(requested).lower() == normalized:
            return
        self._execute_fix(
            key="bad_routing",
            description=f"Bad routing in {mode} -> switch to a valid fallback mode",
            fix_fn=lambda: self._fix_bad_routing(mode, requested, resolved),
            is_hard=False,
        )

    def set_safe_mode(self, enabled: bool) -> None:
        self.safe_mode = bool(enabled)

    def _is_live_active(self) -> bool:
        return bool(
            not getattr(self.app, "_closing", False)
            and (
                getattr(self.app, "_live_transcription_running", False)
                or getattr(self.app, "_live_transcription_starting", False)
                or getattr(self.app, "is_transcribing", False)
            )
        )

    def _can_run(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            if self._active_fix:
                return False
            last = self._last_action_at.get(key, 0.0)
            if now - last < self.cooldown_seconds:
                return False
            self._last_action_at[key] = now
            self._active_fix = True
        return True

    def _finish_run(self) -> None:
        with self._lock:
            self._active_fix = False

    def _execute_fix(self, key: str, description: str, fix_fn: Callable[[], None], *, is_hard: bool) -> None:
        if not self._can_run(key):
            return
        try:
            if self.safe_mode and is_hard:
                self._suggest_fix(description, fix_fn=fix_fn, is_hard=is_hard)
                return
            self._log(f"AutoFix: {description}", "warning" if is_hard else "info")
            fix_fn()
        except Exception as exc:
            self._log(f"AutoFix error for {key}: {exc}", "error")
        finally:
            self._finish_run()

    def _run_on_ui_thread(self, fn: Callable[[], None]) -> None:
        root = getattr(self.app, "root", None)
        if root is None or not root.winfo_exists():
            return
        root.after(0, fn)

    def _current_pcm_queue_size(self) -> int:
        session = getattr(self.app, "live_transcription_session", None)
        if session is None:
            return 0
        queue_ref = getattr(session, "_pcm_queue", None)
        if queue_ref is None:
            return 0
        try:
            return int(queue_ref.qsize())
        except Exception as exc:
            self._log(f"Unable to read PCM queue size: {exc}", "warning")
            return 0

    def _is_fix_still_valid(self, description: str, age_seconds: float) -> bool:
        if age_seconds > 10.0:
            return False
        lowered = description.lower()
        if "restart live transcription" in lowered and not self._is_live_active():
            return False
        return True

    def _clear_suggestion(self) -> None:
        with self._suggestion_lock:
            self.last_suggestion = None
            self._suggestion_time = 0.0
        if hasattr(self.app, "on_autofix_cleared"):
            self.app.on_autofix_cleared()

    def _run_recovery_pipeline(self, issue: str) -> bool:
        if not self._is_live_active():
            return False
        steps = self._get_recovery_steps(issue)
        if not steps:
            return False
        self._log(f"Starting recovery pipeline for {issue}", "warning")
        for step_name, step_fn in steps:
            try:
                self._log(f"Trying recovery step: {step_name}", "info")
                step_fn()
                time.sleep(0.3)
                if self._check_if_recovered(issue):
                    self._log(f"Recovered using: {step_name}", "info")
                    return True
            except Exception as exc:
                self._log(f"Recovery step failed: {step_name} ({exc})", "error")
        self._log(f"Recovery pipeline did not resolve {issue}", "warning")
        return False

    def _get_recovery_steps(self, issue: str) -> list[tuple[str, Callable[[], None]]]:
        if issue == "no_signal":
            return [("Switch to fallback input mode", self._fix_no_signal)]
        if issue == "queue_overflow":
            return [("Allow queue to drain", lambda: None)]
        if issue == "stalled":
            return [("Wait briefly for callback recovery", lambda: None)]
        return []

    def _check_if_recovered(self, issue: str) -> bool:
        if issue == "no_signal":
            return str(getattr(self.app, "_latest_signal_state", "")).strip().lower() not in {"no signal", "unknown"}
        if issue == "queue_overflow":
            return self._current_pcm_queue_size() < 20
        if issue == "stalled":
            watchdog_state = str(getattr(self.app, "debug_watchdog_var", "").get() if hasattr(getattr(self.app, "debug_watchdog_var", None), "get") else getattr(self.app, "debug_watchdog_var", ""))
            return "stalled" not in watchdog_state.lower()
        return False

    def _fix_no_signal(self) -> None:
        if not self._is_live_active():
            self._log("AutoFix: no signal detected. Verify routing or source playback.", "warning")
            return
        fallback_mode = self.app._pick_fallback_mode(self.app.current_mode)
        if fallback_mode is None or fallback_mode == self.app.current_mode:
            self._log("AutoFix: no fallback mode available for signal recovery.", "warning")
            return
        self._log(f"AutoFix: no signal on {self.app.current_mode}. Switching to {fallback_mode}.", "warning")
        self._run_on_ui_thread(lambda: self.app.apply_audio_mode(fallback_mode))

    def _fix_bad_routing(self, mode: str, requested: str, resolved: str) -> None:
        if not self._is_live_active():
            self._log(f"AutoFix: bad routing detected in {mode}, but live transcription is not active.", "warning")
            return
        fallback_mode = self.app._pick_fallback_mode(mode)
        if fallback_mode is None or fallback_mode == self.app.current_mode:
            self._log(
                f"AutoFix: bad routing in {mode} (requested={requested or '-'}, resolved={resolved or '-'}), but no fallback mode is available.",
                "warning",
            )
            return
        self._log(
            f"AutoFix: bad routing in {mode} (requested={requested or '-'}, resolved={resolved or '-'}). Switching to {fallback_mode}.",
            "warning",
        )
        self._run_on_ui_thread(lambda: self.app.apply_audio_mode(fallback_mode))

    def _restart_live_session(self, reason: str) -> None:
        if not self._is_live_active():
            self._log(f"AutoFix: detected {reason}, but live transcription is not active.", "warning")
            return

        def _restart() -> None:
            if self.app._closing:
                return
            self._log(f"AutoFix: restarting live transcription after {reason}.", "warning")
            self.app.stop_live_transcription()
            self.app.root.after(900, self.app.start_live_transcription)

        self._run_on_ui_thread(_restart)

    def _suggest_fix(self, description: str, *, fix_fn: Callable[[], None] | None = None, is_hard: bool = False) -> None:
        prefix = "HARD FIX" if is_hard else "SOFT FIX"
        self._log(f"{prefix}: {description}", "warning")
        with self._suggestion_lock:
            self.last_suggestion = (description, fix_fn)
            self._suggestion_time = time.time()
        if hasattr(self.app, "on_autofix_suggestion"):
            self.app.on_autofix_suggestion(description)

    def apply_last_suggestion(self) -> bool:
        with self._suggestion_lock:
            if self.last_suggestion is None:
                return False
            description, fix_fn = self.last_suggestion
            suggestion_age = time.time() - self._suggestion_time
        if not self._is_fix_still_valid(description, suggestion_age):
            self._log("Suggested fix is stale or no longer valid", "warning")
            self._clear_suggestion()
            return False
        with self._lock:
            if self._active_fix:
                return False
            self._active_fix = True
        self._log(f"Applying suggested fix: {description}", "warning")
        try:
            if fix_fn is not None:
                fix_fn()
        except Exception as exc:
            self._log(f"Suggested fix failed: {exc}", "error")
            return False
        finally:
            with self._lock:
                self._active_fix = False
        self._clear_suggestion()
        return True

    def _log(self, message: str, level: str = "info") -> None:
        log_fn = getattr(LOGGER, level, LOGGER.info)
        log_fn(message)


def debug_log(message: str, level: str = "info") -> None:
    log_fn = getattr(LOGGER, level, LOGGER.info)
    log_fn(message)


def _log_uncaught_exception(tag: str, exc_type, exc_value, exc_traceback, **fields: Any) -> None:
    LOGGER.critical(
        _log_message(tag, **fields),
        exc_info=(exc_type, exc_value, exc_traceback),
    )


def install_global_exception_hooks() -> None:
    def _sys_excepthook(exc_type, exc_value, exc_traceback) -> None:
        _log_uncaught_exception(
            "TkCallback",
            exc_type,
            exc_value,
            exc_traceback,
            event="sys_excepthook",
            reason=str(exc_value or exc_type.__name__),
        )

    def _thread_excepthook(args) -> None:
        _log_uncaught_exception(
            "TkCallback",
            args.exc_type,
            args.exc_value,
            args.exc_traceback,
            event="threading_excepthook",
            thread_name=getattr(args.thread, "name", "unknown"),
            reason=str(args.exc_value or args.exc_type.__name__),
        )

    sys.excepthook = _sys_excepthook
    threading.excepthook = _thread_excepthook


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
    sanitized: ConfigDict = {
        **DEFAULT_CONFIG,
        "modes": {
            "mic": dict(DEFAULT_MODE_CONFIGS["mic"]),
            "vac": dict(DEFAULT_MODE_CONFIGS["vac"]),
            "mixed": dict(DEFAULT_MODE_CONFIGS["mixed"]),
        },
    }
    sanitized["mic_device"] = config.get("mic_device", sanitized["mic_device"])
    sanitized["vac_device"] = config.get("vac_device", sanitized["vac_device"])
    sanitized["speaker_device"] = config.get("speaker_device", sanitized["speaker_device"])
    sanitized["vac_playback_device"] = config.get("vac_playback_device", sanitized["vac_playback_device"])
    sanitized["voicemeeter_device"] = config.get("voicemeeter_device", sanitized["voicemeeter_device"])
    sanitized["mixed_playback_device"] = config.get("mixed_playback_device", sanitized["mixed_playback_device"])
    sanitized["restore_devices_on_exit"] = config.get("restore_devices_on_exit", sanitized["restore_devices_on_exit"])
    sanitized["wer_mode_enabled"] = config.get("wer_mode_enabled", sanitized["wer_mode_enabled"])
    sanitized["require_signal_check"] = config.get("require_signal_check", sanitized["require_signal_check"])
    sanitized["quality_check_interval_seconds"] = config.get(
        "quality_check_interval_seconds",
        sanitized["quality_check_interval_seconds"],
    )
    sanitized["sample_rate_hz"] = config.get("sample_rate_hz", sanitized["sample_rate_hz"])
    sanitized["last_mode"] = config.get("last_mode", sanitized["last_mode"])
    sanitized["active_mode"] = config.get("active_mode", sanitized["active_mode"])

    raw_modes = config.get("modes", {})
    if not isinstance(raw_modes, dict):
        raw_modes = {}

    def _mode_value(mode_key: ModeKey, field: str, fallback: str) -> str:
        raw_mode_config = raw_modes.get(mode_key, {})
        if isinstance(raw_mode_config, dict):
            candidate = raw_mode_config.get(field, fallback)
            if isinstance(candidate, str):
                return candidate.strip()
        return fallback

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
    mixed_playback_device: Any = sanitized["mixed_playback_device"]
    sanitized["mixed_playback_device"] = (
        mixed_playback_device.strip()
        if isinstance(mixed_playback_device, str)
        else DEFAULT_CONFIG["mixed_playback_device"]
    )

    sanitized["wer_mode_enabled"] = _coerce_bool(
        sanitized.get("wer_mode_enabled"),
        bool(DEFAULT_CONFIG["wer_mode_enabled"]),
    )
    sanitized["require_signal_check"] = _coerce_bool(
        sanitized.get("require_signal_check"),
        bool(DEFAULT_CONFIG["require_signal_check"]),
    )
    sanitized["restore_devices_on_exit"] = _coerce_bool(
        sanitized.get("restore_devices_on_exit"),
        True,
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
    active_mode: Any = sanitized["active_mode"]
    sanitized["active_mode"] = active_mode if active_mode in MODE_NAME_BY_KEY else MODE_KEY_BY_NAME[sanitized["last_mode"]]

    mic_output = sanitized["speaker_device"]
    mixed_output = sanitized["mixed_playback_device"] or mic_output
    sanitized["modes"] = {
        "mic": {
            "input_device": _mode_value("mic", "input_device", sanitized["mic_device"]),
            "output_device": _mode_value("mic", "output_device", mic_output),
        },
        "vac": {
            "input_device": _mode_value("vac", "input_device", sanitized["vac_device"]),
            "output_device": _mode_value("vac", "output_device", sanitized["vac_playback_device"]),
        },
        "mixed": {
            "input_device": _mode_value("mixed", "input_device", sanitized["voicemeeter_device"]),
            "output_device": _mode_value("mixed", "output_device", mixed_output),
        },
    }
    sanitized["mic_device"] = sanitized["modes"]["mic"]["input_device"]
    sanitized["speaker_device"] = sanitized["modes"]["mic"]["output_device"]
    sanitized["vac_device"] = sanitized["modes"]["vac"]["input_device"]
    sanitized["vac_playback_device"] = sanitized["modes"]["vac"]["output_device"]
    sanitized["voicemeeter_device"] = sanitized["modes"]["mixed"]["input_device"]
    sanitized["mixed_playback_device"] = (
        ""
        if sanitized["modes"]["mixed"]["output_device"] == sanitized["speaker_device"]
        else sanitized["modes"]["mixed"]["output_device"]
    )
    sanitized["last_mode"] = MODE_NAME_BY_KEY[sanitized["active_mode"]]

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


def mode_name_to_key(mode_name: ModeName) -> ModeKey:
    return MODE_KEY_BY_NAME[mode_name]


def mode_key_to_name(mode_key: ModeKey) -> ModeName:
    return MODE_NAME_BY_KEY[mode_key]


def get_deepgram_api_key() -> str:
    return os.environ.get("DEEPGRAM_API_KEY", "").strip()


def get_deepgram_config() -> dict[str, Any]:
    return dict(DEEPGRAM_BASE_CONFIG)


def get_live_deepgram_options(sample_rate_hz: int) -> dict[str, Any]:
    return {
        **get_deepgram_config(),
        "language": "en-US",
        "interim_results": True,
        "encoding": "linear16",
        "channels": 1,
        "sample_rate": sample_rate_hz,
        "endpointing": 300,
        "utterance_end_ms": "1000",
        "vad_events": True,
        "no_delay": True,
    }


def get_file_deepgram_query_params() -> dict[str, str]:
    return {
        key: "true" if isinstance(value, bool) and value else str(value)
        for key, value in {
            **get_deepgram_config(),
            "detect_language": True,
        }.items()
    }


def get_deepgram_settings_message() -> str:
    return "Deepgram Settings: Court Transcript Profile"


def get_deepgram_settings_detail() -> str:
    return "Utterances and diarization stay on, while smart formatting and paragraph reshaping stay off for cleaner downstream transcript blocks."


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


def _normalize_utterance_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def build_blocks_from_deepgram_utterances(utterances: list[Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for utterance in utterances:
        words = list(_deepgram_value(utterance, "words", []) or [])
        confidence_values = [
            float(_deepgram_value(word, "confidence", 1.0) or 1.0)
            for word in words
            if _deepgram_value(word, "confidence", None) is not None
        ]
        blocks.append(
            {
                "speaker": _deepgram_value(utterance, "speaker"),
                "start": float(_deepgram_value(utterance, "start", 0.0) or 0.0),
                "end": float(_deepgram_value(utterance, "end", 0.0) or 0.0),
                "confidence": float(_deepgram_value(utterance, "confidence", 0.0) or 0.0),
                "speaker_confidence": (
                    float(_deepgram_value(utterance, "speaker_confidence", 0.0) or 0.0)
                    if _deepgram_value(utterance, "speaker_confidence", None) is not None
                    else None
                ),
                "word_confidence_min": min(confidence_values) if confidence_values else None,
                "text": _normalize_utterance_text(
                    _deepgram_value(utterance, "transcript")
                    or _deepgram_value(utterance, "text")
                    or " ".join(
                        str(
                            _deepgram_value(word, "punctuated_word")
                            or _deepgram_value(word, "word")
                            or ""
                        ).strip()
                        for word in words
                    )
                ),
                "words": words,
            }
        )
    return [block for block in blocks if block["text"]]


def merge_utterances(
    utterances: list[dict[str, Any]],
    gap_threshold_seconds: float = 1.2,
    min_word_count: int = 2,
) -> list[dict[str, Any]]:
    if not utterances:
        return []

    merged: list[dict[str, Any]] = []
    current = dict(utterances[0])
    current_words = list(current.get("words", []) or [])
    current["words"] = current_words

    for utterance in utterances[1:]:
        same_speaker = utterance.get("speaker") == current.get("speaker")
        gap = float(utterance.get("start", 0.0) or 0.0) - float(current.get("end", 0.0) or 0.0)

        if same_speaker and gap <= gap_threshold_seconds:
            current["end"] = utterance.get("end", current.get("end", 0.0))
            current_words.extend(list(utterance.get("words", []) or []))
            current["text"] = _normalize_utterance_text(f"{current.get('text', '')} {utterance.get('text', '')}")
            if utterance.get("confidence") is not None:
                current["confidence"] = max(float(current.get("confidence", 0.0) or 0.0), float(utterance.get("confidence", 0.0) or 0.0))
            current_min_conf = current.get("word_confidence_min")
            utterance_min_conf = utterance.get("word_confidence_min")
            if current_min_conf is None:
                current["word_confidence_min"] = utterance_min_conf
            elif utterance_min_conf is not None:
                current["word_confidence_min"] = min(float(current_min_conf), float(utterance_min_conf))
            continue

        if len(current_words) >= min_word_count:
            merged.append(current)

        current = dict(utterance)
        current_words = list(current.get("words", []) or [])
        current["words"] = current_words

    if len(current_words) >= min_word_count:
        merged.append(current)

    return merged


def smooth_speakers(utterances: list[dict[str, Any]], window: int = 2) -> list[dict[str, Any]]:
    if not utterances:
        return []

    smoothed: list[dict[str, Any]] = []
    for index, utterance in enumerate(utterances):
        current = dict(utterance)
        prev_speakers = [
            utterances[j].get("speaker")
            for j in range(max(0, index - window), index)
            if utterances[j].get("speaker") is not None
        ]
        next_speakers = [
            utterances[j].get("speaker")
            for j in range(index + 1, min(len(utterances), index + 1 + window))
            if utterances[j].get("speaker") is not None
        ]
        surrounding = prev_speakers + next_speakers
        if surrounding:
            most_common = max(set(surrounding), key=surrounding.count)
            if surrounding.count(most_common) >= len(surrounding) * 0.7:
                current["speaker"] = most_common
        smoothed.append(current)
    return smoothed


def prevent_micro_speaker_switch(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not blocks:
        return []

    stabilized = [dict(blocks[0])]
    for block in blocks[1:]:
        current = dict(block)
        previous = stabilized[-1]
        if current.get("speaker") != previous.get("speaker") and len(list(current.get("words", []) or [])) <= 2:
            current["speaker"] = previous.get("speaker")
        stabilized.append(current)
    return stabilized


def stabilize_speaker_blocks(blocks: list[dict[str, Any]], short_flip_max_words: int = 3) -> list[dict[str, Any]]:
    if not blocks:
        return []

    stabilized = [dict(blocks[0])]
    for block in blocks[1:]:
        current = dict(block)
        previous = stabilized[-1]
        previous_speaker = previous.get("speaker")
        current_speaker = current.get("speaker")
        current_word_count = len(_normalize_utterance_text(current.get("text", "")).split())
        if previous_speaker is not None and current_speaker != previous_speaker and current_word_count <= short_flip_max_words:
            current["speaker"] = previous_speaker
        stabilized.append(current)
    return stabilized


def detect_qa_patterns(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    detected: list[dict[str, Any]] = []
    for block in blocks:
        current = dict(block)
        text = _normalize_utterance_text(current.get("text", "")).lower()
        if text.startswith(("yes", "no", "uh huh", "i did", "i do")):
            current["type"] = "A"
        elif text.endswith("?"):
            current["type"] = "Q"
        detected.append(current)
    return detected


def enforce_qa_structure(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enforced: list[dict[str, Any]] = []
    previous_type = ""
    for block in blocks:
        current = dict(block)
        text = _normalize_utterance_text(current.get("text", ""))
        if not text:
            enforced.append(current)
            continue
        if text.endswith("?"):
            current["type"] = "Q"
        elif current.get("type") == "A":
            current["type"] = "A"
        elif previous_type in {"Q", "A"}:
            current["type"] = "A"
        previous_type = str(current.get("type", "") or "").upper()
        enforced.append(current)
    return enforced


def flag_low_confidence_words(words: list[Any], threshold: float = 0.85) -> list[dict[str, Any]]:
    flagged: list[dict[str, Any]] = []
    for word in words:
        confidence = _deepgram_value(word, "confidence", None)
        if confidence is None:
            continue
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            continue
        if confidence_value >= threshold:
            continue
        flagged.append(
            {
                "word": str(_deepgram_value(word, "punctuated_word") or _deepgram_value(word, "word") or "").strip(),
                "start": float(_deepgram_value(word, "start", 0.0) or 0.0),
                "end": float(_deepgram_value(word, "end", 0.0) or 0.0),
                "confidence": confidence_value,
                "speaker": _deepgram_value(word, "speaker"),
            }
        )
    return [item for item in flagged if item["word"]]


def normalize_deepgram_blocks(utterances: list[Any]) -> list[dict[str, Any]]:
    blocks = build_blocks_from_deepgram_utterances(utterances)
    blocks = merge_utterances(blocks)
    blocks = smooth_speakers(blocks)
    blocks = prevent_micro_speaker_switch(blocks)
    blocks = stabilize_speaker_blocks(blocks)
    blocks = detect_qa_patterns(blocks)
    blocks = enforce_qa_structure(blocks)
    return blocks


def format_utterance_blocks(blocks: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for block in blocks:
        text = _normalize_utterance_text(block.get("text", ""))
        if not text:
            continue
        block_type = str(block.get("type", "") or "").strip().upper()
        if block_type in {"Q", "A"}:
            lines.append(f"{block_type}: {text}")
            continue
        speaker = block.get("speaker")
        if speaker in (None, ""):
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
        utterances = payload["results"].get("utterances", [])
        if utterances:
            blocks = normalize_deepgram_blocks(utterances)
            utterance_text = format_utterance_blocks(blocks)
            if utterance_text:
                return utterance_text
        words = alternatives[0].get("words", [])
        speaker_text = _deepgram_words_to_speaker_lines(words)
        if speaker_text:
            return speaker_text
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        log_event("FileSession", level="warning", event="payload_format_fallback", reason=str(exc))
    return extract_transcript_text(payload)


def format_live_result_text(result: Any) -> str:
    try:
        utterances = _deepgram_value(result, "utterances", [])
        if utterances:
            blocks = normalize_deepgram_blocks(utterances)
            utterance_text = format_utterance_blocks(blocks)
            if utterance_text:
                return utterance_text
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
    except Exception as exc:
        log_event("LiveSession", level="warning", event="live_result_format_failed", reason=str(exc))
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


def linear_to_db(value: float, floor_db: float = -100.0) -> float:
    """Convert a linear amplitude in [0.0, 1.0] to dBFS. Clamps to floor_db."""
    if value <= 0:
        return floor_db
    return max(floor_db, 20.0 * math.log10(value))


def classify_speech_signal(rms_db: float, peak_db: float) -> tuple[str, str, str]:
    """
    Classify a measured signal against published speech target ranges.

    Returns (state, feedback, color) where:
        state     -- one of: 'no_signal', 'too_quiet', 'optimal', 'too_loud', 'clipping'
        feedback  -- short user-facing string
        color     -- hex color for the meter/status text
    """
    if peak_db >= PEAK_DB_CLIP_WARNING:
        return "clipping", "Clipping risk — reduce input level", "#E53935"
    if rms_db < RMS_DB_SILENCE_FLOOR:
        return "no_signal", "No signal detected — check routing, mute, and source playback", "#9E9E9E"
    if rms_db < RMS_DB_TOO_QUIET:
        return "too_quiet", "Too quiet — increase mic gain or source volume", "#F9A825"
    if rms_db > RMS_DB_OPTIMAL_MAX:
        return "too_loud", "Too loud — reduce input level", "#FB8C00"
    return "optimal", "Optimal speech level", "#43A047"


class SignalStateTracker:
    def __init__(self, max_history: int = 8, majority_ratio: float = 0.6):
        self.max_history = max(3, int(max_history))
        self.majority_ratio = max(0.5, min(1.0, float(majority_ratio)))
        self.history: list[str] = []
        self.current_state = ""

    def update(self, state: str) -> tuple[str, bool]:
        normalized = str(state or "no_signal").strip().lower() or "no_signal"
        self.history.append(normalized)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        if not self.current_state:
            self.current_state = normalized
            return self.current_state, True

        counts = {candidate: self.history.count(candidate) for candidate in set(self.history)}
        candidate = max(counts, key=counts.get)
        threshold = max(2, math.ceil(len(self.history) * self.majority_ratio))
        if counts[candidate] >= threshold and candidate != self.current_state:
            self.current_state = candidate
            return self.current_state, True
        return self.current_state, False


def analyze_live_input_signal(raw_bytes: bytes) -> dict[str, Any] | None:
    try:
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
    except Exception as exc:
        log_event("DeviceManager", level="warning", event="analyze_live_input_signal_failed", reason=str(exc))
        return None
    if samples.size == 0:
        return None

    normalized = samples / 32768.0
    rms = float(np.sqrt(np.mean(normalized * normalized)))
    peak = float(np.max(np.abs(normalized)))
    rms_db = linear_to_db(rms)
    peak_db = linear_to_db(peak)

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
        "rms_db": rms_db,
        "peak_db": peak_db,
        "clipping_hard": peak_db >= PEAK_DB_CLIP_HARD,
        "color": color,
        "detail": detail,
    }


def generate_silence_chunk(frames: int = 1024) -> np.ndarray:
    return np.zeros((max(1, int(frames)), 1), dtype=np.float32)


def build_silence_input_device(sample_rate_hz: int = LIVE_TRANSCRIPTION_SAMPLE_RATE_HZ) -> ActiveAudioDevice:
    return {
        "name": NULL_INPUT_DEVICE_NAME,
        "index": NULL_INPUT_DEVICE_INDEX,
        "info": {
            "name": NULL_INPUT_DEVICE_NAME,
            "max_input_channels": 1,
            "default_samplerate": int(sample_rate_hz),
            "is_synthetic": True,
        },
        "sample_rate": int(sample_rate_hz),
    }


def evaluate_live_signal_readiness(signal: dict[str, Any] | None, device_name: str) -> tuple[bool, str, dict[str, Any] | None]:
    normalized_name = normalize_audio_device_name(device_name)
    if signal is None:
        return False, f"Unable to read audio from {normalized_name} before starting live transcription.", None

    rms_db = float(signal.get("rms_db", linear_to_db(float(signal.get("rms", 0.0)))))
    peak_db = float(signal.get("peak_db", linear_to_db(float(signal.get("peak", 0.0)))))
    speech_state, feedback, _color = classify_speech_signal(rms_db, peak_db)
    enriched_signal = {
        **signal,
        "device_name": normalized_name,
        "speech_state": speech_state,
        "feedback": feedback,
    }

    if speech_state == "no_signal":
        return False, feedback, enriched_signal

    if speech_state == "too_quiet":
        return True, feedback, enriched_signal

    if speech_state in {"too_loud", "clipping"}:
        return True, feedback, enriched_signal

    return True, f"Input verified on {normalized_name}. {feedback}", enriched_signal


def normalize_audio_device_name(name: str) -> str:
    return re.sub(r",\s*(MME|Windows .+|DirectSound|WDM-KS)$", "", name, flags=re.IGNORECASE).strip()


def resolve_input_device(name: str) -> tuple[int | None, dict[str, Any] | None]:
    target = normalize_audio_device_name(name)
    try:
        devices = sd.query_devices()
    except Exception as exc:
        log_event("Resolver", level="warning", event="query_input_devices_failed", requested=name, reason=str(exc))
        return None, None

    exact_match: tuple[int, dict[str, Any]] | None = None
    partial_match: tuple[int, dict[str, Any]] | None = None
    token_match: tuple[int, dict[str, Any]] | None = None
    target_tokens = [token for token in re.split(r"[^a-z0-9]+", target.lower()) if len(token) >= 4]

    for index, device in enumerate(devices):
        if int(device.get("max_input_channels", 0)) <= 0:
            continue
        raw_name = str(device.get("name", "")).strip()
        normalized = normalize_audio_device_name(raw_name)
        if normalized == target:
            exact_match = (index, device)
            break
        normalized_lower = normalized.lower()
        target_lower = target.lower()
        if target and (target_lower in normalized_lower or normalized_lower in target_lower) and partial_match is None:
            partial_match = (index, device)
        if token_match is None and target_tokens:
            overlap = sum(1 for token in target_tokens if token in normalized_lower)
            if overlap >= max(1, min(2, len(target_tokens))):
                token_match = (index, device)

    match = exact_match or partial_match or token_match
    if match is None:
        return None, None
    return match[0], match[1]


def resolve_input_device_exact(name: str) -> tuple[int | None, dict[str, Any] | None]:
    target = normalize_audio_device_name(name)
    try:
        devices = sd.query_devices()
    except Exception as exc:
        log_event("Resolver", level="warning", event="query_input_devices_failed", requested=name, reason=str(exc))
        return None, None

    for index, device in enumerate(devices):
        if int(device.get("max_input_channels", 0)) <= 0:
            continue
        raw_name = str(device.get("name", "")).strip()
        normalized = normalize_audio_device_name(raw_name)
        if normalized == target:
            return index, device
    return None, None


def resolve_output_device(name: str) -> tuple[int | None, dict[str, Any] | None]:
    target = normalize_audio_device_name(name)
    try:
        devices = sd.query_devices()
    except Exception as exc:
        log_event("Resolver", level="warning", event="query_output_devices_failed", requested=name, reason=str(exc))
        return None, None

    exact_match: tuple[int, dict[str, Any]] | None = None
    partial_match: tuple[int, dict[str, Any]] | None = None
    token_match: tuple[int, dict[str, Any]] | None = None
    target_tokens = [token for token in re.split(r"[^a-z0-9]+", target.lower()) if len(token) >= 4]

    for index, device in enumerate(devices):
        if int(device.get("max_output_channels", 0)) <= 0:
            continue
        raw_name = str(device.get("name", "")).strip()
        normalized = normalize_audio_device_name(raw_name)
        if normalized == target:
            exact_match = (index, device)
            break
        normalized_lower = normalized.lower()
        target_lower = target.lower()
        if target and (target_lower in normalized_lower or normalized_lower in target_lower) and partial_match is None:
            partial_match = (index, device)
        if token_match is None and target_tokens:
            overlap = sum(1 for token in target_tokens if token in normalized_lower)
            if overlap >= max(1, min(2, len(target_tokens))):
                token_match = (index, device)

    match = exact_match or partial_match or token_match
    if match is None:
        return None, None
    return match[0], match[1]


def get_default_input_device() -> tuple[int | None, dict[str, Any] | None]:
    try:
        device_info = sd.query_devices(kind="input")
    except Exception as exc:
        log_event("DeviceManager", level="warning", event="default_input_query_failed", reason=str(exc))
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
    except Exception as exc:
        log_event("DeviceManager", level="warning", event="default_output_query_failed", reason=str(exc))
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
    return sample_resolved_input_signal(
        device_index,
        sample_rate_hz,
        normalize_audio_device_name(str(device_info.get("name", device_name))),
        duration_seconds=duration_seconds,
        device_info=device_info,
    )


def sample_resolved_input_signal(
    device_index: int,
    sample_rate_hz: int,
    device_name: str,
    duration_seconds: float = 0.35,
    device_info: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    resolved_info = device_info
    if device_index == NULL_INPUT_DEVICE_INDEX or bool((resolved_info or {}).get("is_synthetic")):
        frames = max(1, int(sample_rate_hz * duration_seconds))
        synthetic = generate_silence_chunk(frames=frames)
        pcm16 = (np.asarray(synthetic[:, 0], dtype=np.float32) * 32767.0).astype(np.int16, copy=False)
        signal = analyze_live_input_signal(pcm16.tobytes())
        if signal is None:
            return {
                "state": "silent",
                "rms": 0.0,
                "peak": 0.0,
                "rms_db": linear_to_db(0.0),
                "peak_db": linear_to_db(0.0),
                "clipping_hard": False,
                "color": "#9E9E9E",
                "detail": "Synthetic silence mode is active because no input devices were detected.",
            }
        signal["detail"] = "Synthetic silence mode is active because no input devices were detected."
        return signal
    if resolved_info is None:
        try:
            queried = sd.query_devices(device_index)
        except Exception as exc:
            log_event("Preflight", level="warning", event="resolved_device_query_failed", device=device_name, reason=str(exc))
            queried = None
        if isinstance(queried, dict):
            resolved_info = queried

    actual_sample_rate = int(sample_rate_hz)
    if resolved_info is not None:
        try:
            actual_sample_rate = int(float(resolved_info.get("default_samplerate", sample_rate_hz)))
        except (TypeError, ValueError):
            actual_sample_rate = int(sample_rate_hz)
    if actual_sample_rate <= 0:
        actual_sample_rate = int(sample_rate_hz)

    capture_channels = 1
    if resolved_info is not None:
        try:
            capture_channels = max(1, min(int(resolved_info.get("max_input_channels", 1)), 2))
        except (TypeError, ValueError):
            capture_channels = 1

    frames = max(1, int(actual_sample_rate * duration_seconds))

    try:
        sd.check_input_settings(
            device=device_index,
            samplerate=actual_sample_rate,
            channels=capture_channels,
        )
        recording = sd.rec(
            frames,
            samplerate=actual_sample_rate,
            channels=capture_channels,
            dtype="float32",
            device=device_index,
        )
        sd.wait()
    except Exception as exc:
        log_event("Preflight", level="warning", event="sample_input_signal_failed", device=device_name, reason=str(exc))
        return None

    samples = np.asarray(recording, dtype=np.float64)
    if samples.ndim == 2:
        samples = np.mean(samples, axis=1)
    samples = np.asarray(np.squeeze(samples), dtype=np.float64)
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
    except Exception as exc:
        log_event("Resolver", level="warning", event="list_devices_failed", channel_key=channel_key, reason=str(exc))
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


def infer_microphone_input_device(preferred_name: str, devices: list[str]) -> str:
    return infer_device_by_patterns(
        preferred_name,
        devices,
        required_patterns=[("microphone",), ("mic",)],
        optional_patterns=[("usb",), ("input",)],
        excluded_terms=("cable", "vb-audio", "voicemeeter", "stereo mix", "line in", "aux"),
    )


def is_microphone_like_input(name: str) -> bool:
    lowered = normalize_audio_device_name(name).lower()
    if not lowered:
        return False
    if "microphone" in lowered:
        return True
    return bool(re.search(r"(?<![a-z])mic(?![a-z])", lowered))


def is_generic_onboard_mic_input(name: str) -> bool:
    lowered = normalize_audio_device_name(name).lower()
    return any(
        token in lowered
        for token in (
            "microsoft sound mapper",
            "primary sound capture driver",
            "realtek hd audio mic input",
            "line in",
        )
    )


def classify_mixed_input_caps(name: str) -> set[str]:
    caps: set[str] = set()
    if normalize_audio_device_name(name):
        caps.add("input")
    return caps


def is_valid_mixed_input_device(name: str) -> bool:
    return "input" in classify_mixed_input_caps(name)


def pick_mixed_input_device(preferred_name: str, devices: list[str]) -> str:
    preferred = normalize_audio_device_name(preferred_name)
    if preferred and preferred in devices and is_valid_mixed_input_device(preferred):
        return preferred
    normalized_map = {normalize_audio_device_name(device): device for device in devices}
    if preferred and preferred in normalized_map:
        return normalized_map[preferred]
    ranked = sorted((device for device in devices if is_valid_mixed_input_device(device)), key=_device_rank_key)
    return ranked[0] if ranked else ""


def infer_speaker_output_device(preferred_name: str, devices: list[str]) -> str:
    return infer_device_by_patterns(
        preferred_name,
        devices,
        required_patterns=[("speaker",), ("headphone",), ("realtek",)],
        optional_patterns=[("tv",), ("hdmi",), ("high", "definition", "audio"), ("nvidia",), ("output",)],
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

        creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        if SOUNDVOLUMEVIEW_PATH.exists():
            for role in ("Console", "Multimedia", "Communications"):
                log_event("DeviceManager", event="set_default_device", backend="soundvolumeview", role=role, device=device_name)
                subprocess.run(
                    [str(SOUNDVOLUMEVIEW_PATH), "/SetDefault", device_name, role],
                    check=True,
                    capture_output=True,
                    creationflags=creation_flags,
                )
            return

        if not NIRCMD_PATH.exists():
            raise FileNotFoundError(
                f"Missing both {SOUNDVOLUMEVIEW_PATH.name} and {NIRCMD_PATH.name} in {APP_DIR}."
            )

        for role in ("0", "1", "2"):
            log_event("DeviceManager", event="set_default_device", backend="nircmd", role=role, device=device_name)
            subprocess.run(
                [str(NIRCMD_PATH), "setdefaultsounddevice", device_name, role],
                check=True,
                capture_output=True,
                creationflags=creation_flags,
            )

    @classmethod
    def set_default_recording_device(cls, device_name: str) -> tuple[bool, str]:
        try:
            log_event("DeviceManager", event="switch_recording_requested", device=device_name)
            cls._set_default_device(device_name)
            return True, f"Switched recording device to {device_name} for all Windows audio roles."
        except subprocess.CalledProcessError as exc:
            error_text = exc.stderr.decode("utf-8", errors="ignore").strip() if exc.stderr else ""
            log_failure("DEVICE", mode="recording", device=device_name, reason=error_text or str(exc))
            return False, error_text or f"Failed to switch to {device_name}."
        except Exception as exc:
            LOGGER.exception(_log_message("Failure: DEVICE", mode="recording", device=device_name, reason=str(exc)))
            return False, str(exc)

    @classmethod
    def set_default_playback_device(cls, device_name: str) -> tuple[bool, str]:
        try:
            log_event("DeviceManager", event="switch_playback_requested", device=device_name)
            cls._set_default_device(device_name)
            return True, f"Switched playback device to {device_name} for all Windows audio roles."
        except subprocess.CalledProcessError as exc:
            error_text = exc.stderr.decode("utf-8", errors="ignore").strip() if exc.stderr else ""
            log_failure("DEVICE", mode="playback", device=device_name, reason=error_text or str(exc))
            return False, error_text or f"Failed to switch to {device_name}."
        except Exception as exc:
            LOGGER.exception(_log_message("Failure: DEVICE", mode="playback", device=device_name, reason=str(exc)))
            return False, str(exc)

    @staticmethod
    def toggle_mute() -> tuple[bool, str]:
        if not NIRCMD_PATH.exists():
            return False, f"Missing {NIRCMD_PATH.name} in {APP_DIR}."

        try:
            log_event("DeviceManager", event="toggle_mute_requested", target="default_record")
            subprocess.run(
                [str(NIRCMD_PATH), "mutesysvolume", "2", "default_record"],
                check=True,
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            return True, "Toggled mute on the default recording device."
        except subprocess.CalledProcessError as exc:
            error_text = exc.stderr.decode("utf-8", errors="ignore").strip() if exc.stderr else ""
            log_failure("DEVICE", tag="MuteToggle", device="default_record", reason=f"nircmd mutesysvolume exited with code {exc.returncode}")
            return False, error_text or "Failed to toggle mute."
        except Exception as exc:
            LOGGER.exception(_log_message("Failure: DEVICE", tag="MuteToggle", device="default_record", reason=str(exc)))
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
                "level_text": "RMS: unavailable | Peak: unavailable",
                "status_text": "Audio monitor unavailable",
                "detail_text": f"{requested_device_name or 'Default input'}: {exc}",
                "rms_db": -100.0,
                "peak_db": -100.0,
                "meter_color": "#9E9E9E",
                "meter_feedback": "Audio monitor unavailable",
                "meter_progress": 0.0,
            }

        rms_db = linear_to_db(rms)
        peak_db = linear_to_db(peak)
        speech_state, feedback, meter_color = classify_speech_signal(rms_db, peak_db)

        if speech_state == "no_signal":
            quality = "too_quiet"
            status = "No usable input"
            detail = "Input is effectively silent. Check the selected recording device and gain."
        elif speech_state == "too_quiet":
            quality = "low"
            status = "Low signal"
            detail = "Speech may be too quiet for reliable transcription."
        elif speech_state == "clipping":
            quality = "clipping"
            status = "Clipping risk"
            detail = "Input is peaking too high. Lower mic gain or mixer output."
        elif speech_state == "too_loud":
            quality = "good"
            status = "Strong signal"
            detail = "Signal is usable but getting close to clipping."
        else:
            quality = "excellent"
            status = "Optimal"
            detail = "Signal level looks healthy for speech capture."

        return {
            "quality": quality,
            "level_text": f"RMS: {rms_db:.1f} dB | Peak: {peak_db:.1f} dB",
            "status_text": feedback if speech_state != "optimal" else status,
            "detail_text": (
                f"{detail} Input: {normalize_audio_device_name(str(device_info.get('name', requested_device_name or 'Default input')))}."
            ),
            "rms_db": rms_db,
            "peak_db": peak_db,
            "meter_color": meter_color,
            "meter_feedback": feedback,
            "meter_progress": max(0.0, min(1.0, (rms_db + 60.0) / 60.0)),
        }


class DeepgramFileTranscriber:
    def __init__(self, api_key: str):
        self.api_key = api_key.strip()

    def transcribe_file(self, media_path: Path, output_dir: Path = TRANSCRIPTS_DIR) -> tuple[bool, str]:
        debug_log(
            f"[DeepgramFileTranscriber] Starting file transcription for {media_path} "
            f"with fixed config={get_deepgram_config()}"
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
        query = urllib.parse.urlencode(get_file_deepgram_query_params())

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
        on_transcript,
        on_status,
        on_signal,
    ):
        self.api_key = api_key.strip()
        self.input_device_name = input_device["name"].strip()
        self.input_device_index = int(input_device["index"])
        self.input_device_info = input_device["info"]
        self.sample_rate_hz = self._resolve_session_sample_rate(input_device)
        self.capture_channels = max(1, min(int(self.input_device_info.get("max_input_channels", 1)), 2))
        self.mode_name = mode_name.strip() or "Unknown"
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
        self.mode_switches: list[dict[str, str]] = []
        self.callback_count = 0
        self.last_signal_debug_at = 0.0
        self.last_signal_status = ""
        self._swap_lock = threading.Lock()
        self._reconnect_lock = threading.Lock()
        self._input_stream_factory = sd.InputStream
        self._pcm_queue: queue.Queue[bytes] = queue.Queue(maxsize=LIVE_PCM_QUEUE_MAX_BLOCKS)
        self._persist_queue: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self._sender_thread: threading.Thread | None = None
        self._keepalive_thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._persist_thread: threading.Thread | None = None
        self._reconnect_thread: threading.Thread | None = None
        self._silence_thread: threading.Thread | None = None
        self._last_send_at = 0.0
        self._last_callback_at = 0.0
        self._dropped_blocks = 0
        self._reconnect_attempt = 0
        self._state = LiveSessionState.STOPPED
        self._deepgram_client_factory = None
        self._live_options_cls = None
        self._live_events = None
        self._deepgram_client = None
        self._last_signal_emit_at = 0.0
        self._signal_state_tracker = SignalStateTracker()
        self._audio_processor = AudioProcessor()
        self._uses_synthetic_input = bool(self.input_device_info.get("is_synthetic"))

    def _resolve_session_sample_rate(self, input_device: ActiveAudioDevice) -> int:
        candidate = input_device.get("sample_rate", LIVE_TRANSCRIPTION_SAMPLE_RATE_HZ)
        try:
            sample_rate = int(float(candidate))
        except (TypeError, ValueError):
            sample_rate = LIVE_TRANSCRIPTION_SAMPLE_RATE_HZ
        if sample_rate <= 0:
            sample_rate = LIVE_TRANSCRIPTION_SAMPLE_RATE_HZ
        return sample_rate

    def _close_stream(self, stream) -> None:
        AudioEngine(LOGGER).close_stream(
            stream,
            context="LiveSession",
            device_name=self.actual_device_name,
        )

    def _open_stream_for_device(self, device_index: int, capture_channels: int, sample_rate_hz: int | None = None):
        if device_index == NULL_INPUT_DEVICE_INDEX or self._uses_synthetic_input:
            return None
        engine = AudioEngine(
            LOGGER,
            sounddevice_module=sd,
            input_stream_factory=self._input_stream_factory,
        )
        stream_sample_rate = int(sample_rate_hz or self.sample_rate_hz)
        return engine.start_input_stream(
            device_index=device_index,
            samplerate=stream_sample_rate,
            channels=capture_channels,
            callback=self._audio_callback,
            blocksize=1024,
            dtype="float32",
        )

    def _configure_connection(self, connection) -> None:
        if self._live_events is None:
            return
        connection.on(self._live_events.Open, self._on_open)
        connection.on(self._live_events.Transcript, self._on_transcript)
        connection.on(self._live_events.Error, self._on_error)
        connection.on(self._live_events.Close, self._on_close)

    def _close_connection(self, connection=None) -> None:
        target = connection or self.connection
        if target is None:
            return
        finish_error: list[Exception] = []

        def _finish_target() -> None:
            try:
                target.finish()
            except Exception as exc:
                finish_error.append(exc)

        finish_thread = threading.Thread(target=_finish_target, name="LiveSessionFinish", daemon=True)
        finish_thread.start()
        finish_thread.join(timeout=DEEPGRAM_FINISH_TIMEOUT_SECONDS)
        if finish_thread.is_alive():
            log_event(
                "LiveSession",
                level="warning",
                event="connection_finish_timeout",
                timeout_seconds=DEEPGRAM_FINISH_TIMEOUT_SECONDS,
            )
        elif finish_error:
            log_event("LiveSession", level="warning", event="connection_finish_warning", reason=str(finish_error[0]))
        if target is self.connection:
            self.connection = None

    def _connect_websocket(self) -> tuple[bool, str]:
        if self._deepgram_client_factory is None or self._live_options_cls is None:
            return False, "Deepgram SDK is not initialized for live transcription."

        self._state = LiveSessionState.CONNECTING
        self._deepgram_client = self._deepgram_client_factory(self.api_key)
        connection = self._deepgram_client.listen.websocket.v("1")
        self._configure_connection(connection)

        requested_live_options = get_live_deepgram_options(self.sample_rate_hz)
        supported_names = set(inspect.signature(self._live_options_cls.__init__).parameters.keys())
        live_options_payload = {
            key: value for key, value in requested_live_options.items() if key in supported_names
        }
        omitted_names = sorted(set(requested_live_options.keys()) - set(live_options_payload.keys()))
        if omitted_names:
            log_event("LiveSession", level="warning", event="sdk_options_omitted", omitted=",".join(omitted_names))

        options = self._live_options_cls(**live_options_payload)
        if not connection.start(options):
            self._state = LiveSessionState.STOPPED
            return False, "Failed to start Deepgram live transcription connection."

        self.connection = connection
        self._state = LiveSessionState.RUNNING
        self._last_send_at = time.monotonic()
        return True, ""

    def _start_background_threads(self) -> None:
        thread_specs = (
            ("_sender_thread", self._sender_loop, "LiveSessionSender"),
            ("_keepalive_thread", self._keepalive_loop, "LiveSessionKeepAlive"),
            ("_watchdog_thread", self._watchdog_loop, "LiveSessionWatchdog"),
            ("_persist_thread", self._persist_loop, "LiveSessionPersist"),
        )
        for attr_name, target, thread_name in thread_specs:
            thread = getattr(self, attr_name)
            if thread is not None and thread.is_alive():
                continue
            thread = threading.Thread(target=target, name=thread_name, daemon=True)
            setattr(self, attr_name, thread)
            thread.start()
        if self._uses_synthetic_input and (self._silence_thread is None or not self._silence_thread.is_alive()):
            self._silence_thread = threading.Thread(target=self._silence_loop, name="LiveSessionSilence", daemon=True)
            self._silence_thread.start()

    def _join_thread(self, thread: threading.Thread | None, timeout: float = 2.0) -> None:
        if thread is None:
            return
        thread.join(timeout=timeout)
        if thread.is_alive():
            log_event("LiveSession", level="warning", event="thread_join_timeout", thread_name=thread.name, timeout_seconds=timeout)

    def _enqueue_partial_transcript_write(self, transcript_text: str) -> None:
        if not transcript_text.strip():
            return
        try:
            self._persist_queue.put_nowait(transcript_text)
        except queue.Full:
            pass

    def _persist_transcript_text(self, transcript_text: str) -> None:
        if self.transcript_path is None or not transcript_text.strip():
            return
        try:
            self.transcript_path.write_text(transcript_text, encoding="utf-8")
        except OSError as exc:
            log_event("LiveSession", level="warning", event="partial_transcript_write_failed", reason=str(exc))

    def _sender_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self.running or self.connection is None or self._state != LiveSessionState.RUNNING:
                self._stop_event.wait(0.1)
                continue
            try:
                pcm_bytes = self._pcm_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.connection.send(pcm_bytes)
                self._last_send_at = time.monotonic()
            except Exception as exc:
                log_event("LiveSession", level="warning", event="sender_send_failed", reason=str(exc))
                self._trigger_reconnect(f"audio send failed: {exc}")

    def _send_keepalive(self) -> None:
        if self.connection is None:
            return
        try:
            keep_alive = getattr(self.connection, "keep_alive", None)
            if callable(keep_alive):
                keep_alive()
            else:
                self.connection.send(json.dumps({"type": "KeepAlive"}))
            self._last_send_at = time.monotonic()
            log_event("LiveSession", level="debug", event="keepalive_sent")
        except Exception as exc:
            log_event("LiveSession", level="warning", event="keepalive_failed", reason=str(exc))
            self._trigger_reconnect(f"keepalive failed: {exc}")

    def _keepalive_loop(self) -> None:
        while not self._stop_event.wait(DEEPGRAM_KEEPALIVE_SECONDS):
            if not self.running or self.connection is None or self._state != LiveSessionState.RUNNING:
                continue
            if time.monotonic() - self._last_send_at < DEEPGRAM_KEEPALIVE_IDLE_THRESHOLD:
                continue
            self._send_keepalive()

    def _reopen_stream(self) -> tuple[bool, str]:
        if not self._swap_lock.acquire(timeout=5.0):
            return False, "Timed out waiting to reopen the audio stream."
        try:
            if self._uses_synthetic_input:
                self._last_callback_at = time.monotonic()
                log_event("LiveSession", event="synthetic_audio_reopened", device=self.actual_device_name)
                return True, ""
            old_stream = self.stream
            if old_stream is not None:
                self._close_stream(old_stream)
                self.stream = None
            self.stream = self._open_stream_for_device(self.input_device_index, self.capture_channels, self.sample_rate_hz)
            self._last_callback_at = time.monotonic()
            log_event("LiveSession", event="audio_stream_reopened", device=self.actual_device_name)
            return True, ""
        except Exception as exc:
            log_failure("DEVICE", mode=self.mode_name, device=self.actual_device_name, reason=str(exc), tag="LiveSession")
            return False, f"Failed to reopen audio stream on {self.actual_device_name}: {exc}"
        finally:
            self._swap_lock.release()

    def _watchdog_loop(self) -> None:
        while not self._stop_event.wait(2.0):
            if not self.running or (self.stream is None and not self._uses_synthetic_input):
                continue
            if time.monotonic() - self._last_callback_at < AUDIO_CALLBACK_STALL_SECONDS:
                continue
            log_event("LiveSession", level="warning", event="audio_callback_stalled", device=self.actual_device_name)
            self.on_status(f"Audio callback stalled. Reopening {self.actual_device_name}...")
            reopened, message = self._reopen_stream()
            if reopened:
                self.on_status(f"Audio stream reopened on {self.actual_device_name}.")
                continue
            self.error_message = message
            self.on_status(message)
            self.running = False
            self._state = LiveSessionState.STOPPED
            self._stop_event.set()
            return

    def _persist_loop(self) -> None:
        pending_text = ""
        while not self._stop_event.is_set() or not self._persist_queue.empty():
            try:
                pending_text = self._persist_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            while not self._stop_event.wait(PARTIAL_TRANSCRIPT_DEBOUNCE_SECONDS):
                try:
                    pending_text = self._persist_queue.get_nowait()
                except queue.Empty:
                    break
            self._persist_transcript_text(pending_text)

    def _trigger_reconnect(self, reason: str) -> None:
        if not self.running or self._stop_event.is_set():
            return
        with self._reconnect_lock:
            if self._state == LiveSessionState.RECONNECTING:
                return
            self._state = LiveSessionState.RECONNECTING
            self.error_message = ""
            log_event("LiveSession", level="warning", event="reconnect_requested", reason=reason)
            if self._reconnect_thread is None or not self._reconnect_thread.is_alive():
                self._reconnect_thread = threading.Thread(target=self._reconnect, name="LiveSessionReconnect", daemon=True)
                self._reconnect_thread.start()

    def _reconnect(self) -> None:
        max_attempts = len(RECONNECT_BACKOFF_SECONDS)
        for attempt, delay_seconds in enumerate(RECONNECT_BACKOFF_SECONDS, start=1):
            if self._stop_event.wait(delay_seconds) or not self.running:
                return
            self._reconnect_attempt = attempt
            self.on_status(f"Reconnecting to Deepgram (attempt {attempt} of {max_attempts})...")
            log_event("LiveSession", level="warning", event="reconnect_attempt", attempt=attempt, max_attempts=max_attempts)
            self._close_connection()
            connected, message = self._connect_websocket()
            if connected:
                self._reconnect_attempt = 0
                marker = f"[Reconnected to Deepgram at {time.strftime('%H:%M:%S')}]"
                self.final_lines.append(marker)
                combined = "\n".join(self.final_lines)
                self._persist_transcript_text(combined)
                self.on_transcript(combined, self.current_interim)
                self.on_status("Live transcription reconnected.")
                log_event("LiveSession", event="reconnect_success", attempt=attempt)
                return
            log_event("LiveSession", level="warning", event="reconnect_failed", attempt=attempt, reason=message)

        self.error_message = "Deepgram live transcription disconnected and could not reconnect."
        self.running = False
        self._state = LiveSessionState.STOPPED
        self._stop_event.set()
        self.on_status(self.error_message)
        log_failure("DEPENDENCY", mode=self.mode_name, device=self.actual_device_name, reason=self.error_message, tag="LiveSession")

    def start(self) -> tuple[bool, str]:
        log_event("LiveSession", event="start_requested", mode=self.mode_name, configured_device=self.input_device_name)
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
            log_failure("DEPENDENCY", mode=self.mode_name, device=self.input_device_name, reason=str(exc))
            return False, f"Deepgram SDK is not available in this Python environment: {exc}"

        self._deepgram_client_factory = DeepgramClient
        self._live_options_cls = LiveOptions
        self._live_events = LiveTranscriptionEvents
        self.transcript_path = build_live_transcript_output_path()
        self.metadata_path = build_live_transcript_metadata_path(self.transcript_path)
        self._stop_event.clear()
        self._state = LiveSessionState.CONNECTING
        connected, message = self._connect_websocket()
        if not connected:
            log_failure("DEPENDENCY", mode=self.mode_name, device=self.input_device_name, reason=message or "Deepgram websocket start returned false")
            return False, message

        try:
            if self.stream is not None:
                self._close_stream(self.stream)
                self.stream = None
            self.stream = self._open_stream_for_device(self.input_device_index, self.capture_channels, self.sample_rate_hz)
        except Exception as exc:
            self._close_connection()
            log_failure("DEVICE", mode=self.mode_name, device=self.input_device_name, reason=str(exc))
            return False, f"Failed to open input stream on {self.input_device_name}: {exc}"

        self.running = True
        self._last_callback_at = time.monotonic()
        self._last_signal_emit_at = 0.0
        self._signal_state_tracker = SignalStateTracker()
        self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.actual_device_name = normalize_audio_device_name(str(self.input_device_info.get("name", self.input_device_name)))
        self._start_background_threads()
        log_event(
            "LiveSession",
            event="running",
            mode=self.mode_name,
            resolved_device=self.actual_device_name,
            device_index=self.input_device_index,
            sample_rate=self.sample_rate_hz,
        )
        self._write_metadata(status="running")
        return True, f"Live transcription started from {self.actual_device_name}."

    def stop(self) -> tuple[bool, str]:
        log_event("LiveSession", event="stop_requested", mode=self.mode_name, resolved_device=self.actual_device_name)
        self.running = False
        self._state = LiveSessionState.STOPPED
        self._stop_event.set()
        acquired = self._swap_lock.acquire(timeout=5.0)
        if not acquired:
            log_failure("DEVICE", mode=self.mode_name, device=self.actual_device_name, reason="swap lock timeout during stop")
        try:
            if self.stream is not None:
                self._close_stream(self.stream)
                self.stream = None
            self._close_connection()
        finally:
            if acquired:
                self._swap_lock.release()

        for thread_name in (
            "_sender_thread",
            "_keepalive_thread",
            "_watchdog_thread",
            "_persist_thread",
            "_reconnect_thread",
            "_silence_thread",
        ):
            self._join_thread(getattr(self, thread_name), timeout=2.0)
            setattr(self, thread_name, None)

        transcript_body = "\n".join(line for line in self.final_lines if line.strip()).strip()
        transcript_text = transcript_body or "[No final transcript captured]"
        output_path = self.transcript_path or build_live_transcript_output_path()
        self.stopped_at = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            output_path.write_text(transcript_text, encoding="utf-8")
        except OSError as exc:
            log_failure("DEVICE", mode=self.mode_name, device=self.actual_device_name, reason=str(exc))
            return False, f"Failed to save live transcript: {exc}"

        self._write_metadata(status="error" if self.error_message else "completed")

        if self.error_message:
            log_event("LiveSession", level="warning", event="stopped_with_error", reason=self.error_message)
            return False, f"{self.error_message}\nPartial transcript saved to {output_path.name}"
        log_event("LiveSession", event="completed", transcript_file=output_path.name)
        return True, f"Live transcript saved to {output_path.name}"

    def switch_input_device(
        self,
        new_device: ActiveAudioDevice,
        new_mode_name: str,
    ) -> tuple[bool, str]:
        if not self.running or self.connection is None:
            return False, "Session is not running"

        if not self._swap_lock.acquire(timeout=5.0):
            log_failure("DEVICE", tag="HotSwitch", mode=self.mode_name, device=self.actual_device_name, reason="swap lock timeout")
            return False, "Hot switch timed out waiting for the audio stream lock."

        started_at = time.time()
        old_stream = self.stream
        old_index = self.input_device_index
        old_info = self.input_device_info
        old_name = self.input_device_name
        old_actual = self.actual_device_name
        old_mode = self.mode_name
        old_channels = self.capture_channels
        try:
            log_event(
                "HotSwitch",
                event="start",
                from_mode=old_mode,
                from_device=old_actual,
                to_mode=new_mode_name,
                to_device=new_device["name"],
            )
            if old_stream is not None:
                self._close_stream(old_stream)
                log_event("LiveSession", event="stream_closed", device=old_actual)
            self.stream = None

            new_info = new_device.get("info") or {}
            new_index = int(new_device["index"])
            new_channels = max(1, min(int(new_info.get("max_input_channels", 1)), 2))
            new_sample_rate = self._resolve_session_sample_rate(new_device)
            self._uses_synthetic_input = bool(new_info.get("is_synthetic")) or new_index == NULL_INPUT_DEVICE_INDEX
            new_stream = self._open_stream_for_device(new_index, new_channels, new_sample_rate)
            if self._uses_synthetic_input and (self._silence_thread is None or not self._silence_thread.is_alive()):
                self._silence_thread = threading.Thread(target=self._silence_loop, name="LiveSessionSilence", daemon=True)
                self._silence_thread.start()

            self.stream = new_stream
            self.input_device_index = new_index
            self.input_device_info = new_info
            self.input_device_name = new_device["name"].strip()
            self.actual_device_name = normalize_audio_device_name(str(new_info.get("name", self.input_device_name)))
            self.mode_name = new_mode_name.strip() or self.mode_name
            self.sample_rate_hz = new_sample_rate
            self.capture_channels = new_channels

            marker = f"[Switched to {self.mode_name} mode at {time.strftime('%H:%M:%S')} - {self.actual_device_name}]"
            self.final_lines.append(marker)
            self.mode_switches.append(
                {
                    "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "from_mode": old_mode,
                    "from_device": old_actual,
                    "to_mode": self.mode_name,
                    "to_device": self.actual_device_name,
                }
            )
            self._write_partial_transcript()
            self._write_metadata(status="running")
            self.on_transcript("\n".join(self.final_lines), self.current_interim)
            self.on_status(f"Switched to {self.mode_name} - capturing from {self.actual_device_name}")
            self.on_signal(
                {
                    "device_name": self.actual_device_name,
                    "mode_name": self.mode_name,
                    "state": "active",
                    "rms": 0.0,
                    "peak": 0.0,
                    "color": "#8AB4F8",
                    "detail": f"Capture stream moved to {self.actual_device_name}.",
                }
            )
            elapsed_ms = int((time.time() - started_at) * 1000)
            log_event("HotSwitch", event="complete", from_mode=old_mode, to_mode=self.mode_name, elapsed_ms=elapsed_ms)
            log_event("ModeSwitch", from_mode=old_mode, to_mode=self.mode_name, old_device=old_actual, new_device=self.actual_device_name, hot=True)
            return True, f"Switched to {self.mode_name} - capturing from {self.actual_device_name}"
        except Exception as exc:
            LOGGER.exception(_log_message("Failure: DEVICE", mode=new_mode_name, device=new_device.get("name", ""), reason=str(exc)))
            try:
                restored_stream = self._open_stream_for_device(old_index, old_channels, self.sample_rate_hz)
                self.stream = restored_stream
                self.input_device_index = old_index
                self.input_device_info = old_info
                self.input_device_name = old_name
                self.actual_device_name = old_actual
                self.mode_name = old_mode
                self.sample_rate_hz = self._resolve_session_sample_rate(
                    {
                        "name": old_name,
                        "index": old_index,
                        "info": old_info,
                        "sample_rate": self.sample_rate_hz,
                    }
                )
                self.capture_channels = old_channels
                self._uses_synthetic_input = bool((old_info or {}).get("is_synthetic")) or old_index == NULL_INPUT_DEVICE_INDEX
                return False, f"Hot switch failed; restored {old_actual}."
            except Exception as restore_exc:
                LOGGER.exception(_log_message("Failure: DEVICE", mode=old_mode, device=old_actual, reason=str(restore_exc)))
                self.error_message = "Hot switch failed and old device could not be restored; session ended."
                self.connection = None
                self.stream = None
                self.running = False
                return False, self.error_message
        finally:
            self._swap_lock.release()

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if not self.running:
            return
        self.callback_count += 1
        self._last_callback_at = time.monotonic()
        if status:
            log_event("LiveSession", level="warning", event="audio_callback_status", status=status)
            self.on_status(f"Audio stream status: {status}")
        raw_pcm_bytes = self._pcm16_bytes_from_input(indata)
        self._report_input_signal(raw_pcm_bytes, frames)
        processed = self._audio_processor.process(indata)
        if processed is None:
            if indata is None:
                pcm_bytes = raw_pcm_bytes
            else:
                return
        else:
            pcm_bytes = self._pcm16_bytes_from_samples(processed)
        try:
            self._pcm_queue.put_nowait(pcm_bytes)
        except queue.Full:
            try:
                self._pcm_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._pcm_queue.put_nowait(pcm_bytes)
            except queue.Full:
                pass
            self._dropped_blocks += 1
            if self._dropped_blocks % DROPPED_BLOCKS_LOG_EVERY == 0:
                log_event("LiveSession", level="warning", event="pcm_queue_drops", count=self._dropped_blocks)

    def _silence_loop(self) -> None:
        frames = 1024
        sleep_seconds = frames / float(self.sample_rate_hz or LIVE_TRANSCRIPTION_SAMPLE_RATE_HZ)
        while not self._stop_event.is_set():
            if not self.running or self._state != LiveSessionState.RUNNING or not self._uses_synthetic_input:
                self._stop_event.wait(0.1)
                continue
            chunk = generate_silence_chunk(frames=frames)
            self._last_callback_at = time.monotonic()
            raw_pcm_bytes = self._pcm16_bytes_from_input(chunk)
            self._report_input_signal(raw_pcm_bytes, frames)
            try:
                self._pcm_queue.put_nowait(raw_pcm_bytes)
            except queue.Full:
                try:
                    self._pcm_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._pcm_queue.put_nowait(raw_pcm_bytes)
                except queue.Full:
                    pass
            self._stop_event.wait(sleep_seconds)

    def _pcm16_bytes_from_input(self, indata: np.ndarray) -> bytes:
        samples = np.asarray(indata, dtype=np.float32)
        if samples.ndim == 2:
            samples = np.mean(samples, axis=1)
        samples = np.squeeze(samples)
        if samples.ndim == 0:
            samples = np.asarray([float(samples)], dtype=np.float32)
        samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
        samples = np.clip(samples, -1.0, 1.0)
        return self._pcm16_bytes_from_samples(samples)

    def _pcm16_bytes_from_samples(self, samples: np.ndarray) -> bytes:
        pcm16 = (np.asarray(samples, dtype=np.float32) * 32767.0).astype(np.int16, copy=False)
        return pcm16.tobytes()

    def _on_open(self, client, open=None, **kwargs) -> None:
        log_event("LiveSession", event="websocket_opened", mode=self.mode_name, device=self.actual_device_name)
        self.on_status("Deepgram live connection opened.")

    def _on_close(self, client, close=None, **kwargs) -> None:
        log_event("LiveSession", event="websocket_closed", mode=self.mode_name, device=self.actual_device_name)
        self.on_status("Deepgram live connection closed.")
        if self.running:
            self._trigger_reconnect("websocket closed")

    def _on_error(self, client, error=None, **kwargs) -> None:
        self.error_message = str(error) if error else "Deepgram live transcription error."
        log_failure("DEPENDENCY", mode=self.mode_name, device=self.actual_device_name, reason=self.error_message)
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
            log_event("LiveSession", event="final_transcript", segment_count=len(self.final_lines), preview=transcript[:120])
            self._write_partial_transcript()
            combined = "\n".join(self.final_lines)
            self.on_transcript(combined, "")
        else:
            self.current_interim = transcript
            log_event("LiveSession", level="debug", event="interim_transcript", preview=transcript[:120])
            combined = "\n".join(self.final_lines)
            self.on_transcript(combined, self.current_interim)

    def _write_partial_transcript(self) -> None:
        transcript_body = "\n".join(line for line in self.final_lines if line.strip()).strip()
        self._enqueue_partial_transcript_write(transcript_body)

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
            "deepgram_config": get_deepgram_config(),
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "final_segment_count": len(self.final_lines),
            "error_message": self.error_message,
            "transcript_file": self.transcript_path.name if self.transcript_path else "",
            "mode_switches": self.mode_switches,
        }
        try:
            self.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            log_failure("DEVICE", mode=self.mode_name, device=self.actual_device_name, reason=str(exc), tag="LiveSession")
        else:
            log_event("LiveSession", event="metadata_updated", metadata_file=self.metadata_path.name)

    def _report_input_signal(self, raw_bytes: bytes, frames: int) -> None:
        now = time.time()
        signal = analyze_live_input_signal(raw_bytes)
        if signal is None:
            return
        rms_db = float(signal.get("rms_db", linear_to_db(float(signal.get("rms", 0.0)))))
        peak_db = float(signal.get("peak_db", linear_to_db(float(signal.get("peak", 0.0)))))
        speech_state, feedback, speech_color = classify_speech_signal(rms_db, peak_db)
        should_emit_signal = self._last_signal_emit_at == 0.0 or (now - self._last_signal_emit_at) >= 1.0
        if should_emit_signal:
            self.on_signal(
                {
                    **signal,
                    "device_name": self.actual_device_name,
                    "mode_name": self.mode_name,
                    "speech_state": speech_state,
                    "feedback": feedback,
                    "meter_color": speech_color,
                }
            )
            self._last_signal_emit_at = now

        rms = float(signal["rms"])
        peak = float(signal["peak"])
        state = str(signal["state"])
        if not self._signal_state_tracker.current_state and self.last_signal_status:
            self._signal_state_tracker.current_state = self.last_signal_status
        stable_state, state_changed = self._signal_state_tracker.update(speech_state)
        if state_changed:
            log_level = "debug" if stable_state == "no_signal" else "info"
            log_event(
                "AudioSignal",
                level=log_level,
                event="state_change",
                device=self.actual_device_name,
                from_state=self.last_signal_status or "unknown",
                to=stable_state,
                instantaneous_state=speech_state,
                rms=f"{rms:.5f}",
                peak=f"{peak:.5f}",
                rms_db=f"{rms_db:.1f}",
                peak_db=f"{peak_db:.1f}",
            )
            self.last_signal_status = stable_state
            status_message = f"Live input {stable_state} on {self.actual_device_name} (RMS {rms_db:.1f} dB, Peak {peak_db:.1f} dB)"
            if stable_state == "no_signal":
                status_message = f"Temporary silence on {self.actual_device_name} (RMS {rms_db:.1f} dB, Peak {peak_db:.1f} dB)"
            elif stable_state != "optimal":
                status_message = f"{feedback}. {status_message}"
            self.on_status(status_message)
        if now - self.last_signal_debug_at >= 2.0:
            self.last_signal_debug_at = now
            log_event(
                "AudioSignal",
                level="debug",
                device=self.actual_device_name,
                mode=self.mode_name,
                rms=f"{rms:.5f}",
                peak=f"{peak:.5f}",
                rms_db=f"{rms_db:.1f}",
                peak_db=f"{peak_db:.1f}",
                frames=frames,
                state=state,
                speech_state=speech_state,
                stable_state=stable_state,
            )


class App:
    def __init__(self) -> None:
        log_event(
            "App",
            event="startup",
            version="3.0",
            python=sys.version.split()[0],
            platform=sys.platform,
            app_dir=str(APP_DIR),
            frozen=getattr(sys, "frozen", False),
        )
        self.config = load_config()
        self.detected_input_devices = list_input_devices()
        self.detected_output_devices = list_output_devices()
        self._hydrate_config_from_detected_devices()
        self.device_manager = AudioDeviceManager()
        self.current_mode: ModeName = mode_key_to_name(self.config["active_mode"])
        self.active_audio_device: ActiveAudioDevice | None = None
        self.audio_state = AudioState()
        self.is_muted = False
        self.settings_window: ctk.CTkToplevel | None = None
        self.setup_notes_label = None
        self.setup_notes_visible = False
        self._closing = False
        self._vac_test_running = False
        self._transcription_running = False
        self._live_transcription_running = False
        self._live_transcription_starting = False
        self._live_transcription_stopping = False
        self.live_transcription_session: LiveTranscriptionSession | None = None
        self.live_transcript_final_text = ""
        self.live_transcript_interim_text = ""
        self.live_signal_status_text = "Waiting to sample the selected input."
        self._vac_test_forced_monitoring = False
        self._pending_vac_test = False
        self._audio_switch_in_progress = False
        self._pending_mode_button: ModeName | None = None
        self._pending_live_start_mode: ModeName | None = None
        self._pending_mode_after_live_stop: ModeName | None = None
        self._best_mode_running = False
        self._latest_signal_state = "Unknown"
        self._resume_monitor_after_live = False
        self._last_detected_refresh_at = 0.0
        self._resolved_input_name_cache: dict[tuple[str, str, tuple[str, ...]], str] = {}
        self._last_mixed_unavailable_reason: str | None = None
        self._slow_verification_count = 0
        self._slow_verification_advisory_logged = False
        self.device_detector = AutoAudioEngine(logger=LOGGER)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Virtual Audio Control")
        self.root.geometry("920x760")
        self.root.minsize(820, 640)
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.report_callback_exception = self._log_tk_callback_exception

        self.status_var = ctk.StringVar(value="Ready - Monitoring active" if self.config["wer_mode_enabled"] else "Ready")
        self.mode_var = ctk.StringVar(value=self.current_mode)
        self.require_signal_check_var = ctk.BooleanVar(value=bool(self.config["require_signal_check"]))
        self.monitor_status_var = ctk.StringVar(value="Excellent")
        self.monitor_level_var = ctk.StringVar(value="RMS: -∞ dB | Peak: -∞ dB")
        self.monitor_detail_var = ctk.StringVar(value="No issues detected")
        self.monitor_summary_var = ctk.StringVar(value="Monitoring active")
        self.monitor_recommendation_var = ctk.StringVar(value="No issues detected\nAll systems optimal")
        self.footer_var = ctk.StringVar(value="v3.0 Pro | Real-Time WER Optimization")
        self.runtime_audio_var = ctk.StringVar(value="Active Input: Detecting... | Active Output: Detecting... | Signal: Unknown")
        self.active_source_var = ctk.StringVar(value="⚠️ Source: None — Click a mode to begin.")
        self.debug_session_var = ctk.StringVar(value="Session: Idle")
        self.debug_device_var = ctk.StringVar(value="Devices: Detecting...")
        self.debug_signal_var = ctk.StringVar(value="Signal: Unknown")
        self.debug_queue_var = ctk.StringVar(value="Queue: -")
        self.debug_watchdog_var = ctk.StringVar(value="Audio: OK")
        self.debug_routing_var = ctk.StringVar(value="Routing: -")
        self.input_truth_var = ctk.StringVar(value="Input truth scan pending...")
        self.safe_mode_var = ctk.BooleanVar(value=True)
        self.shortcuts_var = ctk.StringVar(
            value="Shortcuts: Ctrl+1 Monitor | Ctrl+2 Routing | Ctrl+3 Transcribe | Ctrl+4 Settings | Ctrl+M Mute | Ctrl+Shift+1/2/3 Modes | F5 Refresh"
        )
        self._debug_log_queue: queue.Queue[tuple[str, int]] = queue.Queue(maxsize=500)
        self._debug_log_handler: UILogHandler | None = None
        self.autofix = AutoFixEngine(self)
        self.autofix.set_safe_mode(bool(self.safe_mode_var.get()))

        self.mic_var = ctk.StringVar(value=self.config["mic_device"])
        self.vac_var = ctk.StringVar(value=self.config["vac_device"])
        self.speaker_var = ctk.StringVar(value=self.config["speaker_device"])
        self.vac_playback_var = ctk.StringVar(value=self.config["vac_playback_device"])
        self.mix_var = ctk.StringVar(value=self.config["voicemeeter_device"])
        self.mixed_playback_var = ctk.StringVar(value=self.config["mixed_playback_device"])
        # FIX 3: Self-healing last_mode validation.
        # At this point self.current_mode == self.config["last_mode"] and all
        # StringVars (mic_var, vac_var, mix_var, speaker_var, vac_playback_var)
        # that _resolve_mode_devices reads from are initialized. If the saved mode's
        # input device is absent (e.g. Voicemeeter not running for Mixed mode,
        # CABLE Output missing for VAC), downgrade self.current_mode IN MEMORY ONLY
        # so the UI boots in a mode that can actually run.
        # self.config["last_mode"] is deliberately left untouched so that a later
        # launch with the original device present restores the saved preference.
        self._sanitize_mixed_input_configuration()
        self._validate_last_mode_against_detected_devices()
        self.direct_recording_var = ctk.StringVar(value=self.config["mic_device"])
        self.direct_playback_var = ctk.StringVar(value=self.config["speaker_device"])
        self.restore_devices_on_exit_var = ctk.BooleanVar(value=bool(self.config["restore_devices_on_exit"]))
        self.wer_enabled_var = ctk.BooleanVar(value=bool(self.config["wer_mode_enabled"]))
        self._original_default_input_name: str | None = None
        self._original_default_output_name: str | None = None
        self._capture_launch_audio_defaults()
        try:
            self.active_audio_device = self.resolve_active_device(self._current_live_input_device_name())
            self._lock_active_input_device(self.active_audio_device)
        except Exception as exc:
            log_event("Startup", level="warning", event="initial_active_device_unresolved", mode=self.current_mode, reason=str(exc))
            self.active_audio_device = None
            self._clear_active_input_device_lock()
        self.monitor = AudioQualityMonitor(
            sample_rate_hz=int(self.config["sample_rate_hz"]),
            interval_seconds=float(self.config["quality_check_interval_seconds"]),
            callback=self._queue_quality_update,
            device_provider=self.get_active_audio_device,
        )
        log_run_header(self.config)

        self._build_ui()
        self._install_debug_log_handler()
        self._reconcile_startup_mode()
        self._refresh_mode_hint()
        self._refresh_detection_summary()
        self._refresh_runtime_audio_status()
        self._queue_input_truth_refresh()

        if self.wer_enabled_var.get():
            self.monitor.start()

    def _build_ui(self) -> None:
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.minsize(900, 650)

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

        self.active_source_label = ctk.CTkLabel(
            header_frame,
            textvariable=self.active_source_var,
            font=("Arial", 10, "bold"),
            text_color="#9E9E9E",
            anchor="w",
        )
        self.active_source_label.grid(row=2, column=0, sticky="w", pady=(4, 0))

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

        self.monitor_tab = self._create_tab_scroll_frame("Monitor")
        self.routing_tab = self._create_tab_scroll_frame("Routing")
        self.transcribe_tab = self._create_tab_scroll_frame("Transcribe")
        self.settings_tab = self._create_tab_scroll_frame("Settings")
        self.debug_tab = self._create_tab_scroll_frame("Debug")

        self._build_monitor_tab()
        self._build_routing_tab()
        self._build_transcribe_tab()
        self._build_settings_tab()
        self._build_debug_tab()

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
            textvariable=self.shortcuts_var,
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
        self.root.bind("<Control-1>", lambda event: self._select_tab("Monitor"))
        self.root.bind("<Control-2>", lambda event: self._select_tab("Routing"))
        self.root.bind("<Control-3>", lambda event: self._select_tab("Transcribe"))
        self.root.bind("<Control-4>", lambda event: self._select_tab("Settings"))
        self.root.bind("<Control-m>", lambda event: self._handle_shortcut(self.toggle_mute))
        self.root.bind("<Control-M>", lambda event: self._handle_shortcut(self.toggle_mute))
        self.root.bind("<Control-Shift-KeyPress-1>", lambda event: self._handle_shortcut(lambda: self.apply_audio_mode("Microphone")))
        self.root.bind("<Control-Shift-KeyPress-2>", lambda event: self._handle_shortcut(lambda: self.apply_audio_mode("VAC")))
        self.root.bind("<Control-Shift-KeyPress-3>", lambda event: self._handle_shortcut(lambda: self.apply_audio_mode("Mixed")))
        self.root.bind("<F5>", lambda event: self.refresh_detected_devices())

    def _handle_shortcut(self, callback) -> str:
        callback()
        return "break"

    def _select_tab(self, tab_name: str) -> str:
        self.tabview.set(tab_name)
        return "break"

    def _log_tk_callback_exception(self, exc, val, tb) -> None:
        _log_uncaught_exception(
            "TkCallback",
            exc,
            val,
            tb,
            event="report_callback_exception",
            reason=str(val or exc.__name__),
        )
        try:
            self.status_var.set("An unexpected error occurred. See logs/errors.log.")
        except Exception as exc:
            log_event("TkCallback", level="warning", event="status_update_failed", reason=str(exc))

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

    def _build_monitor_tab(self) -> None:
        # --- Tab 1: Monitor ---
        self.monitor_tab.grid_columnconfigure(0, weight=1)

        mode_frame = ctk.CTkFrame(self.monitor_tab)
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
            text=self.current_mode,
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

        self.active_device_label = ctk.CTkLabel(
            mode_text_frame,
            text="ACTIVE INPUT: Detecting...",
            text_color="#8AB4F8",
            font=("Arial", 10, "bold"),
            anchor="w",
        )
        self.active_device_label.pack(anchor="w", pady=(6, 0))

        self.signal_label = ctk.CTkLabel(
            mode_text_frame,
            text="Signal: Unknown",
            text_color="#C6C6C6",
            font=("Arial", 9, "bold"),
            anchor="w",
        )
        self.signal_label.pack(anchor="w", pady=(2, 0))

        self.meter = AudioLevelMeter(mode_frame, width=760, height=78)
        self.meter.pack(fill="x", padx=12, pady=(0, 10))

        controls_frame = ctk.CTkFrame(self.monitor_tab)
        controls_frame.pack(fill="x", padx=6, pady=(0, 8))
        self._add_section_title(controls_frame, "Run Controls")

        mode_buttons = ctk.CTkFrame(controls_frame, fg_color="transparent")
        mode_buttons.pack(fill="x", padx=12, pady=(0, 10))
        for column in range(4):
            mode_buttons.grid_columnconfigure(column, weight=1)

        self.live_hot_switch_chip = ctk.CTkLabel(
            controls_frame,
            text="Live - clicking a mode will hot-switch",
            font=("Arial", 9, "bold"),
            text_color="#0B1F16",
            fg_color="#7DDC9A",
            corner_radius=10,
            padx=10,
            pady=4,
        )

        self.btn_mic = ctk.CTkButton(
            mode_buttons,
            text="Microphone",
            command=lambda: self.run_mode("Microphone"),
            height=40,
            font=("Arial", 10, "bold"),
            fg_color="#1565C0",
            hover_color="#0D47A1",
        )
        self.btn_mic.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.btn_vac = ctk.CTkButton(
            mode_buttons,
            text="VAC",
            command=lambda: self.run_mode("VAC"),
            height=40,
            font=("Arial", 10, "bold"),
            fg_color="#2E7D32",
            hover_color="#1B5E20",
        )
        self.btn_vac.grid(row=0, column=1, sticky="ew", padx=4)

        self.btn_mix = ctk.CTkButton(
            mode_buttons,
            text="Mixed",
            command=lambda: self.run_mode("Mixed"),
            height=40,
            font=("Arial", 10, "bold"),
            fg_color="#8E24AA",
            hover_color="#6A1B9A",
        )
        self.btn_mix.grid(row=0, column=2, sticky="ew", padx=4)

        self.mute_button = ctk.CTkButton(
            mode_buttons,
            text="Mute Toggle",
            command=self.toggle_mute,
            height=40,
            font=("Arial", 10, "bold"),
            fg_color="#D32F2F",
            hover_color="#B71C1C",
        )
        self.mute_button.grid(row=0, column=3, sticky="ew", padx=(4, 0))

        self.wer_status_frame = ctk.CTkFrame(self.monitor_tab, fg_color="#1A1A1A")
        self.wer_status_frame.pack(fill="x", padx=6, pady=(0, 8))

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

        self.mode_badge_label = ctk.CTkLabel(
            status_row,
            text="VAC",
            width=110,
            height=28,
            corner_radius=12,
            font=("Arial", 10, "bold"),
            fg_color="#2E7D32",
        )
        self.mode_badge_label.pack(side="right")

        self.warnings_box = ctk.CTkTextbox(
            self.wer_status_frame,
            height=52,
            font=("Arial", 9),
            fg_color="#121212",
            wrap="word",
        )
        self.warnings_box.pack(fill="x", padx=12, pady=(0, 10))
        self._set_warnings_text(self.monitor_recommendation_var.get())

    def _build_routing_tab(self) -> None:
        # --- Tab 2: Routing ---
        routing_frame = ctk.CTkFrame(self.routing_tab)
        routing_frame.pack(fill="x", padx=8, pady=(0, 12))
        self._add_section_title(routing_frame, "Direct Audio Device Control", "Set Windows input and output devices, then verify routing inline.")

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
            text="Normal",
            command=lambda: self.apply_device_preset("normal"),
            height=34,
            font=("Arial", 9, "bold"),
        ).pack(side="left", padx=(0, 6), expand=True, fill="x")

        ctk.CTkButton(
            preset_row,
            text="VAC",
            command=lambda: self.apply_device_preset("vac"),
            height=34,
            font=("Arial", 9, "bold"),
            fg_color="#2E7D32",
            hover_color="#1B5E20",
        ).pack(side="left", padx=6, expand=True, fill="x")

        ctk.CTkButton(
            preset_row,
            text="Voicemeeter",
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

        self.btn_auto_best_mode = ctk.CTkButton(
            utility_row,
            text="Auto Best Mode",
            command=self.auto_select_best_mode,
            height=34,
            font=("Arial", 10, "bold"),
            fg_color="#7B1FA2",
            hover_color="#5E1380",
        )
        self.btn_auto_best_mode.pack(side="left", padx=6, expand=True, fill="x")

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
        # --- Tab 3: Transcribe ---
        api_key_ready = bool(get_deepgram_api_key())

        options_frame = ctk.CTkFrame(self.transcribe_tab)
        options_frame.pack(fill="x", padx=8, pady=(0, 12))
        self._add_section_title(options_frame, "Deepgram")

        ctk.CTkLabel(
            options_frame,
            text=get_deepgram_settings_message(),
            font=("Arial", 11, "bold"),
            anchor="w",
            justify="left",
        ).pack(fill="x", padx=14, pady=(0, 4))

        ctk.CTkLabel(
            options_frame,
            text=get_deepgram_settings_detail(),
            font=("Arial", 10),
            text_color="#C6C6C6",
            wraplength=760,
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=14, pady=(0, 8))

        self.transcription_status_label = ctk.CTkLabel(
            options_frame,
            text=("Deepgram API key detected" if api_key_ready else "Deepgram API key missing from .env"),
            font=("Arial", 10, "bold"),
            text_color="#66BB6A" if api_key_ready else "#F9A825",
            wraplength=760,
            justify="left",
            anchor="w",
        )
        self.transcription_status_label.pack(fill="x", padx=14, pady=(0, 10))

        file_frame = ctk.CTkFrame(self.transcribe_tab)
        file_frame.pack(fill="x", padx=8, pady=(0, 12))
        self._add_section_title(file_frame, "File Transcription")

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
                + get_deepgram_settings_message()
                + "\n"
                + get_deepgram_settings_detail()
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

        live_options_row = ctk.CTkFrame(live_frame, fg_color="transparent")
        live_options_row.pack(fill="x", padx=12, pady=(0, 10))
        live_options_row.grid_columnconfigure(0, weight=1)
        live_options_row.grid_columnconfigure(1, weight=1)

        self.signal_check_toggle = ctk.CTkSwitch(
            live_options_row,
            text="Signal Required to Start",
            variable=self.require_signal_check_var,
            command=self.save_form_config,
            onvalue=True,
            offvalue=False,
        )
        self.signal_check_toggle.grid(row=0, column=0, sticky="w", padx=(0, 8))

        ctk.CTkButton(
            live_options_row,
            text="Auto Best Mode",
            command=self.auto_select_best_mode,
            height=32,
            font=("Arial", 10, "bold"),
            fg_color="#7B1FA2",
            hover_color="#5E1380",
        ).grid(row=0, column=1, sticky="e")

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

    def _build_settings_tab(self) -> None:
        # --- Tab 4: Settings ---
        settings_frame = ctk.CTkFrame(self.settings_tab)
        settings_frame.pack(fill="x", padx=8, pady=(0, 12))
        self._add_section_title(settings_frame, "Device Defaults", "Saved values used by mode switching and live transcription.")

        self.detected_input_devices_label = ctk.CTkLabel(
            settings_frame,
            text="Input devices: checking...",
            font=("Arial", 10),
            wraplength=760,
            justify="left",
            text_color="#C6C6C6",
            anchor="w",
        )
        self.detected_input_devices_label.pack(fill="x", padx=14, pady=(0, 6))

        self.detected_output_devices_label = ctk.CTkLabel(
            settings_frame,
            text="Output devices: checking...",
            font=("Arial", 10),
            wraplength=760,
            justify="left",
            text_color="#C6C6C6",
            anchor="w",
        )
        self.detected_output_devices_label.pack(fill="x", padx=14, pady=(0, 10))
        self._refresh_detection_summary()

        self.mic_menu = self._add_device_selector(settings_frame, "Microphone device", self.mic_var, self.detected_input_devices)
        self.vac_menu = self._add_device_selector(settings_frame, "VAC recording device", self.vac_var, self.detected_input_devices)
        self.speaker_menu = self._add_device_selector(settings_frame, "Speaker playback device", self.speaker_var, self.detected_output_devices)
        self.vac_playback_menu = self._add_device_selector(settings_frame, "VAC playback target", self.vac_playback_var, self.detected_output_devices)
        self.mix_menu = self._add_device_selector(settings_frame, "Voicemeeter device", self.mix_var, self._group_mix_device_values(self.mix_var.get()))
        self.mix_var.trace_add("write", self._on_mix_var_changed)

        self.mix_validation_label = ctk.CTkLabel(
            settings_frame,
            text="Voicemeeter devices are listed first. Other inputs are not recommended for Mixed mode.",
            font=("Arial", 9),
            text_color="#C6C6C6",
            wraplength=760,
            justify="left",
            anchor="w",
        )
        self.mix_validation_label.pack(fill="x", padx=14, pady=(0, 6))

        self.input_truth_label = ctk.CTkLabel(
            settings_frame,
            textvariable=self.input_truth_var,
            font=("Consolas", 9),
            text_color="#C6C6C6",
            wraplength=760,
            justify="left",
            anchor="w",
        )
        self.input_truth_label.pack(fill="x", padx=14, pady=(0, 10))

        actions = ctk.CTkFrame(settings_frame, fg_color="transparent")
        actions.pack(fill="x", padx=12, pady=(10, 12))
        for column in range(4):
            actions.grid_columnconfigure(column, weight=1)

        ctk.CTkButton(actions, text="Refresh Devices", command=self.refresh_detected_devices, height=34).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkButton(actions, text="Open config.json", command=self.open_config_file, height=34).grid(row=0, column=1, sticky="ew", padx=6)
        ctk.CTkButton(actions, text="Refresh Signal Preview", command=self._queue_input_truth_refresh, height=34).grid(row=0, column=2, sticky="ew", padx=6)
        ctk.CTkButton(actions, text="Save Settings", command=self.save_settings, height=34).grid(row=0, column=3, sticky="ew", padx=(6, 0))

        help_frame = ctk.CTkFrame(self.settings_tab)
        help_frame.pack(fill="x", padx=8, pady=(0, 12))
        self._add_section_title(help_frame, "Help")
        self.help_text_box = ctk.CTkTextbox(
            help_frame,
            height=220,
            font=("Arial", 10),
            wrap="word",
        )
        self.help_text_box.pack(fill="x", padx=14, pady=(0, 14))
        self.help_text_box.insert(
            "1.0",
            "\n".join(
                [
                    "1. Set your conferencing app microphone to 'Same as System' when routing through Windows defaults.",
                    "2. Use Monitor to confirm levels and WER health before starting a recording session.",
                    "3. Microphone mode is best for direct speech when room noise is acceptable.",
                    "4. VAC mode is best for playback-only transcription because the path stays digital end to end.",
                    "5. Mixed mode expects Voicemeeter to be configured before you switch into it.",
                    "6. The Routing tab changes Windows defaults for all roles, so stop live transcription first.",
                    "7. Use Test VAC Routing and the inline routing meter together to confirm signal presence.",
                    "8. Live Transcription shows the active input source, status, signal level, and transcript output.",
                    "9. Save Settings after changing device defaults so mode switching uses the updated assignments.",
                    "10. If monitoring or transcription fails, refresh devices and verify the active recording path first.",
                ]
            ),
        )
        self.help_text_box.configure(state="disabled")

    def _build_debug_tab(self) -> None:
        dashboard_frame = ctk.CTkFrame(self.debug_tab)
        dashboard_frame.pack(fill="both", expand=True, padx=8, pady=(0, 12))
        self._add_section_title(
            dashboard_frame,
            "Debug Dashboard",
            "Live session state, active devices, signal summary, and in-app logs.",
        )

        summary_frame = ctk.CTkFrame(dashboard_frame, fg_color="transparent")
        summary_frame.pack(fill="x", padx=14, pady=(0, 10))

        ctk.CTkLabel(summary_frame, textvariable=self.debug_session_var, anchor="w", justify="left").pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(summary_frame, textvariable=self.debug_device_var, anchor="w", justify="left").pack(fill="x", pady=(0, 4))
        self.debug_signal_label = ctk.CTkLabel(summary_frame, textvariable=self.debug_signal_var, anchor="w", justify="left")
        self.debug_signal_label.pack(fill="x", pady=(0, 4))
        self.debug_queue_label = ctk.CTkLabel(summary_frame, textvariable=self.debug_queue_var, anchor="w", justify="left")
        self.debug_queue_label.pack(fill="x", pady=(0, 4))
        self.debug_watchdog_label = ctk.CTkLabel(summary_frame, textvariable=self.debug_watchdog_var, anchor="w", justify="left")
        self.debug_watchdog_label.pack(fill="x", pady=(0, 4))
        self.debug_routing_label = ctk.CTkLabel(summary_frame, textvariable=self.debug_routing_var, anchor="w", justify="left")
        self.debug_routing_label.pack(fill="x", pady=(0, 6))
        self.debug_suggestion_var = ctk.StringVar(value="Suggested Fix: None")
        self.debug_suggestion_label = ctk.CTkLabel(summary_frame, textvariable=self.debug_suggestion_var, anchor="w", justify="left")
        self.debug_suggestion_label.pack(fill="x", pady=(0, 6))

        self.safe_mode_switch = ctk.CTkSwitch(
            summary_frame,
            text="Safe Mode (recommended)",
            variable=self.safe_mode_var,
            command=self._on_safe_mode_toggle,
            onvalue=True,
            offvalue=False,
        )
        self.safe_mode_switch.pack(anchor="w")

        self.apply_fix_button = ctk.CTkButton(
            summary_frame,
            text="Apply Suggested Fix",
            command=self._on_apply_fix,
            state="disabled",
        )
        self.apply_fix_button.pack(anchor="w", pady=(6, 0))

        self.debug_log_box = ctk.CTkTextbox(
            dashboard_frame,
            height=320,
            font=("Consolas", 10),
            wrap="word",
        )
        self.debug_log_box.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        self.debug_log_box.insert("1.0", "Debug log stream will appear here.")
        self.debug_log_box.configure(state="disabled")

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

    def _group_mix_device_values(self, current_value: str) -> list[str]:
        voicemeeter_devices = [device for device in self.detected_input_devices if "voicemeeter" in normalize_audio_device_name(device).lower()]
        mixed_capable_devices = [
            device
            for device in self.detected_input_devices
            if device not in voicemeeter_devices and is_valid_mixed_input_device(device)
        ]
        other_devices = [
            device
            for device in self.detected_input_devices
            if device not in voicemeeter_devices and device not in mixed_capable_devices
        ]
        ordered = [*voicemeeter_devices, *mixed_capable_devices, *other_devices]
        return self._menu_values_for(current_value, ordered)

    def _is_voicemeeter_choice_valid(self, device_name: str) -> bool:
        return is_valid_mixed_input_device(device_name)

    def _warn_non_voicemeeter_choice(self, selected_device: str) -> bool:
        choice = messagebox.askyesno(
            "Mixed Input Not Recognized",
            (
                "The device you selected "
                f"({selected_device}) does not look like a mixed-capable loopback source.\n\n"
                'Mixed mode works best with devices such as "CABLE Output (VB-Audio Virtual Cable)", '
                '"Stereo Mix", or a Voicemeeter virtual input. Without one, Mixed mode may remain unavailable.\n\n'
                "Choose Yes to save anyway, or No to pick a different device."
            ),
        )
        log_event(
            "Settings",
            event="voicemeeter_field_non_voicemeeter",
            device=selected_device,
            user_choice="save_anyway" if choice else "pick_a_different_device",
        )
        return bool(choice)

    def _on_mix_var_changed(self, *_args) -> None:
        selected = self.mix_var.get().strip()
        label = getattr(self, "mix_validation_label", None)
        if label is None or not label.winfo_exists():
            return
        if selected and not self._is_voicemeeter_choice_valid(selected):
            label.configure(
                text=f"Warning: {selected} does not look like a loopback/mixed input. Mixed mode may remain unavailable.",
                text_color="#F9A825",
            )
        else:
            label.configure(
                text="Voicemeeter, VB Cable, and Stereo Mix inputs are listed first for Mixed mode.",
                text_color="#C6C6C6",
            )

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

    def _enqueue_debug_log(self, message: str, level: int = logging.INFO) -> None:
        queue_ref = getattr(self, "_debug_log_queue", None)
        if queue_ref is None:
            return
        try:
            queue_ref.put_nowait((message, level))
        except queue.Full:
            try:
                queue_ref.get_nowait()
            except queue.Empty:
                return
            try:
                queue_ref.put_nowait((message, level))
            except queue.Full:
                return

    def _update_debug_queue(self, size: int) -> None:
        if size < 10:
            color = "#4CAF50"
        elif size < 30:
            color = "#F9A825"
        else:
            color = "#D32F2F"
        self.debug_queue_var.set(f"Queue: {size}")
        label = getattr(self, "debug_queue_label", None)
        if label is not None and label.winfo_exists():
            label.configure(text_color=color)

    def _update_debug_watchdog(self, state: str) -> None:
        normalized = str(state or "ok").strip().lower()
        color_map = {
            "ok": "#4CAF50",
            "stalled": "#D32F2F",
        }
        self.debug_watchdog_var.set(f"Audio: {normalized.upper()}")
        label = getattr(self, "debug_watchdog_label", None)
        if label is not None and label.winfo_exists():
            label.configure(text_color=color_map.get(normalized, "#C6C6C6"))

    def _update_debug_routing(self, mode: str, requested: str, resolved: str, fallback: bool) -> None:
        status = "FALLBACK" if fallback else "DIRECT"
        self.debug_routing_var.set(
            f"Routing: {mode} | {status}\nReq: {requested or '-'}\nRes: {resolved or '-'}"
        )

    def _on_safe_mode_toggle(self) -> None:
        enabled = bool(self.safe_mode_var.get())
        self.autofix.set_safe_mode(enabled)
        debug_log(
            "Safe Mode ENABLED (no automatic disruptive fixes)"
            if enabled
            else "Safe Mode DISABLED (AutoFix fully automatic)"
        )

    def on_autofix_suggestion(self, description: str) -> None:
        self.debug_suggestion_var.set(f"Suggested Fix: {description}")
        button = getattr(self, "apply_fix_button", None)
        if button is not None and button.winfo_exists():
            button.configure(state="normal")

    def on_autofix_cleared(self) -> None:
        self.debug_suggestion_var.set("Suggested Fix: None")
        button = getattr(self, "apply_fix_button", None)
        if button is not None and button.winfo_exists():
            button.configure(state="disabled")

    def _on_apply_fix(self) -> None:
        success = self.autofix.apply_last_suggestion()
        if success:
            debug_log("Suggested fix applied", level="info")
        else:
            debug_log("No suggested fix to apply", level="warning")

    def _poll_debug_log_queue(self) -> None:
        if self._closing:
            return
        log_box = getattr(self, "debug_log_box", None)
        root = getattr(self, "root", None)
        if log_box is None or root is None or not log_box.winfo_exists() or not root.winfo_exists():
            return
        drained: list[tuple[str, int]] = []
        while True:
            try:
                drained.append(self._debug_log_queue.get_nowait())
            except queue.Empty:
                break
        if drained:
            log_box.configure(state="normal")
            if log_box.get("1.0", "end-1c") == "Debug log stream will appear here.":
                log_box.delete("1.0", "end")
            for message, level in drained:
                prefix = "ERROR: " if level >= logging.ERROR else ("WARN: " if level >= logging.WARNING else "")
                log_box.insert("end", prefix + message + "\n")
            log_box.see("end")
            log_box.configure(state="disabled")
        root.after(250, self._poll_debug_log_queue)

    def _install_debug_log_handler(self) -> None:
        if self._debug_log_handler is not None:
            return
        handler = UILogHandler(
            self._enqueue_debug_log,
            self._update_debug_watchdog,
            self.autofix.on_watchdog_stalled,
        )
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        LOGGER.addHandler(handler)
        self._debug_log_handler = handler
        self._update_debug_watchdog("ok")
        self._poll_debug_log_queue()

    def _refresh_detected_devices(self, *, force: bool = False) -> None:
        """Refresh the cached device list. Call this before any action that needs fresh state."""
        now = time.monotonic()
        if not force and (now - self._last_detected_refresh_at) < 0.2:
            return
        engine = getattr(self, "device_detector", None) or AutoAudioEngine(logger=LOGGER)
        try:
            engine.refresh_devices()
            new_inputs = engine.list_input_names()
            new_outputs = engine.list_output_names()
        except Exception as exc:
            log_event("Devices", level="warning", event="auto_refresh_failed", reason=str(exc))
            new_inputs = list_input_devices()
            new_outputs = list_output_devices()
        if new_inputs != self.detected_input_devices:
            LOGGER.debug(
                "[Devices] event=inputs_changed old_count=%d new_count=%d",
                len(self.detected_input_devices),
                len(new_inputs),
            )
            cache = getattr(self, "_resolved_input_name_cache", None)
            if cache is not None:
                cache.clear()
        self.detected_input_devices = new_inputs
        self.detected_output_devices = new_outputs
        self._last_detected_refresh_at = now

    def _clear_active_input_device_lock(self) -> None:
        state = getattr(self, "audio_state", None)
        if state is None:
            return
        state.selected_device_name = None
        state.resolved_device_index = None
        state.locked = False

    def _lock_active_input_device(self, active_device: ActiveAudioDevice | None) -> None:
        state = getattr(self, "audio_state", None)
        if state is None:
            return
        if active_device is None:
            self._clear_active_input_device_lock()
            return
        state.selected_device_name = normalize_audio_device_name(active_device["name"])
        state.resolved_device_index = int(active_device["index"])
        state.locked = True
        log_event(
            "AUDIO",
            event="active_input_locked",
            device=state.selected_device_name,
            index=state.resolved_device_index,
        )

    def _format_input_truth_entry(self, device_name: str, rms_db: float | None) -> str:
        if rms_db is None:
            return f"{device_name} -> unavailable ?"
        if rms_db < -80.0:
            status = "NO SIGNAL"
        elif rms_db > -3.0:
            status = "CLIPPING"
        else:
            status = "OK"
        return f"{device_name} -> {rms_db:.1f} dB [{status}]"

    def _sample_input_truth_db(self, device_index: int, device_name: str, *, duration_seconds: float = 0.2) -> float | None:
        try:
            signal = sample_resolved_input_signal(
                device_index,
                LIVE_TRANSCRIPTION_SAMPLE_RATE_HZ,
                device_name,
                duration_seconds=duration_seconds,
            )
        except Exception as exc:
            log_event("SignalPreview", level="warning", event="sample_failed", device=device_name, index=device_index, reason=str(exc))
            return None
        if signal is None:
            return None
        return float(signal.get("rms_db", linear_to_db(float(signal.get("rms", 0.0)))))

    def _apply_input_truth_snapshot(self, rows: list[str]) -> None:
        text = "Input truth:\n" + ("\n".join(rows) if rows else "No input devices detected.")
        self.input_truth_var.set(text)

    def _queue_input_truth_refresh(self) -> None:
        root = getattr(self, "root", None)
        if root is None:
            return
        self.input_truth_var.set("Input truth: scanning live signal across detected inputs...")

        def _worker() -> None:
            rows: list[str] = []
            try:
                devices = sd.query_devices()
            except Exception as exc:
                rows = [f"Unable to enumerate input devices: {exc}"]
            else:
                for index, device in enumerate(devices):
                    if int(device.get("max_input_channels", 0)) <= 0:
                        continue
                    device_name = normalize_audio_device_name(str(device.get("name", f"Input {index}")))
                    rms_db = self._sample_input_truth_db(index, device_name)
                    rows.append(self._format_input_truth_entry(device_name, rms_db))

            if self._closing:
                return
            try:
                root.after(0, lambda: self._apply_input_truth_snapshot(rows))
            except Exception as exc:
                log_event("SignalPreview", level="warning", event="queue_refresh_failed", reason=str(exc))

        threading.Thread(target=_worker, daemon=True, name="input-truth-scan").start()

    def _active_source_summary(self) -> tuple[str, str]:
        if self.active_audio_device is None:
            return "⚠️ Source: None — Click a mode to begin.", "#9E9E9E"

        device_name = self.active_audio_device["name"]
        if not self._active_device_matches_mode(self.current_mode):
            return f"⚠️ Source: Mismatch — UI says {self.current_mode}, Windows says {device_name}", "#FB8C00"

        color = MODE_UI.get(self.current_mode, MODE_UI["Microphone"])["accent"]
        return f" Source: {self.current_mode} — {device_name}", color

    def _reconcile_startup_mode(self) -> None:
        """Validate that the loaded mode is actually available. Fall back if not."""
        fallback_mode: ModeName | None = None
        reason = ""
        if self.current_mode == "Mixed" and not self._is_mixed_mode_available():
            fallback_mode = "Microphone"
            reason = self._mixed_unavailable_reason(self.mix_var.get().strip())
        elif self.current_mode == "VAC" and not any("cable output" in normalize_audio_device_name(device).lower() for device in self.detected_input_devices):
            fallback_mode = "Microphone"
            reason = "VAC recording device not detected on boot"
        elif self.current_mode == "Microphone" and self.mic_var.get().strip() not in self.detected_input_devices:
            log_event("Startup", level="warning", event="mode_warning", loaded_mode="Microphone", reason="Configured microphone not detected on boot")

        if fallback_mode is not None:
            log_event("Startup", event="mode_unavailable", loaded_mode=self.current_mode, fallback=fallback_mode, reason=reason)
            self.current_mode = fallback_mode
            self.config["last_mode"] = fallback_mode
            self.mode_var.set(fallback_mode)
            save_config(self.config)
            try:
                self.active_audio_device = self.resolve_active_device(self._current_live_input_device_name())
                self._lock_active_input_device(self.active_audio_device)
            except Exception as exc:
                log_event("Startup", level="warning", event="resolve_fallback_failed", mode=fallback_mode, reason=str(exc))
                self.active_audio_device = None
                self._clear_active_input_device_lock()
            if reason:
                self.status_var.set(reason)
                self.debug_routing_var.set(f"Routing: Mixed | UNAVAILABLE\nReason: {reason}")

        self._refresh_run_control_buttons()

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
        self.debug_device_var.set(f"Devices: input={input_name} | output={output_name}")
        active_device_label = getattr(self, "active_device_label", None)
        if active_device_label is not None and active_device_label.winfo_exists():
            state = getattr(self, "audio_state", None)
            if state is not None and state.locked and state.selected_device_name:
                active_device_label.configure(text=f"ACTIVE INPUT: {state.selected_device_name} (locked #{state.resolved_device_index})")
            else:
                active_device_label.configure(text=f"ACTIVE INPUT: {input_name} (unlocked)")
        signal_label = getattr(self, "signal_label", None)
        if signal_label is not None and signal_label.winfo_exists():
            signal_label.configure(text=f"Signal: {self._latest_signal_state}")
        active_source_label = getattr(self, "active_source_label", None)
        if active_source_label is not None and active_source_label.winfo_exists():
            summary_text, summary_color = self._active_source_summary()
            self.active_source_var.set(summary_text)
            active_source_label.configure(text_color=summary_color)

    def _log_mixed_unavailable(self, requested: str, reason: str, *, log_failure_level: str = "debug") -> None:
        if log_failure_level == "error":
            log_failure("ROUTING", mode="Mixed", requested=requested, resolved="", reason=reason)
            self._last_mixed_unavailable_reason = reason
            return
        if self._last_mixed_unavailable_reason == reason:
            log_event("Mixed", level="debug", event="still_unavailable", reason=reason, requested=requested)
            return
        log_failure("ROUTING", mode="Mixed", requested=requested, resolved="", reason=reason)
        self._last_mixed_unavailable_reason = reason

    def _mark_mixed_available(self) -> None:
        if self._last_mixed_unavailable_reason is not None:
            log_event("Mixed", event="became_available", previous_reason=self._last_mixed_unavailable_reason)
        self._last_mixed_unavailable_reason = None

    def _is_mixed_mode_available(self, *, log_failure_level: str = "debug") -> bool:
        self._refresh_detected_devices()
        configured = self.mix_var.get().strip()
        if configured:
            resolver_log_level = "error" if log_failure_level == "error" else "debug"
            if self._resolve_detected_input_name(configured, "Mixed", log_failure_level=resolver_log_level):
                self._mark_mixed_available()
                return True
            return False
        available = any(is_valid_mixed_input_device(device) for device in self.detected_input_devices)
        if available:
            self._mark_mixed_available()
            return True
        self._log_mixed_unavailable(
            configured,
            self._mixed_unavailable_reason(configured),
            log_failure_level=log_failure_level,
        )
        return False

    def _resolve_detected_input_name(self, requested_name: str, mode_name: ModeName, *, log_failure_level: str = "error") -> str:
        requested = requested_name.strip()
        cache = getattr(self, "_resolved_input_name_cache", None)
        if cache is None:
            cache = {}
            self._resolved_input_name_cache = cache
        cache_key = (str(mode_name), requested, tuple(self.detected_input_devices))
        if cache_key in cache:
            resolved = cache[cache_key]
            if mode_name == "Mixed" and resolved:
                self._mark_mixed_available()
            return resolved
        log_event("Resolver", mode=mode_name, requested=requested or "<empty>", candidates_count=len(self.detected_input_devices))
        if not requested:
            if mode_name == "Mixed":
                self._log_mixed_unavailable(requested, "No input device configured for Mixed mode.", log_failure_level=log_failure_level)
            return ""
        if requested in self.detected_input_devices:
            if mode_name == "Mixed" and not is_valid_mixed_input_device(requested):
                self._log_mixed_unavailable(
                    requested,
                    self._mixed_unavailable_reason(requested),
                    log_failure_level=log_failure_level,
                )
            else:
                log_event("Resolver", mode=mode_name, requested=requested, resolved=requested, match_type="exact")
                if mode_name == "Mixed":
                    self._mark_mixed_available()
                cache[cache_key] = requested
                return requested

        normalized_requested = normalize_audio_device_name(requested)
        for device in self.detected_input_devices:
            normalized_device = normalize_audio_device_name(device)
            if normalized_device == normalized_requested:
                if mode_name == "Mixed" and not is_valid_mixed_input_device(normalized_device):
                    continue
                log_event("Resolver", mode=mode_name, requested=requested, resolved=device, match_type="normalized")
                if mode_name == "Mixed":
                    self._mark_mixed_available()
                cache[cache_key] = device
                return device

        resolved_index, resolved_info = resolve_input_device(requested)
        if resolved_index is not None and resolved_info is not None:
            resolved_name = normalize_audio_device_name(str(resolved_info.get("name", requested)))
            for device in self.detected_input_devices:
                if normalize_audio_device_name(device) == resolved_name:
                    if mode_name == "Mixed" and not is_valid_mixed_input_device(resolved_name):
                        break
                    log_event("Resolver", mode=mode_name, requested=requested, resolved=device, match_type="resolved")
                    if mode_name == "Mixed":
                        self._mark_mixed_available()
                    cache[cache_key] = device
                    return device

        if mode_name == "Mixed":
            self._log_mixed_unavailable(
                requested,
                self._mixed_unavailable_reason(requested),
                log_failure_level=log_failure_level,
            )
        else:
            log_event("Resolver", mode=mode_name, requested=requested, resolved="", match_type="none", level="warning")
        return ""

    def _refresh_run_control_buttons(self) -> None:
        active_name = self.current_mode
        button_map = {
            "Microphone": getattr(self, "btn_mic", None),
            "VAC": getattr(self, "btn_vac", None),
            "Mixed": getattr(self, "btn_mix", None),
            "Mute Toggle": getattr(self, "mute_button", None),
        }
        for mode_name, button in button_map.items():
            if button is None or not button.winfo_exists():
                continue
            is_active = (mode_name == active_name) or (mode_name == "Mute Toggle" and self.is_muted)
            target_text = mode_name
            if mode_name == "Mute Toggle":
                target_text = "Muted — Click to Unmute" if self.is_muted else "Mute Toggle"
            elif self._audio_switch_in_progress and self._pending_mode_button == mode_name:
                target_text = "Switching..."
            button.configure(
                border_width=2 if is_active else 0,
                border_color="#E0E0E0" if is_active else button.cget("fg_color"),
                text=target_text,
            )
        mixed_button = getattr(self, "btn_mix", None)
        if mixed_button is not None and mixed_button.winfo_exists():
            mixed_available = self._is_mixed_mode_available()
            mixed_button.configure(
                state="normal" if mixed_available else "disabled",
                text="Switching..." if self._audio_switch_in_progress and self._pending_mode_button == "Mixed" else ("Mixed" if mixed_available else "Mixed Unavailable"),
            )

    def _menu_values_for(self, current_value: str, devices: list[str]) -> list[str]:
        values = list(devices)
        if current_value and current_value not in values:
            values.insert(0, current_value)
        if not values:
            values = [current_value or "No devices detected"]
        return values

    def _validate_last_mode_against_detected_devices(self) -> None:
        """Downgrade self.current_mode in memory if the saved last_mode's input device is absent.

        Checks whether the input device configured for the current mode is present in
        self.detected_input_devices. If not, picks the next available mode using the
        same fallback chain as runtime (current_mode -> VAC -> Microphone).

        Does NOT modify self.config or write to disk; the user's saved preference is
        preserved so it can be restored naturally on a later launch when the original
        device (e.g. Voicemeeter, CABLE Output) is present again.
        """
        saved_mode: ModeName = self.current_mode
        if saved_mode not in ("Microphone", "VAC", "Mixed"):
            return
        if saved_mode != "Microphone":
            preferred_mic, _preferred_playback = self._resolve_mode_devices("Microphone")
            if preferred_mic:
                preferred_mic_resolved = self._resolve_detected_input_name(preferred_mic, "Microphone") or preferred_mic
                if preferred_mic_resolved in self.detected_input_devices:
                    log_event(
                        "StartupValidation",
                        event="last_mode_reset_to_microphone",
                        saved_mode=saved_mode,
                        effective_mode="Microphone",
                        preferred_device=preferred_mic_resolved,
                        note="startup always prefers direct microphone mode when available",
                    )
                    self.current_mode = "Microphone"
                    mode_var = getattr(self, "mode_var", None)
                    if mode_var is not None:
                        try:
                            mode_var.set("Microphone")
                        except Exception as exc:
                            log_event("StartupValidation", level="warning", event="mode_var_update_failed", reason=str(exc))
                    return
        configured_device, _playback = self._resolve_mode_devices(saved_mode)
        if not configured_device:
            log_event(
                "StartupValidation",
                event="no_device_configured",
                saved_mode=saved_mode,
            )
            return
        resolved = self._resolve_detected_input_name(configured_device, saved_mode) or configured_device
        if resolved in self.detected_input_devices:
            return
        # Device is absent. Walk the fallback chain starting from saved_mode.
        fallback_chain: dict[str, list[ModeName]] = {
            "Mixed": ["VAC", "Microphone"],
            "VAC": ["Microphone"],
            "Microphone": [],
        }
        for candidate in fallback_chain.get(saved_mode, []):
            candidate_device, _candidate_playback = self._resolve_mode_devices(candidate)
            if not candidate_device:
                continue
            candidate_resolved = self._resolve_detected_input_name(candidate_device, candidate) or candidate_device
            if candidate_resolved in self.detected_input_devices:
                log_event(
                    "StartupValidation",
                    event="last_mode_downgraded",
                    saved_mode=saved_mode,
                    effective_mode=candidate,
                    missing_device=configured_device,
                    note="config.json left untouched; saved preference preserved",
                )
                self.current_mode = candidate
                mode_var = getattr(self, "mode_var", None)
                if mode_var is not None:
                    try:
                        mode_var.set(candidate)
                    except Exception as exc:
                        log_event("StartupValidation", level="warning", event="mode_var_update_failed", reason=str(exc))
                return
        log_event(
            "StartupValidation",
            event="no_fallback_available",
            saved_mode=saved_mode,
            missing_device=configured_device,
        )

    def _mixed_unavailable_reason(self, requested: str) -> str:
        configured = normalize_audio_device_name(requested).strip()
        if configured and not is_valid_mixed_input_device(configured):
            return (
                f'Mixed disabled: configured device "{requested}" is not a usable input. '
                "Select any detected recording input in Settings to enable Mixed mode."
            )
        return "No detected recording inputs are available for Mixed mode."

    def _sanitize_mixed_input_configuration(self) -> None:
        configured = self.mix_var.get().strip()
        if not configured or self._is_voicemeeter_choice_valid(configured):
            return
        replacement = pick_mixed_input_device(configured, self.detected_input_devices)
        new_value = replacement if replacement and replacement in self.detected_input_devices and self._is_voicemeeter_choice_valid(replacement) else ""
        log_event(
            "StartupValidation",
            level="warning",
            event="mixed_device_reset",
            previous_device=configured,
            replacement=new_value or "<empty>",
            reason="Configured Mixed device is not mixed-capable.",
        )
        self.config["voicemeeter_device"] = new_value
        self.mix_var.set(new_value)
        save_config(self.config)

    def _hydrate_config_from_detected_devices(self) -> None:
        devices = self.detected_input_devices
        current_mic = self.config.get("mic_device", "")
        current_vac = self.config.get("vac_device", "")
        current_speaker = self.config.get("speaker_device", "")
        current_vac_playback = self.config.get("vac_playback_device", "")
        current_mixed_playback = self.config.get("mixed_playback_device", "")
        current_mix = self.config.get("voicemeeter_device", "")
        output_devices = self.detected_output_devices
        engine = getattr(self, "device_detector", None) or AutoAudioEngine(logger=LOGGER)
        try:
            engine.refresh_devices()
        except Exception as exc:
            log_event("Devices", level="warning", event="auto_hydrate_refresh_failed", reason=str(exc))
            engine = None

        def _match_detected_device(candidate_name: str, detected_names: list[str]) -> str:
            normalized_candidate = normalize_audio_device_name(candidate_name)
            for detected_name in detected_names:
                if normalize_audio_device_name(detected_name) == normalized_candidate:
                    return detected_name
            return ""

        best_mic_entry = engine.select_best_input_device("Microphone") if engine is not None else None
        best_vac_input_entry = engine.select_best_input_device("VAC") if engine is not None else None
        best_mixed_input_entry = engine.select_best_input_device("Mixed") if engine is not None else None
        best_speaker_output_entry = engine.select_best_output_device("Microphone") if engine is not None else None
        best_vac_output_entry = engine.select_best_output_device("VAC") if engine is not None else None

        best_mic = _match_detected_device(best_mic_entry.name, devices) if best_mic_entry is not None else ""
        best_vac_input = _match_detected_device(best_vac_input_entry.name, devices) if best_vac_input_entry is not None else ""
        best_mixed_input = _match_detected_device(best_mixed_input_entry.name, devices) if best_mixed_input_entry is not None else ""
        best_speaker_output = _match_detected_device(best_speaker_output_entry.name, output_devices) if best_speaker_output_entry is not None else ""
        best_vac_output = _match_detected_device(best_vac_output_entry.name, output_devices) if best_vac_output_entry is not None else ""
        default_input_name = ""
        try:
            _default_input_index, default_input_info = get_default_input_device()
        except Exception as exc:
            log_event("Devices", level="warning", event="default_input_for_hydrate_failed", reason=str(exc))
            default_input_info = None
        if default_input_info is not None:
            candidate_default_input = normalize_audio_device_name(str(default_input_info.get("name", "")))
            default_input_name = _match_detected_device(candidate_default_input, devices)
        preferred_default_mic = default_input_name if is_microphone_like_input(default_input_name) else ""

        if current_mic not in devices or not is_microphone_like_input(current_mic):
            self.config["mic_device"] = preferred_default_mic or best_mic or infer_microphone_input_device(current_mic, devices)
        elif (
            preferred_default_mic
            and normalize_audio_device_name(current_mic) != normalize_audio_device_name(preferred_default_mic)
            and is_generic_onboard_mic_input(current_mic)
            and not is_generic_onboard_mic_input(preferred_default_mic)
        ):
            log_event(
                "Devices",
                event="mic_device_replaced_with_default_input",
                previous=current_mic,
                replacement=preferred_default_mic,
                reason="Configured mic looked like a stale generic onboard input while Windows default input is a better microphone candidate.",
            )
            self.config["mic_device"] = preferred_default_mic

        if current_vac not in devices or not all(token in current_vac.lower() for token in ("cable", "output")):
            self.config["vac_device"] = best_vac_input or infer_vac_recording_device(DEFAULT_CONFIG["vac_device"], devices)

        if current_speaker not in output_devices or not any(token in current_speaker.lower() for token in ("speaker", "headphone", "realtek")):
            self.config["speaker_device"] = best_speaker_output or infer_speaker_output_device(current_speaker, output_devices)

        if current_vac_playback not in output_devices or not all(token in current_vac_playback.lower() for token in ("cable", "input")):
            self.config["vac_playback_device"] = best_vac_output or infer_vac_playback_device(DEFAULT_CONFIG["vac_playback_device"], output_devices)

        if current_mixed_playback and current_mixed_playback not in output_devices:
            self.config["mixed_playback_device"] = ""

        if current_mix not in devices or not is_valid_mixed_input_device(current_mix):
            self.config["voicemeeter_device"] = best_mixed_input or pick_mixed_input_device(DEFAULT_CONFIG["voicemeeter_device"], devices)

        save_config(self.config)

    def refresh_detected_devices(self) -> None:
        if self._live_transcription_running or self._live_transcription_starting:
            self.status_var.set("Stop live transcription before refreshing device lists.")
            return
        debug_log("[App] Refreshing detected input and output devices")
        self._refresh_detected_devices(force=True)
        self._hydrate_config_from_detected_devices()
        self.mic_var.set(self.config["mic_device"])
        self.vac_var.set(self.config["vac_device"])
        self.speaker_var.set(self.config["speaker_device"])
        self.vac_playback_var.set(self.config["vac_playback_device"])
        self.mix_var.set(self.config["voicemeeter_device"])
        self.mixed_playback_var.set(self.config.get("mixed_playback_device", ""))
        self.direct_recording_var.set(self._normalize_direct_device_selection(self.direct_recording_var.get(), self.detected_input_devices))
        self.direct_playback_var.set(self._normalize_direct_device_selection(self.direct_playback_var.get(), self.detected_output_devices))
        self._refresh_device_menus()
        self._refresh_detection_summary()
        self._refresh_mode_hint()
        self._refresh_runtime_audio_status()
        self._queue_input_truth_refresh()
        self.status_var.set("Refreshed detected Windows recording and playback devices.")

    def _normalize_direct_device_selection(self, current_value: str, devices: list[str]) -> str:
        if current_value in devices:
            return current_value
        if devices:
            return devices[0]
        return current_value

    def _capture_launch_audio_defaults(self) -> None:
        try:
            _input_index, input_info = get_default_input_device()
            _output_index, output_info = get_default_output_device()
            if input_info and isinstance(input_info.get("name"), str):
                self._original_default_input_name = input_info["name"].strip() or None
            if output_info and isinstance(output_info.get("name"), str):
                self._original_default_output_name = output_info["name"].strip() or None
            log_event(
                "App",
                event="captured_default_devices",
                input=self._original_default_input_name or "",
                output=self._original_default_output_name or "",
            )
        except Exception as exc:
            log_event("App", level="warning", event="capture_defaults_failed", reason=str(exc))

    def use_current_windows_output_for_speakers(self) -> None:
        output_index, output_info = get_default_output_device()
        if output_index is None or output_info is None:
            self.status_var.set("Unable to read the current Windows output device.")
            return
        device_name = normalize_audio_device_name(str(output_info.get("name", "")))
        if not device_name:
            self.status_var.set("Windows reported an empty output device name.")
            return
        self.speaker_var.set(device_name)
        self.config["speaker_device"] = device_name
        save_config(self.config)
        self._refresh_mode_hint()
        self._refresh_runtime_audio_status()
        self.status_var.set(f"Speaker playback device set to the current Windows output: {device_name}.")

    def _restore_original_default_devices(self) -> tuple[bool, bool]:
        try:
            restored_input = False
            restored_output = False
            if self._original_default_input_name:
                ok, _message = self.device_manager.set_default_recording_device(self._original_default_input_name)
                restored_input = ok
            if self._original_default_output_name:
                ok, _message = self.device_manager.set_default_playback_device(self._original_default_output_name)
                restored_output = ok
            log_event(
                "App",
                event="restore_defaults",
                input=self._original_default_input_name or "",
                output=self._original_default_output_name or "",
                restored_input=restored_input,
                restored_output=restored_output,
            )
        except Exception as exc:
            log_event("App", level="warning", event="restore_defaults_failed", reason=str(exc))
            restored_input = False
            restored_output = False
        return restored_input, restored_output

    def reset_windows_audio(self) -> None:
        if self._live_transcription_running or self._live_transcription_starting:
            messagebox.showwarning(
                "Live Transcription Running",
                "Stop live transcription before resetting Windows audio.",
            )
            return

        input_restored, output_restored = self._restore_original_default_devices()
        if not input_restored:
            mic_name = str(self.config.get("mic_device", "")).strip()
            if mic_name:
                self.device_manager.set_default_recording_device(mic_name)
        if not output_restored:
            speaker_name = str(self.config.get("speaker_device", "")).strip()
            if speaker_name:
                self.device_manager.set_default_playback_device(speaker_name)

        self.status_var.set("Windows audio defaults reset.")
        messagebox.showinfo(
            "Audio Reset",
            "Windows default recording and playback devices have been restored.",
        )

    def _current_transcript_text(self) -> tuple[str, str]:
        live_text = self.live_transcript_final_text.strip()
        if live_text:
            live_name = ""
            if self.live_transcription_session is not None and self.live_transcription_session.transcript_path is not None:
                live_name = self.live_transcription_session.transcript_path.name
            return live_text, f"live session ({live_name})" if live_name else "live session"

        try:
            transcript_files = [path for path in TRANSCRIPTS_DIR.glob("*.txt") if path.is_file()]
        except OSError:
            transcript_files = []
        if not transcript_files:
            return "", ""
        newest = max(transcript_files, key=lambda path: path.stat().st_mtime)
        try:
            return newest.read_text(encoding="utf-8"), f"most recent file ({newest.name})"
        except OSError:
            return "", ""

    def copy_transcript_to_clipboard(self) -> None:
        text, source = self._current_transcript_text()
        if not text:
            self.status_var.set("No transcript text is available to copy.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_var.set(f"Copied transcript from {source} to the clipboard.")

    def save_transcript_as(self) -> None:
        text, source = self._current_transcript_text()
        if not text:
            self.status_var.set("No transcript text is available to save.")
            return
        target_path = filedialog.asksaveasfilename(
            title="Save Transcript As",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not target_path:
            return
        try:
            Path(target_path).write_text(text, encoding="utf-8")
        except OSError as exc:
            self.status_var.set(f"Failed to save transcript: {exc}")
            return
        self.status_var.set(f"Saved transcript from {source} to {Path(target_path).name}.")

    def _refresh_device_menus(self) -> None:
        menu_specs = [
            ("mic_menu", self.mic_var, self.detected_input_devices),
            ("vac_menu", self.vac_var, self.detected_input_devices),
            ("speaker_menu", self.speaker_var, self.detected_output_devices),
            ("vac_playback_menu", self.vac_playback_var, self.detected_output_devices),
            ("mixed_playback_menu", self.mixed_playback_var, ["", *self.detected_output_devices]),
            ("route_vac_playback_menu", self.vac_playback_var, self.detected_output_devices),
            ("mix_menu", self.mix_var, self._group_mix_device_values(self.mix_var.get())),
            ("route_mix_menu", self.mix_var, self._group_mix_device_values(self.mix_var.get())),
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
            except Exception as exc:
                log_event("App", level="warning", event="refresh_device_menu_failed", menu_name=menu_name, reason=str(exc))
                continue

    def _refresh_detection_summary(self) -> None:
        if self.detected_input_devices:
            joined_inputs = ", ".join(self.detected_input_devices[:8])
            if len(self.detected_input_devices) > 8:
                joined_inputs += ", ..."
            input_text = f"Input devices: {joined_inputs}"
        else:
            input_text = "Input devices: none. Check Windows audio drivers or device connection."

        if self.detected_output_devices:
            joined_outputs = ", ".join(self.detected_output_devices[:6])
            if len(self.detected_output_devices) > 6:
                joined_outputs += ", ..."
            output_text = f"Output devices: {joined_outputs}"
        else:
            output_text = "Output devices: none detected."

        input_label = getattr(self, "detected_input_devices_label", None)
        if input_label is not None and input_label.winfo_exists():
            input_label.configure(text=input_text)

        output_label = getattr(self, "detected_output_devices_label", None)
        if output_label is not None and output_label.winfo_exists():
            output_label.configure(text=output_text)

    def save_form_config(self) -> None:
        self.config["mic_device"] = self.mic_var.get().strip()
        self.config["vac_device"] = self.vac_var.get().strip()
        self.config["speaker_device"] = self.speaker_var.get().strip()
        self.config["vac_playback_device"] = self.vac_playback_var.get().strip()
        self.config["voicemeeter_device"] = self.mix_var.get().strip()
        self.config["mixed_playback_device"] = self.mixed_playback_var.get().strip()
        self.config["modes"]["mic"] = {
            "input_device": self.config["mic_device"],
            "output_device": self.config["speaker_device"],
        }
        self.config["modes"]["vac"] = {
            "input_device": self.config["vac_device"],
            "output_device": self.config["vac_playback_device"],
        }
        self.config["modes"]["mixed"] = {
            "input_device": self.config["voicemeeter_device"],
            "output_device": self.config["mixed_playback_device"] or self.config["speaker_device"],
        }
        self.config["active_mode"] = mode_name_to_key(self.current_mode)
        self.config["last_mode"] = self.current_mode
        self.config["restore_devices_on_exit"] = bool(self.restore_devices_on_exit_var.get())
        self.config["require_signal_check"] = bool(self.require_signal_check_var.get())
        self.config["wer_mode_enabled"] = bool(self.wer_enabled_var.get())
        save_config(self.config)
        self.status_var.set(f"Saved configuration to {CONFIG_PATH.name}.")

    def save_settings(self) -> None:
        selected_mix = self.mix_var.get().strip()
        if selected_mix and not self._is_voicemeeter_choice_valid(selected_mix):
            if not self._warn_non_voicemeeter_choice(selected_mix):
                self.status_var.set("Pick a different mixed-capable input before saving Mixed mode settings.")
                return
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

    def run_mode(self, mode_name: ModeName) -> None:
        if self._audio_switch_in_progress:
            self.status_var.set("Wait for the current audio device switch to finish.")
            return
        if self._transcription_running:
            self.status_var.set("Wait for the current file transcription job to finish.")
            return
        if self._live_transcription_running or self._live_transcription_starting:
            self._pending_live_start_mode = mode_name
            self.apply_audio_mode(mode_name)
            return

        self._pending_live_start_mode = mode_name
        self.apply_audio_mode(mode_name)

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
        normalized_name = normalize_audio_device_name(device_name)
        if normalized_name == NULL_INPUT_DEVICE_NAME:
            return build_silence_input_device(int(self.config["sample_rate_hz"]))
        device_index, device_info = resolve_input_device_exact(device_name)
        if device_index is None or device_info is None:
            raise RuntimeError(f"Failed to resolve device: {device_name}")

        sample_rate = int(float(device_info.get("default_samplerate", self.config["sample_rate_hz"])))
        if sample_rate <= 0:
            sample_rate = int(self.config["sample_rate_hz"])

        return {
            "name": normalize_audio_device_name(str(device_info.get("name", device_name))),
            "index": device_index,
            "info": device_info,
            "sample_rate": sample_rate,
        }

    def verify_active_device(self) -> bool:
        if self.active_audio_device is None:
            return False
        state = getattr(self, "audio_state", None)
        if state is not None and state.locked and state.resolved_device_index is not None:
            return int(self.active_audio_device["index"]) == int(state.resolved_device_index)
        current_index, _ = resolve_input_device(self.active_audio_device["name"])
        return current_index == self.active_audio_device["index"]

    def _expected_input_device_for_mode(self, mode_name: ModeName) -> str:
        configured_input, _playback = self._resolve_mode_devices(mode_name)
        if not configured_input:
            return ""
        resolved_name = self._resolve_detected_input_name(configured_input, mode_name)
        return normalize_audio_device_name(resolved_name or configured_input)

    def _active_device_matches_mode(self, mode_name: ModeName, active_device: ActiveAudioDevice | None = None) -> bool:
        candidate_device = active_device or self.active_audio_device
        if candidate_device is None:
            return False
        if bool(candidate_device["info"].get("is_synthetic")):
            return True
        expected_name = self._expected_input_device_for_mode(mode_name)
        active_name = normalize_audio_device_name(candidate_device["name"])
        if not expected_name:
            return False
        return active_name == expected_name

    def _probe_vac_route(self, duration_seconds: float = 0.35) -> tuple[dict[str, Any] | None, str]:
        active_device = self.active_audio_device
        playback_target = self.vac_playback_var.get().strip()
        if active_device is None:
            return None, "No active VAC recording device is bound."
        if not playback_target:
            return None, "No VAC playback target is configured."

        output_index, output_info = resolve_output_device(playback_target)
        if output_index is None or output_info is None:
            return None, f"Failed to resolve VAC playback target: {playback_target}"

        sample_rate = 48000
        try:
            sample_rate = int(float(output_info.get("default_samplerate", sample_rate)))
        except (TypeError, ValueError):
            sample_rate = 48000
        if sample_rate <= 0:
            sample_rate = 48000

        frequency_hz = 880.0
        amplitude = 0.22

        try:
            timeline = np.linspace(0.0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
            envelope = np.minimum(1.0, timeline * 6.0) * np.minimum(1.0, (duration_seconds - timeline) * 6.0)
            tone = (np.sin(2 * np.pi * frequency_hz * timeline) * envelope * amplitude).astype(np.float32)
            sd.play(tone, samplerate=sample_rate, device=output_index, blocking=False)
            time.sleep(0.12)
            signal = sample_resolved_input_signal(
                active_device["index"],
                active_device["sample_rate"],
                active_device["name"],
                duration_seconds=max(0.35, duration_seconds),
                device_info=active_device["info"],
            )
            sd.wait()
            return signal, normalize_audio_device_name(str(output_info.get("name", playback_target)))
        except Exception as exc:
            return None, f"VAC probe failed: {exc}"
        finally:
            try:
                sd.stop()
            except Exception as exc:
                log_event("Preflight", level="warning", step="vac_probe_cleanup", reason=str(exc))

    def _ask_yes_no_sync(self, title: str, message: str) -> bool:
        decision = {"value": False}
        ready = threading.Event()

        def _prompt() -> None:
            try:
                decision["value"] = bool(messagebox.askyesno(title, message))
            finally:
                ready.set()

        self.root.after(0, _prompt)
        ready.wait()
        return bool(decision["value"])

    def _run_preflight(self, mode_name: ModeName, active_device: ActiveAudioDevice) -> tuple[bool, str, dict[str, Any]]:
        self._refresh_detected_devices()
        diagnostics: dict[str, Any] = {
            "mode": mode_name,
            "device": active_device["name"],
            "device_index": active_device["index"],
            "sample_rate_hz": active_device["sample_rate"],
        }
        log_event("Preflight", step="1_start", mode=mode_name)
        if bool(active_device["info"].get("is_synthetic")):
            diagnostics.update(
                {
                    "exists": False,
                    "accessible": False,
                    "supported": False,
                    "signal_state": "silent",
                    "speech_state": "no_signal",
                    "feedback": "No valid input device is active. Synthetic silence cannot be used for live transcription.",
                }
            )
            self._queue_live_signal_update(
                {
                    "state": "silent",
                    "rms": 0.0,
                    "peak": 0.0,
                    "rms_db": linear_to_db(0.0),
                    "peak_db": linear_to_db(0.0),
                    "color": "#9E9E9E",
                    "detail": diagnostics["feedback"],
                    "device_name": active_device["name"],
                    "speech_state": "no_signal",
                    "feedback": diagnostics["feedback"],
                    "meter_color": "#9E9E9E",
                }
            )
            log_failure("DEVICE", reason=diagnostics["feedback"], **diagnostics)
            log_event("Preflight", step="2_synthetic_input", device=active_device["name"], result="failed")
            return False, "DEVICE", diagnostics

        normalized_active = normalize_audio_device_name(active_device["name"])
        exists = any(normalize_audio_device_name(device) == normalized_active for device in self.detected_input_devices)
        diagnostics["exists"] = exists
        log_event("Preflight", step="2_device_exists", device=active_device["name"], exists=exists)
        if not exists:
            log_failure("DEVICE", reason="Device is not present in detected_input_devices", **diagnostics)
            return False, "DEVICE", diagnostics

        try:
            sd.check_input_settings(
                device=active_device["index"],
                samplerate=active_device["sample_rate"],
                channels=1,
            )
            accessible = True
            supported = True
            check_error = ""
        except Exception as exc:
            accessible = False
            supported = False
            check_error = str(exc)
        diagnostics["accessible"] = accessible
        diagnostics["supported"] = supported
        log_event("Preflight", step="3_device_accessible", device=active_device["name"], accessible=accessible)
        log_event("Preflight", step="4_sample_rate", device=active_device["name"], supported=supported, rate_hz=active_device["sample_rate"])
        if not accessible:
            log_failure("DEVICE", reason=check_error or "Input settings check failed", **diagnostics)
            return False, "DEVICE", diagnostics

        if mode_name == "VAC":
            playback_target = self._resolve_mode_devices("VAC")[1]
            normalized_playback = normalize_audio_device_name(playback_target)
            if playback_target and normalized_playback and normalized_playback == normalized_active:
                diagnostics["feedback"] = "VAC recording and playback devices resolve to the same endpoint."
                log_failure("ROUTING", reason=diagnostics["feedback"], **diagnostics)
                return False, "ROUTING", diagnostics

        signal_duration = 1.0
        signal = sample_resolved_input_signal(
            active_device["index"],
            active_device["sample_rate"],
            active_device["name"],
            duration_seconds=signal_duration,
            device_info=active_device["info"],
        )
        if signal is None and mode_name == "VAC":
            time.sleep(0.2)
            signal = sample_resolved_input_signal(
                active_device["index"],
                active_device["sample_rate"],
                active_device["name"],
                duration_seconds=0.4,
                device_info=active_device["info"],
            )
            log_event("Preflight", step="5b_vac_reprobe", device=active_device["name"], signal_available=bool(signal))
        if (signal is None or str(signal.get("state", "")).lower() == "silent") and mode_name == "VAC":
            probe_signal, probe_target = self._probe_vac_route(duration_seconds=0.4)
            if probe_signal is not None:
                signal = probe_signal
            else:
                log_event("Preflight", step="5b_vac_reprobe", device=active_device["name"], signal_available=False, probe_target=probe_target, level="warning")

        rms = 0.0 if signal is None else float(signal.get("rms", 0.0))
        peak = 0.0 if signal is None else float(signal.get("peak", 0.0))
        rms_db = linear_to_db(rms) if signal is None else float(signal.get("rms_db", linear_to_db(rms)))
        peak_db = linear_to_db(peak) if signal is None else float(signal.get("peak_db", linear_to_db(peak)))
        speech_state, feedback, speech_color = classify_speech_signal(rms_db, peak_db)
        signal_state = "unavailable" if signal is None else str(signal.get("state", "unknown"))
        diagnostics["rms"] = rms
        diagnostics["peak"] = peak
        diagnostics["rms_db"] = rms_db
        diagnostics["peak_db"] = peak_db
        diagnostics["signal_state"] = signal_state
        diagnostics["speech_state"] = speech_state
        diagnostics["feedback"] = feedback
        log_event(
            "Preflight",
            step="5_signal_sample",
            device=active_device["name"],
            rms=f"{rms:.5f}",
            peak=f"{peak:.5f}",
            rms_db=f"{rms_db:.1f}",
            peak_db=f"{peak_db:.1f}",
            state=signal_state,
        )

        passes_threshold = rms_db >= PREFLIGHT_MIN_SIGNAL_DB and speech_state != "no_signal"
        diagnostics["threshold_db"] = PREFLIGHT_MIN_SIGNAL_DB
        diagnostics["passes_threshold"] = passes_threshold
        log_event("Preflight", step="6_signal_threshold", threshold_db=PREFLIGHT_MIN_SIGNAL_DB, rms_db=f"{rms_db:.1f}", passes=passes_threshold)
        if signal is not None:
            self._queue_live_signal_update(
                {
                    **signal,
                    "device_name": active_device["name"],
                    "speech_state": speech_state,
                    "feedback": feedback,
                    "meter_color": speech_color,
                }
            )
        if not passes_threshold:
            log_failure("SIGNAL", reason=feedback, **diagnostics)
            return False, "SIGNAL", diagnostics

        if speech_state in {"too_quiet", "too_loud", "clipping"}:
            log_event(
                "Preflight",
                step="6b_signal_quality",
                level="warning",
                state=speech_state,
                rms_db=f"{rms_db:.1f}",
                peak_db=f"{peak_db:.1f}",
                feedback=feedback,
            )

        expected_name = self._expected_input_device_for_mode(mode_name)
        resolved_mode_match = normalize_audio_device_name(active_device["name"]) == normalize_audio_device_name(expected_name)
        diagnostics["resolved_mode_match"] = resolved_mode_match
        log_event("Preflight", step="7_mode_vs_resolved", mode=mode_name, resolved_mode_match=resolved_mode_match)
        if not resolved_mode_match:
            log_failure("ROUTING", reason="Resolved input does not match the requested mode", **diagnostics)
            return False, "ROUTING", diagnostics

        if mode_name == "Mixed":
            mixed_input_available = any(is_valid_mixed_input_device(device) for device in self.detected_input_devices)
            diagnostics["mixed_input_available"] = mixed_input_available
            log_event("Preflight", step="8_mixed_input_required", required=True, available=mixed_input_available)
            if not mixed_input_available:
                log_failure("DEPENDENCY", reason="No mixed-capable input device is currently detected.", **diagnostics)
                return False, "DEPENDENCY", diagnostics
        else:
            log_event("Preflight", step="8_mixed_input_required", required=False, available=True)

        log_event("Preflight", step="9_done", result="ok")
        return True, "", diagnostics

    def _current_live_input_device_name(self) -> str:
        if self.active_audio_device is not None:
            return self.active_audio_device["name"]
        if self.current_mode == "VAC":
            return self.vac_var.get().strip() or "Not configured"
        if self.current_mode == "Mixed":
            return self.mix_var.get().strip() or "Not configured"
        return self.mic_var.get().strip() or "Not configured"

    def _routing_fallback_order(self, mode_name: ModeName) -> list[ModeName]:
        orders: dict[ModeName, list[ModeName]] = {
            "VAC": ["Mixed", "Microphone"],
            "Mixed": ["VAC", "Microphone"],
            "Microphone": ["VAC", "Mixed"],
        }
        return orders.get(mode_name, ["VAC", "Mixed", "Microphone"])

    def _select_working_live_input(self, mode_name: ModeName) -> tuple[ActiveAudioDevice | None, ModeName | None, dict[str, Any] | None]:
        detector = getattr(self, "device_detector", None) or AutoAudioEngine(logger=LOGGER)
        router = RoutingManager(
            detector,
            sample_resolved_input_signal,
            logger=LOGGER,
        )
        fallback_order = self._routing_fallback_order(mode_name)
        preferred_names = {candidate_mode: self._expected_input_device_for_mode(candidate_mode) for candidate_mode in fallback_order}
        selected_entry, selected_mode, signal = router.select_working_device(
            fallback_order,
            LIVE_TRANSCRIPTION_SAMPLE_RATE_HZ,
            preferred_names=preferred_names,
        )
        if selected_entry is None or selected_mode is None:
            return None, None, signal
        try:
            active_device = self.resolve_active_device(selected_entry.name)
        except Exception as exc:
            log_event("Routing", level="warning", event="fallback_resolve_failed", mode=selected_mode, device=selected_entry.name, reason=str(exc))
            return None, None, signal
        return active_device, selected_mode, signal

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
                    + get_deepgram_settings_message()
                    + "\n"
                    + get_deepgram_settings_detail()
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
        except Exception as exc:
            log_event("App", level="warning", event="live_transcript_scroll_warning", reason=str(exc))

    def _queue_live_transcript_update(self, final_text: str, interim_text: str) -> None:
        if self._closing:
            return
        try:
            self.root.after(0, lambda: self._apply_live_transcript_update(final_text, interim_text))
        except Exception as exc:
            log_event("App", level="warning", event="queue_live_transcript_update_failed", reason=str(exc))
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
        except Exception as exc:
            log_event("App", level="warning", event="queue_live_status_update_failed", reason=str(exc))
            return

    def _queue_live_signal_update(self, payload: dict[str, Any]) -> None:
        if self._closing:
            return
        try:
            self.root.after(0, lambda: self._apply_live_signal_update(payload))
        except Exception as exc:
            log_event("App", level="warning", event="queue_live_signal_update_failed", reason=str(exc))
            return

    def _apply_live_status_update(self, message: str) -> None:
        lowered = message.lower()
        is_problem = any(token in lowered for token in ("error", "failed", "unable"))
        is_warning = any(token in lowered for token in ("low input signal", "clipping", "lower the source level", "may be weak"))
        self.live_transcription_status_label.configure(
            text=message,
            text_color="#F57C00" if (is_problem or is_warning) else ("#66BB6A" if self._live_transcription_running else ("#F9A825" if self._live_transcription_starting else "#C6C6C6")),
        )
        if self._live_transcription_running:
            session_state = "Running"
        elif self._live_transcription_starting:
            session_state = "Starting"
        elif self._live_transcription_stopping:
            session_state = "Stopping"
        else:
            session_state = "Idle"
        self.debug_session_var.set(f"Session: {session_state} | Status: {message}")
        self.status_var.set(message)

    def _apply_live_signal_update(self, payload: dict[str, Any]) -> None:
        device_name = str(payload.get("device_name", "selected input"))
        rms = float(payload.get("rms", 0.0))
        peak = float(payload.get("peak", 0.0))
        rms_db = float(payload.get("rms_db", linear_to_db(rms)))
        peak_db = float(payload.get("peak_db", linear_to_db(peak)))
        detail = str(payload.get("detail", "")).strip()
        state = str(payload.get("state", "unknown")).strip().lower()
        speech_state = str(payload.get("speech_state", "")).strip().lower() or classify_speech_signal(rms_db, peak_db)[0]
        feedback = str(payload.get("feedback", "")).strip() or classify_speech_signal(rms_db, peak_db)[1]
        signal_text = f"Input signal: {feedback}. {detail} Device: {device_name}. RMS {rms_db:.1f} dB, Peak {peak_db:.1f} dB."
        self.live_signal_status_text = signal_text
        self.live_signal_status_label.configure(
            text=signal_text,
            text_color=str(payload.get("meter_color", payload.get("color", "#C6C6C6"))),
        )
        signal_state = {
            "no_signal": "No signal",
            "too_quiet": "Low signal",
            "too_loud": "Too loud",
            "clipping": "Clipping",
            "optimal": "Active",
            "silent": "No signal",
            "low": "Low signal",
            "active": "Active",
        }.get(speech_state or state, "Unknown")
        self.debug_signal_var.set(f"Signal: {signal_state} | RMS {rms_db:.1f} dB | Peak {peak_db:.1f} dB")
        signal_label = getattr(self, "debug_signal_label", None)
        if signal_label is not None and signal_label.winfo_exists():
            signal_label.configure(text_color=str(payload.get("meter_color", payload.get("color", "#C6C6C6"))))
        session = self.live_transcription_session
        queue_size = 0
        if session is not None:
            queue_ref = getattr(session, "_pcm_queue", None)
            if queue_ref is not None:
                try:
                    queue_size = int(queue_ref.qsize())
                except Exception:
                    queue_size = 0
        self._update_debug_queue(queue_size)
        if speech_state in {"optimal", "too_quiet", "too_loud", "clipping"}:
            self._update_debug_watchdog("ok")
        self.autofix.on_signal(speech_state, rms_db, peak_db)
        self.autofix.on_queue(queue_size)
        self._refresh_runtime_audio_status(signal_state=signal_state)

    def _silent_signal_message(self, mode_name: str, device_name: str, *, relaxed: bool = False) -> str:
        if mode_name == "VAC":
            base = (
                f"No audio detected on {device_name}.\n\n"
                "VAC mode requires system audio to be actively playing.\n\n"
                "Fix:\n"
                "1. Set your conferencing app microphone to 'Same as System'\n"
                f"2. Set Windows Output to '{self.vac_playback_var.get().strip() or 'the configured CABLE Input target'}'\n"
                "3. Play audio (YouTube, media, etc.)\n"
                "4. Try again"
            )
        elif mode_name == "Mixed":
            base = (
                f"No audio was detected on {device_name}. Mixed mode requires an active loopback/mixed source such as "
                "VB Cable, Stereo Mix, or Voicemeeter to be routing mic or system audio into the selected input."
            )
        else:
            base = (
                f"No audio was detected on {device_name}. Check mute state, mic gain, and whether the selected "
                "microphone is the active Windows recording source."
            )
        if relaxed:
            return base + " Starting anyway because Signal Required to Start is turned off."
        return base

    def _set_live_controls_state(self, *, running: bool = False, starting: bool = False, stopping: bool = False) -> None:
        self._live_transcription_running = running
        self._live_transcription_starting = starting
        self._live_transcription_stopping = stopping
        hot_switch_chip = getattr(self, "live_hot_switch_chip", None)
        if hot_switch_chip is not None and hot_switch_chip.winfo_exists():
            if running and not stopping:
                hot_switch_chip.pack(anchor="e", padx=12, pady=(0, 8))
            else:
                hot_switch_chip.pack_forget()
        for button_name in ("btn_start_live", "transcribe_btn_start_live"):
            button = getattr(self, button_name, None)
            if button is not None and button.winfo_exists():
                default_text = "Start" if button_name == "btn_start_live" else "Start Live Transcription"
                button.configure(
                    state="disabled" if (running or starting or stopping) else "normal",
                    text="Starting..." if starting else default_text,
                )
        for button_name in ("btn_stop_live", "transcribe_btn_stop_live"):
            button = getattr(self, button_name, None)
            if button is not None and button.winfo_exists():
                default_text = "Stop" if button_name == "btn_stop_live" else "Stop Live Transcription"
                button.configure(
                    state="disabled" if (stopping or not running) else "normal",
                    text="Stopping..." if stopping else default_text,
                )

    def start_live_transcription(self) -> None:
        log_section("Starting Live Transcription")
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
            message = "No active input device is bound. Apply a mode and verify the selected device before starting live transcription."
            self.live_transcription_status_label.configure(text=message, text_color="#F57C00")
            self.status_var.set(message)
            messagebox.showerror("No Active Input Device", message)
            return
        if not self._active_device_matches_mode(self.current_mode):
            expected_name = self._expected_input_device_for_mode(self.current_mode) or "the selected mode input"
            actual_name = active_device["name"]
            message = (
                f"Active device mismatch. Current mode is {self.current_mode}, expected {expected_name}, "
                f"but the active input is {actual_name}. Reapply the mode before starting live transcription."
            )
            self.live_transcription_status_label.configure(text=message, text_color="#F57C00")
            self.status_var.set(message)
            messagebox.showerror("Live Input Mismatch", message)
            return

        input_device_name = active_device["name"]
        debug_log(
            f"[App] Starting live transcription mode={self.current_mode} input={input_device_name} "
            f"index={active_device['index']} sample_rate={active_device['sample_rate']}"
        )
        self._update_debug_watchdog("ok")
        self._update_debug_queue(0)
        self._update_debug_routing(self.current_mode, input_device_name, input_device_name, False)
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
            ),
            daemon=True,
        ).start()

    def _start_live_transcription_worker(
        self,
        api_key: str,
        active_device: ActiveAudioDevice,
        mode_name: str,
    ) -> None:
        input_device_name = active_device["name"]
        if not self.verify_active_device():
            success = False
            message = f"Device mismatch detected before live transcription start: {input_device_name}"
            session = None
            try:
                self.root.after(0, lambda: self._finish_start_live_transcription(success, message, session))
            except Exception as exc:
                log_event("App", level="warning", event="queue_live_start_failure_failed", reason=str(exc))
                return
            return

        preflight_ok, failure_code, diagnostics = self._run_preflight(mode_name, active_device)
        if not preflight_ok:
            success = False
            message = str(diagnostics.get("feedback", "")).strip() or self._silent_signal_message(mode_name, input_device_name)
            session = None
            try:
                self.root.after(
                    0,
                    lambda: self._finish_start_live_transcription(
                        success,
                        message,
                        session,
                        failure_code=failure_code,
                        mode_name=mode_name,
                        device_name=input_device_name,
                    ),
                )
            except Exception as exc:
                log_event("App", level="warning", event="queue_live_start_failure_failed", reason=str(exc), failure_code=failure_code)
                return
            return

        session = LiveTranscriptionSession(
            api_key=api_key,
            input_device=active_device,
            mode_name=mode_name,
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
            requested_name = self._expected_input_device_for_mode(self.current_mode)
            self._update_debug_routing(mode_name, requested_name, active_device["name"], False)
            final_message = message
            feedback = str(diagnostics.get("feedback", "")).strip()
            if feedback and diagnostics.get("speech_state") != "optimal":
                final_message = f"{final_message} {feedback}".strip()
            self.root.after(0, lambda: self._finish_start_live_transcription(success, final_message, session))
        except Exception as exc:
            log_event("App", level="warning", event="queue_live_start_success_failed", reason=str(exc))
            if success:
                session.stop()
            return

    def _finish_start_live_transcription(
        self,
        success: bool,
        message: str,
        session: LiveTranscriptionSession | None,
        *,
        failure_code: str = "",
        mode_name: str = "",
        device_name: str = "",
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
            dialog_message = (
                self._silent_signal_message(mode_name, device_name)
                if failure_code == "SIGNAL" and mode_name and device_name
                else message
            )
            messagebox.showerror("Live Transcription Failed", dialog_message)
            self.status_var.set(message)
            return

        self.live_transcription_session = session
        debug_log(f"[App] Live transcription started successfully: {message}")
        self._set_live_controls_state(running=True, starting=False)
        lowered = message.lower()
        is_warning = any(token in lowered for token in ("clipping", "too quiet", "too loud", "no signal"))
        self.live_transcription_status_label.configure(text=message, text_color="#F57C00" if is_warning else "#66BB6A")
        signal_state = (
            "Clipping" if "clipping" in lowered else
            ("No signal" if "no signal" in lowered else
             ("Too loud" if "too loud" in lowered else
              ("Low signal" if "too quiet" in lowered else "Active")))
        )
        self._refresh_runtime_audio_status(signal_state=signal_state)
        self.status_var.set(message)

    def stop_live_transcription(self) -> None:
        log_section("Stopping Live Transcription")
        if self._live_transcription_starting:
            self.status_var.set("Live transcription is still starting. Wait a moment and stop it again.")
            return
        if self._live_transcription_stopping:
            self.status_var.set("Live transcription is already stopping.")
            return

        if not self._live_transcription_running or self.live_transcription_session is None:
            self.status_var.set("Live transcription is not running.")
            return

        debug_log("[App] Stopping live transcription")
        session = self.live_transcription_session
        self._set_live_controls_state(running=True, starting=False, stopping=True)
        self.live_transcription_status_label.configure(text="Stopping live transcription...", text_color="#F9A825")
        self.status_var.set("Stopping live transcription...")
        threading.Thread(
            target=self._stop_live_transcription_worker,
            args=(session,),
            daemon=True,
        ).start()

    def _stop_live_transcription_worker(self, session: LiveTranscriptionSession) -> None:
        success, message = session.stop()
        if self._closing:
            return
        try:
            self.root.after(0, lambda: self._finish_stop_live_transcription(session, success, message))
        except Exception as exc:
            log_event("App", level="warning", event="queue_live_stop_finish_failed", reason=str(exc))

    def _finish_stop_live_transcription(self, session: LiveTranscriptionSession, success: bool, message: str) -> None:
        if self.live_transcription_session is session:
            self.live_transcription_session = None
        self._set_live_controls_state(running=False, starting=False, stopping=False)
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
        if success and self._pending_mode_after_live_stop is not None and not self._closing:
            next_mode = self._pending_mode_after_live_stop
            self._pending_mode_after_live_stop = None
            try:
                self.root.after(0, lambda: self.apply_audio_mode(next_mode))
            except Exception as exc:
                log_event("App", level="warning", event="queue_mode_apply_after_live_stop_failed", mode=next_mode, reason=str(exc))

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
        transcriber = DeepgramFileTranscriber(get_deepgram_api_key())
        success, message = transcriber.transcribe_file(media_path)
        if self._closing:
            return
        try:
            self.root.after(0, lambda: self._finish_file_transcription(success, message))
        except Exception as exc:
            log_event("App", level="warning", event="queue_file_transcription_finish_failed", reason=str(exc))
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
        mode_key = mode_name_to_key(mode_name)
        mode_config = self.config.get("modes", {}).get(mode_key, {})
        configured_input = str(mode_config.get("input_device", "")).strip()
        configured_output = str(mode_config.get("output_device", "")).strip()
        if mode_name == "Mixed":
            configured_input = self.mix_var.get().strip() or configured_input
            configured_output = self.mixed_playback_var.get().strip() or configured_output
        elif mode_name == "VAC":
            configured_input = self.vac_var.get().strip() or configured_input
            configured_output = self.vac_playback_var.get().strip() or configured_output
        else:
            configured_input = self.mic_var.get().strip() or configured_input
            configured_output = self.speaker_var.get().strip() or configured_output
        if configured_input or configured_output:
            if mode_name == "Mixed" and not configured_output:
                configured_output = self.speaker_var.get().strip()
            return configured_input, configured_output
        if mode_name == "Microphone":
            return self.mic_var.get().strip(), self.speaker_var.get().strip()
        if mode_name == "VAC":
            return self.vac_var.get().strip(), self.vac_playback_var.get().strip()
        mixed_playback = self.mixed_playback_var.get().strip() or self.speaker_var.get().strip()
        return self.mix_var.get().strip(), mixed_playback

    def _pick_fallback_mode(self, original_mode: ModeName) -> "ModeName | None":
        """Return the next-best available mode whose input device is currently detected.

        Fallback chain (only traversed from the original_mode downward):
            Mixed -> VAC -> Microphone
        Returns None if no mode in the chain has a present input device.
        Does NOT mutate state, config, or Windows defaults.
        """
        chain: dict[str, list[ModeName]] = {
            "Mixed": ["VAC", "Microphone"],
            "VAC": ["Microphone"],
            "Microphone": [],
        }
        for candidate in chain.get(original_mode, []):
            candidate_device, _candidate_playback = self._resolve_mode_devices(candidate)
            if not candidate_device:
                continue
            resolved = self._resolve_detected_input_name(candidate_device, candidate) or candidate_device
            if resolved in self.detected_input_devices:
                return candidate
        return None

    def apply_audio_mode(self, mode_name: ModeName) -> None:
        log_section(f"Applying Audio Mode: {mode_name}")
        if self._audio_switch_in_progress:
            self.status_var.set("Audio device switch already in progress.")
            return
        self._refresh_detected_devices()
        if self._live_transcription_running or self._live_transcription_starting:
            self._pending_mode_after_live_stop = mode_name
            self._pending_live_start_mode = mode_name
            self.status_var.set(f"Stopping live transcription so {mode_name} mode can be applied cleanly...")
            self.stop_live_transcription()
            return
        if mode_name == "Mixed" and not self._is_mixed_mode_available(log_failure_level="error"):
            message = (
                "No mixed-capable input device is currently detected. Connect or enable a source such as "
                "VB Cable, Stereo Mix, or Voicemeeter, then click Refresh Devices."
            )
            log_failure("DEPENDENCY", mode=mode_name, device=self.mix_var.get().strip(), reason=message)
            messagebox.showerror("Mixed Mode Unavailable", message)
            self.status_var.set("Mixed mode is unavailable because no mixed-capable input device is currently detected.")
            return

        log_event("ApplyMode", requested=mode_name, live=False)
        self.save_form_config()
        input_device, playback_target = self._resolve_mode_devices(mode_name)
        log_event(
            "ApplyMode",
            requested=mode_name,
            mode_available=(mode_name != "Mixed" or self._is_mixed_mode_available(log_failure_level="debug")),
            resolved_input=input_device,
            resolved_playback=playback_target,
            detected_inputs=len(self.detected_input_devices),
        )
        if mode_name in {"Microphone", "VAC", "Mixed"}:
            resolved_name = self._resolve_detected_input_name(
                input_device,
                mode_name,
                log_failure_level="error" if mode_name == "Mixed" else "debug",
            )
            if resolved_name:
                input_device = resolved_name
                if mode_name == "Microphone":
                    self.mic_var.set(resolved_name)
                elif mode_name == "VAC":
                    self.vac_var.set(resolved_name)
                else:
                    self.mix_var.set(resolved_name)
        if not input_device:
            self.status_var.set(f"No input device configured for {mode_name} mode.")
            return
        if mode_name == "Mixed" and input_device not in self.detected_input_devices:
            self.status_var.set(
                f"Mixed mode is configured for '{input_device}', but that device is not currently detected. Check your VB Cable, Stereo Mix, or Voicemeeter source and refresh devices."
            )
            return
        if not playback_target:
            self.status_var.set(f"No playback device configured for {mode_name} mode.")
            return
        self._audio_switch_in_progress = True
        self._pending_mode_button = mode_name
        self._refresh_run_control_buttons()
        self.status_var.set(f"Switching audio device... {mode_name} mode requested.")
        threading.Thread(
            target=self._apply_audio_mode_worker,
            args=(mode_name, input_device, playback_target),
            daemon=True,
        ).start()

    def _apply_audio_mode_hot(self, mode_name: ModeName) -> None:
        if self.live_transcription_session is None:
            self.status_var.set("Live session is not available for hot switching.")
            return
        if mode_name == self.current_mode:
            self.status_var.set(f"{mode_name} is already the active mode.")
            return
        if mode_name == "Mixed" and not self._is_mixed_mode_available(log_failure_level="error"):
            message = (
                "No mixed-capable input device is currently detected. Connect or enable a source such as "
                "VB Cable, Stereo Mix, or Voicemeeter, then click Refresh Devices."
            )
            log_failure("DEPENDENCY", mode=mode_name, device=self.mix_var.get().strip(), reason=message)
            messagebox.showerror("Mixed Mode Unavailable", message)
            return

        self.save_form_config()
        input_device, playback_target = self._resolve_mode_devices(mode_name)
        resolved_name = self._resolve_detected_input_name(input_device, mode_name, log_failure_level="error")
        if resolved_name:
            input_device = resolved_name
        if not input_device or not playback_target:
            self.status_var.set(f"Hot switch could not resolve devices for {mode_name}.")
            return

        self._audio_switch_in_progress = True
        self._pending_mode_button = mode_name
        self._refresh_run_control_buttons()
        self.status_var.set(f"Hot-switching live transcription to {mode_name}...")
        threading.Thread(
            target=self._apply_audio_mode_hot_worker,
            args=(mode_name, input_device, playback_target),
            daemon=True,
        ).start()

    def _apply_audio_mode_worker(self, mode_name: ModeName, input_device: str, playback_target: str) -> None:
        resolved_device: ActiveAudioDevice | None = None
        outcome = ModeSwitchOutcome.HARD_FAILURE
        try:
            resolved_device = self.resolve_active_device(input_device)
        except Exception as exc:
            message = str(exc)
        else:
            ok_record, record_message = self.device_manager.set_default_recording_device(input_device)
            if not ok_record:
                message = record_message
            else:
                ok_playback, playback_message = self.device_manager.set_default_playback_device(playback_target)
                if not ok_playback:
                    message = f"{record_message} Playback switch failed: {playback_message}"
                else:
                    verification_result = self._wait_for_active_input_device(resolved_device["name"])
                    if verification_result == DeviceVerificationResult.DEVICE_UNAVAILABLE:
                        message = (
                            f"Windows completed the switch request, but {resolved_device['name']} no longer appears "
                            f"in the available input devices."
                        )
                    elif verification_result == DeviceVerificationResult.CONFIRMED:
                        outcome = ModeSwitchOutcome.SUCCESS
                        message = f"{mode_name} mode active | Input: {resolved_device['name']} | Output: {playback_target}"
                    elif verification_result in (
                        DeviceVerificationResult.EVENTUALLY_CONFIRMED,
                        DeviceVerificationResult.TIMED_OUT,
                    ):
                        outcome = ModeSwitchOutcome.SUCCESS_WITH_WARNING
                        message = (
                            f"Switched Windows defaults, but verification for {resolved_device['name']} was slow."
                        )
                    else:
                        message = f"Unable to verify that {resolved_device['name']} became the active input device."
        if outcome != ModeSwitchOutcome.HARD_FAILURE and resolved_device is None:
            outcome = ModeSwitchOutcome.HARD_FAILURE
            message = f"Unable to resolve an active audio device for {mode_name} mode."

        if self._closing:
            return

        def _dispatched_finish() -> None:
            try:
                self._finish_apply_audio_mode(outcome, mode_name, resolved_device, playback_target, message)
            except Exception:
                LOGGER.exception("[ApplyMode] event=dispatched_finish_crashed mode=%s", mode_name)

        try:
            self.root.after(0, _dispatched_finish)
        except Exception as exc:
            log_event("ApplyMode", level="warning", event="queue_finish_failed", mode=mode_name, reason=str(exc))
            return

    def _apply_audio_mode_hot_worker(self, mode_name: ModeName, input_device: str, playback_target: str) -> None:
        message = ""
        outcome = ModeSwitchOutcome.HARD_FAILURE
        new_active_device: ActiveAudioDevice | None = None
        continue_anyway = False
        try:
            ok_record, record_message = self.device_manager.set_default_recording_device(input_device)
            if not ok_record:
                message = record_message
                return
            ok_playback, playback_message = self.device_manager.set_default_playback_device(playback_target)
            if not ok_playback:
                message = playback_message
                return
            verification_result = self._wait_for_active_input_device(input_device)
            if verification_result == DeviceVerificationResult.DEVICE_UNAVAILABLE:
                message = f"Windows completed the switch request, but {input_device} is no longer available."
                return
            new_active_device = self.resolve_active_device(input_device)
            preflight_ok, failure_code, _diagnostics = self._run_preflight(mode_name, new_active_device)
            if not preflight_ok:
                log_event("HotSwitch", level="warning", event="preflight_warning", mode=mode_name, failure_code=failure_code)
                feedback = str(_diagnostics.get("feedback", "")).strip()
                continue_anyway = self._ask_yes_no_sync(
                    "Signal Issue on New Mode",
                    (feedback + "\n\n" if feedback else "") + f"The new {mode_name} source did not pass preflight checks. Continue the hot switch anyway?",
                )
                if not continue_anyway:
                    message = f"Hot switch to {mode_name} was cancelled after preflight warning."
                    return
            assert self.live_transcription_session is not None
            success, message = self.live_transcription_session.switch_input_device(new_active_device, mode_name)
            if success:
                outcome = (
                    ModeSwitchOutcome.SUCCESS
                    if verification_result == DeviceVerificationResult.CONFIRMED
                    else ModeSwitchOutcome.SUCCESS_WITH_WARNING
                )
        except Exception as exc:
            LOGGER.exception(_log_message("Failure: UNKNOWN", mode=mode_name, device=input_device, reason=str(exc)))
            message = str(exc)
        finally:
            if self._closing:
                return

            def _dispatched_finish() -> None:
                try:
                    self._finish_apply_audio_mode_hot(outcome, mode_name, new_active_device, playback_target, message)
                except Exception:
                    LOGGER.exception("[ApplyMode] event=dispatched_finish_crashed mode=%s", mode_name)

            try:
                self.root.after(0, _dispatched_finish)
            except Exception as exc:
                log_event("HotSwitch", level="warning", event="queue_finish_failed", mode=mode_name, reason=str(exc))

    def _finish_apply_audio_mode(
        self,
        outcome: ModeSwitchOutcome,
        mode_name: ModeName,
        resolved_device: ActiveAudioDevice | None,
        playback_target: str,
        message: str,
    ) -> None:
        LOGGER.info("[ApplyMode] event=finish_enter mode=%s outcome=%s worker_thread=%s", mode_name, outcome.value, threading.current_thread().name)
        try:
            self._audio_switch_in_progress = False
            self._pending_mode_button = None
            self._refresh_run_control_buttons()
            if outcome == ModeSwitchOutcome.HARD_FAILURE or resolved_device is None:
                self._pending_live_start_mode = None
                self._pending_vac_test = False
                self._refresh_runtime_audio_status()
                if hasattr(self, "status_label") and self.status_label.winfo_exists():
                    self.status_label.configure(text_color="#E53935")
                self.status_var.set(message)
                LOGGER.error("[ApplyMode] event=finish_hard_failure mode=%s reason=%s", mode_name, message)
                LOGGER.info(
                    "[ApplyMode] event=finish_complete mode=%s current_mode=%s active_device=%s header_chip=%s",
                    mode_name,
                    self.current_mode,
                    self.active_audio_device["name"] if self.active_audio_device else "None",
                    self.mode_var.get() if hasattr(self, "mode_var") else "unset",
                )
                return

            self.active_audio_device = resolved_device
            self._lock_active_input_device(resolved_device)
            if not self.verify_active_device():
                self._pending_vac_test = False
                self._refresh_runtime_audio_status()
                if hasattr(self, "status_label") and self.status_label.winfo_exists():
                    self.status_label.configure(text_color="#E53935")
                self.status_var.set("Device mismatch detected after switching modes.")
                LOGGER.error("[ApplyMode] event=finish_hard_failure mode=%s reason=%s", mode_name, "Device mismatch detected after switching modes.")
                LOGGER.info(
                    "[ApplyMode] event=finish_complete mode=%s current_mode=%s active_device=%s header_chip=%s",
                    mode_name,
                    self.current_mode,
                    self.active_audio_device["name"] if self.active_audio_device else "None",
                    self.mode_var.get() if hasattr(self, "mode_var") else "unset",
                )
                return

            self.current_mode = mode_name
            self.config["active_mode"] = mode_name_to_key(mode_name)
            self.config["last_mode"] = mode_name
            save_config(self.config)
            self.mode_var.set(mode_name)
            self.active_audio_device = resolved_device
            self._lock_active_input_device(resolved_device)
            self.direct_recording_var.set(resolved_device["name"])
            self.direct_playback_var.set(playback_target)
            self.active_device_label.configure(text=f"ACTIVE INPUT: {resolved_device['name']}")
            self._apply_mode_theme(mode_name)
            self._refresh_run_control_buttons()
            self._refresh_runtime_audio_status()
            self._refresh_mode_hint()
            self._update_debug_routing(mode_name, resolved_device["name"], resolved_device["name"], False)
            if hasattr(self, "status_label") and self.status_label.winfo_exists():
                success_color = MODE_UI.get(mode_name, {}).get("text_color", "#66BB6A")
                self.status_label.configure(text_color="#F9A825" if outcome == ModeSwitchOutcome.SUCCESS_WITH_WARNING else success_color)
            if outcome == ModeSwitchOutcome.SUCCESS_WITH_WARNING:
                LOGGER.warning("[ApplyMode] event=finish_success_with_warning mode=%s note=%s", mode_name, message)
                self.status_var.set(f"⚠ {mode_name} mode active (Windows verification was slow). {message}")
            else:
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
            if self._pending_live_start_mode == mode_name and not self._live_transcription_running and not self._live_transcription_starting:
                self._pending_live_start_mode = None
                try:
                    self.root.after(0, self.start_live_transcription)
                except Exception as exc:
                    log_event("App", level="warning", event="queue_run_mode_start_failed", mode=mode_name, reason=str(exc))
            LOGGER.info(
                "[ApplyMode] event=finish_complete mode=%s current_mode=%s active_device=%s header_chip=%s",
                mode_name,
                self.current_mode,
                self.active_audio_device["name"] if self.active_audio_device else "None",
                self.mode_var.get() if hasattr(self, "mode_var") else "unset",
            )
        except Exception:
            LOGGER.exception("[ApplyMode] event=finish_crashed mode=%s", mode_name)

    def _finish_apply_audio_mode_hot(
        self,
        outcome: ModeSwitchOutcome,
        mode_name: ModeName,
        resolved_device: ActiveAudioDevice | None,
        playback_target: str,
        message: str,
    ) -> None:
        LOGGER.info("[ApplyMode] event=finish_enter mode=%s outcome=%s worker_thread=%s", mode_name, outcome.value, threading.current_thread().name)
        try:
            self._audio_switch_in_progress = False
            self._pending_mode_button = None
            self._refresh_run_control_buttons()
            if outcome == ModeSwitchOutcome.HARD_FAILURE or resolved_device is None:
                if message:
                    self.status_var.set(message)
                    if hasattr(self, "status_label") and self.status_label.winfo_exists():
                        self.status_label.configure(text_color="#E53935")
                    messagebox.showerror("Hot Switch Failed", message)
                LOGGER.error("[ApplyMode] event=finish_hard_failure mode=%s reason=%s", mode_name, message)
                LOGGER.info(
                    "[ApplyMode] event=finish_complete mode=%s current_mode=%s active_device=%s header_chip=%s",
                    mode_name,
                    self.current_mode,
                    self.active_audio_device["name"] if self.active_audio_device else "None",
                    self.mode_var.get() if hasattr(self, "mode_var") else "unset",
                )
                return

            self.active_audio_device = resolved_device
            self._lock_active_input_device(resolved_device)
            self.current_mode = mode_name
            self.config["last_mode"] = mode_name
            save_config(self.config)
            self.mode_var.set(mode_name)
            self.direct_recording_var.set(resolved_device["name"])
            self.direct_playback_var.set(playback_target)
            self.active_device_label.configure(text=f"ACTIVE INPUT: {resolved_device['name']}")
            self._apply_mode_theme(mode_name)
            self._refresh_run_control_buttons()
            self._refresh_runtime_audio_status(signal_state="Active")
            self._refresh_mode_hint()
            self._update_debug_routing(mode_name, resolved_device["name"], resolved_device["name"], False)
            live_color = "#F9A825" if outcome == ModeSwitchOutcome.SUCCESS_WITH_WARNING else MODE_UI.get(mode_name, {}).get("text_color", "#66BB6A")
            self.live_transcription_status_label.configure(text=message, text_color=live_color)
            if hasattr(self, "status_label") and self.status_label.winfo_exists():
                self.status_label.configure(text_color=live_color)
            if outcome == ModeSwitchOutcome.SUCCESS_WITH_WARNING:
                LOGGER.warning("[ApplyMode] event=finish_success_with_warning mode=%s note=%s", mode_name, message)
                self.status_var.set(f"⚠ Hot-switched to {mode_name} - transcript continuing. {message}")
            else:
                self.status_var.set(f"Hot-switched to {mode_name} - transcript continuing.")
            LOGGER.info(
                "[ApplyMode] event=finish_complete mode=%s current_mode=%s active_device=%s header_chip=%s",
                mode_name,
                self.current_mode,
                self.active_audio_device["name"] if self.active_audio_device else "None",
                self.mode_var.get() if hasattr(self, "mode_var") else "unset",
            )
        except Exception:
            LOGGER.exception("[ApplyMode] event=finish_crashed mode=%s", mode_name)

    def _apply_mode_theme(self, mode_name: str) -> None:
        mode_ui = MODE_UI.get(mode_name)
        if mode_ui is None:
            LOGGER.error("[ApplyMode] event=unknown_mode_in_ui_map mode=%s known=%s", mode_name, list(MODE_UI.keys()))
            accent = "#616161"
            bright = "#9E9E9E"
            label = mode_name
        else:
            accent = str(mode_ui["accent"])
            bright = str(mode_ui["text_color"])
            label = str(mode_ui["label"])

        for widget_name in ("header_mode_chip", "mode_badge_label"):
            widget = getattr(self, widget_name, None)
            if widget is None:
                LOGGER.warning("[ApplyMode] event=badge_widget_missing name=%s", widget_name)
                continue
            try:
                widget.configure(text=label, fg_color=accent)
            except Exception:
                LOGGER.exception("[ApplyMode] event=badge_configure_failed name=%s", widget_name)

        source_label = getattr(self, "active_source_label", None)
        if source_label is None:
            LOGGER.warning("[ApplyMode] event=source_label_missing")
        else:
            try:
                source_label.configure(text_color=bright)
            except Exception:
                LOGGER.exception("[ApplyMode] event=source_label_configure_failed")

        LOGGER.debug("[ApplyMode] event=theme_applied mode=%s accent=%s bright=%s", mode_name, accent, bright)

    def _update_mode_badges(self, mode_name: str) -> None:
        self._apply_mode_theme(mode_name)

    def _wait_for_active_input_device(
        self,
        expected_device_name: str,
        timeout_seconds: float = DEVICE_VERIFY_INITIAL_TIMEOUT_SECONDS,
    ) -> DeviceVerificationResult:
        normalized_expected = normalize_audio_device_name(expected_device_name)
        start_time = time.time()
        deadline = start_time + timeout_seconds

        while time.time() < deadline:
            _index, device_info = get_default_input_device()
            if device_info is not None:
                active_name = normalize_audio_device_name(str(device_info.get("name", "")))
                if active_name == normalized_expected:
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    LOGGER.info("[DeviceVerify] event=confirmed device=%s elapsed_ms=%d", _quote_log_value(expected_device_name), elapsed_ms)
                    return DeviceVerificationResult.CONFIRMED
            time.sleep(0.1)

        extended_deadline = time.time() + DEVICE_VERIFY_EXTENDED_TIMEOUT_SECONDS
        while time.time() < extended_deadline:
            _index, device_info = get_default_input_device()
            if device_info is not None:
                active_name = normalize_audio_device_name(str(device_info.get("name", "")))
                if active_name == normalized_expected:
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    LOGGER.warning(
                        "[DeviceVerify] event=eventually_confirmed device=%s elapsed_ms=%d initial_window_exceeded=True",
                        _quote_log_value(expected_device_name),
                        elapsed_ms,
                    )
                    self._slow_verification_count += 1
                    if self._slow_verification_count > 1 and not self._slow_verification_advisory_logged:
                        log_event(
                            "DeviceVerify",
                            event="consistently_slow",
                            elapsed_ms=elapsed_ms,
                            note="Your system verifies device changes slower than average. This is normal on some Windows installs.",
                        )
                        self._slow_verification_advisory_logged = True
                    return DeviceVerificationResult.EVENTUALLY_CONFIRMED
            time.sleep(0.2)

        try:
            devices = sd.query_devices()
        except Exception as exc:
            LOGGER.warning("[DeviceVerify] event=query_failed device=%s reason=%s", _quote_log_value(expected_device_name), _quote_log_value(exc))
            return DeviceVerificationResult.TIMED_OUT

        expected_present = any(
            normalize_audio_device_name(str(device.get("name", ""))) == normalized_expected
            for device in devices
        )
        elapsed_ms = int((time.time() - start_time) * 1000)
        if not expected_present:
            LOGGER.error("[DeviceVerify] event=device_unavailable device=%s elapsed_ms=%d", _quote_log_value(expected_device_name), elapsed_ms)
            return DeviceVerificationResult.DEVICE_UNAVAILABLE
        LOGGER.warning("[DeviceVerify] event=timed_out device=%s elapsed_ms=%d", _quote_log_value(expected_device_name), elapsed_ms)
        return DeviceVerificationResult.TIMED_OUT

    def test_vac_routing(self) -> None:
        if self._vac_test_running:
            self.status_var.set("VAC routing test is already running.")
            return

        debug_log("[App] Starting VAC routing test")
        self._pending_vac_test = True
        self.apply_audio_mode("VAC")

    def auto_select_best_mode(self) -> None:
        if self._best_mode_running:
            self.status_var.set("Best mode detection is already running.")
            return
        if self._audio_switch_in_progress:
            self.status_var.set("Wait for the current audio device switch to finish before detecting the best mode.")
            return
        if self._live_transcription_running or self._live_transcription_starting:
            self.status_var.set("Stop live transcription before auto-detecting the best mode.")
            return
        self._best_mode_running = True
        self.status_var.set("Sampling Microphone, VAC, and Mixed inputs to find the best mode...")
        best_button = getattr(self, "btn_auto_best_mode", None)
        if best_button is not None and best_button.winfo_exists():
            best_button.configure(state="disabled", text="Sampling...")
        threading.Thread(target=self._auto_best_mode_worker, daemon=True).start()

    def _auto_best_mode_worker(self) -> None:
        state_rank = {"active": 3, "low": 2, "clipping": 1, "silent": 0}
        samples: list[tuple[int, float, ModeName, str, dict[str, Any] | None]] = []

        for mode_name in ("Microphone", "VAC", "Mixed"):
            input_name, _playback_target = self._resolve_mode_devices(mode_name)
            resolved_name = self._resolve_detected_input_name(input_name, mode_name)
            if not resolved_name:
                continue
            signal = sample_input_signal(
                resolved_name,
                LIVE_TRANSCRIPTION_SAMPLE_RATE_HZ,
                duration_seconds=0.5 if mode_name == "VAC" else 0.35,
            )
            state = "" if signal is None else str(signal.get("state", "")).lower()
            rms = 0.0 if signal is None else float(signal.get("rms", 0.0))
            samples.append((state_rank.get(state, -1), rms, mode_name, resolved_name, signal))

        if not samples:
            success = False
            message = "No usable input devices were available to evaluate the best mode."
            best_mode = None
        else:
            samples.sort(key=lambda item: (item[0], item[1]), reverse=True)
            _rank, _rms, best_mode, resolved_name, signal = samples[0]
            if signal is None or str(signal.get("state", "")).lower() == "silent":
                success = False
                message = "All sampled modes are currently silent. Start audio or speak into the mic, then try Auto Best Mode again."
            else:
                success = True
                detail = "" if signal is None else str(signal.get("detail", "")).strip()
                message = f"Best mode detected: {best_mode} using {resolved_name}. {detail}".strip()

        try:
            self.root.after(0, lambda: self._finish_auto_best_mode(success, best_mode, message))
        except Exception as exc:
            log_event("App", level="warning", event="queue_auto_best_mode_failed", reason=str(exc))
            return

    def _finish_auto_best_mode(self, success: bool, best_mode: ModeName | None, message: str) -> None:
        self._best_mode_running = False
        best_button = getattr(self, "btn_auto_best_mode", None)
        if best_button is not None and best_button.winfo_exists():
            best_button.configure(state="normal", text="Auto Best Mode")
        self.status_var.set(message)
        if success and best_mode is not None:
            self.apply_audio_mode(best_mode)

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
                    device_info=active_device["info"],
                )
            sd.wait()
            signal_ok, signal_message, signal_payload = evaluate_live_signal_readiness(
                signal,
                active_device["name"] if active_device is not None else "selected VAC input",
            )
            if signal_payload is not None:
                try:
                    self._queue_live_signal_update(signal_payload)
                except Exception as exc:
                    log_event("App", level="warning", event="vac_test_signal_update_failed", reason=str(exc))
            if signal is None:
                result_message = "VAC routing test completed, but the app could not verify input signal on the selected VAC device."
            elif not signal_ok:
                result_message = "VAC routing test finished, but no audio was detected on the selected input. Do not assume routing is working."
            else:
                state = str(signal.get("state", "")).lower()
                if state == "low":
                    result_message = "VAC routing test finished with only low signal on the selected input. Routing may be weak or misconfigured."
                elif state == "clipping":
                    result_message = "VAC routing test completed, but the selected input is clipping. Lower playback or mixer output."
                else:
                    result_message = "VAC routing test completed with confirmed signal on the selected input."
        except Exception as exc:
            result_message = f"VAC routing test failed: {exc}"
        finally:
            try:
                sd.stop()
            except Exception as exc:
                log_event("App", level="warning", event="vac_test_stop_warning", reason=str(exc))
            if self._closing:
                return
            try:
                self.root.after(0, lambda: self._finish_vac_test(result_message))
            except Exception as exc:
                log_event("App", level="warning", event="queue_vac_test_finish_failed", reason=str(exc))
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
        active_device_name = self.active_audio_device["name"] if self.active_audio_device is not None else self._current_live_input_device_name()
        log_event("MuteToggle", event="before", device=active_device_name, muted=self.is_muted, mode=self.current_mode)
        ok, message = self.device_manager.toggle_mute()
        if not ok:
            log_failure("DEVICE", tag="MuteToggle", device=active_device_name, mode=self.current_mode, reason=message)
            self.status_var.set(message)
            messagebox.showerror("Mute Toggle Failed", message)
            return

        # nircmd default_record follows the current default recording device on purpose.
        self.is_muted = not self.is_muted
        self.mute_button.configure(
            fg_color="#5F2120" if self.is_muted else "#D32F2F",
            hover_color="#471816" if self.is_muted else "#B71C1C",
        )
        self._refresh_run_control_buttons()
        log_event("MuteToggle", event="after", device=active_device_name, muted=self.is_muted, mode=self.current_mode)
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
            self.mode_badge_label.configure(text=self.current_mode, fg_color=MODE_UI.get(self.current_mode, MODE_UI["Microphone"])["accent"])
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
            self.monitor_wer_label.configure(text=MODE_UI.get(self.current_mode, MODE_UI["Microphone"])["badge"].replace("WER ", ""), text_color="#9E9E9E")
            self.mode_badge_label.configure(text=self.current_mode, fg_color="#4A4A4A")
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
        except Exception as exc:
            log_event("App", level="warning", event="queue_quality_update_failed", reason=str(exc))
            return

    def _apply_quality_update(self, result: dict[str, Any]) -> None:
        if self._closing or not self.wer_enabled_var.get():
            return
        quality = result["quality"]
        ui_config = MODE_UI.get(self.current_mode, MODE_UI["Microphone"])
        self.monitor_status_var.set(result["status_text"])
        self.monitor_level_var.set(result["level_text"])
        self.monitor_detail_var.set(result["detail_text"])
        color = str(result.get("meter_color", QUALITY_COLORS.get(quality, "#9E9E9E")))
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
        self.monitor_summary_var.set(
            f"Quality: {result['status_text']} | WER mode: {'On' if self.wer_enabled_var.get() else 'Off'}")
        progress = float(result.get("meter_progress", QUALITY_PROGRESS.get(quality, 0.0)))
        rms_text, peak_text = self._split_levels(result["level_text"])
        status_text = str(result.get("meter_feedback", "Monitoring" if quality != "error" else "Unavailable"))
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
        self.mode_badge_label.configure(text=self.current_mode, fg_color=ui_config["accent"])
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
        self.mode_display.configure(text=ui_config["title"], text_color=ui_config["accent"])
        self.mode_badge_label.configure(text=self.current_mode, fg_color=ui_config["accent"])
        self.mode_device_label.configure(text=self._current_mode_device_summary())
        if hasattr(self, "header_mode_chip") and self.header_mode_chip.winfo_exists():
            self.header_mode_chip.configure(text=self.current_mode, fg_color=ui_config["accent"])
        self.monitor_wer_label.configure(text=ui_config["badge"].replace("WER ", ""))
        self._refresh_run_control_buttons()
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
        if self._debug_log_handler is not None:
            LOGGER.removeHandler(self._debug_log_handler)
            self._debug_log_handler = None
        self._close_settings_window()
        if self.live_transcription_session is not None:
            self.live_transcription_session.stop()
            self.live_transcription_session = None
        self.monitor.stop()
        if self.root.winfo_exists():
            self.save_form_config()
            if self.config.get("restore_devices_on_exit"):
                self._restore_original_default_devices()
            self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    ensure_local_venv()
    install_global_exception_hooks()
    App().run()
