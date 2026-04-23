from __future__ import annotations

import inspect
import logging
import math
import os
import queue
import sys
import threading
import time
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from tkinter import messagebox
from typing import Any

import customtkinter as ctk
import numpy as np
import sounddevice as sd


APP_DIR = Path(__file__).resolve().parent
VENV_PYTHON = APP_DIR / ".venv" / "Scripts" / "python.exe"
ENV_PATH = APP_DIR / ".env"
LOGS_DIR = APP_DIR / "logs"
TRANSCRIPTS_DIR = APP_DIR / "transcripts"
LOG_PATH = LOGS_DIR / "virtual_audio.log"
ERROR_LOG_PATH = LOGS_DIR / "errors.log"

CONFIG = {
    "input_device": "CABLE Output (VB-Audio Virtual Cable)",
    "sample_rate": 16000,
    "channels": 1,
}

_ACTIVE_AUDIO_CALLBACK = None


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
    if os.environ.get("PYCHARM_HOSTED"):
        return
    current_python = Path(sys.executable).resolve()
    target_python = VENV_PYTHON.resolve()
    if current_python == target_python:
        return
    os.execv(str(target_python), [str(target_python), str(APP_DIR / "app.py"), *sys.argv[1:]])


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("virtual_audio_simple")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(threadName)s | %(message)s")

    file_handler = RotatingFileHandler(LOG_PATH, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    error_handler = RotatingFileHandler(ERROR_LOG_PATH, maxBytes=1 * 1024 * 1024, backupCount=2, encoding="utf-8")
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


load_dotenv_file(ENV_PATH)
LOGGER = setup_logging()


def log_error(message: str, exc: BaseException | str | None = None) -> None:
    if exc is None:
        LOGGER.error(message)
        return
    if isinstance(exc, BaseException):
        LOGGER.error("%s: %s", message, exc)
        LOGGER.debug("Traceback:\n%s", "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        return
    LOGGER.error("%s: %s", message, exc)


def show_error_popup(message: str) -> None:
    try:
        messagebox.showerror("Virtual Audio Control", message)
    except Exception:
        log_error("Unable to show error popup", message)


def safe_thread(fn):
    def wrapper(*args, **kwargs):
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            log_error("Thread crash", exc)
    return wrapper


def normalize_device_name(name: str) -> str:
    return " ".join(str(name or "").strip().lower().split())


def _is_wasapi_device(name: str) -> bool:
    return "wasapi" in normalize_device_name(name)


def list_audio_devices() -> None:
    devices = sd.query_devices()
    for index, device in enumerate(devices):
        print(index, device["name"], device["max_input_channels"], device["max_output_channels"])


def get_deepgram_api_key() -> str:
    return os.environ.get("DEEPGRAM_API_KEY", "").strip()


def compute_rms_db(audio: np.ndarray) -> float:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 2:
        samples = np.mean(samples, axis=1)
    samples = np.squeeze(samples)
    if samples.size == 0:
        return -200.0
    rms = float(np.sqrt(np.mean(samples ** 2)))
    return float(20 * np.log10(rms + 1e-10))


def signal_state_from_db(rms_db: float) -> str:
    return "No Signal" if rms_db < -80.0 else "Active"


def pcm16_bytes(audio: np.ndarray) -> bytes:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 2:
        samples = np.mean(samples, axis=1)
    samples = np.squeeze(samples)
    samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32767.0).astype(np.int16, copy=False).tobytes()


def resolve_input_device(device_name: str) -> tuple[int | None, dict[str, Any] | None]:
    target = normalize_device_name(device_name)
    exact_matches: list[tuple[int, dict[str, Any]]] = []
    partial_matches: list[tuple[int, dict[str, Any]]] = []
    for index, device in enumerate(sd.query_devices()):
        if int(device.get("max_input_channels", 0)) <= 0:
            continue
        current_name = str(device.get("name", "")).strip()
        normalized_name = normalize_device_name(current_name)
        if normalized_name == target:
            exact_matches.append((index, dict(device)))
            continue
        if target and target in normalized_name:
            partial_matches.append((index, dict(device)))

    matches = exact_matches or partial_matches
    if not matches:
        return None, None

    for index, info in matches:
        if _is_wasapi_device(str(info.get("name", ""))):
            LOGGER.info("Using WASAPI input device [%s]: %s", index, info.get("name", ""))
            return index, info

    index, info = matches[0]
    LOGGER.info("Using input device [%s]: %s", index, info.get("name", ""))
    return index, info


def get_vac_device() -> int:
    device_index, device_info = resolve_input_device(CONFIG["input_device"])
    if device_index is None or device_info is None:
        raise RuntimeError(f"VAC input device not found: {CONFIG['input_device']}")
    return device_index


def start_audio_stream():
    global _ACTIVE_AUDIO_CALLBACK
    if _ACTIVE_AUDIO_CALLBACK is None:
        log_error("Audio stream failed", "Audio callback is not configured.")
        show_error_popup("Audio callback is not configured.")
        return None
    try:
        device_index = get_vac_device()
        stream = sd.InputStream(
            device=device_index,
            channels=1,
            samplerate=CONFIG["sample_rate"],
            callback=_ACTIVE_AUDIO_CALLBACK,
            blocksize=8000,
            dtype="float32",
        )
        stream.start()
        return stream
    except Exception as exc:
        log_error("Audio stream failed", exc)
        show_error_popup(str(exc))
        return None


def _deepgram_value(node: Any, key: str, default: Any = None) -> Any:
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def format_live_result_text(result: Any) -> str:
    try:
        channel = _deepgram_value(result, "channel")
        alternatives = _deepgram_value(channel, "alternatives", [])
        if not alternatives:
            return ""
        return str(_deepgram_value(alternatives[0], "transcript", "")).strip()
    except Exception as exc:
        log_error("Deepgram transcript format failed", exc)
        return ""


class DeepgramLiveClient:
    def __init__(self, api_key: str, ui_queue: queue.Queue[tuple[str, Any]]):
        self.api_key = api_key.strip()
        self.ui_queue = ui_queue
        self.connection = None
        self.final_lines: list[str] = []
        self.interim_text = ""
        self.running = False

    def start(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "Missing DEEPGRAM_API_KEY in .env."
        try:
            from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents
        except Exception as exc:
            log_error("Deepgram SDK unavailable", exc)
            return False, f"Deepgram SDK is not available: {exc}"

        try:
            client = DeepgramClient(self.api_key)
            connection = client.listen.websocket.v("1")
            connection.on(LiveTranscriptionEvents.Open, self._on_open)
            connection.on(LiveTranscriptionEvents.Transcript, self._on_transcript)
            connection.on(LiveTranscriptionEvents.Error, self._on_error)
            connection.on(LiveTranscriptionEvents.Close, self._on_close)

            supported_names = set(inspect.signature(LiveOptions.__init__).parameters.keys())
            raw_options = {
                "model": "nova-3",
                "language": "en-US",
                "smart_format": True,
                "punctuate": True,
                "interim_results": True,
                "encoding": "linear16",
                "channels": CONFIG["channels"],
                "sample_rate": CONFIG["sample_rate"],
            }
            options = LiveOptions(**{key: value for key, value in raw_options.items() if key in supported_names})
            if not connection.start(options):
                return False, "Failed to start Deepgram live transcription connection."
        except Exception as exc:
            log_error("Deepgram connection failed", exc)
            return False, f"Deepgram connection failed: {exc}"

        self.connection = connection
        self.running = True
        return True, "Deepgram live transcription connected."

    def stop(self) -> None:
        self.running = False
        if self.connection is not None:
            try:
                self.connection.finish()
            except Exception as exc:
                log_error("Deepgram finish failed", exc)
        self.connection = None

    def send(self, pcm_bytes: bytes) -> None:
        if not self.running or self.connection is None:
            return
        self.connection.send(pcm_bytes)

    def current_transcript(self) -> str:
        sections = [line for line in self.final_lines if line.strip()]
        if self.interim_text.strip():
            sections.append(self.interim_text.strip())
        return "\n".join(sections).strip()

    def _on_open(self, client, open=None, **kwargs) -> None:
        self.ui_queue.put(("status", "Deepgram connected."))

    def _on_close(self, client, close=None, **kwargs) -> None:
        self.ui_queue.put(("status", "Deepgram connection closed."))

    def _on_error(self, client, error=None, **kwargs) -> None:
        message = str(error) if error else "Deepgram live transcription error."
        log_error("Deepgram error", message)
        self.ui_queue.put(("error", message))

    def _on_transcript(self, client, result=None, **kwargs) -> None:
        if result is None:
            return
        transcript = format_live_result_text(result)
        if not transcript:
            return
        if getattr(result, "is_final", False):
            self.final_lines.append(transcript)
            self.interim_text = ""
        else:
            self.interim_text = transcript
        self.ui_queue.put(("transcript", self.current_transcript()))


class SimpleAudioApp:
    def __init__(self) -> None:
        self.ui_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=32)
        self.audio_stop_event = threading.Event()
        self.sender_thread: threading.Thread | None = None
        self.stream = None
        self.deepgram: DeepgramLiveClient | None = None
        self.running = False
        self.rms_db = -200.0
        self.signal_state = "No Signal"
        self.device_index, self.device_info = resolve_input_device(CONFIG["input_device"])
        self.locked_device_name = str(self.device_info.get("name", CONFIG["input_device"])) if self.device_info else CONFIG["input_device"]
        self.transcript_text = ""

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.root = ctk.CTk()
        self.root.title("Audio Control Panel Pro")
        self.root.geometry("860x620")
        self.root.minsize(760, 520)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.input_var = ctk.StringVar(value=self.locked_device_name)
        self.rms_var = ctk.StringVar(value="RMS: -200.0 dB")
        self.status_var = ctk.StringVar(value="Idle")

        self._build_ui()
        self.root.after(100, self._process_ui_queue)
        self.root.after(250, self._report_startup_device_status)

    def _build_ui(self) -> None:
        frame = ctk.CTkFrame(self.root, corner_radius=16)
        frame.pack(fill="both", expand=True, padx=18, pady=18)

        ctk.CTkLabel(
            frame,
            text="Audio Control Panel Pro",
            font=("Arial", 24, "bold"),
            anchor="w",
        ).pack(fill="x", padx=18, pady=(18, 10))

        ctk.CTkLabel(frame, text="Input Device:", font=("Arial", 13, "bold"), anchor="w").pack(fill="x", padx=18)
        self.input_label = ctk.CTkLabel(
            frame,
            textvariable=self.input_var,
            font=("Consolas", 12),
            anchor="w",
            justify="left",
            wraplength=760,
        )
        self.input_label.pack(fill="x", padx=18, pady=(0, 12))

        button_row = ctk.CTkFrame(frame, fg_color="transparent")
        button_row.pack(fill="x", padx=18, pady=(0, 12))
        button_row.grid_columnconfigure(0, weight=1)
        button_row.grid_columnconfigure(1, weight=1)

        self.start_button = ctk.CTkButton(
            button_row,
            text="Start Transcription",
            command=self.start_transcription,
            height=40,
            font=("Arial", 12, "bold"),
            fg_color="#1565C0",
            hover_color="#0D47A1",
        )
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.stop_button = ctk.CTkButton(
            button_row,
            text="Stop",
            command=self.stop_transcription,
            height=40,
            font=("Arial", 12, "bold"),
            fg_color="#B71C1C",
            hover_color="#7F1010",
            state="disabled",
        )
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        ctk.CTkLabel(frame, text="Signal Level:", font=("Arial", 13, "bold"), anchor="w").pack(fill="x", padx=18)
        self.rms_label = ctk.CTkLabel(frame, textvariable=self.rms_var, font=("Consolas", 12), anchor="w")
        self.rms_label.pack(fill="x", padx=18)

        ctk.CTkLabel(frame, text="Status:", font=("Arial", 13, "bold"), anchor="w").pack(fill="x", padx=18, pady=(12, 0))
        self.status_label = ctk.CTkLabel(frame, textvariable=self.status_var, font=("Arial", 12), anchor="w")
        self.status_label.pack(fill="x", padx=18, pady=(0, 12))

        ctk.CTkLabel(frame, text="Transcription:", font=("Arial", 13, "bold"), anchor="w").pack(fill="x", padx=18)
        self.transcript_box = ctk.CTkTextbox(frame, font=("Consolas", 11), wrap="word")
        self.transcript_box.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        self.transcript_box.insert("1.0", "(live text here)")
        self.transcript_box.configure(state="disabled")

    def _report_startup_device_status(self) -> None:
        if self.device_index is None or self.device_info is None:
            message = (
                "Configured input device not found:\n"
                f"{CONFIG['input_device']}\n\n"
                "Install VB-Audio Virtual Cable and make sure Zoom audio is routed into CABLE Input."
            )
            self.status_var.set("Configured VAC input device not found.")
            show_error_popup(message)
        else:
            self.status_var.set("Ready")

    def _set_transcript_text(self, text: str) -> None:
        display = text if text.strip() else "(live text here)"
        self.transcript_box.configure(state="normal")
        self.transcript_box.delete("1.0", "end")
        self.transcript_box.insert("1.0", display)
        self.transcript_box.configure(state="disabled")
        self.transcript_box.see("end")

    def _process_ui_queue(self) -> None:
        while True:
            try:
                kind, payload = self.ui_queue.get_nowait()
            except queue.Empty:
                break

            if kind == "signal":
                rms_db, state_text = payload
                self.rms_db = float(rms_db)
                self.signal_state = str(state_text)
                self.rms_var.set(f"RMS: {self.rms_db:.1f} dB")
                if self.running:
                    self.status_var.set(self.signal_state)
            elif kind == "status":
                self.status_var.set(str(payload))
            elif kind == "transcript":
                self.transcript_text = str(payload)
                self._set_transcript_text(self.transcript_text)
            elif kind == "error":
                self.status_var.set(str(payload))
                show_error_popup(str(payload))

        self.root.after(100, self._process_ui_queue)

    def audio_callback(self, indata, frames, time_info, status) -> None:
        try:
            rms_db = compute_rms_db(indata)
            state_text = signal_state_from_db(rms_db)
            self.ui_queue.put(("signal", (rms_db, state_text)))
            if status:
                log_error("Audio callback status", str(status))
            if not self.running:
                return
            pcm_bytes = pcm16_bytes(indata)
            try:
                self.audio_queue.put_nowait(pcm_bytes)
            except queue.Full:
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.audio_queue.put_nowait(pcm_bytes)
                except queue.Full:
                    pass
        except Exception as exc:
            log_error("Audio callback failed", exc)
            self.ui_queue.put(("error", f"Audio callback failed: {exc}"))

    @safe_thread
    def _audio_sender_loop(self) -> None:
        while not self.audio_stop_event.is_set():
            try:
                pcm_bytes = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.deepgram is None or not self.running:
                continue
            try:
                self.deepgram.send(pcm_bytes)
            except Exception as exc:
                log_error("Deepgram send failed", exc)
                self.ui_queue.put(("error", f"Deepgram send failed: {exc}"))
                self.ui_queue.put(("status", "Deepgram send failed."))
                self.audio_stop_event.set()

    def start_transcription(self) -> None:
        global _ACTIVE_AUDIO_CALLBACK

        if self.running:
            return
        if self.device_index is None or self.device_info is None:
            self._report_startup_device_status()
            return

        api_key = get_deepgram_api_key()
        if not api_key:
            show_error_popup("Missing DEEPGRAM_API_KEY in .env.")
            self.status_var.set("Missing DEEPGRAM_API_KEY.")
            return

        self.deepgram = DeepgramLiveClient(api_key, self.ui_queue)
        success, message = self.deepgram.start()
        if not success:
            self.deepgram = None
            self.status_var.set(message)
            show_error_popup(message)
            return

        self.audio_stop_event.clear()
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        _ACTIVE_AUDIO_CALLBACK = self.audio_callback
        self.stream = start_audio_stream()
        if self.stream is None:
            if self.deepgram is not None:
                self.deepgram.stop()
                self.deepgram = None
            self.status_var.set("Audio stream failed to start.")
            return

        self.running = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set("Active")
        self._set_transcript_text("")
        self.sender_thread = threading.Thread(target=self._audio_sender_loop, daemon=True, name="audio-sender")
        self.sender_thread.start()

    def stop_transcription(self) -> None:
        if not self.running and self.stream is None and self.deepgram is None:
            self.status_var.set("Idle")
            return

        self.running = False
        self.audio_stop_event.set()

        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as exc:
                log_error("Audio stream stop failed", exc)
            self.stream = None

        if self.deepgram is not None:
            try:
                self.deepgram.stop()
            finally:
                self._save_transcript_snapshot(self.deepgram.current_transcript())
                self.deepgram = None

        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.status_var.set("Stopped")

    def _save_transcript_snapshot(self, transcript_text: str) -> None:
        text = transcript_text.strip()
        if not text:
            return
        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        path = TRANSCRIPTS_DIR / f"live_transcript_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        try:
            path.write_text(text, encoding="utf-8")
        except OSError as exc:
            log_error("Failed to save transcript", exc)

    def on_close(self) -> None:
        self.stop_transcription()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    ensure_local_venv()
    if "--list-devices" in sys.argv:
        list_audio_devices()
        return
    app = SimpleAudioApp()
    app.run()


if __name__ == "__main__":
    main()
