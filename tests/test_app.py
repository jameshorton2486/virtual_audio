from __future__ import annotations

import os
import queue
import subprocess
import tempfile
import threading
import unittest
import logging
import re
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pathlib import Path

import app
import numpy as np
from audio.auto_audio_engine import AutoAudioEngine
from audio.detector import AudioDeviceEntry
from audio.routing import RoutingManager


class DummyVar:
    def __init__(self, value):
        self.value = value
        self._traces = []

    def get(self):
        return self.value

    def set(self, value) -> None:
        self.value = value
        for callback in self._traces:
            callback()

    def trace_add(self, _mode, callback) -> None:
        self._traces.append(lambda: callback())


class DummyButton:
    def __init__(self, fg_color="#111111"):
        self.props = {"fg_color": fg_color, "text": "", "state": "normal", "border_width": 0, "border_color": fg_color}

    def winfo_exists(self) -> bool:
        return True

    def cget(self, name):
        return self.props[name]

    def configure(self, **kwargs) -> None:
        self.props.update(kwargs)


class DummyRoot:
    def __init__(self):
        self.exists = True
        self.clipboard = ""

    def after(self, _delay, callback) -> None:
        callback()

    def winfo_exists(self) -> bool:
        return self.exists

    def destroy(self) -> None:
        self.exists = False

    def clipboard_clear(self) -> None:
        self.clipboard = ""

    def clipboard_append(self, value: str) -> None:
        self.clipboard += value

    def update(self) -> None:
        return None


class CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def bind_app_method(obj, method_name: str):
    return getattr(app.App, method_name).__get__(obj, app.App)


class DeviceHelpersTests(unittest.TestCase):
    def test_reset_logs_on_startup_archives_existing_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            logs_dir = Path(temp_dir)
            log_path = logs_dir / "virtual_audio.log"
            error_log_path = logs_dir / "errors.log"
            log_path.write_text("info", encoding="utf-8")
            error_log_path.write_text("error", encoding="utf-8")

            with patch.object(app, "LOGS_DIR", logs_dir), patch.object(app, "LOG_PATH", log_path), patch.object(
                app,
                "ERROR_LOG_PATH",
                error_log_path,
            ), patch("app.time.strftime", return_value="20260419_180000"):
                app.reset_logs_on_startup(archive=True)

            self.assertFalse(log_path.exists())
            self.assertFalse(error_log_path.exists())
            self.assertEqual((logs_dir / "virtual_audio_20260419_180000.log").read_text(encoding="utf-8"), "info")
            self.assertEqual((logs_dir / "errors_20260419_180000.log").read_text(encoding="utf-8"), "error")

    def test_log_run_header_includes_config_context(self) -> None:
        handler = CaptureHandler()
        app.LOGGER.addHandler(handler)
        try:
            with patch("app.time.strftime", return_value="2026-04-19 18:30:00"):
                app.log_run_header(
                    {
                        "last_mode": "VAC",
                        "sample_rate_hz": 24000,
                        "wer_mode_enabled": True,
                    }
                )
        finally:
            app.LOGGER.removeHandler(handler)

        self.assertTrue(any("NEW APPLICATION RUN" in message for message in handler.messages))
        self.assertTrue(any("Time: 2026-04-19 18:30:00" in message for message in handler.messages))
        self.assertTrue(any("Last Mode: VAC" in message for message in handler.messages))
        self.assertTrue(any("Sample Rate: 24000" in message for message in handler.messages))
        self.assertTrue(any("WER Mode: True" in message for message in handler.messages))

    def test_mixed_unavailable_reason_mentions_invalid_configured_device(self) -> None:
        app_stub = app.App.__new__(app.App)
        app_stub._mixed_unavailable_reason = bind_app_method(app_stub, "_mixed_unavailable_reason")

        reason = app_stub._mixed_unavailable_reason("Microphone (Realtek HD Audio Mic input)")

        self.assertIn("recording inputs", reason)

    def test_sanitize_mixed_input_configuration_preserves_valid_vac_device(self) -> None:
        app_stub = app.App.__new__(app.App)
        app_stub.detected_input_devices = [
            "Microphone (Realtek HD Audio Mic input)",
            "CABLE Output (VB-Audio Virtual Cable)",
        ]
        app_stub.config = {"voicemeeter_device": "CABLE Output (VB-Audio Virtual Cable)"}
        app_stub.mix_var = DummyVar("CABLE Output (VB-Audio Virtual Cable)")
        app_stub._is_voicemeeter_choice_valid = bind_app_method(app_stub, "_is_voicemeeter_choice_valid")
        app_stub._sanitize_mixed_input_configuration = bind_app_method(app_stub, "_sanitize_mixed_input_configuration")

        with patch.object(app, "save_config") as save_config:
            app_stub._sanitize_mixed_input_configuration()

        self.assertEqual(app_stub.mix_var.get(), "CABLE Output (VB-Audio Virtual Cable)")
        self.assertEqual(app_stub.config["voicemeeter_device"], "CABLE Output (VB-Audio Virtual Cable)")
        save_config.assert_not_called()

    def test_ui_log_handler_updates_watchdog_on_stall_event(self) -> None:
        enqueued: list[tuple[str, int]] = []
        watchdog_states: list[str] = []
        stalled_calls: list[str] = []
        handler = app.UILogHandler(
            lambda message, level: enqueued.append((message, level)),
            lambda state: watchdog_states.append(state),
            lambda: stalled_calls.append("called"),
        )
        handler.setFormatter(logging.Formatter("%(message)s"))

        record = logging.LogRecord(
            name="virtual_audio",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg="[LiveSession] event=audio_callback_stalled device='CABLE Output'",
            args=(),
            exc_info=None,
        )

        handler.emit(record)

        self.assertEqual(watchdog_states, ["stalled"])
        self.assertEqual(stalled_calls, ["called"])
        self.assertEqual(enqueued[0][0], "[LiveSession] event=audio_callback_stalled device='CABLE Output'")

    def test_autofix_no_signal_switches_to_fallback_mode(self) -> None:
        fake_app = SimpleNamespace(
            _closing=False,
            _live_transcription_running=True,
            _live_transcription_starting=False,
            current_mode="VAC",
            root=DummyRoot(),
            apply_audio_mode=Mock(),
            _pick_fallback_mode=Mock(return_value="Microphone"),
        )

        autofix = app.AutoFixEngine(fake_app, cooldown_seconds=0.0)
        autofix.on_signal("no_signal", -100.0, -100.0)

        fake_app.apply_audio_mode.assert_called_once_with("Microphone")

    def test_autofix_queue_overflow_is_suggestion_in_safe_mode(self) -> None:
        suggestion_calls: list[str] = []
        fake_app = SimpleNamespace(
            _closing=False,
            _live_transcription_running=True,
            _live_transcription_starting=False,
            current_mode="VAC",
            root=DummyRoot(),
            stop_live_transcription=Mock(),
            start_live_transcription=Mock(),
            on_autofix_suggestion=lambda description: suggestion_calls.append(description),
            live_transcription_session=SimpleNamespace(_pcm_queue=queue.Queue()),
            debug_watchdog_var=DummyVar("Audio: OK"),
            _latest_signal_state="Active",
        )
        for _ in range(25):
            fake_app.live_transcription_session._pcm_queue.put_nowait(b"x")

        autofix = app.AutoFixEngine(fake_app, cooldown_seconds=0.0)
        autofix.on_queue(50)
        autofix.on_queue(50)
        autofix.on_queue(50)

        fake_app.stop_live_transcription.assert_not_called()
        fake_app.start_live_transcription.assert_not_called()
        self.assertEqual(len(suggestion_calls), 1)
        self.assertIsNotNone(autofix.last_suggestion)

    def test_autofix_queue_overflow_restarts_live_session_when_safe_mode_disabled(self) -> None:
        fake_app = SimpleNamespace(
            _closing=False,
            _live_transcription_running=True,
            _live_transcription_starting=False,
            current_mode="VAC",
            root=DummyRoot(),
            stop_live_transcription=Mock(),
            start_live_transcription=Mock(),
            live_transcription_session=SimpleNamespace(_pcm_queue=queue.Queue()),
            debug_watchdog_var=DummyVar("Audio: OK"),
            _latest_signal_state="Active",
        )
        for _ in range(30):
            fake_app.live_transcription_session._pcm_queue.put_nowait(b"x")

        autofix = app.AutoFixEngine(fake_app, cooldown_seconds=0.0)
        autofix.set_safe_mode(False)
        autofix.on_queue(50)
        autofix.on_queue(50)
        autofix.on_queue(50)

        fake_app.stop_live_transcription.assert_called_once()
        fake_app.start_live_transcription.assert_called_once()

    def test_autofix_queue_overflow_ignores_single_spike(self) -> None:
        fake_app = SimpleNamespace(
            _closing=False,
            _live_transcription_running=True,
            _live_transcription_starting=False,
            current_mode="VAC",
            root=DummyRoot(),
            stop_live_transcription=Mock(),
            start_live_transcription=Mock(),
            live_transcription_session=SimpleNamespace(_pcm_queue=queue.Queue()),
            debug_watchdog_var=DummyVar("Audio: OK"),
            _latest_signal_state="Active",
        )

        autofix = app.AutoFixEngine(fake_app, cooldown_seconds=0.0)
        autofix.set_safe_mode(False)
        autofix.on_queue(50)
        autofix.on_queue(10)
        autofix.on_queue(50)
        autofix.on_queue(50)

        fake_app.stop_live_transcription.assert_not_called()
        fake_app.start_live_transcription.assert_not_called()

    def test_autofix_active_fix_blocks_overlap(self) -> None:
        fake_app = SimpleNamespace(
            _closing=False,
            _live_transcription_running=True,
            _live_transcription_starting=False,
            current_mode="VAC",
            root=DummyRoot(),
            debug_watchdog_var=DummyVar("Audio: STALLED"),
        )

        autofix = app.AutoFixEngine(fake_app, cooldown_seconds=0.0)
        autofix._active_fix = True

        with patch.object(autofix, "_suggest_fix") as suggest_fix:
            autofix.on_watchdog_stalled()

        suggest_fix.assert_not_called()

    def test_autofix_disruptive_fix_skips_when_idle(self) -> None:
        suggestion_calls: list[str] = []
        fake_app = SimpleNamespace(
            _closing=False,
            _live_transcription_running=False,
            _live_transcription_starting=False,
            current_mode="VAC",
            root=DummyRoot(),
            stop_live_transcription=Mock(),
            start_live_transcription=Mock(),
            on_autofix_suggestion=lambda description: suggestion_calls.append(description),
            debug_watchdog_var=DummyVar("Audio: STALLED"),
        )

        autofix = app.AutoFixEngine(fake_app, cooldown_seconds=0.0)
        autofix.on_watchdog_stalled()

        self.assertEqual(len(suggestion_calls), 1)
        fake_app.stop_live_transcription.assert_not_called()
        fake_app.start_live_transcription.assert_not_called()

    def test_apply_suggested_fix_executes_and_clears(self) -> None:
        executed = {"ran": False}
        cleared_calls: list[str] = []

        def fake_fix():
            executed["ran"] = True

        fake_app = SimpleNamespace(
            on_autofix_suggestion=lambda description: None,
            on_autofix_cleared=lambda: cleared_calls.append("cleared"),
            _closing=False,
            _live_transcription_running=True,
            _live_transcription_starting=False,
        )

        autofix = app.AutoFixEngine(fake_app, cooldown_seconds=0.0)
        autofix._suggest_fix("Restart live transcription", fix_fn=fake_fix, is_hard=True)

        self.assertIsNotNone(autofix.last_suggestion)
        result = autofix.apply_last_suggestion()

        self.assertTrue(result)
        self.assertTrue(executed["ran"])
        self.assertIsNone(autofix.last_suggestion)
        self.assertEqual(cleared_calls, ["cleared"])

    def test_apply_suggested_fix_refuses_stale_entry(self) -> None:
        executed = {"ran": False}
        cleared_calls: list[str] = []

        def fake_fix():
            executed["ran"] = True

        fake_app = SimpleNamespace(
            on_autofix_suggestion=lambda description: None,
            on_autofix_cleared=lambda: cleared_calls.append("cleared"),
            _live_transcription_running=True,
            _live_transcription_starting=False,
            _closing=False,
        )

        autofix = app.AutoFixEngine(fake_app, cooldown_seconds=0.0)
        autofix._suggest_fix("Restart live transcription", fix_fn=fake_fix, is_hard=True)
        autofix._suggestion_time -= 11.0

        result = autofix.apply_last_suggestion()

        self.assertFalse(result)
        self.assertFalse(executed["ran"])
        self.assertIsNone(autofix.last_suggestion)
        self.assertEqual(cleared_calls, ["cleared"])

    def test_recovery_pipeline_queue_overflow_succeeds_without_restart_when_queue_recovers(self) -> None:
        queue_ref: queue.Queue[bytes] = queue.Queue()
        for _ in range(25):
            queue_ref.put_nowait(b"x")
        fake_app = SimpleNamespace(
            _closing=False,
            _live_transcription_running=True,
            _live_transcription_starting=False,
            current_mode="VAC",
            root=DummyRoot(),
            live_transcription_session=SimpleNamespace(_pcm_queue=queue_ref),
            debug_watchdog_var=DummyVar("Audio: OK"),
            _latest_signal_state="Active",
        )

        autofix = app.AutoFixEngine(fake_app, cooldown_seconds=0.0)
        with patch("app.time.sleep", lambda _seconds: [queue_ref.get_nowait() for _ in range(min(queue_ref.qsize(), 10))]):
            result = autofix._run_recovery_pipeline("queue_overflow")

        self.assertTrue(result)

    def test_on_apply_fix_runs_through_autofix_and_updates_ui(self) -> None:
        app_stub = app.App.__new__(app.App)
        app_stub.debug_suggestion_var = DummyVar("Suggested Fix: Restart")
        app_stub.apply_fix_button = DummyButton()
        app_stub.on_autofix_cleared = bind_app_method(app_stub, "on_autofix_cleared")
        app_stub._on_apply_fix = bind_app_method(app_stub, "_on_apply_fix")

        autofix = SimpleNamespace(apply_last_suggestion=Mock(return_value=True))
        app_stub.autofix = autofix

        app_stub._on_apply_fix()

        autofix.apply_last_suggestion.assert_called_once()

    def test_auto_audio_engine_classifies_input_devices(self) -> None:
        engine = AutoAudioEngine()
        engine.devices = [
            {"name": "Microphone (Realtek HD Audio Mic input)", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "CABLE Output (VB-Audio Virtual Cable)", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "VoiceMeeter Output (VB-Audio VoiceMeeter VAIO)", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "Stereo Mix", "max_input_channels": 1, "max_output_channels": 0},
        ]

        categories = engine.classify_input_devices()

        self.assertEqual([entry.name for entry in categories["mic"]], ["Microphone (Realtek HD Audio Mic input)"])
        self.assertEqual([entry.name for entry in categories["vac"]], ["CABLE Output (VB-Audio Virtual Cable)"])
        self.assertEqual([entry.name for entry in categories["voicemeeter"]], ["VoiceMeeter Output (VB-Audio VoiceMeeter VAIO)"])
        self.assertEqual([entry.name for entry in categories["other"]], ["Stereo Mix"])

    def test_auto_audio_engine_does_not_treat_microsoft_sound_mapper_as_mic(self) -> None:
        engine = AutoAudioEngine()
        engine.devices = [
            {"name": "Microsoft Sound Mapper - Input", "max_input_channels": 2, "max_output_channels": 0},
            {"name": "Microphone (Razer Seiren V3 Mini)", "max_input_channels": 1, "max_output_channels": 0},
        ]

        categories = engine.classify_input_devices()

        self.assertEqual([entry.name for entry in categories["mic"]], ["Microphone (Razer Seiren V3 Mini)"])
        self.assertEqual([entry.name for entry in categories["other"]], ["Microsoft Sound Mapper - Input"])

    def test_auto_audio_engine_selects_best_device_by_mode(self) -> None:
        engine = AutoAudioEngine()
        engine.devices = [
            {"name": "Microphone (Realtek HD Audio Mic input)", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "CABLE Output (VB-Audio Virtual Cable)", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "VoiceMeeter Output (VB-Audio VoiceMeeter VAIO)", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "Speakers (Realtek Audio)", "max_input_channels": 0, "max_output_channels": 2},
            {"name": "CABLE Input (VB-Audio Virtual Cable)", "max_input_channels": 0, "max_output_channels": 2},
        ]

        self.assertEqual(engine.select_best_input_device("Microphone").name, "Microphone (Realtek HD Audio Mic input)")
        self.assertEqual(engine.select_best_input_device("VAC").name, "CABLE Output (VB-Audio Virtual Cable)")
        self.assertEqual(engine.select_best_input_device("Mixed").name, "VoiceMeeter Output (VB-Audio VoiceMeeter VAIO)")
        self.assertEqual(engine.select_best_output_device("Microphone").name, "Speakers (Realtek Audio)")
        self.assertEqual(engine.select_best_output_device("VAC").name, "CABLE Input (VB-Audio Virtual Cable)")

    def test_routing_manager_uses_mode_specific_preferred_names(self) -> None:
        mixed_entry = AudioDeviceEntry(index=0, info={"name": "Stereo Mix"}, name="Stereo Mix")
        mic_entry = AudioDeviceEntry(index=1, info={"name": "Microphone (USB Mic)"}, name="Microphone (USB Mic)")
        calls: list[tuple[str | None, str]] = []

        class FakeDetector:
            def refresh_devices(self) -> None:
                return None

            def list_candidate_input_devices(self, preferred_name: str = "", mode_name=None):
                calls.append((mode_name, preferred_name))
                if mode_name == "Mixed":
                    return [mixed_entry]
                if mode_name == "Microphone":
                    return [mic_entry]
                return [mixed_entry]

            def build_null_input_entry(self) -> AudioDeviceEntry:
                return AudioDeviceEntry(index=-1, info={"name": "Silence"}, name="Silence")

        def signal_sampler(index, sample_rate_hz, device_name, duration_seconds=0.35, device_info=None):
            if device_name == "Microphone (USB Mic)":
                return {"rms_db": -18.0, "peak_db": -6.0, "state": "active"}
            return {"rms_db": -80.0, "peak_db": -80.0, "state": "silent"}

        manager = RoutingManager(FakeDetector(), signal_sampler)

        selected_entry, selected_mode, signal = manager.select_working_device(
            ["Mixed", "Microphone"],
            sample_rate_hz=16000,
            preferred_names={
                "Mixed": "Stereo Mix",
                "Microphone": "Microphone (USB Mic)",
            },
        )

        self.assertEqual(calls, [("Mixed", "Stereo Mix"), ("Microphone", "Microphone (USB Mic)")])
        self.assertEqual(selected_mode, "Microphone")
        self.assertEqual(selected_entry, mic_entry)
        self.assertEqual(signal["state"], "active")

    def test_refresh_detected_devices_prefers_auto_audio_engine(self) -> None:
        app_stub = SimpleNamespace(
            detected_input_devices=[],
            detected_output_devices=[],
            _last_detected_refresh_at=0.0,
        )

        fake_devices = [
            {"name": "Microphone (Realtek HD Audio Mic input)", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "Speakers (Realtek Audio)", "max_input_channels": 0, "max_output_channels": 2},
        ]

        with patch("app.time.monotonic", return_value=1.0), patch("app.AutoAudioEngine.refresh_devices", autospec=True) as refresh_mock:
            def _refresh(engine_self):
                engine_self.devices = fake_devices
                return fake_devices

            refresh_mock.side_effect = _refresh
            app.App._refresh_detected_devices(app_stub, force=True)

        self.assertEqual(app_stub.detected_input_devices, ["Microphone (Realtek HD Audio Mic input)"])
        self.assertEqual(app_stub.detected_output_devices, ["Speakers (Realtek Audio)"])

    def test_extract_transcript_text_returns_primary_transcript(self) -> None:
        payload = {
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {
                                "transcript": "hello world",
                            }
                        ]
                    }
                ]
            }
        }

        self.assertEqual(app.extract_transcript_text(payload), "hello world")

    def test_extract_transcript_text_handles_missing_data(self) -> None:
        self.assertEqual(app.extract_transcript_text({}), "")

    def test_format_deepgram_payload_text_uses_speaker_labels_when_words_present(self) -> None:
        payload = {
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {
                                "transcript": "hello world",
                                "words": [
                                    {"speaker": 0, "punctuated_word": "Hello"},
                                    {"speaker": 0, "punctuated_word": "there."},
                                    {"speaker": 1, "punctuated_word": "General"},
                                    {"speaker": 1, "punctuated_word": "Kenobi."},
                                ],
                            }
                        ]
                    }
                ]
            }
        }

        self.assertEqual(
            app.format_deepgram_payload_text(payload),
            "Speaker 0: Hello there.\nSpeaker 1: General Kenobi.",
        )

    def test_format_deepgram_payload_text_prefers_utterance_pipeline(self) -> None:
        payload = {
            "results": {
                "utterances": [
                    {
                        "speaker": 0,
                        "start": 0.0,
                        "end": 0.4,
                        "transcript": "What time did you arrive?",
                        "words": [{"word": "What"}, {"word": "time"}, {"word": "arrive"}],
                    },
                    {
                        "speaker": 1,
                        "start": 0.6,
                        "end": 0.9,
                        "transcript": "Yes I did",
                        "words": [{"word": "Yes"}, {"word": "I"}, {"word": "did"}],
                    },
                ],
                "channels": [
                    {
                        "alternatives": [
                            {
                                "transcript": "fallback text",
                                "words": [{"speaker": 0, "punctuated_word": "Fallback"}],
                            }
                        ]
                    }
                ],
            }
        }

        self.assertEqual(
            app.format_deepgram_payload_text(payload),
            "Q: What time did you arrive?\nA: Yes I did",
        )

    def test_merge_utterances_drops_tiny_fragment_and_merges_close_same_speaker(self) -> None:
        utterances = [
            {"speaker": 0, "start": 0.0, "end": 0.5, "text": "Yes I do", "words": [1, 2, 3]},
            {"speaker": 0, "start": 1.0, "end": 1.4, "text": "remember that", "words": [4, 5]},
            {"speaker": 1, "start": 3.0, "end": 3.1, "text": "okay", "words": [6]},
        ]

        result = app.merge_utterances(utterances)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["speaker"], 0)
        self.assertEqual(result[0]["text"], "Yes I do remember that")

    def test_prevent_micro_speaker_switch_reverts_tiny_flip(self) -> None:
        blocks = [
            {"speaker": 0, "text": "Please state your name.", "words": [1, 2, 3, 4]},
            {"speaker": 1, "text": "Yes.", "words": [5]},
        ]

        result = app.prevent_micro_speaker_switch(blocks)

        self.assertEqual(result[1]["speaker"], 0)

    def test_stabilize_speaker_blocks_reverts_short_flip(self) -> None:
        blocks = [
            {"speaker": 0, "text": "Please state your full legal name."},
            {"speaker": 1, "text": "Yes sir."},
            {"speaker": 0, "text": "Where were you employed then?"},
        ]

        result = app.stabilize_speaker_blocks(blocks)

        self.assertEqual(result[1]["speaker"], 0)

    def test_enforce_qa_structure_marks_following_answer_after_question(self) -> None:
        blocks = [
            {"speaker": 0, "text": "What time did you arrive?"},
            {"speaker": 1, "text": "Around 8 a.m."},
            {"speaker": 1, "text": "And then I waited."},
        ]

        result = app.enforce_qa_structure(blocks)

        self.assertEqual(result[0]["type"], "Q")
        self.assertEqual(result[1]["type"], "A")
        self.assertEqual(result[2]["type"], "A")

    def test_flag_low_confidence_words_returns_only_low_confidence_items(self) -> None:
        words = [
            {"word": "hello", "start": 0.0, "end": 0.2, "confidence": 0.95, "speaker": 0},
            {"word": "their", "start": 0.3, "end": 0.5, "confidence": 0.62, "speaker": 1},
        ]

        result = app.flag_low_confidence_words(words, threshold=0.85)

        self.assertEqual(
            result,
            [
                {
                    "word": "their",
                    "start": 0.3,
                    "end": 0.5,
                    "confidence": 0.62,
                    "speaker": 1,
                }
            ],
        )

    def test_normalize_deepgram_blocks_applies_full_alignment_pipeline(self) -> None:
        utterances = [
            {
                "speaker": 0,
                "start": 0.0,
                "end": 0.5,
                "transcript": "What did you say?",
                "confidence": 0.9,
                "words": [{"word": "What", "confidence": 0.9}, {"word": "say", "confidence": 0.88}],
            },
            {
                "speaker": 1,
                "start": 0.7,
                "end": 1.0,
                "transcript": "Yes.",
                "confidence": 0.92,
                "words": [{"word": "Yes", "confidence": 0.92}],
            },
            {
                "speaker": 1,
                "start": 1.1,
                "end": 1.5,
                "transcript": "I was there.",
                "confidence": 0.93,
                "words": [{"word": "I", "confidence": 0.93}, {"word": "there", "confidence": 0.84}],
            },
        ]

        result = app.normalize_deepgram_blocks(utterances)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["type"], "Q")
        self.assertEqual(result[1]["type"], "A")
        self.assertEqual(result[1]["speaker"], 0)

    def test_build_transcript_output_paths_uses_transcripts_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            media_path = Path(temp_dir) / "zoom recording.mp4"
            media_path.write_text("placeholder", encoding="utf-8")
            output_dir = Path(temp_dir) / "transcripts"

            transcript_path, payload_path = app.build_transcript_output_paths(media_path, output_dir=output_dir)

            self.assertEqual(transcript_path.parent, output_dir)
            self.assertEqual(payload_path.parent, output_dir)
            self.assertEqual(transcript_path.suffix, ".txt")
            self.assertEqual(payload_path.suffix, ".json")
            self.assertIn("zoom_recording", transcript_path.name)

    def test_build_live_transcript_output_path_uses_timestamped_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = app.build_live_transcript_output_path(output_dir=Path(temp_dir))

            self.assertEqual(output_path.parent, Path(temp_dir))
            self.assertEqual(output_path.suffix, ".txt")
            self.assertIn("live_transcript_", output_path.name)

    def test_build_live_transcript_metadata_path_uses_same_stem(self) -> None:
        transcript_path = Path("transcripts") / "live_transcript_20260414_210000.txt"
        metadata_path = app.build_live_transcript_metadata_path(transcript_path)

        self.assertEqual(metadata_path.parent, transcript_path.parent)
        self.assertEqual(metadata_path.suffix, ".json")
        self.assertEqual(metadata_path.stem, transcript_path.stem)

    def test_normalize_audio_device_name_strips_backend_suffix(self) -> None:
        self.assertEqual(
            app.normalize_audio_device_name("CABLE Output (VB-Audio Virtual Cable), Windows WASAPI"),
            "CABLE Output (VB-Audio Virtual Cable)",
        )

    def test_infer_vac_recording_device_prefers_cable_output(self) -> None:
        devices = [
            "CABLE Input (VB-Audio Virtual Cable)",
            "CABLE Output (VB-Audio Virtual Cable)",
            "VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)",
        ]

        result = app.infer_vac_recording_device(app.DEFAULT_CONFIG["vac_device"], devices)

        self.assertEqual(result, "CABLE Output (VB-Audio Virtual Cable)")

    def test_infer_vac_playback_device_prefers_cable_input(self) -> None:
        devices = [
            "CABLE Output (VB-Audio Virtual Cable)",
            "Speakers (Realtek Audio)",
            "CABLE Input (VB-Audio Virtual Cable)",
        ]

        result = app.infer_vac_playback_device(app.DEFAULT_CONFIG["vac_playback_device"], devices)

        self.assertEqual(result, "CABLE Input (VB-Audio Virtual Cable)")

    def test_infer_speaker_output_device_avoids_virtual_outputs(self) -> None:
        devices = [
            "CABLE Input (VB-Audio Virtual Cable)",
            "VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)",
            "Speakers (Realtek Audio)",
        ]

        result = app.infer_speaker_output_device(app.DEFAULT_CONFIG["speaker_device"], devices)

        self.assertEqual(result, "Speakers (Realtek Audio)")

    def test_infer_speaker_output_device_accepts_tv_hdmi_outputs(self) -> None:
        devices = [
            "CABLE Input (VB-Audio Virtual Cable)",
            "SAMSUNG TV (NVIDIA High Definition Audio)",
        ]

        result = app.infer_speaker_output_device("Nonexistent Speakers", devices)

        self.assertEqual(result, "SAMSUNG TV (NVIDIA High Definition Audio)")

    def test_infer_microphone_input_device_falls_back_to_real_nonvirtual_input(self) -> None:
        devices = [
            "Stereo Mix (Realtek(R) Audio)",
            "Microphone (Realtek HD Audio Mic input)",
            "CABLE Output (VB-Audio Virtual Cable)",
        ]

        result = app.infer_microphone_input_device("VB-Audio Point", devices)

        self.assertEqual(result, "Microphone (Realtek HD Audio Mic input)")

    def test_set_default_recording_device_updates_all_windows_roles(self) -> None:
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0)

        with patch("pathlib.Path.exists", return_value=True), patch("app.subprocess.run", side_effect=fake_run):
            ok, message = app.AudioDeviceManager.set_default_recording_device("CABLE Output (VB-Audio Virtual Cable)")

        self.assertTrue(ok)
        self.assertIn("all Windows audio roles", message)
        self.assertEqual(
            calls,
            [
                [str(app.SOUNDVOLUMEVIEW_PATH), "/SetDefault", "CABLE Output (VB-Audio Virtual Cable)", "Console"],
                [str(app.SOUNDVOLUMEVIEW_PATH), "/SetDefault", "CABLE Output (VB-Audio Virtual Cable)", "Multimedia"],
                [str(app.SOUNDVOLUMEVIEW_PATH), "/SetDefault", "CABLE Output (VB-Audio Virtual Cable)", "Communications"],
            ],
        )

    def test_set_default_playback_device_updates_all_windows_roles(self) -> None:
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0)

        with patch("pathlib.Path.exists", return_value=True), patch("app.subprocess.run", side_effect=fake_run):
            ok, message = app.AudioDeviceManager.set_default_playback_device("CABLE Input (VB-Audio Virtual Cable)")

        self.assertTrue(ok)
        self.assertIn("playback device", message)
        self.assertEqual(
            calls,
            [
                [str(app.SOUNDVOLUMEVIEW_PATH), "/SetDefault", "CABLE Input (VB-Audio Virtual Cable)", "Console"],
                [str(app.SOUNDVOLUMEVIEW_PATH), "/SetDefault", "CABLE Input (VB-Audio Virtual Cable)", "Multimedia"],
                [str(app.SOUNDVOLUMEVIEW_PATH), "/SetDefault", "CABLE Input (VB-Audio Virtual Cable)", "Communications"],
            ],
        )

    def test_sanitize_config_recovers_from_invalid_values(self) -> None:
        result = app.sanitize_config(
            {
                "mic_device": "   ",
                "vac_device": 42,
                "speaker_device": "",
                "vac_playback_device": None,
                "voicemeeter_device": None,
                "mixed_playback_device": None,
                "restore_devices_on_exit": None,
                "deepgram_smart_format": "false",
                "deepgram_diarize": "true",
                "deepgram_paragraphs": "false",
                "deepgram_filler_words": "true",
                "deepgram_numerals": "false",
                "wer_mode_enabled": "false",
                "quality_check_interval_seconds": 0,
                "sample_rate_hz": "invalid",
                "last_mode": "Speakerphone",
            }
        )

        self.assertEqual(result["mic_device"], app.DEFAULT_CONFIG["mic_device"])
        self.assertEqual(result["vac_device"], app.DEFAULT_CONFIG["vac_device"])
        self.assertEqual(result["speaker_device"], app.DEFAULT_CONFIG["speaker_device"])
        self.assertEqual(result["vac_playback_device"], app.DEFAULT_CONFIG["vac_playback_device"])
        self.assertEqual(result["voicemeeter_device"], app.DEFAULT_CONFIG["voicemeeter_device"])
        self.assertEqual(result["mixed_playback_device"], "")
        self.assertTrue(result["restore_devices_on_exit"])
        self.assertFalse(result["wer_mode_enabled"])
        self.assertEqual(
            result["quality_check_interval_seconds"],
            app.DEFAULT_CONFIG["quality_check_interval_seconds"],
        )
        self.assertEqual(result["sample_rate_hz"], app.DEFAULT_CONFIG["sample_rate_hz"])
        self.assertEqual(result["last_mode"], app.DEFAULT_CONFIG["last_mode"])

    def test_infer_device_prefers_exact_match(self) -> None:
        devices = ["Microphone (Yeti Stereo Microphone)", "CABLE Output (VB-Audio Virtual Cable)"]
        result = app.infer_device("CABLE Output (VB-Audio Virtual Cable)", devices, ["cable"])
        self.assertEqual(result, "CABLE Output (VB-Audio Virtual Cable)")

    def test_infer_device_prefers_real_microphone_over_mapper(self) -> None:
        devices = [
            "Microsoft Sound Mapper - Input",
            "Primary Sound Capture Driver",
            "Microphone (Yeti Stereo Microphone)",
            "Microphone (Realtek HD Audio Mic input)",
        ]
        result = app.infer_device("Microphone (Realtek Audio)", devices, ["microphone"])
        self.assertEqual(result, "Microphone (Yeti Stereo Microphone)")

    def test_infer_device_keeps_vac_placeholder_when_not_detected(self) -> None:
        devices = ["Microphone (Yeti Stereo Microphone)", "Stereo Mix (Realtek HD Audio Stereo input)"]
        result = app.infer_device(app.DEFAULT_CONFIG["vac_device"], devices, ["cable"])
        self.assertEqual(result, app.DEFAULT_CONFIG["vac_device"])

    def test_list_input_devices_normalizes_backend_suffixes(self) -> None:
        mocked_devices = [
            {"name": "Microphone (Yeti Stereo Microphone), Windows WASAPI", "max_input_channels": 2},
            {"name": "Microphone (Yeti Stereo Microphone), MME", "max_input_channels": 2},
            {"name": "SAMSUNG (NVIDIA High Definition Audio), MME", "max_input_channels": 0},
            {"name": "CABLE Output (VB-Audio Virtual Cable), Windows WDM-KS", "max_input_channels": 2},
        ]
        with patch("app.sd.query_devices", return_value=mocked_devices):
            result = app.list_input_devices()

        self.assertEqual(
            result,
            ["Microphone (Yeti Stereo Microphone)", "CABLE Output (VB-Audio Virtual Cable)"],
        )

    def test_list_output_devices_normalizes_backend_suffixes(self) -> None:
        mocked_devices = [
            {"name": "Speakers (Realtek Audio), Windows WASAPI", "max_output_channels": 2},
            {"name": "Speakers (Realtek Audio), MME", "max_output_channels": 2},
            {"name": "CABLE Input (VB-Audio Virtual Cable), Windows WDM-KS", "max_output_channels": 2},
            {"name": "Microphone (Yeti Stereo Microphone), MME", "max_output_channels": 0},
        ]
        with patch("app.sd.query_devices", return_value=mocked_devices):
            result = app.list_output_devices()

        self.assertEqual(
            result,
            ["Speakers (Realtek Audio)", "CABLE Input (VB-Audio Virtual Cable)"],
        )


class AudioQualityMonitorTests(unittest.TestCase):
    def test_sample_quality_handles_non_finite_or_extreme_values(self) -> None:
        results: list[dict[str, object]] = []
        monitor = app.AudioQualityMonitor(24000, 2.0, results.append)
        recording = np.array([[np.nan], [np.inf], [-np.inf], [1e20]], dtype=np.float32)

        with patch("app.sd.rec", return_value=recording), patch("app.sd.wait", return_value=None):
            result = monitor._sample_quality()

        self.assertNotEqual(result["quality"], "error")
        self.assertIn("RMS", result["level_text"])


class LiveSignalTests(unittest.TestCase):
    def test_linear_to_db_floor(self) -> None:
        self.assertEqual(app.linear_to_db(0.0), -100.0)
        self.assertEqual(app.linear_to_db(1.0), 0.0)
        self.assertAlmostEqual(app.linear_to_db(0.5), -6.02, places=2)

    def test_classify_speech_signal_optimal(self) -> None:
        self.assertEqual(
            app.classify_speech_signal(-18.0, -10.0),
            ("optimal", "Optimal speech level", "#43A047"),
        )

    def test_classify_speech_signal_too_quiet(self) -> None:
        self.assertEqual(
            app.classify_speech_signal(-35.0, -30.0),
            ("too_quiet", "Too quiet — increase mic gain or source volume", "#F9A825"),
        )

    def test_classify_speech_signal_no_signal(self) -> None:
        self.assertEqual(
            app.classify_speech_signal(-60.0, -55.0),
            ("no_signal", "No signal detected — check routing, mute, and source playback", "#9E9E9E"),
        )

    def test_classify_speech_signal_clipping_wins_over_too_loud(self) -> None:
        self.assertEqual(
            app.classify_speech_signal(-8.0, -1.0),
            ("clipping", "Clipping risk — reduce input level", "#E53935"),
        )

    def test_analyze_live_input_signal_detects_silence(self) -> None:
        raw_bytes = (np.zeros(1024, dtype=np.int16)).tobytes()

        result = app.analyze_live_input_signal(raw_bytes)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["state"], "silent")

    def test_analyze_live_input_signal_detects_low_signal(self) -> None:
        raw_bytes = (np.full(1024, 100, dtype=np.int16)).tobytes()

        result = app.analyze_live_input_signal(raw_bytes)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["state"], "low")

    def test_analyze_live_input_signal_detects_active_signal(self) -> None:
        raw_bytes = (np.full(1024, 2500, dtype=np.int16)).tobytes()

        result = app.analyze_live_input_signal(raw_bytes)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["state"], "active")

    def test_analyze_live_input_signal_detects_clipping(self) -> None:
        raw_bytes = (np.full(1024, 32000, dtype=np.int16)).tobytes()

        result = app.analyze_live_input_signal(raw_bytes)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["state"], "clipping")

    def test_analyze_live_input_signal_includes_db_fields(self) -> None:
        raw_bytes = (np.full(1024, 16384, dtype=np.int16)).tobytes()

        result = app.analyze_live_input_signal(raw_bytes)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("rms_db", result)
        self.assertIn("peak_db", result)
        self.assertIn("clipping_hard", result)
        expected_db = app.linear_to_db(16384 / 32768.0)
        self.assertAlmostEqual(float(result["rms_db"]), expected_db, delta=0.5)
        self.assertAlmostEqual(float(result["peak_db"]), expected_db, delta=0.5)

    def test_report_input_signal_emits_signal_callback_before_deepgram_send(self) -> None:
        transcript_updates: list[tuple[str, str]] = []
        status_updates: list[str] = []
        signal_updates: list[dict[str, object]] = []
        session = app.LiveTranscriptionSession(
            api_key="test-key",
            input_device={
                "name": "CABLE Output (VB-Audio Virtual Cable)",
                "index": 1,
                "info": {"name": "CABLE Output (VB-Audio Virtual Cable)", "default_samplerate": 24000},
                "sample_rate": 24000,
            },
            mode_name="VAC",
            on_transcript=transcript_updates.append,
            on_status=status_updates.append,
            on_signal=signal_updates.append,
        )
        session.actual_device_name = "CABLE Output (VB-Audio Virtual Cable)"
        raw_bytes = (np.full(1024, 2500, dtype=np.int16)).tobytes()

        with patch("app.time.time", side_effect=[10.0, 11.1, 12.2]):
            session._report_input_signal(raw_bytes, 1024)
            session._report_input_signal(raw_bytes, 1024)
            session._report_input_signal(raw_bytes, 1024)

        self.assertGreaterEqual(len(signal_updates), 1)
        self.assertEqual(signal_updates[-1]["state"], "active")
        self.assertEqual(signal_updates[-1]["device_name"], "CABLE Output (VB-Audio Virtual Cable)")
        self.assertEqual(signal_updates[-1]["mode_name"], "VAC")
        self.assertEqual(len(status_updates), 1)
        self.assertIn("Live input optimal", status_updates[0])
        self.assertIn("RMS", status_updates[0])

    def test_report_input_signal_alternating_boundary_values_do_not_commit_state_change(self) -> None:
        session = app.LiveTranscriptionSession(
            api_key="test-key",
            input_device={
                "name": "CABLE Output (VB-Audio Virtual Cable)",
                "index": 1,
                "info": {"name": "CABLE Output (VB-Audio Virtual Cable)", "default_samplerate": 24000},
                "sample_rate": 24000,
            },
            mode_name="VAC",
            on_transcript=lambda *_args: None,
            on_status=Mock(),
            on_signal=Mock(),
        )
        session.actual_device_name = "CABLE Output (VB-Audio Virtual Cable)"
        session.last_signal_status = "optimal"
        signal_payloads = [
            {"rms": 0.02, "peak": 0.03, "rms_db": -24.8, "peak_db": -10.0, "state": "active"},
            {"rms": 0.005, "peak": 0.01, "rms_db": -25.2, "peak_db": -20.0, "state": "low"},
            {"rms": 0.02, "peak": 0.03, "rms_db": -24.7, "peak_db": -10.0, "state": "active"},
            {"rms": 0.005, "peak": 0.01, "rms_db": -25.3, "peak_db": -20.0, "state": "low"},
            {"rms": 0.02, "peak": 0.03, "rms_db": -24.9, "peak_db": -10.0, "state": "active"},
            {"rms": 0.005, "peak": 0.01, "rms_db": -25.1, "peak_db": -20.0, "state": "low"},
        ]

        with patch("app.analyze_live_input_signal", side_effect=signal_payloads), patch(
            "app.time.time",
            side_effect=[10.0, 10.3, 10.6, 10.9, 11.2, 11.5],
        ):
            for _ in signal_payloads:
                session._report_input_signal(b"raw", 1024)

        self.assertEqual(session.last_signal_status, "optimal")
        self.assertEqual(session.on_status.call_count, 0)
        self.assertEqual(session.on_signal.call_count, 2)

    def test_format_live_result_text_uses_speaker_labels_when_words_present(self) -> None:
        result = SimpleNamespace(
            channel=SimpleNamespace(
                alternatives=[
                    SimpleNamespace(
                        transcript="hello there general kenobi",
                        words=[
                            SimpleNamespace(speaker=0, punctuated_word="Hello"),
                            SimpleNamespace(speaker=0, punctuated_word="there."),
                            SimpleNamespace(speaker=1, punctuated_word="General"),
                            SimpleNamespace(speaker=1, punctuated_word="Kenobi."),
                        ],
                    )
                ]
            )
        )

        self.assertEqual(
            app.format_live_result_text(result),
            "Speaker 0: Hello there.\nSpeaker 1: General Kenobi.",
        )

    def test_format_live_result_text_uses_utterances_when_available(self) -> None:
        result = SimpleNamespace(
            utterances=[
                SimpleNamespace(
                    speaker=0,
                    start=0.0,
                    end=0.5,
                    transcript="Can you state your name?",
                    words=[SimpleNamespace(word="Can"), SimpleNamespace(word="you"), SimpleNamespace(word="name")],
                ),
                SimpleNamespace(
                    speaker=1,
                    start=0.8,
                    end=1.2,
                    transcript="Yes I can",
                    words=[SimpleNamespace(word="Yes"), SimpleNamespace(word="I"), SimpleNamespace(word="can")],
                ),
            ],
            channel=SimpleNamespace(alternatives=[]),
        )

        self.assertEqual(
            app.format_live_result_text(result),
            "Q: Can you state your name?\nA: Yes I can",
        )


class AppBehaviorTests(unittest.TestCase):
    def _make_app_stub(self):
        stub = app.App.__new__(app.App)
        stub.current_mode = "Microphone"
        stub.config = {
            "last_mode": "Microphone",
            "active_mode": "mic",
            "restore_devices_on_exit": True,
            "mic_device": "Microphone (Realtek HD Audio Mic input)",
            "vac_device": "CABLE Output (VB-Audio Virtual Cable)",
            "speaker_device": "Speakers (Realtek Audio)",
            "vac_playback_device": "CABLE Input (VB-Audio Virtual Cable)",
            "voicemeeter_device": "VoiceMeeter Output (VB-Audio VoiceMeeter VAIO)",
            "mixed_playback_device": "",
            "modes": {
                "mic": {
                    "input_device": "Microphone (Realtek HD Audio Mic input)",
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
            },
        }
        stub.is_muted = False
        stub._audio_switch_in_progress = False
        stub._pending_mode_button = None
        stub._pending_live_start_mode = None
        stub._pending_mode_after_live_stop = None
        stub._pending_vac_test = False
        stub._closing = False
        stub._transcription_running = False
        stub._live_transcription_running = False
        stub._live_transcription_starting = False
        stub._live_transcription_stopping = False
        stub._resume_monitor_after_live = False
        stub._last_detected_refresh_at = 0.0
        stub._resolved_input_name_cache = {}
        stub._last_mixed_unavailable_reason = None
        stub._slow_verification_count = 0
        stub._slow_verification_advisory_logged = False
        stub._latest_signal_state = "Unknown"
        stub.detected_input_devices = [
            "Microphone (Realtek HD Audio Mic input)",
            "CABLE Output (VB-Audio Virtual Cable)",
        ]
        stub.detected_output_devices = ["Speakers (Realtek Audio)", "CABLE Input (VB-Audio Virtual Cable)"]
        stub.mix_var = DummyVar("VoiceMeeter Output (VB-Audio VoiceMeeter VAIO)")
        stub.mic_var = DummyVar("Microphone (Realtek HD Audio Mic input)")
        stub.vac_var = DummyVar("CABLE Output (VB-Audio Virtual Cable)")
        stub.speaker_var = DummyVar("Speakers (Realtek Audio)")
        stub.vac_playback_var = DummyVar("CABLE Input (VB-Audio Virtual Cable)")
        stub.mixed_playback_var = DummyVar("")
        stub.restore_devices_on_exit_var = DummyVar(False)
        stub.require_signal_check_var = DummyVar(True)
        stub.status_var = DummyVar("")
        stub.mode_var = DummyVar("Microphone")
        stub.active_source_var = DummyVar("")
        stub.direct_recording_var = DummyVar("")
        stub.direct_playback_var = DummyVar("")
        stub.runtime_audio_var = DummyVar("")
        stub.debug_session_var = DummyVar("Session: Idle")
        stub.debug_device_var = DummyVar("Devices: Detecting...")
        stub.debug_signal_var = DummyVar("Signal: Unknown")
        stub.debug_queue_var = DummyVar("Queue: -")
        stub.debug_watchdog_var = DummyVar("Audio: OK")
        stub.debug_routing_var = DummyVar("Routing: -")
        stub.mode_badge_label = DummyButton("#1565C0")
        stub.active_device_label = DummyButton("#1565C0")
        stub.active_source_label = DummyButton("#1565C0")
        stub.live_transcription_status_label = DummyButton()
        stub.live_signal_status_label = DummyButton()
        stub.debug_signal_label = DummyButton()
        stub.debug_queue_label = DummyButton()
        stub.debug_watchdog_label = DummyButton()
        stub.debug_routing_label = DummyButton()
        stub.status_label = DummyButton()
        stub.btn_mic = DummyButton("#1565C0")
        stub.btn_vac = DummyButton("#2E7D32")
        stub.btn_mix = DummyButton("#8E24AA")
        stub.mute_button = DummyButton("#D32F2F")
        stub.root = DummyRoot()
        stub._original_default_input_name = "Microphone (Realtek HD Audio Mic input)"
        stub._original_default_output_name = "Speakers (Realtek Audio)"
        stub._queue_live_signal_update = lambda payload: None
        stub._refresh_mode_hint = lambda: None
        stub._refresh_runtime_audio_status = lambda signal_state=None: None
        stub._refresh_live_transcription_labels = lambda: None
        stub.save_form_config = lambda: None
        stub._wait_for_active_input_device = lambda expected: True
        stub.device_manager = SimpleNamespace(
            set_default_recording_device=lambda device: (True, f"record {device}"),
            set_default_playback_device=lambda device: (True, f"play {device}"),
        )
        stub._probe_vac_route = lambda duration_seconds=0.4: (None, "probe")
        stub._resolve_mode_devices = bind_app_method(stub, "_resolve_mode_devices")
        stub._expected_input_device_for_mode = bind_app_method(stub, "_expected_input_device_for_mode")
        stub._resolve_detected_input_name = bind_app_method(stub, "_resolve_detected_input_name")
        stub._is_mixed_mode_available = bind_app_method(stub, "_is_mixed_mode_available")
        stub._set_live_controls_state = bind_app_method(stub, "_set_live_controls_state")
        stub._update_debug_queue = bind_app_method(stub, "_update_debug_queue")
        stub._update_debug_watchdog = bind_app_method(stub, "_update_debug_watchdog")
        stub._update_debug_routing = bind_app_method(stub, "_update_debug_routing")
        stub._refresh_run_control_buttons = bind_app_method(stub, "_refresh_run_control_buttons")
        stub._run_preflight = bind_app_method(stub, "_run_preflight")
        stub._finish_apply_audio_mode_hot = bind_app_method(stub, "_finish_apply_audio_mode_hot")
        stub._finish_apply_audio_mode = bind_app_method(stub, "_finish_apply_audio_mode")
        stub._reconcile_startup_mode = bind_app_method(stub, "_reconcile_startup_mode")
        stub._refresh_detected_devices = bind_app_method(stub, "_refresh_detected_devices")
        stub._active_source_summary = bind_app_method(stub, "_active_source_summary")
        stub._active_device_matches_mode = bind_app_method(stub, "_active_device_matches_mode")
        stub._apply_mode_theme = bind_app_method(stub, "_apply_mode_theme")
        stub._log_mixed_unavailable = bind_app_method(stub, "_log_mixed_unavailable")
        stub._mark_mixed_available = bind_app_method(stub, "_mark_mixed_available")
        stub._mixed_unavailable_reason = bind_app_method(stub, "_mixed_unavailable_reason")
        stub._sanitize_mixed_input_configuration = bind_app_method(stub, "_sanitize_mixed_input_configuration")
        stub.use_current_windows_output_for_speakers = bind_app_method(stub, "use_current_windows_output_for_speakers")
        stub._restore_original_default_devices = bind_app_method(stub, "_restore_original_default_devices")
        stub.reset_windows_audio = bind_app_method(stub, "reset_windows_audio")
        stub._current_transcript_text = bind_app_method(stub, "_current_transcript_text")
        stub.copy_transcript_to_clipboard = bind_app_method(stub, "copy_transcript_to_clipboard")
        stub.save_transcript_as = bind_app_method(stub, "save_transcript_as")
        stub.run_mode = bind_app_method(stub, "run_mode")
        stub.stop_live_transcription = bind_app_method(stub, "stop_live_transcription")
        stub._finish_stop_live_transcription = bind_app_method(stub, "_finish_stop_live_transcription")
        stub.resolve_active_device = lambda name: {
            "name": name,
            "index": 1,
            "info": {"name": name, "max_input_channels": 1},
            "sample_rate": 24000,
        }
        stub.verify_active_device = lambda: True
        stub._wait_for_active_input_device = lambda expected: app.DeviceVerificationResult.CONFIRMED
        stub.active_audio_device = {
            "name": "Microphone (Realtek HD Audio Mic input)",
            "index": 0,
            "info": {"name": "Microphone (Realtek HD Audio Mic input)", "max_input_channels": 1},
            "sample_rate": 24000,
        }
        stub.live_transcription_session = SimpleNamespace(
            switch_input_device=Mock(return_value=(True, "switched")),
            _pcm_queue=queue.Queue(),
        )
        stub.live_transcript_final_text = ""
        stub.start_live_transcription = Mock()
        stub.monitor = SimpleNamespace(start=Mock(), stop=Mock())
        stub.wer_enabled_var = DummyVar(True)
        stub.autofix = SimpleNamespace(
            on_signal=Mock(),
            on_queue=Mock(),
            on_watchdog_stalled=Mock(),
            on_bad_routing=Mock(),
        )
        return stub

    def test_refresh_run_control_buttons_includes_mute(self) -> None:
        app_stub = self._make_app_stub()
        app_stub.is_muted = True

        app_stub._refresh_run_control_buttons()

        self.assertEqual(app_stub.mute_button.props["text"], "Muted — Click to Unmute")
        self.assertEqual(app_stub.mute_button.props["border_width"], 2)

    def test_run_mode_queues_live_start_after_apply(self) -> None:
        app_stub = self._make_app_stub()
        app_stub.apply_audio_mode = Mock()

        app.App.run_mode(app_stub, "VAC")

        self.assertEqual(app_stub._pending_live_start_mode, "VAC")
        app_stub.apply_audio_mode.assert_called_once_with("VAC")

    def test_run_mode_preserves_requested_restart_mode_when_live_running(self) -> None:
        app_stub = self._make_app_stub()
        app_stub._live_transcription_running = True
        app_stub.apply_audio_mode = Mock()

        app.App.run_mode(app_stub, "Mixed")

        self.assertEqual(app_stub._pending_live_start_mode, "Mixed")
        app_stub.apply_audio_mode.assert_called_once_with("Mixed")

    def test_mixed_mode_accepts_vac_keyword(self) -> None:
        app_stub = self._make_app_stub()

        self.assertTrue(app_stub._is_mixed_mode_available())
        self.assertTrue(app_stub._resolve_detected_input_name(app_stub.mix_var.get(), "Mixed"))

    def test_mixed_mode_resolver_requires_exact_or_normalized_match(self) -> None:
        app_stub = self._make_app_stub()
        app_stub.detected_input_devices = ["VoiceMeeter Output (VB-Audio VoiceMeeter VAIO)"]
        handler = CaptureHandler()
        app.LOGGER.addHandler(handler)
        try:
            app_stub._resolve_detected_input_name(app_stub.mix_var.get(), "Mixed")
        finally:
            app.LOGGER.removeHandler(handler)

        self.assertFalse(any("match_type=fallback_selected" in message for message in handler.messages))
        self.assertTrue(any("match_type=exact" in message for message in handler.messages))

    def test_reconcile_startup_mode_sets_clear_mixed_unavailable_status(self) -> None:
        app_stub = self._make_app_stub()
        app_stub.current_mode = "Mixed"
        app_stub.mix_var.set("CABLE Output (VB-Audio Virtual Cable)")
        app_stub._reconcile_startup_mode = bind_app_method(app_stub, "_reconcile_startup_mode")
        app_stub._is_mixed_mode_available = Mock(return_value=False)

        with patch.object(app, "save_config"):
            app_stub._reconcile_startup_mode()

        self.assertEqual(app_stub.current_mode, "Microphone")
        self.assertIn("No detected recording inputs are available for Mixed mode.", app_stub.status_var.get())
        self.assertIn("UNAVAILABLE", app_stub.debug_routing_var.get())

    def test_resolve_detected_input_name_uses_cache_for_repeated_exact_match(self) -> None:
        app_stub = self._make_app_stub()
        handler = CaptureHandler()
        app.LOGGER.addHandler(handler)
        try:
            first = app_stub._resolve_detected_input_name("CABLE Output (VB-Audio Virtual Cable)", "VAC")
            second = app_stub._resolve_detected_input_name("CABLE Output (VB-Audio Virtual Cable)", "VAC")
        finally:
            app.LOGGER.removeHandler(handler)

        self.assertEqual(first, "CABLE Output (VB-Audio Virtual Cable)")
        self.assertEqual(second, "CABLE Output (VB-Audio Virtual Cable)")
        self.assertEqual(sum("match_type=exact" in message for message in handler.messages), 1)
        self.assertEqual(sum(message.startswith("[Resolver]") and "candidates_count=" in message for message in handler.messages), 1)

    def test_preflight_emits_all_steps(self) -> None:
        app_stub = self._make_app_stub()
        app_stub.detected_input_devices.append("VoiceMeeter Output (VB-Audio VoiceMeeter VAIO)")
        active_device = {
            "name": "VoiceMeeter Output (VB-Audio VoiceMeeter VAIO)",
            "index": 2,
            "info": {"name": "VoiceMeeter Output (VB-Audio VoiceMeeter VAIO)", "max_input_channels": 1},
            "sample_rate": 24000,
        }
        handler = CaptureHandler()
        app.LOGGER.addHandler(handler)
        try:
            with patch.object(app_stub, "_refresh_detected_devices", return_value=None), patch("app.sd.check_input_settings", return_value=None), patch(
                "app.sample_resolved_input_signal",
                return_value={"state": "active", "rms": 0.01, "peak": 0.1, "color": "#66BB6A", "detail": "ok"},
            ):
                ok, failure_code, _diagnostics = app_stub._run_preflight("Mixed", active_device)
        finally:
            app.LOGGER.removeHandler(handler)

        self.assertTrue(ok)
        self.assertEqual(failure_code, "")
        for step in range(1, 10):
            self.assertTrue(any(f"step={step}_" in message or f"step={step}" in message for message in handler.messages), msg=f"missing step {step}")

    def test_preflight_step6_uses_db_threshold(self) -> None:
        app_stub = self._make_app_stub()
        active_device = {
            "name": "Microphone (Realtek HD Audio Mic input)",
            "index": 1,
            "info": {"name": "Microphone (Realtek HD Audio Mic input)", "max_input_channels": 1},
            "sample_rate": 24000,
        }
        handler = CaptureHandler()
        app.LOGGER.addHandler(handler)
        try:
            with patch.object(app_stub, "_refresh_detected_devices", return_value=None), patch("app.sd.check_input_settings", return_value=None), patch(
                "app.sample_resolved_input_signal",
                return_value={
                    "state": "silent",
                    "rms": 0.0001,
                    "peak": 0.0002,
                    "rms_db": -48.0,
                    "peak_db": -46.0,
                    "color": "#9E9E9E",
                    "detail": "silent",
                    "clipping_hard": False,
                },
            ):
                ok, failure_code, diagnostics = app_stub._run_preflight("Microphone", active_device)
        finally:
            app.LOGGER.removeHandler(handler)

        self.assertFalse(ok)
        self.assertEqual(failure_code, "SIGNAL")
        self.assertEqual(diagnostics["rms_db"], -48.0)
        self.assertTrue(any(message.startswith("[Failure: SIGNAL]") for message in handler.messages))
        self.assertTrue(any("threshold_db=-80.0" in message for message in handler.messages))

    def test_preflight_rejects_synthetic_input(self) -> None:
        app_stub = self._make_app_stub()
        active_device = {
            "name": "Silence",
            "index": -1,
            "info": {"name": "Silence", "is_synthetic": True},
            "sample_rate": 24000,
        }

        ok, failure_code, diagnostics = app_stub._run_preflight("Microphone", active_device)

        self.assertFalse(ok)
        self.assertEqual(failure_code, "DEVICE")
        self.assertEqual(diagnostics["speech_state"], "no_signal")

    def test_preflight_rejects_vac_input_output_loop(self) -> None:
        app_stub = self._make_app_stub()
        app_stub.config["modes"]["vac"]["output_device"] = "CABLE Output (VB-Audio Virtual Cable)"
        app_stub.vac_playback_var.set("CABLE Output (VB-Audio Virtual Cable)")
        active_device = {
            "name": "CABLE Output (VB-Audio Virtual Cable)",
            "index": 1,
            "info": {"name": "CABLE Output (VB-Audio Virtual Cable)", "max_input_channels": 1},
            "sample_rate": 24000,
        }

        with patch.object(app_stub, "_refresh_detected_devices", return_value=None), patch("app.sd.check_input_settings", return_value=None):
            ok, failure_code, diagnostics = app_stub._run_preflight("VAC", active_device)

        self.assertFalse(ok)
        self.assertEqual(failure_code, "ROUTING")
        self.assertIn("same endpoint", diagnostics["feedback"])

    def test_apply_audio_mode_hot_switch_invokes_session_swap(self) -> None:
        app_stub = self._make_app_stub()
        app_stub._live_transcription_running = True
        app_stub.live_transcription_session = SimpleNamespace(switch_input_device=Mock(return_value=(True, "switched")))

        with patch.object(app, "save_config", return_value=None), patch.object(
            app_stub, "_run_preflight", return_value=(True, "", {})
        ):
            app.App._apply_audio_mode_hot_worker(app_stub, "VAC", "CABLE Output (VB-Audio Virtual Cable)", "CABLE Input (VB-Audio Virtual Cable)")

        app_stub.live_transcription_session.switch_input_device.assert_called_once()
        args = app_stub.live_transcription_session.switch_input_device.call_args[0]
        self.assertEqual(args[0]["name"], "CABLE Output (VB-Audio Virtual Cable)")
        self.assertEqual(args[1], "VAC")

    def test_apply_audio_mode_stops_live_session_before_switching_modes(self) -> None:
        app_stub = self._make_app_stub()
        app_stub._live_transcription_running = True
        app_stub.stop_live_transcription = Mock()

        app.App.apply_audio_mode(app_stub, "VAC")

        self.assertEqual(app_stub._pending_mode_after_live_stop, "VAC")
        self.assertEqual(app_stub._pending_live_start_mode, "VAC")
        app_stub.stop_live_transcription.assert_called_once()

    def test_finish_callback_logs_entry_and_completion(self) -> None:
        app_stub = self._make_app_stub()
        handler = CaptureHandler()
        app.LOGGER.addHandler(handler)
        try:
            with patch.object(app, "save_config", return_value=None):
                app_stub._finish_apply_audio_mode(
                    app.ModeSwitchOutcome.SUCCESS,
                    "VAC",
                    {
                        "name": "CABLE Output (VB-Audio Virtual Cable)",
                        "index": 1,
                        "info": {"name": "CABLE Output (VB-Audio Virtual Cable)", "max_input_channels": 1},
                        "sample_rate": 24000,
                    },
                    "CABLE Input (VB-Audio Virtual Cable)",
                    "ok",
                )
        finally:
            app.LOGGER.removeHandler(handler)

        self.assertTrue(any("[ApplyMode] event=finish_enter mode=VAC" in message for message in handler.messages))
        self.assertTrue(any("[ApplyMode] event=finish_complete mode=VAC" in message for message in handler.messages))

    def test_finish_callback_logs_crash_on_exception(self) -> None:
        app_stub = self._make_app_stub()
        app_stub.active_device_label = Mock()
        app_stub.active_device_label.configure.side_effect = RuntimeError("boom")
        app_stub.active_device_label.winfo_exists.return_value = True
        handler = CaptureHandler()
        app.LOGGER.addHandler(handler)
        try:
            with patch.object(app, "save_config", return_value=None):
                app_stub._finish_apply_audio_mode(
                    app.ModeSwitchOutcome.SUCCESS,
                    "VAC",
                    {
                        "name": "CABLE Output (VB-Audio Virtual Cable)",
                        "index": 1,
                        "info": {"name": "CABLE Output (VB-Audio Virtual Cable)", "max_input_channels": 1},
                        "sample_rate": 24000,
                    },
                    "CABLE Input (VB-Audio Virtual Cable)",
                    "ok",
                )
        finally:
            app.LOGGER.removeHandler(handler)

        self.assertTrue(any("[ApplyMode] event=finish_crashed mode=VAC" in message for message in handler.messages))

    def test_wait_for_active_input_device_confirmed(self) -> None:
        app_stub = self._make_app_stub()

        class FakeClock:
            def __init__(self, step: float = 0.1):
                self.current = -step
                self.step = step

            def __call__(self) -> float:
                self.current += self.step
                return self.current

        with patch("app.time.time", side_effect=FakeClock()), patch("app.time.sleep", return_value=None), patch(
            "app.get_default_input_device",
            return_value=(1, {"name": "CABLE Output (VB-Audio Virtual Cable)"}),
        ):
            result = app.App._wait_for_active_input_device(app_stub, "CABLE Output (VB-Audio Virtual Cable)")

        self.assertEqual(result, app.DeviceVerificationResult.CONFIRMED)

    def test_wait_for_active_input_device_eventually_confirmed(self) -> None:
        app_stub = self._make_app_stub()

        class FakeClock:
            def __init__(self, step: float = 0.3):
                self.current = -step
                self.step = step

            def __call__(self) -> float:
                self.current += self.step
                return self.current

        responses = [(1, {"name": "Microphone (Realtek HD Audio Mic input)"})] * 6 + [
            (1, {"name": "CABLE Output (VB-Audio Virtual Cable)"})
        ]
        with patch("app.time.time", side_effect=FakeClock()), patch("app.time.sleep", return_value=None), patch(
            "app.get_default_input_device",
            side_effect=responses,
        ):
            result = app.App._wait_for_active_input_device(app_stub, "CABLE Output (VB-Audio Virtual Cable)")

        self.assertEqual(result, app.DeviceVerificationResult.EVENTUALLY_CONFIRMED)

    def test_wait_for_active_input_device_timed_out_but_exists(self) -> None:
        app_stub = self._make_app_stub()

        class FakeClock:
            def __init__(self, step: float = 0.5):
                self.current = -step
                self.step = step

            def __call__(self) -> float:
                self.current += self.step
                return self.current

        with patch("app.time.time", side_effect=FakeClock()), patch("app.time.sleep", return_value=None), patch(
            "app.get_default_input_device",
            return_value=(1, {"name": "Microphone (Realtek HD Audio Mic input)"}),
        ), patch(
            "app.sd.query_devices",
            return_value=[{"name": "CABLE Output (VB-Audio Virtual Cable)"}],
        ):
            result = app.App._wait_for_active_input_device(app_stub, "CABLE Output (VB-Audio Virtual Cable)")

        self.assertEqual(result, app.DeviceVerificationResult.TIMED_OUT)

    def test_wait_for_active_input_device_unavailable(self) -> None:
        app_stub = self._make_app_stub()

        class FakeClock:
            def __init__(self, step: float = 0.5):
                self.current = -step
                self.step = step

            def __call__(self) -> float:
                self.current += self.step
                return self.current

        with patch("app.time.time", side_effect=FakeClock()), patch("app.time.sleep", return_value=None), patch(
            "app.get_default_input_device",
            return_value=(1, {"name": "Microphone (Realtek HD Audio Mic input)"}),
        ), patch(
            "app.sd.query_devices",
            return_value=[{"name": "Something Else"}],
        ):
            result = app.App._wait_for_active_input_device(app_stub, "CABLE Output (VB-Audio Virtual Cable)")

        self.assertEqual(result, app.DeviceVerificationResult.DEVICE_UNAVAILABLE)

    def test_finish_apply_audio_mode_repaints_on_success_with_warning(self) -> None:
        app_stub = self._make_app_stub()
        handler = CaptureHandler()
        app.LOGGER.addHandler(handler)
        try:
            with patch.object(app, "save_config", return_value=None):
                app_stub._finish_apply_audio_mode(
                    app.ModeSwitchOutcome.SUCCESS_WITH_WARNING,
                    "VAC",
                    {
                        "name": "CABLE Output (VB-Audio Virtual Cable)",
                        "index": 1,
                        "info": {"name": "CABLE Output (VB-Audio Virtual Cable)", "max_input_channels": 1},
                        "sample_rate": 24000,
                    },
                    "CABLE Input (VB-Audio Virtual Cable)",
                    "verification lagged",
                )
        finally:
            app.LOGGER.removeHandler(handler)

        self.assertEqual(app_stub.current_mode, "VAC")
        self.assertEqual(app_stub.mode_badge_label.props["text"], "VAC")
        self.assertEqual(app_stub.active_source_label.props["text_color"], "#66BB6A")
        self.assertEqual(app_stub.status_label.props["text_color"], "#F9A825")

    def test_finish_apply_audio_mode_starts_live_when_run_mode_pending(self) -> None:
        app_stub = self._make_app_stub()
        app_stub._pending_live_start_mode = "VAC"

        with patch.object(app, "save_config", return_value=None):
            app_stub._finish_apply_audio_mode(
                app.ModeSwitchOutcome.SUCCESS,
                "VAC",
                {
                    "name": "CABLE Output (VB-Audio Virtual Cable)",
                    "index": 1,
                    "info": {"name": "CABLE Output (VB-Audio Virtual Cable)", "max_input_channels": 1},
                    "sample_rate": 24000,
                },
                "CABLE Input (VB-Audio Virtual Cable)",
                "ok",
            )

        self.assertIsNone(app_stub._pending_live_start_mode)
        app_stub.start_live_transcription.assert_called_once()

    def test_finish_apply_audio_mode_clears_pending_live_start_on_failure(self) -> None:
        app_stub = self._make_app_stub()
        app_stub._pending_live_start_mode = "VAC"

        app_stub._finish_apply_audio_mode(
            app.ModeSwitchOutcome.HARD_FAILURE,
            "VAC",
            None,
            "CABLE Input (VB-Audio Virtual Cable)",
            "failed",
        )

        self.assertIsNone(app_stub._pending_live_start_mode)
        app_stub.start_live_transcription.assert_not_called()

    def test_stop_live_transcription_dispatches_background_worker(self) -> None:
        app_stub = self._make_app_stub()
        session = SimpleNamespace(stop=Mock(return_value=(True, "stopped ok")))
        app_stub.live_transcription_session = session
        app_stub._live_transcription_running = True

        started_targets: list[tuple[object, tuple[object, ...]]] = []

        class FakeThread:
            def __init__(self, *, target=None, args=(), daemon=None, name=None):
                self.target = target
                self.args = args

            def start(self):
                started_targets.append((self.target, self.args))

        with patch("app.threading.Thread", FakeThread):
            app.App.stop_live_transcription(app_stub)

        self.assertTrue(app_stub._live_transcription_stopping)
        self.assertEqual(app_stub.status_var.get(), "Stopping live transcription...")
        self.assertEqual(len(started_targets), 1)
        self.assertEqual(started_targets[0][1], (session,))
        session.stop.assert_not_called()

    def test_finish_stop_live_transcription_resets_ui_and_restarts_monitor(self) -> None:
        app_stub = self._make_app_stub()
        session = SimpleNamespace(stop=Mock(return_value=(True, "stopped ok")))
        app_stub.live_transcription_session = session
        app_stub._resume_monitor_after_live = True
        app_stub._set_live_controls_state(running=True, stopping=True)

        app.App._finish_stop_live_transcription(app_stub, session, True, "stopped ok")

        self.assertIsNone(app_stub.live_transcription_session)
        self.assertFalse(app_stub._live_transcription_running)
        self.assertFalse(app_stub._live_transcription_stopping)
        self.assertEqual(app_stub.status_var.get(), "stopped ok")
        app_stub.monitor.start.assert_called_once()

    def test_finish_stop_live_transcription_applies_pending_mode(self) -> None:
        app_stub = self._make_app_stub()
        session = SimpleNamespace(stop=Mock(return_value=(True, "stopped ok")))
        app_stub.live_transcription_session = session
        app_stub._pending_mode_after_live_stop = "VAC"
        app_stub.apply_audio_mode = Mock()

        app.App._finish_stop_live_transcription(app_stub, session, True, "stopped ok")

        self.assertIsNone(app_stub._pending_mode_after_live_stop)
        app_stub.apply_audio_mode.assert_called_once_with("VAC")

    def test_start_live_transcription_worker_fails_fast_on_preflight_error(self) -> None:
        app_stub = self._make_app_stub()
        app_stub._finish_start_live_transcription = Mock()
        app_stub.verify_active_device = Mock(return_value=True)
        app_stub._run_preflight = Mock(return_value=(False, "SIGNAL", {"feedback": "VAC silent", "speech_state": "no_signal"}))

        with patch("app.LiveTranscriptionSession") as session_cls:
            app.App._start_live_transcription_worker(
                app_stub,
                "test-key",
                {
                    "name": "CABLE Output (VB-Audio Virtual Cable)",
                    "index": 1,
                    "info": {"name": "CABLE Output (VB-Audio Virtual Cable)", "max_input_channels": 1},
                    "sample_rate": 24000,
                },
                "VAC",
            )

        session_cls.assert_not_called()
        finish_args = app_stub._finish_start_live_transcription.call_args.args
        self.assertFalse(finish_args[0])
        self.assertEqual(finish_args[1], "VAC silent")

    def test_finish_apply_audio_mode_sets_status_color_on_success(self) -> None:
        app_stub = self._make_app_stub()

        with patch.object(app, "save_config", return_value=None):
            app_stub._finish_apply_audio_mode(
                app.ModeSwitchOutcome.SUCCESS,
                "VAC",
                {
                    "name": "CABLE Output (VB-Audio Virtual Cable)",
                    "index": 1,
                    "info": {"name": "CABLE Output (VB-Audio Virtual Cable)", "max_input_channels": 1},
                    "sample_rate": 24000,
                },
                "CABLE Input (VB-Audio Virtual Cable)",
                "ok",
            )

        self.assertEqual(app_stub.status_label.props["text_color"], "#66BB6A")

    def test_finish_apply_audio_mode_does_not_repaint_on_hard_failure(self) -> None:
        app_stub = self._make_app_stub()

        with patch("app.messagebox.showerror", return_value=None), patch.object(app, "save_config", return_value=None):
            app_stub._finish_apply_audio_mode(
                app.ModeSwitchOutcome.HARD_FAILURE,
                "VAC",
                {
                    "name": "CABLE Output (VB-Audio Virtual Cable)",
                    "index": 1,
                    "info": {"name": "CABLE Output (VB-Audio Virtual Cable)", "max_input_channels": 1},
                    "sample_rate": 24000,
                },
                "CABLE Input (VB-Audio Virtual Cable)",
                "hard fail",
            )

        self.assertEqual(app_stub.current_mode, "Microphone")

    def test_apply_mode_theme_vac_sets_green(self) -> None:
        app_stub = self._make_app_stub()

        app_stub._apply_mode_theme("VAC")

        self.assertEqual(app_stub.mode_badge_label.props["text"], "VAC")
        self.assertEqual(app_stub.mode_badge_label.props["fg_color"], "#2E7D32")
        self.assertEqual(app_stub.active_source_label.props["text_color"], "#66BB6A")

    def test_apply_mode_theme_microphone_sets_blue(self) -> None:
        app_stub = self._make_app_stub()

        app_stub._apply_mode_theme("Microphone")

        self.assertEqual(app_stub.mode_badge_label.props["text"], "Microphone")
        self.assertEqual(app_stub.mode_badge_label.props["fg_color"], "#1565C0")

    def test_apply_mode_theme_source_label_uses_brighter_color(self) -> None:
        app_stub = self._make_app_stub()

        app_stub._apply_mode_theme("Microphone")

        self.assertEqual(app_stub.active_source_label.props["text_color"], "#42A5F5")

    def test_apply_mode_theme_unknown_mode_fallback(self) -> None:
        app_stub = self._make_app_stub()
        handler = CaptureHandler()
        app.LOGGER.addHandler(handler)
        try:
            app_stub._apply_mode_theme("Unknown")
        finally:
            app.LOGGER.removeHandler(handler)

        self.assertEqual(app_stub.mode_badge_label.props["fg_color"], "#616161")
        self.assertEqual(app_stub.active_source_label.props["text_color"], "#9E9E9E")
        self.assertTrue(any("[ApplyMode] event=unknown_mode_in_ui_map mode=Unknown" in message for message in handler.messages))

    def test_apply_mode_theme_missing_badge_logs_and_continues(self) -> None:
        app_stub = self._make_app_stub()
        app_stub.mode_badge_label = None
        handler = CaptureHandler()
        previous_level = app.LOGGER.level
        app.LOGGER.setLevel(logging.DEBUG)
        app.LOGGER.addHandler(handler)
        try:
            app_stub._apply_mode_theme("VAC")
        finally:
            app.LOGGER.removeHandler(handler)
            app.LOGGER.setLevel(previous_level)

        self.assertTrue(any("[ApplyMode] event=badge_widget_missing name=mode_badge_label" in message for message in handler.messages))
        self.assertTrue(any("[ApplyMode] event=theme_applied mode=VAC accent=#2E7D32 bright=#66BB6A" in message for message in handler.messages))

    def test_mixed_with_generic_input_is_unavailable_and_logs_routing_error(self) -> None:
        app_stub = self._make_app_stub()
        app_stub.detected_input_devices = ["Microphone (Realtek HD Audio Mic input)"]
        app_stub.mix_var.set("VoiceMeeter Output (VB-Audio VoiceMeeter VAIO)")
        handler = CaptureHandler()
        previous_level = app.LOGGER.level
        app.LOGGER.setLevel(logging.DEBUG)
        app.LOGGER.addHandler(handler)
        try:
            with patch.object(app_stub, "_refresh_detected_devices", return_value=None):
                self.assertFalse(app_stub._is_mixed_mode_available(log_failure_level="debug"))
                self.assertFalse(app_stub._is_mixed_mode_available(log_failure_level="debug"))
                self.assertFalse(app_stub._is_mixed_mode_available(log_failure_level="debug"))
        finally:
            app.LOGGER.removeHandler(handler)
            app.LOGGER.setLevel(previous_level)

        error_messages = [message for message in handler.messages if message.startswith("[Failure: ROUTING]")]
        self.assertGreaterEqual(len(error_messages), 1)

    def test_mixed_user_click_surfaces_missing_mixed_route(self) -> None:
        app_stub = self._make_app_stub()
        app_stub.detected_input_devices = ["Microphone (Realtek HD Audio Mic input)"]
        app_stub.mix_var.set("VoiceMeeter Output (VB-Audio VoiceMeeter VAIO)")
        handler = CaptureHandler()
        app.LOGGER.addHandler(handler)
        try:
            with patch.object(app_stub, "_refresh_detected_devices", return_value=None):
                for _ in range(3):
                    with patch("app.messagebox.showerror", return_value=None):
                        app.App.apply_audio_mode(app_stub, "Mixed")
        finally:
            app.LOGGER.removeHandler(handler)

        error_messages = [message for message in handler.messages if message.startswith("[Failure: ROUTING]")]
        self.assertGreaterEqual(len(error_messages), 1)

    def test_mixed_became_available_logs_transition(self) -> None:
        app_stub = self._make_app_stub()
        app_stub.detected_input_devices = []
        app_stub.mix_var.set("VoiceMeeter Output (VB-Audio VoiceMeeter VAIO)")
        with patch.object(app_stub, "_refresh_detected_devices", return_value=None):
            app_stub._is_mixed_mode_available(log_failure_level="debug")
        app_stub.detected_input_devices.append("VoiceMeeter Output (VB-Audio VoiceMeeter VAIO)")
        handler = CaptureHandler()
        app.LOGGER.addHandler(handler)
        try:
            with patch.object(app_stub, "_refresh_detected_devices", return_value=None):
                self.assertTrue(app_stub._is_mixed_mode_available(log_failure_level="debug"))
        finally:
            app.LOGGER.removeHandler(handler)

        self.assertTrue(any("[Mixed] event=became_available" in message for message in handler.messages))

    def test_verification_extended_window_is_5000ms(self) -> None:
        self.assertEqual(app.DEVICE_VERIFY_EXTENDED_TIMEOUT_SECONDS, 5.0)

    def test_consistently_slow_advisory_fires_once(self) -> None:
        app_stub = self._make_app_stub()
        handler = CaptureHandler()
        app.LOGGER.addHandler(handler)
        try:
            class FakeClock:
                def __init__(self, step: float = 0.3):
                    self.current = -step
                    self.step = step

                def __call__(self) -> float:
                    self.current += self.step
                    return self.current

            responses = [(1, {"name": "Microphone (Realtek HD Audio Mic input)"})] * 6 + [
                (1, {"name": "CABLE Output (VB-Audio Virtual Cable)"})
            ]
            for _ in range(3):
                with patch("app.time.time", side_effect=FakeClock()), patch("app.time.sleep", return_value=None), patch(
                    "app.get_default_input_device",
                    side_effect=list(responses),
                ):
                    result = app.App._wait_for_active_input_device(app_stub, "CABLE Output (VB-Audio Virtual Cable)")
                    self.assertEqual(result, app.DeviceVerificationResult.EVENTUALLY_CONFIRMED)
        finally:
            app.LOGGER.removeHandler(handler)

        advisory_messages = [message for message in handler.messages if "[DeviceVerify] event=consistently_slow" in message]
        self.assertEqual(len(advisory_messages), 1)

    def test_vac_silent_signal_message_includes_zoom_note(self) -> None:
        app_stub = self._make_app_stub()
        message = app.App._silent_signal_message(app_stub, "VAC", "CABLE Output (VB-Audio Virtual Cable)")

        self.assertIn("Same as System", message)

    def test_reconcile_startup_mode_falls_back_from_mixed_when_vm_absent(self) -> None:
        app_stub = self._make_app_stub()
        app_stub.current_mode = "Mixed"
        app_stub.config["last_mode"] = "Mixed"
        app_stub.mode_var.set("Mixed")
        app_stub._is_mixed_mode_available = lambda: False

        with patch.object(app, "save_config", return_value=None):
            app_stub._reconcile_startup_mode()

        self.assertEqual(app_stub.current_mode, "Microphone")
        self.assertEqual(app_stub.config["last_mode"], "Microphone")

    def test_voicemeeter_field_warns_on_non_voicemeeter_device(self) -> None:
        app_stub = self._make_app_stub()
        app_stub.refresh_detected_devices = lambda: None
        app_stub.tabview = SimpleNamespace(set=lambda _name: None)
        app_stub.mix_var.set("Input (VB-Audio Point)")

        with patch("app.messagebox.askyesno", return_value=True) as prompt, patch.object(app_stub, "save_form_config", return_value=None):
            app.App.save_settings(app_stub)

        prompt.assert_not_called()

    def test_use_current_windows_output_for_speakers_updates_setting(self) -> None:
        app_stub = self._make_app_stub()

        with patch(
            "app.get_default_output_device",
            return_value=(3, {"name": "Samsung TV (NVIDIA High Definition Audio)"}),
        ), patch.object(app, "save_config", return_value=None):
            app.App.use_current_windows_output_for_speakers(app_stub)

        self.assertEqual(app_stub.speaker_var.get(), "Samsung TV (NVIDIA High Definition Audio)")
        self.assertEqual(app_stub.config["speaker_device"], "Samsung TV (NVIDIA High Definition Audio)")
        self.assertIn("Samsung TV", app_stub.status_var.get())

    def test_resolve_mode_devices_uses_speaker_fallback_then_mixed_override(self) -> None:
        app_stub = self._make_app_stub()

        recording_device, playback_device = app.App._resolve_mode_devices(app_stub, "Mixed")
        self.assertEqual(recording_device, "VoiceMeeter Output (VB-Audio VoiceMeeter VAIO)")
        self.assertEqual(playback_device, "Speakers (Realtek Audio)")

        app_stub.mixed_playback_var.set("Samsung TV (NVIDIA High Definition Audio)")
        recording_device, playback_device = app.App._resolve_mode_devices(app_stub, "Mixed")
        self.assertEqual(recording_device, "VoiceMeeter Output (VB-Audio VoiceMeeter VAIO)")
        self.assertEqual(playback_device, "Samsung TV (NVIDIA High Definition Audio)")

    def test_restore_original_default_devices_uses_captured_names_when_enabled(self) -> None:
        app_stub = self._make_app_stub()
        app_stub.restore_devices_on_exit_var.set(True)
        app_stub.device_manager = SimpleNamespace(
            set_default_recording_device=Mock(return_value=(True, "record restored")),
            set_default_playback_device=Mock(return_value=(True, "play restored")),
        )
        app_stub.config["restore_devices_on_exit"] = True

        app.App._restore_original_default_devices(app_stub)

        app_stub.device_manager.set_default_recording_device.assert_called_once_with("Microphone (Realtek HD Audio Mic input)")
        app_stub.device_manager.set_default_playback_device.assert_called_once_with("Speakers (Realtek Audio)")

    def test_current_transcript_text_prefers_live_text(self) -> None:
        app_stub = self._make_app_stub()
        app_stub.live_transcript_final_text = "Live transcript text"
        app_stub.live_transcription_session = SimpleNamespace(transcript_path=Path("live_transcript_123.txt"))

        text, source = app.App._current_transcript_text(app_stub)

        self.assertEqual(text, "Live transcript text")
        self.assertEqual(source, "live session (live_transcript_123.txt)")

    def test_current_transcript_text_falls_back_to_newest_file(self) -> None:
        app_stub = self._make_app_stub()
        app_stub.live_transcript_final_text = ""
        app_stub.live_transcription_session = None

        with tempfile.TemporaryDirectory() as tmpdir:
            transcript_dir = Path(tmpdir)
            older = transcript_dir / "older.txt"
            newer = transcript_dir / "newer.txt"
            older.write_text("older text", encoding="utf-8")
            newer.write_text("newer text", encoding="utf-8")
            os.utime(older, (1, 1))
            os.utime(newer, (2, 2))

            with patch.object(app, "TRANSCRIPTS_DIR", transcript_dir):
                text, source = app.App._current_transcript_text(app_stub)

        self.assertEqual(text, "newer text")
        self.assertEqual(source, "most recent file (newer.txt)")

    def test_current_transcript_text_returns_empty_when_nothing_available(self) -> None:
        app_stub = self._make_app_stub()
        app_stub.live_transcript_final_text = ""
        app_stub.live_transcription_session = None

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(app, "TRANSCRIPTS_DIR", Path(tmpdir)):
            text, source = app.App._current_transcript_text(app_stub)

        self.assertEqual((text, source), ("", ""))

    def test_reset_windows_audio_falls_back_to_configured_devices_when_originals_missing(self) -> None:
        app_stub = self._make_app_stub()
        app_stub._original_default_input_name = None
        app_stub._original_default_output_name = None
        app_stub.device_manager = SimpleNamespace(
            set_default_recording_device=Mock(return_value=(True, "record restored")),
            set_default_playback_device=Mock(return_value=(True, "play restored")),
        )

        with patch("app.messagebox.showinfo", return_value=None):
            app.App.reset_windows_audio(app_stub)

        app_stub.device_manager.set_default_recording_device.assert_called_once_with("Microphone (Realtek HD Audio Mic input)")
        app_stub.device_manager.set_default_playback_device.assert_called_once_with("Speakers (Realtek Audio)")
        self.assertEqual(app_stub.status_var.get(), "Windows audio defaults reset.")

    def test_reset_windows_audio_warns_and_skips_when_live_transcription_running(self) -> None:
        app_stub = self._make_app_stub()
        app_stub._live_transcription_running = True
        app_stub.device_manager = SimpleNamespace(
            set_default_recording_device=Mock(return_value=(True, "record restored")),
            set_default_playback_device=Mock(return_value=(True, "play restored")),
        )

        with patch("app.messagebox.showwarning", return_value=None) as warning, patch(
            "app.messagebox.showinfo",
            return_value=None,
        ):
            app.App.reset_windows_audio(app_stub)

        warning.assert_called_once()
        app_stub.device_manager.set_default_recording_device.assert_not_called()
        app_stub.device_manager.set_default_playback_device.assert_not_called()

    def test_device_refresh_logs_changes(self) -> None:
        app_stub = self._make_app_stub()
        handler = CaptureHandler()
        previous_level = app.LOGGER.level
        app.LOGGER.setLevel(logging.DEBUG)
        app.LOGGER.addHandler(handler)
        try:
            with patch("app.list_input_devices", return_value=["Microphone (Realtek HD Audio Mic input)", "New Device"]), patch(
                "app.list_output_devices",
                return_value=["Speakers (Realtek Audio)"],
            ):
                app_stub._refresh_detected_devices(force=True)
        finally:
            app.LOGGER.removeHandler(handler)
            app.LOGGER.setLevel(previous_level)

        self.assertTrue(any("[Devices] event=inputs_changed" in message for message in handler.messages))

    def test_hydrate_config_self_heals_stale_fake_device_names(self) -> None:
        app_stub = self._make_app_stub()
        app_stub.config.update(
            {
                "mic_device": "VB-Audio Point",
                "speaker_device": "Fake Speakers",
                "vac_device": "Old Cable",
                "vac_playback_device": "Fake Playback",
                "mixed_playback_device": "Ghost TV",
                "voicemeeter_device": "VB-Audio Matrix",
            }
        )
        app_stub.detected_input_devices = [
            "Microphone (Realtek HD Audio Mic input)",
            "CABLE Output (VB-Audio Virtual Cable)",
        ]
        app_stub.detected_output_devices = [
            "CABLE Input (VB-Audio Virtual Cable)",
            "SAMSUNG TV (NVIDIA High Definition Audio)",
        ]

        with patch.object(app, "save_config", return_value=None), patch("app.get_default_input_device", return_value=(None, None)):
            app.App._hydrate_config_from_detected_devices(app_stub)

        self.assertEqual(app_stub.config["mic_device"], "Microphone (Realtek HD Audio Mic input)")
        self.assertEqual(app_stub.config["vac_device"], "CABLE Output (VB-Audio Virtual Cable)")
        self.assertEqual(app_stub.config["speaker_device"], "SAMSUNG TV (NVIDIA High Definition Audio)")
        self.assertEqual(app_stub.config["vac_playback_device"], "CABLE Input (VB-Audio Virtual Cable)")
        self.assertEqual(app_stub.config["mixed_playback_device"], "")
        self.assertEqual(app_stub.config["voicemeeter_device"], "CABLE Output (VB-Audio Virtual Cable)")

    def test_hydrate_config_prefers_windows_default_usb_mic_over_generic_realtek_input(self) -> None:
        app_stub = self._make_app_stub()
        app_stub.config["mic_device"] = "Microphone (Realtek HD Audio Mic input)"
        app_stub.detected_input_devices = [
            "Microsoft Sound Mapper - Input",
            "Microphone (Razer Seiren V3 Mini)",
            "Microphone (Realtek HD Audio Mic input)",
        ]
        app_stub.detected_output_devices = ["Speakers (Realtek Audio)"]

        with patch.object(app, "save_config", return_value=None), patch(
            "app.get_default_input_device",
            return_value=(1, {"name": "Microphone (Razer Seiren V3 Mini)"}),
        ):
            app.App._hydrate_config_from_detected_devices(app_stub)

        self.assertEqual(app_stub.config["mic_device"], "Microphone (Razer Seiren V3 Mini)")

    def test_no_silent_excepts(self) -> None:
        source = Path(app.APP_DIR / "app.py").read_text(encoding="utf-8")
        self.assertEqual(len(re.findall(r"except Exception:\s*(?:pass|return)", source)), 0)

    def test_log_format_tagged(self) -> None:
        handler = CaptureHandler()
        app.LOGGER.addHandler(handler)
        try:
            app.log_event("App", event="test")
            app.log_failure("DEVICE", mode="VAC", device="CABLE Output", reason="test")
        finally:
            app.LOGGER.removeHandler(handler)

        self.assertTrue(handler.messages)
        self.assertTrue(all(message.startswith("[") for message in handler.messages))


class LiveSessionSwitchTests(unittest.TestCase):
    def test_switch_input_device_restores_old_stream_on_failure(self) -> None:
        transcript_updates: list[tuple[str, str]] = []
        status_updates: list[str] = []
        signal_updates: list[dict[str, object]] = []
        session = app.LiveTranscriptionSession(
            api_key="test-key",
            input_device={
                "name": "Microphone (Realtek HD Audio Mic input)",
                "index": 1,
                "info": {"name": "Microphone (Realtek HD Audio Mic input)", "max_input_channels": 1},
                "sample_rate": 24000,
            },
            mode_name="Microphone",
            on_transcript=transcript_updates.append,
            on_status=status_updates.append,
            on_signal=signal_updates.append,
        )

        class DummyStream:
            def __init__(self, device):
                self.device = device

            def start(self):
                return None

            def stop(self):
                return None

            def close(self):
                return None

        session.connection = object()
        session.running = True
        session.stream = DummyStream(1)
        old_stream = session.stream

        def stream_factory(**kwargs):
            if kwargs["device"] == 2:
                raise RuntimeError("new device failed")
            return DummyStream(kwargs["device"])

        session._input_stream_factory = stream_factory

        with patch("app.sd.check_input_settings", return_value=None):
            ok, message = session.switch_input_device(
                {
                    "name": "CABLE Output (VB-Audio Virtual Cable)",
                    "index": 2,
                    "info": {"name": "CABLE Output (VB-Audio Virtual Cable)", "max_input_channels": 1},
                    "sample_rate": 24000,
                },
                "VAC",
            )

        self.assertFalse(ok)
        self.assertIn("restored", message.lower())
        self.assertTrue(session.running)
        self.assertEqual(session.input_device_index, 1)
        self.assertNotEqual(session.stream, old_stream)


class LiveSessionResilienceTests(unittest.TestCase):
    def _make_session(self) -> app.LiveTranscriptionSession:
        return app.LiveTranscriptionSession(
            api_key="test-key",
            input_device={
                "name": "CABLE Output (VB-Audio Virtual Cable)",
                "index": 1,
                "info": {"name": "CABLE Output (VB-Audio Virtual Cable)", "max_input_channels": 1},
                "sample_rate": 24000,
            },
            mode_name="VAC",
            on_transcript=lambda *_args: None,
            on_status=lambda *_args: None,
            on_signal=lambda *_args: None,
        )

    def test_audio_callback_drops_oldest_block_when_pcm_queue_is_full(self) -> None:
        session = self._make_session()
        session.running = True
        session._pcm_queue = queue.Queue(maxsize=2)

        with patch.object(session, "_report_input_signal", return_value=None), patch.object(
            session,
            "_pcm16_bytes_from_input",
            side_effect=[b"oldest", b"middle", b"newest"],
        ):
            session._audio_callback(None, 1024, None, None)
            session._audio_callback(None, 1024, None, None)
            session._audio_callback(None, 1024, None, None)

        self.assertEqual(session._dropped_blocks, 1)
        self.assertEqual(session._pcm_queue.get_nowait(), b"middle")
        self.assertEqual(session._pcm_queue.get_nowait(), b"newest")

    def test_on_close_triggers_reconnect_while_running(self) -> None:
        session = self._make_session()
        session.running = True
        session._trigger_reconnect = Mock()

        session._on_close(client=Mock())

        session._trigger_reconnect.assert_called_once_with("websocket closed")

    def test_keepalive_loop_sends_keepalive_during_idle_periods(self) -> None:
        session = self._make_session()
        session.running = True
        session._state = app.LiveSessionState.RUNNING
        session._last_send_at = 0.0

        connection = Mock()
        connection.keep_alive = None

        def send(payload) -> None:
            self.assertIn("KeepAlive", payload)
            session._stop_event.set()

        connection.send.side_effect = send
        session.connection = connection

        with patch.object(app, "DEEPGRAM_KEEPALIVE_SECONDS", 0.01), patch.object(
            app,
            "DEEPGRAM_KEEPALIVE_IDLE_THRESHOLD",
            0.0,
        ), patch("app.time.monotonic", return_value=10.0):
            thread = threading.Thread(target=session._keepalive_loop, daemon=True)
            thread.start()
            thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        connection.send.assert_called()

    def test_reconnect_consumes_backoff_sequence_and_caps_at_attempt_six(self) -> None:
        session = self._make_session()
        session.running = True
        session.current_interim = ""
        session.final_lines = []
        status_updates: list[str] = []
        session.on_status = status_updates.append
        session.on_transcript = Mock()
        session._persist_transcript_text = Mock()
        session._close_connection = Mock()

        waits: list[float] = []

        def fake_wait(delay: float) -> bool:
            waits.append(delay)
            return False

        session._stop_event = Mock()
        session._stop_event.wait.side_effect = fake_wait

        connect_results = [(False, "nope")] * 5 + [(True, "")]
        session._connect_websocket = Mock(side_effect=connect_results)

        with patch("app.time.strftime", return_value="12:34:56"):
            session._reconnect()

        self.assertEqual(waits, list(app.RECONNECT_BACKOFF_SECONDS))
        self.assertEqual(session._connect_websocket.call_count, 6)
        self.assertEqual(session._reconnect_attempt, 0)
        self.assertIn("[Reconnected to Deepgram at 12:34:56]", session.final_lines)
        self.assertTrue(any("attempt 6 of 6" in message for message in status_updates))
        self.assertIn("Live transcription reconnected.", status_updates)


if __name__ == "__main__":
    unittest.main()
