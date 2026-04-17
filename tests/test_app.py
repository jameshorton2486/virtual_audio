from __future__ import annotations

import subprocess
import tempfile
import unittest
import logging
import re
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pathlib import Path

import app
import numpy as np


class DummyVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value) -> None:
        self.value = value


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

    def after(self, _delay, callback) -> None:
        callback()

    def winfo_exists(self) -> bool:
        return self.exists


class CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def bind_app_method(obj, method_name: str):
    return getattr(app.App, method_name).__get__(obj, app.App)


class DeviceHelpersTests(unittest.TestCase):
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
                [str(app.NIRCMD_PATH), "setdefaultsounddevice", "CABLE Output (VB-Audio Virtual Cable)", "0"],
                [str(app.NIRCMD_PATH), "setdefaultsounddevice", "CABLE Output (VB-Audio Virtual Cable)", "1"],
                [str(app.NIRCMD_PATH), "setdefaultsounddevice", "CABLE Output (VB-Audio Virtual Cable)", "2"],
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
                [str(app.NIRCMD_PATH), "setdefaultsounddevice", "CABLE Input (VB-Audio Virtual Cable)", "0"],
                [str(app.NIRCMD_PATH), "setdefaultsounddevice", "CABLE Input (VB-Audio Virtual Cable)", "1"],
                [str(app.NIRCMD_PATH), "setdefaultsounddevice", "CABLE Input (VB-Audio Virtual Cable)", "2"],
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

        with patch("app.time.time", return_value=10.0):
            session._report_input_signal(raw_bytes, 1024)

        self.assertEqual(len(signal_updates), 1)
        self.assertEqual(signal_updates[0]["state"], "active")
        self.assertEqual(signal_updates[0]["device_name"], "CABLE Output (VB-Audio Virtual Cable)")
        self.assertEqual(signal_updates[0]["mode_name"], "VAC")
        self.assertEqual(len(status_updates), 1)
        self.assertIn("Live input active", status_updates[0])

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
        stub.config = {"last_mode": "Microphone"}
        stub.is_muted = False
        stub._audio_switch_in_progress = False
        stub._pending_mode_button = None
        stub._closing = False
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
        stub.require_signal_check_var = DummyVar(True)
        stub.status_var = DummyVar("")
        stub.mode_var = DummyVar("Microphone")
        stub.direct_recording_var = DummyVar("")
        stub.direct_playback_var = DummyVar("")
        stub.runtime_audio_var = DummyVar("")
        stub.live_transcription_status_label = DummyButton()
        stub.btn_mic = DummyButton("#1565C0")
        stub.btn_vac = DummyButton("#2E7D32")
        stub.btn_mix = DummyButton("#8E24AA")
        stub.mute_button = DummyButton("#D32F2F")
        stub.root = DummyRoot()
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
        stub._refresh_run_control_buttons = bind_app_method(stub, "_refresh_run_control_buttons")
        stub._run_preflight = bind_app_method(stub, "_run_preflight")
        stub._finish_apply_audio_mode_hot = bind_app_method(stub, "_finish_apply_audio_mode_hot")
        stub.resolve_active_device = lambda name: {
            "name": name,
            "index": 1,
            "info": {"name": name, "max_input_channels": 1},
            "sample_rate": 24000,
        }
        stub.active_audio_device = {
            "name": "Microphone (Realtek HD Audio Mic input)",
            "index": 0,
            "info": {"name": "Microphone (Realtek HD Audio Mic input)", "max_input_channels": 1},
            "sample_rate": 24000,
        }
        stub.live_transcription_session = SimpleNamespace(switch_input_device=Mock(return_value=(True, "switched")))
        return stub

    def test_refresh_run_control_buttons_includes_mute(self) -> None:
        app_stub = self._make_app_stub()
        app_stub.is_muted = True

        app_stub._refresh_run_control_buttons()

        self.assertEqual(app_stub.mute_button.props["text"], "Muted — Click to Unmute")
        self.assertEqual(app_stub.mute_button.props["border_width"], 2)

    def test_mixed_mode_requires_voicemeeter_keyword(self) -> None:
        app_stub = self._make_app_stub()

        self.assertFalse(app_stub._is_mixed_mode_available())
        self.assertEqual(app_stub._resolve_detected_input_name(app_stub.mix_var.get(), "Mixed"), "")

    def test_mixed_mode_resolver_logs_failure_code(self) -> None:
        app_stub = self._make_app_stub()
        handler = CaptureHandler()
        app.LOGGER.addHandler(handler)
        try:
            app_stub._resolve_detected_input_name(app_stub.mix_var.get(), "Mixed")
        finally:
            app.LOGGER.removeHandler(handler)

        self.assertTrue(any(message.startswith("[Failure: ROUTING]") for message in handler.messages))

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
            with patch("app.sd.check_input_settings", return_value=None), patch(
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


if __name__ == "__main__":
    unittest.main()
