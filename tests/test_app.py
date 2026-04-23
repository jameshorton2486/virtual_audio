from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import app


class AppUtilityTests(unittest.TestCase):
    def test_compute_rms_db_reports_no_signal_for_silence(self) -> None:
        db = app.compute_rms_db(np.zeros(1600, dtype=np.float32))
        self.assertLess(db, -80.0)
        self.assertEqual(app.signal_state_from_db(db), "No Signal")

    def test_compute_rms_db_reports_active_for_audio(self) -> None:
        samples = np.full(1600, 0.1, dtype=np.float32)
        db = app.compute_rms_db(samples)
        self.assertGreater(db, -80.0)
        self.assertEqual(app.signal_state_from_db(db), "Active")

    @patch("app.sd.query_devices")
    def test_resolve_input_device_uses_exact_match(self, query_devices: Mock) -> None:
        query_devices.return_value = [
            {"name": "Microphone", "max_input_channels": 1},
            {"name": app.CONFIG["input_device"], "max_input_channels": 1},
        ]

        index, info = app.resolve_input_device(app.CONFIG["input_device"])

        self.assertEqual(index, 1)
        self.assertEqual(info["name"], app.CONFIG["input_device"])

    @patch("app.sd.query_devices")
    def test_resolve_input_device_prefers_wasapi_backend(self, query_devices: Mock) -> None:
        query_devices.return_value = [
            {"name": "CABLE Output (VB-Audio Virtual Cable), Windows DirectSound", "max_input_channels": 1},
            {"name": "CABLE Output (VB-Audio Virtual Cable), Windows WASAPI", "max_input_channels": 1},
        ]

        index, info = app.resolve_input_device(app.CONFIG["input_device"])

        self.assertEqual(index, 1)
        self.assertIn("WASAPI", info["name"])

    @patch("app.sd.query_devices")
    def test_list_audio_devices_prints_enumeration(self, query_devices: Mock) -> None:
        query_devices.return_value = [
            {"name": "Device A", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "Device B", "max_input_channels": 0, "max_output_channels": 2},
        ]

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            app.list_audio_devices()

        output = buffer.getvalue()
        self.assertIn("Device A", output)
        self.assertIn("Device B", output)

    @patch("app.show_error_popup")
    @patch("app.get_vac_device", return_value=7)
    @patch("app.sd.InputStream")
    def test_start_audio_stream_returns_none_on_failure(self, input_stream: Mock, _get_vac_device: Mock, show_error_popup: Mock) -> None:
        input_stream.side_effect = RuntimeError("boom")
        app._ACTIVE_AUDIO_CALLBACK = lambda *args, **kwargs: None

        stream = app.start_audio_stream()

        self.assertIsNone(stream)
        show_error_popup.assert_called_once()

    @patch("app.get_vac_device", return_value=7)
    @patch("app.sd.InputStream")
    def test_start_audio_stream_uses_smaller_blocksize(self, input_stream: Mock, _get_vac_device: Mock) -> None:
        stream = Mock()
        input_stream.return_value = stream
        app._ACTIVE_AUDIO_CALLBACK = lambda *args, **kwargs: None

        result = app.start_audio_stream()

        self.assertEqual(result, stream)
        input_stream.assert_called_once()
        self.assertEqual(input_stream.call_args.kwargs["blocksize"], 1600)
        stream.start.assert_called_once()

    def test_safe_thread_catches_exception(self) -> None:
        calls: list[str] = []

        @app.safe_thread
        def crash() -> None:
            calls.append("started")
            raise RuntimeError("thread failure")

        crash()

        self.assertEqual(calls, ["started"])

    def test_format_live_result_text_uses_speaker_labels_when_diarized(self) -> None:
        result = SimpleNamespace(
            channel=SimpleNamespace(
                alternatives=[
                    SimpleNamespace(
                        transcript="hello there general kenobi",
                        words=[
                            SimpleNamespace(word="hello", punctuated_word="Hello", speaker=0),
                            SimpleNamespace(word="there", punctuated_word="there.", speaker=0),
                            SimpleNamespace(word="general", punctuated_word="General", speaker=1),
                            SimpleNamespace(word="kenobi", punctuated_word="Kenobi.", speaker=1),
                        ],
                    )
                ]
            )
        )

        text = app.format_live_result_text(result)

        self.assertEqual(text, "[Speaker 0] Hello there.\n[Speaker 1] General Kenobi.")

    def test_format_live_result_text_falls_back_when_no_speaker_words_present(self) -> None:
        result = {
            "channel": {
                "alternatives": [
                    {
                        "transcript": "plain transcript fallback",
                        "words": [],
                    }
                ]
            }
        }

        text = app.format_live_result_text(result)

        self.assertEqual(text, "plain transcript fallback")

    def test_build_deepgram_live_options_enables_requested_features(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            options = app.build_deepgram_live_options()

        self.assertEqual(options["model"], "nova-3")
        self.assertEqual(options["language"], "en-US")
        self.assertTrue(options["diarize"])
        self.assertTrue(options["filler_words"])
        self.assertTrue(options["numerals"])

    def test_build_deepgram_live_options_merges_session_keyterms(self) -> None:
        with patch.dict("os.environ", {"DEEPGRAM_KEYTERMS": "Zoom,Deepgram"}, clear=False):
            options = app.build_deepgram_live_options(["Gregory Ernest Stone", "Zoom"])

        self.assertEqual(options["keyterm"], ["Zoom", "Deepgram", "Gregory Ernest Stone"])

    def test_build_deepgram_live_options_uses_keyterms_for_nova_3(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "DEEPGRAM_MODEL": "nova-3",
                "DEEPGRAM_KEYTERMS": "Zoom, VB-Audio Virtual Cable, Deepgram",
            },
            clear=False,
        ):
            options = app.build_deepgram_live_options()

        self.assertEqual(options["keyterm"], ["Zoom", "VB-Audio Virtual Cable", "Deepgram"])
        self.assertNotIn("keywords", options)

    def test_build_deepgram_live_options_falls_back_to_keywords_for_non_nova_3(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "DEEPGRAM_MODEL": "nova-2",
                "DEEPGRAM_KEYWORDS": "Zoom:2, Deepgram:3",
            },
            clear=False,
        ):
            options = app.build_deepgram_live_options()

        self.assertEqual(options["keywords"], ["Zoom:2", "Deepgram:3"])
        self.assertNotIn("keyterm", options)

    def test_parse_claude_json_payload_handles_fenced_json(self) -> None:
        payload = app.parse_claude_json_payload(
            """```json
            {"proper_nouns":["Gregory Ernest Stone"],"legal_terms":[],"likely_domain_terms":[],"spelling_variants":[]}
            ```"""
        )

        self.assertEqual(payload["proper_nouns"], ["Gregory Ernest Stone"])

    def test_extract_notice_session_keyterms_uses_proper_and_legal_terms_only(self) -> None:
        payload = {
            "proper_nouns": ["Gregory Ernest Stone", "Zoom"],
            "legal_terms": ["oral deposition"],
            "likely_domain_terms": ["[inferred] MRI"],
            "spelling_variants": [],
        }

        terms = app.extract_notice_session_keyterms(payload)

        self.assertEqual(terms, ["Gregory Ernest Stone", "Zoom", "oral deposition"])

    def test_normalize_cause_number(self) -> None:
        self.assertEqual(app.normalize_cause_number("2024-CI-27841"), "2024-CI-27841")
        self.assertEqual(app.normalize_cause_number("  2024-ci-27841  "), "2024-CI-27841")
        self.assertEqual(app.normalize_cause_number("2024--CI---27841"), "2024-CI-27841")
        self.assertEqual(app.normalize_cause_number("2025CI08060"), "2025CI08060")

    def test_witness_slug(self) -> None:
        self.assertEqual(app.witness_slug("Gregory Ernest Stone"), "stone_gregory_ernest")
        self.assertEqual(app.witness_slug("John Smith Jr."), "smith_john")
        self.assertEqual(app.witness_slug("Dr. Bianca Caram"), "caram_bianca")
        self.assertEqual(app.witness_slug("Stone"), "stone_unknown")
        self.assertEqual(app.witness_slug(""), "unknown_unknown")

    def test_parse_case_identity_from_stone_response(self) -> None:
        response = {
            "proper_nouns": [
                "2024-CI-27841",
                "Gregory Ernest Stone",
                "Thomas D. Jones",
                "Bexar County",
            ]
        }

        identity = app.parse_case_identity(response)

        self.assertEqual(identity["cause_number"], "2024-CI-27841")
        self.assertEqual(identity["witness_slug"], "stone_gregory_ernest")

    def test_parse_case_identity_skips_firm_as_witness(self) -> None:
        response = {
            "proper_nouns": [
                "2024-CI-27841",
                "Thomas D. Jones, PC",
                "Gregory Ernest Stone",
            ]
        }

        identity = app.parse_case_identity(response)

        self.assertEqual(identity["deponent_full_name"], "Gregory Ernest Stone")

    def test_levenshtein(self) -> None:
        self.assertEqual(app.levenshtein("", "abc"), 3)
        self.assertEqual(app.levenshtein("2025-CVA-001596D2", "2025CVA001596D2"), 2)
        self.assertEqual(app.levenshtein("foo", "foo"), 0)

    def test_check_fuzzy_match_exact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            parent = app.Path(tmpdir)
            (parent / "2024-CI-27841").mkdir()

            match = app.check_fuzzy_match("2024-CI-27841", parent)

        self.assertEqual(match, "2024-CI-27841")

    def test_check_fuzzy_match_near(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            parent = app.Path(tmpdir)
            (parent / "2025-CVA-001596D2").mkdir()

            match = app.check_fuzzy_match("2025CVA001596D2", parent)

        self.assertEqual(match, "2025-CVA-001596D2")

    def test_check_fuzzy_match_distant(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            parent = app.Path(tmpdir)
            (parent / "2024-CI-27841").mkdir()

            match = app.check_fuzzy_match("2025-CI-10000", parent)

        self.assertIsNone(match)

    def test_save_transcript_snapshot_uses_current_witness_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app_stub = SimpleNamespace(current_witness_folder=app.Path(tmpdir))
            with patch("app.time.strftime", return_value="20260423_151500"):
                app.SimpleAudioApp._save_transcript_snapshot(app_stub, "hello transcript")

            output_path = app.Path(tmpdir) / "live_transcript_20260423_151500.txt"
            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.read_text(encoding="utf-8"), "hello transcript")


class DeepgramLiveClientTests(unittest.TestCase):
    def test_keepalive_fires_after_idle_interval(self) -> None:
        ui_queue: "queue.Queue[tuple[str, object]]" = app.queue.Queue()
        client = app.DeepgramLiveClient("test-key", ui_queue)
        connection = Mock()

        def mark_stopped() -> None:
            client.running = False

        connection.keep_alive.side_effect = mark_stopped
        client.connection = connection
        client.running = True
        client._last_send_ts = 0.0

        with patch("app.time.monotonic", side_effect=[10.0, 10.0]), patch("app.time.sleep", return_value=None):
            client._keepalive_loop()

        connection.keep_alive.assert_called_once()

    def test_on_close_while_running_triggers_reconnect_event(self) -> None:
        ui_queue: "queue.Queue[tuple[str, object]]" = app.queue.Queue()
        client = app.DeepgramLiveClient("test-key", ui_queue)
        client.running = True
        client._reconnect_event = Mock()

        client._on_close(None)

        client._reconnect_event.set.assert_called_once()

    def test_successful_reconnect_preserves_transcript_buffers(self) -> None:
        ui_queue: "queue.Queue[tuple[str, object]]" = app.queue.Queue()
        client = app.DeepgramLiveClient("test-key", ui_queue)
        client.running = True
        client.final_lines = ["[Speaker 0] Existing final"]
        client.interim_text = "[Speaker 1] Existing interim"
        connection = Mock()
        connection.start.return_value = True

        with patch.object(client, "_create_connection", return_value=connection), patch(
            "app.time.monotonic",
            return_value=123.0,
        ):
            ok = client._attempt_reconnect()

        self.assertTrue(ok)
        self.assertEqual(client.connection, connection)
        self.assertEqual(client.final_lines, ["[Speaker 0] Existing final"])
        self.assertEqual(client.interim_text, "[Speaker 1] Existing interim")
        self.assertEqual(ui_queue.get_nowait(), ("status", "Reconnected."))

    def test_failed_reconnect_stops_client_and_pushes_error(self) -> None:
        ui_queue: "queue.Queue[tuple[str, object]]" = app.queue.Queue()
        client = app.DeepgramLiveClient("test-key", ui_queue)
        client.running = True
        connection = Mock()
        connection.start.return_value = False

        with patch.object(client, "_create_connection", return_value=connection), patch("app.time.sleep", return_value=None):
            ok = client._attempt_reconnect()

        self.assertFalse(ok)
        self.assertFalse(client.running)
        self.assertEqual(ui_queue.get_nowait(), ("error", "Deepgram disconnected - click Start to resume."))

    def test_request_reconnect_sets_event(self) -> None:
        ui_queue: "queue.Queue[tuple[str, object]]" = app.queue.Queue()
        client = app.DeepgramLiveClient("test-key", ui_queue)
        client.running = True
        client._reconnect_event = Mock()
        client._reconnect_event.is_set.return_value = False

        client.request_reconnect("Reconnecting...")

        client._reconnect_event.is_set.assert_called_once()
        client._reconnect_event.set.assert_called_once()
        self.assertEqual(ui_queue.get_nowait(), ("status", "Reconnecting..."))


class AudioSenderLoopTests(unittest.TestCase):
    def test_sender_loop_retries_pending_chunk_after_send_failure(self) -> None:
        app_stub = SimpleNamespace(
            audio_stop_event=app.threading.Event(),
            audio_queue=app.queue.Queue(),
            deepgram=None,
            running=True,
        )
        deepgram = Mock()
        calls: list[bytes] = []

        def send_side_effect(chunk: bytes) -> None:
            calls.append(chunk)
            if len(calls) == 1:
                raise RuntimeError("temporary drop")
            app_stub.audio_stop_event.set()

        deepgram.send.side_effect = send_side_effect
        app_stub.deepgram = deepgram
        app_stub.audio_queue.put(b"chunk-1")

        with patch.object(app, "log_error"), patch("app.time.sleep", return_value=None):
            app.SimpleAudioApp._audio_sender_loop(app_stub)

        self.assertEqual(calls, [b"chunk-1", b"chunk-1"])
        deepgram.request_reconnect.assert_called_once_with("Deepgram send failed. Reconnecting...")

    def test_sender_loop_does_not_stop_on_send_failure(self) -> None:
        app_stub = SimpleNamespace(
            audio_stop_event=app.threading.Event(),
            audio_queue=app.queue.Queue(),
            deepgram=None,
            running=True,
        )
        deepgram = Mock()

        def send_side_effect(_chunk: bytes) -> None:
            app_stub.running = False
            app_stub.audio_stop_event.set()
            raise RuntimeError("temporary drop")

        deepgram.send.side_effect = send_side_effect
        app_stub.deepgram = deepgram
        app_stub.audio_queue.put(b"chunk-1")

        with patch.object(app, "log_error"), patch("app.time.sleep", return_value=None):
            app.SimpleAudioApp._audio_sender_loop(app_stub)

        self.assertTrue(app_stub.audio_stop_event.is_set())
        deepgram.request_reconnect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
