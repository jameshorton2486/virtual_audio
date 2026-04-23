from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
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
    @patch("app.sd.InputStream")
    def test_start_audio_stream_returns_none_on_failure(self, input_stream: Mock, show_error_popup: Mock) -> None:
        input_stream.side_effect = RuntimeError("boom")
        app._ACTIVE_AUDIO_CALLBACK = lambda *args, **kwargs: None

        stream = app.start_audio_stream()

        self.assertIsNone(stream)
        show_error_popup.assert_called_once()

    def test_safe_thread_catches_exception(self) -> None:
        calls: list[str] = []

        @app.safe_thread
        def crash() -> None:
            calls.append("started")
            raise RuntimeError("thread failure")

        crash()

        self.assertEqual(calls, ["started"])


if __name__ == "__main__":
    unittest.main()
