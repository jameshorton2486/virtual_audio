import subprocess
import unittest
from unittest.mock import patch

import app
import numpy as np


class DeviceHelpersTests(unittest.TestCase):
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

        with patch("app.NIRCMD_PATH.exists", return_value=True), patch("app.subprocess.run", side_effect=fake_run):
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

        with patch("app.NIRCMD_PATH.exists", return_value=True), patch("app.subprocess.run", side_effect=fake_run):
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


if __name__ == "__main__":
    unittest.main()
