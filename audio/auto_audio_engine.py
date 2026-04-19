from audio.detector import AudioDeviceEntry, AutoModeName, DeviceDetector, normalize_device_name


class AutoAudioEngine(DeviceDetector):
    """Compatibility wrapper for existing callers and tests."""
