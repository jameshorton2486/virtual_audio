from __future__ import annotations

import time

import numpy as np
import sounddevice as sd


TARGET_NAME = "CABLE Output (VB-Audio Virtual Cable)"
SAMPLE_RATE = 16000
CHANNELS = 1


def find_vac() -> int:
    devices = sd.query_devices()
    for index, device in enumerate(devices):
        name = str(device.get("name", ""))
        if TARGET_NAME in name and "WASAPI" in name and int(device.get("max_input_channels", 0)) > 0:
            print(f"Using [{index}] {name}")
            return index
    raise RuntimeError("VAC WASAPI input device not found")


def callback(indata, frames, time_info, status) -> None:
    if status:
        print(f"status: {status}")
    rms = float(np.sqrt(np.mean(np.asarray(indata, dtype=np.float32) ** 2)))
    db = float(20 * np.log10(rms + 1e-10))
    print(f"RMS dB: {db:.2f}")


def main() -> None:
    device_index = find_vac()
    print("Listening... press Ctrl+C to stop")
    with sd.InputStream(
        device=device_index,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        callback=callback,
    ):
        try:
            while True:
                time.sleep(0.25)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
