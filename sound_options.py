from __future__ import annotations

import sounddevice as sd


print("\n=== INPUT DEVICES ===")
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] > 0:
        print(f"{i}: {d['name']} (inputs={d['max_input_channels']}, sr={d['default_samplerate']})")

print("\n=== OUTPUT DEVICES ===")
for i, d in enumerate(sd.query_devices()):
    if d["max_output_channels"] > 0:
        print(f"{i}: {d['name']} (outputs={d['max_output_channels']}, sr={d['default_samplerate']})")

print("\n=== DEFAULT DEVICES ===")
print(sd.default.device)
