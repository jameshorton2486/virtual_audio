from __future__ import annotations

from typing import Any, Callable

import sounddevice as sd


class AudioEngine:
    def __init__(self, logger=None, *, sounddevice_module=sd, input_stream_factory=None):
        self.logger = logger
        self.sd = sounddevice_module
        self._input_stream_factory = input_stream_factory or sounddevice_module.InputStream

    def close_stream(self, stream, *, context: str = "AudioEngine", device_name: str = "") -> None:
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception as exc:
            if self.logger is not None:
                self.logger.warning("[%s] stream_close_warning device=%s reason=%s", context, device_name or "unknown", exc)

    def start_input_stream(
        self,
        *,
        device_index: int,
        samplerate: int,
        channels: int,
        callback: Callable[..., Any],
        blocksize: int = 1024,
        dtype: str = "float32",
    ):
        self.sd.check_input_settings(
            device=device_index,
            samplerate=samplerate,
            channels=channels,
        )
        stream = self._input_stream_factory(
            samplerate=samplerate,
            blocksize=blocksize,
            device=device_index,
            channels=channels,
            dtype=dtype,
            callback=callback,
        )
        stream.start()
        if self.logger is not None:
            self.logger.info(
                "[AudioEngine] Started input stream device=%s samplerate=%s channels=%s blocksize=%s",
                device_index,
                samplerate,
                channels,
                blocksize,
            )
        return stream
