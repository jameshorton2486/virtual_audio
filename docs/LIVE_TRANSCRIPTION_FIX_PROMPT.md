You are working in the `virtual_audio` Windows desktop app. This repo is not a generic audio app. Read [app.py](./app.py), [tests/test_app.py](./tests/test_app.py), [README.md](./README.md), and [requirements.txt](./requirements.txt) before changing anything.

Objective: preserve the app's existing Windows audio-mode switching behavior while fixing or extending live transcription resilience. The current architecture already moved the dangerous work off the realtime audio callback. Any further edits must keep that design intact.

Repo facts you must treat as true:

1. `requirements.txt` pins `deepgram-sdk==3.7.0`.
2. The live transcription path is `LiveTranscriptionSession` in `app.py`.
3. Live transcription uses `self._deepgram_client.listen.websocket.v("1")`.
4. Live sample rate is intentionally `LIVE_TRANSCRIPTION_SAMPLE_RATE_HZ = 16000`.
5. The app intentionally changes Windows default recording and playback devices when the user selects `Microphone`, `VAC`, or `Mixed` mode. That is a core feature, not a bug.
6. The app also supports hot switching during an active live session, preflight signal checks, metadata output, and file transcription. Preserve all of that.

Current live-session architecture that must remain intact:

1. `_audio_callback` only converts input to PCM, reports signal state, and enqueues audio into `_pcm_queue`.
2. `_sender_loop` is the only place that calls `connection.send(...)` for PCM blocks.
3. `_keepalive_loop` sends Deepgram keepalive traffic when the stream is idle.
4. `_reconnect` retries websocket startup with exponential backoff using `RECONNECT_BACKOFF_SECONDS`.
5. `_persist_loop` debounces and writes partial transcript text outside the Deepgram callback thread.
6. `_watchdog_loop` detects stalled audio callbacks and reopens the input stream.

Acceptance criteria:

1. Never move websocket sends back into the sounddevice callback.
2. Never perform synchronous disk writes on the Deepgram websocket callback thread.
3. Preserve `switch_input_device`, hot mode switching, `_run_preflight`, `_probe_vac_route`, metadata writing, and file transcription behavior.
4. Keep existing mode semantics:
   `Microphone` and `Mixed` restore playback to the configured speaker device.
   `VAC` routes playback to the configured VAC playback target.
5. `python -m pytest -q` must pass after your changes.
6. If you add tests, put them in `tests/test_app.py` unless there is a compelling reason not to.

Do not do any of the following:

1. Do not create a parallel `audio_engine/` subsystem or split live transcription into a separate framework.
2. Do not remove or disable `set_default_recording_device` / `set_default_playback_device`.
3. Do not rewrite the app into passive capture only.
4. Do not change live transcription from 16 kHz to 24 kHz.
5. Do not increase block size to something high-latency like 8000 frames.
6. Do not remove `sd.InputStream` usage from `LiveTranscriptionSession`.
7. Do not break hot-switching between `Microphone`, `VAC`, and `Mixed`.
8. Do not replace the current UI flow or mode-application flow unless the change is strictly required and covered by tests.
9. Do not regress the Mixed-mode availability checks or their logging behavior.
10. Do not treat Windows audio-device switching as accidental behavior. It is intentional.

If you need to improve the live path further, keep changes surgical:

1. Prefer extending `LiveTranscriptionSession` over introducing new top-level architecture.
2. Reuse the existing queue/thread/state-machine model.
3. Maintain or improve the tests for reconnects, keepalives, callback stalls, queue overflow handling, and mode application.
4. Keep user-visible status updates concise and compatible with the existing UI labels and warning coloring.

Before finishing, provide:

1. A concise summary of what changed.
2. Any risks or follow-up tests worth running manually.
3. The exact test command you ran and whether it passed.
