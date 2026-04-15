# Virtual Audio Control

Windows desktop utility for switching recording devices between a microphone, a virtual audio cable, and a mixed Voicemeeter path.

## What It Includes

- Simple desktop UI built with `customtkinter`
- Device switching through `nircmd.exe`
- Built-in config editor for device names
- Auto-detection and refresh for Windows recording devices
- Auto-detection for VAC recording and playback endpoints
- Live audio quality panel intended to help reduce transcription errors
- Built-in `Test VAC Routing` action that sends a short tone through the cable
- Built-in file transcription for Zoom recordings and saved media using Deepgram
- Built-in live transcription panel for active microphone, VAC, or mixed input
- Packaging command for a standalone `.exe`

## Setup

1. Install Python 3.11+ on Windows.
2. Put `nircmd.exe` in this folder.
3. Create the virtual environment:

```powershell
python -m venv .venv
```

4. Install the dependencies into that environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

5. Add your Deepgram API key in a local `.env` file:

```powershell
Copy-Item .env.example .env
```

Then edit `.env` and set:

```text
DEEPGRAM_API_KEY=your_real_key_here
```

6. Run the app:

```powershell
.\.venv\Scripts\python.exe app.py
```

`app.py` also auto-restarts itself inside `.venv` on Windows when launched from a normal shell, so `python app.py` will still hand off to the local project environment if it exists. PyCharm-hosted runs keep the interpreter you selected in the IDE.
You can also just double-click `run_app.bat`.

## Important Zoom Setting

In Zoom, set `Microphone` to `Same as System`. Otherwise Windows default-device switching will not affect the meeting input.

## How The Mode Buttons Work

- Clicking a mode button changes the Windows default recording device for all three Windows audio roles.
- `VAC` mode selects `CABLE Output (VB-Audio Virtual Cable)` as the recording device and `CABLE Input (VB-Audio Virtual Cable)` as the Windows playback device.
- `Microphone` and `Mixed` modes restore playback to the configured speaker device.
- If an individual app is pinned to a different output device instead of `Default`, change that app back to the Windows default output.
- Use `Test VAC Routing` to send a short tone through the cable and confirm the app meter responds.

## Configure Device Names

Edit the values in `config.json` or use the fields in the app. The app also has a `Refresh Devices` button that re-scans Windows input and output devices and updates the dropdowns.

## Environment Variables

- Store local secrets such as `DEEPGRAM_API_KEY` in `.env`
- `.env` is ignored by Git
- Use `.env.example` as the template

## File Transcription

- Use the `Transcribe File` button in the app for saved Zoom recordings, downloaded videos, and other media files
- Supported file types include `wav`, `mp3`, `m4a`, `mp4`, `webm`, `flac`, `ogg`, and more
- Transcript text and the full Deepgram JSON response are saved into the local `transcripts` folder

## Live Transcription

- Use `Start Live Transcription` to stream the currently active recording device to Deepgram in real time
- `Microphone` mode streams your live mic
- `VAC` mode streams routed playback audio such as Zoom output or other system audio
- `Mixed` mode streams the configured Voicemeeter mixed path
- The rolling transcript appears in the app, auto-scrolls, and is saved automatically into the local `transcripts` folder when you stop
- Each live session also saves a matching `.json` metadata file with mode, device, timestamps, and status
- While live transcription is running, mode and device switching are intentionally blocked so the capture source stays stable

## Recommended Live Workflows

- Zoom deposition audio only: switch to `VAC`, make sure Windows playback is routed to `CABLE Input`, then start live transcription
- Your own live speech: switch to `Microphone`, confirm your mic is the active recording device, then start live transcription
- Narration over routed playback: switch to `Mixed`, verify Voicemeeter routing first, then start live transcription

## WER Notes

- `VAC` mode is usually the cleanest path for transcription when you only need playback audio.
- `Mixed` mode is useful when you need mic narration over playback, but the final quality depends on Voicemeeter routing and levels.
- The live meter is a basic signal-health check. It is not a true WER estimator.

## Build a Standalone EXE

```powershell
.\.venv\Scripts\python.exe -m pip install pyinstaller
.\.venv\Scripts\pyinstaller.exe --onefile --noconsole --name VirtualAudioControl app.py
```

After the build finishes, copy these files together before distributing:

- `dist\VirtualAudioControl.exe`
- `nircmd.exe`
- `config.json`

## Troubleshooting

- If switching fails, check that `nircmd.exe` is present and the device names are exact.
- If the monitor says the signal is silent, confirm the current default recording device is actually receiving audio.
- If the app closes but audio switching worked, review Python/package installation first.
