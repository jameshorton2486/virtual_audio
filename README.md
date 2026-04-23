# Virtual Audio Control

A simple, stable Zoom transcription tool.

This app captures Zoom audio from `CABLE Output (VB-Audio Virtual Cable)`, sends it to Deepgram for live transcription, and shows:

- the active input device
- live RMS / signal level
- live transcription output

It does not switch Windows devices and does not interfere with your microphone. Your Razer mic can still be used directly in Zoom while the app transcribes Zoom playback through VAC.

## Install

```powershell
git clone <repo>
cd virtual_audio
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Requirements

- Python 3.10+
- VB-Audio Virtual Cable installed
- Zoom installed

## Setup

Open Zoom and go to Audio Settings.

Set:

- Speaker: `CABLE Input (VB-Audio Virtual Cable)`
- Microphone: `Razer Seiren V3 Mini`

The app input is fixed to:

- `CABLE Output (VB-Audio Virtual Cable)`

## Run App

```powershell
python app.py
```

To list detected audio devices:

```powershell
python app.py --list-devices
```

## Expected Result

- Zoom audio is transcribed
- Your microphone still works in Zoom
- The app shows signal activity from VAC
- The app does not switch devices or override Windows audio

## UI

The app contains one panel only:

- fixed input device display
- `Start Transcription`
- `Stop`
- RMS / signal status
- live transcription box

## Deepgram

Create a local `.env` file from `.env.example` and set:

```text
DEEPGRAM_API_KEY=your_real_key_here
```

## Testing

The simplified app is designed so that:

- it does not crash when there is no signal
- it does not exit automatically because of device switching logic
- RMS updates when audio plays
- Deepgram receives audio from the VAC stream
