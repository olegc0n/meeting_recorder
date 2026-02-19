# AI Meeting Transcriber

A modern desktop application built with PySide6 that records system audio (meeting loopback) and transcribes it in real-time using faster-whisper.

## Features

- **Real-time Audio Recording**: Captures system audio from selected speakers via loopback
- **Live Transcription**: Transcribes audio using faster-whisper (Whisper model)
- **Multi-language Support**: Supports English, Spanish, French, German, Japanese, and auto-detection
- **Modern GUI**: Clean, intuitive interface built with PySide6
- **Thread-safe Architecture**: Non-blocking audio capture and transcription using QThread

## Requirements

- Python 3.8+
- PySide6
- soundcard
- numpy
- faster-whisper

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python main.py
```

## Usage

1. **Select Audio Source**: Choose your speaker/headphone device from the dropdown
2. **Select Language**: Choose the language for transcription (or "auto" for detection)
3. **Start Recording**: Click "START RECORDING" to begin capturing and transcribing audio
4. **View Transcription**: Transcribed text appears in real-time in the log area
5. **Stop Recording**: Click "STOP RECORDING" when done
6. **Clear Log**: Use "CLEAR LOG" to empty the transcription area

## Technical Details

- **Audio Sample Rate**: 16000 Hz (required by Whisper)
- **Buffer Size**: ~3 seconds of audio before transcription
- **Model Size**: "tiny" (fastest, can be changed in `workers.py`)
- **Threading**: Separate threads for audio capture and transcription to prevent GUI freezing

## Troubleshooting

- **No loopback device found**: Ensure your audio driver supports loopback recording. On Windows, you may need to enable "Stereo Mix" in recording devices.
- **Model loading slow**: The first run downloads the Whisper model (~75MB for tiny). Subsequent runs are faster.
- **No audio captured**: Verify the selected speaker device has a corresponding loopback microphone.

## Project Structure

- `main.py`: GUI application and main window
- `workers.py`: QThread classes for audio capture and transcription
- `utils.py`: Helper functions for device enumeration and loopback matching
