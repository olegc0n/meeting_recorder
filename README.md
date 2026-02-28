# AI Meeting Transcriber

A modern desktop application built with PySide6 that captures system audio via loopback, transcribes it in real-time with faster-whisper, optionally translates non-English speech, and periodically generates AI-powered meeting summaries using a DeepSeek LLM.

## Features

- **Real-time Audio Capture** — records system audio from any speaker/headphone device via loopback (powered by [SoundCard](https://github.com/bastibe/SoundCard))
- **Live Transcription** — uses [faster-whisper](https://github.com/guillaumekln/faster-whisper) (CTranslate2 Whisper) with configurable model size, device, compute type, beam size, and VAD filtering
- **Silence-based Segmentation** — intelligently triggers transcription on detected pauses instead of waiting for a fixed buffer, reducing latency
- **Rolling Context Prompt** — feeds previous transcript back to Whisper as an initial prompt for improved continuity and accuracy across chunks
- **Offline Translation** — translates non-English transcripts to English on-the-fly using [Argos Translate](https://github.com/argosopentech/argos-translate) (no cloud API needed)
- **LLM Meeting Analysis** — periodically sends the last ~15 minutes of transcript to a DeepSeek LLM that extracts key points, action items, and open questions in Markdown
- **Multi-language Support** — English, Spanish, French, German, Japanese, and automatic language detection
- **Statistics Panel** — slide-out panel showing real-time performance metrics (RTF, WPS, latency, total words, chunk count, model load time)
- **Configurable Settings** — in-app settings dialog to tune transcription model, compute options, buffer duration, VAD, initial prompt, and LLM analysis interval
- **Persistent Preferences** — audio source and all settings are saved/restored between sessions via `QSettings`
- **Modern Dark UI** — Catppuccin Mocha-inspired theme with rounded cards, smooth animations, and a responsive splitter layout

## Requirements

- Python 3.8+
- A system audio driver that supports loopback recording

## Installation

1. Clone the repository and create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
# venv\Scripts\activate    # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. **(Optional)** For LLM meeting analysis, create a `.env` file in the project root:

```
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com   # optional, this is the default
```

4. Run the application:

```bash
python main.py
```

## Usage

1. **Select Audio Source** — choose your speaker/headphone device from the dropdown; the app will automatically locate the corresponding loopback microphone
2. **Select Language** — pick the spoken language (`en`, `es`, `fr`, `de`, `ja`) or `auto` for detection
3. **Start Recording** — click **START RECORDING** to begin capturing and transcribing audio
4. **View Transcription** — transcribed text appears in real-time in the *Transcription* panel with chunk timestamps
5. **View Translation** — when the source language is not English, translations appear in the *Translation* panel
6. **View LLM Analysis** — AI-generated key points, action items, and open questions update periodically in the *LLM Meeting Analysis* panel
7. **Statistics** — click the 📊 button to open the slide-out stats panel
8. **Settings** — click the ⚙ button to adjust model size, device, buffer duration, VAD, and LLM options
9. **Stop Recording** — click **STOP RECORDING** when done; any remaining audio in the buffer is transcribed automatically
10. **Clear** — use the *Clear Log*, *Clear Translation*, and *Clear Analysis* buttons to reset individual panels

## Technical Details

| Parameter | Default | Notes |
|---|---|---|
| Audio sample rate | 16 000 Hz | Required by Whisper |
| Buffer duration | 3.0 s | Configurable 1–5 s in Settings |
| Silence threshold | RMS < 0.01 | Triggers early transcription on pauses |
| Overlap | 1.0 s | Kept between chunks to avoid cutting phrases |
| Whisper model | `tiny` | `tiny`, `base`, `small`, `medium` via Settings |
| Compute type | `int8` | `int8` (CPU), `float16` / `float32` (GPU) |
| LLM interval | 30 s | Configurable 10–300 s in Settings |
| LLM transcript window | 15 min | Rolling window sent to DeepSeek |

### Threading Architecture

The application runs four independent `QThread` workers to keep the GUI responsive:

| Worker | Responsibility |
|---|---|
| `AudioWorker` | Captures loopback audio and pushes chunks to a queue |
| `TranscriberWorker` | Pulls audio from the queue, runs Whisper inference, emits text |
| `TranslationWorker` | Translates transcribed text to English via Argos Translate |
| `LLMAnalysisWorker` | Periodically calls DeepSeek to summarize the meeting transcript |

## Project Structure

| File | Description |
|---|---|
| `main.py` | GUI application, main window, stylesheet, and action wiring |
| `workers.py` | `QThread` workers for audio capture, transcription, translation, and LLM analysis |
| `utils.py` | Audio device enumeration and loopback microphone matching |
| `config.py` | Settings persistence via `QSettings` (load/save transcription config & audio source) |
| `settings_dialog.py` | Modal settings dialog for transcription and LLM options |
| `stats_panel.py` | Animated slide-out statistics panel with real-time metrics |
| `requirements.txt` | Python dependencies |

## Dependencies

| Package | Purpose |
|---|---|
| PySide6 | Qt GUI framework |
| soundcard | Cross-platform audio loopback capture |
| numpy | Audio buffer manipulation |
| faster-whisper | CTranslate2-based Whisper speech recognition |
| argostranslate | Offline machine translation |
| openai | OpenAI-compatible client for DeepSeek API |
| python-dotenv | Loads API keys from `.env` |

## Troubleshooting

- **No loopback device found** — ensure your audio driver supports loopback recording. On Linux, PulseAudio/PipeWire monitor devices are used. On Windows, enable "Stereo Mix" in recording devices.
- **Model loading slow** — the first run downloads the Whisper model (~75 MB for `tiny`). Subsequent runs use the cached model.
- **No audio captured** — verify the selected speaker device has a corresponding loopback microphone in the status bar.
- **LLM analysis not appearing** — check that `DEEPSEEK_API_KEY` is set in `.env` and the API is reachable. Errors are shown in the status bar.
- **Translation not working** — Argos Translate downloads language packages on first use; an internet connection is required for the initial download.
