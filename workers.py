"""
Worker threads for audio capture, transcription, and LLM analysis.
"""

import os
import time
import threading
import logging
import logging.handlers
from datetime import datetime, timedelta
import soundcard as sc
import numpy as np
from queue import Queue, Empty
from PySide6.QtCore import QThread, Signal
from faster_whisper import WhisperModel
from typing import Optional
import argostranslate.package
import argostranslate.translate
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
#  LLM debug logger — writes to llm_debug.log (rotates at 5 MB, keeps 3 files)
# ---------------------------------------------------------------------------
_llm_logger = logging.getLogger("llm_debug")
_llm_logger.setLevel(logging.DEBUG)
if not _llm_logger.handlers:
    _log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_debug.log")
    _fh = logging.handlers.RotatingFileHandler(
        _log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    _fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    _llm_logger.addHandler(_fh)

# ---------------------------------------------------------------------------
#  General app logger — writes to app.log (rotates at 5 MB, keeps 3 files)
# ---------------------------------------------------------------------------
_log = logging.getLogger("app")
_log.setLevel(logging.INFO)
if not _log.handlers:
    _app_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.log")
    _app_fh = logging.handlers.RotatingFileHandler(
        _app_log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    _app_fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    _log.addHandler(_app_fh)

class AudioWorker(QThread):
    """
    Worker thread that captures audio from a loopback device.
    Pushes raw audio chunks to a queue for processing.
    """

    error_occurred = Signal(str)

    def __init__(self, device_id: str, sample_rate: int = 16000, chunk_size: int = 1024):
        """
        Initialize the audio worker.

        Args:
            device_id: The ID of the loopback microphone device
            sample_rate: Audio sample rate (default 16000 for Whisper)
            chunk_size: Number of samples per chunk
        """
        super().__init__()
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = Queue()
        self.running = False
        self.microphone = None

    def run(self):
        """Main recording loop."""
        try:
            # Open loopback device (API exposes it as a "microphone" that captures speaker output)
            self.microphone = sc.get_microphone(self.device_id, include_loopback=True)
            self.running = True
            _log.info("AudioWorker started  device_id=%s  sample_rate=%d  chunk_size=%d",
                      self.device_id, self.sample_rate, self.chunk_size)

            # Record audio in chunks
            with self.microphone.recorder(samplerate=self.sample_rate) as mic:
                while self.running:
                    # Read audio chunk
                    audio_data = mic.record(numframes=self.chunk_size)

                    # Convert to mono if stereo (take mean of channels)
                    if len(audio_data.shape) > 1:
                        audio_data = np.mean(audio_data, axis=1)

                    # Ensure float32 format
                    audio_data = audio_data.astype(np.float32)

                    # Push to queue
                    if self.running:
                        self.audio_queue.put(audio_data)

        except Exception as e:
            error_msg = f"Audio capture error: {str(e)}"
            _log.error("AudioWorker error: %s", e, exc_info=True)
            self.error_occurred.emit(error_msg)
            print(error_msg)

    def stop(self):
        """Stop the recording loop."""
        _log.info("AudioWorker stopping")
        self.running = False

    def get_queue(self) -> Queue:
        """Get the audio queue for reading."""
        return self.audio_queue


# ---------------------------------------------------------------------------
#  Performance health levels
# ---------------------------------------------------------------------------
HEALTH_OK = "ok"
HEALTH_WARN = "warn"
HEALTH_CRITICAL = "critical"

# Thresholds (can be tuned or exposed in settings later)
WHISPER_RTF_WARN = 0.8    # real-time factor above which we warn
WHISPER_RTF_CRIT = 1.0    # real-time factor above which it's critical
QUEUE_BACKLOG_WARN = 20   # audio chunks piled up
QUEUE_BACKLOG_CRIT = 50
LLM_RESPONSE_WARN_S = 15  # seconds
LLM_RESPONSE_CRIT_S = 30

# DeepSeek pricing (USD per 1 million tokens) — update if pricing changes
DEEPSEEK_INPUT_PRICE_PER_1M = 0.27
DEEPSEEK_OUTPUT_PRICE_PER_1M = 1.10


class TranscriberWorker(QThread):
    """
    Worker thread that transcribes audio using faster-whisper.
    Reads audio chunks from a queue and emits transcribed text.
    """

    new_text = Signal(str, str, str, str)  # text, chunk_start, chunk_end, t_received
    status_update = Signal(str)
    error_occurred = Signal(str)
    stats_updated = Signal(dict)
    # Emits dict with component health: {"component": str, "level": str, "detail": str}
    performance_alert = Signal(dict)

    def __init__(
        self,
        audio_queue: Queue,
        language: str = "en",
        model_size: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
        buffer_duration: float = 3.0,
        vad_filter: bool = True,
        use_initial_prompt: bool = True,
    ):
        super().__init__()
        self.audio_queue = audio_queue
        self.language = language if language != "auto" else None
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.buffer_duration = buffer_duration
        self.vad_filter = vad_filter
        self.use_initial_prompt = use_initial_prompt
        self.model: Optional[WhisperModel] = None
        self.running = False

        self.sample_rate = 16000
        self.buffer_samples = int(self.sample_rate * buffer_duration)
        self.audio_buffer = np.array([], dtype=np.float32)

        # Rolling context fed back to Whisper as initial_prompt for better continuity
        self.context_prompt: str = ""
        self._max_prompt_words: int = 30

        # Seconds of audio that were kept as overlap from the previous chunk.
        # Segments whose end timestamp falls within this window are already
        # transcribed and must be skipped.
        self._overlap_s: float = 0.0

        # Wall-clock time when the current audio chunk started accumulating
        self._chunk_start_time: Optional[datetime] = None

        # Silence-based early segmentation: trigger transcription when a pause
        # is detected, rather than always waiting for the full buffer to fill.
        self.silence_threshold: float = 0.01   # RMS energy below this = silence
        self.silence_frames_s: float = 0.65    # seconds of silence required
        self.min_audio_s: float = 1.5          # minimum NEW audio before silence trigger applies

    def run(self):
        """Main transcription loop."""
        try:
            self.status_update.emit("Loading model...")
            t0 = time.perf_counter()
            device = self.device
            if device == "auto":
                device = "cuda"
            _log.info(
                "TranscriberWorker loading model  model=%s  device=%s  compute_type=%s  "
                "beam_size=%d  buffer_duration=%.1fs  vad=%s  initial_prompt=%s",
                self.model_size, device, self.compute_type,
                self.beam_size, self.buffer_duration, self.vad_filter, self.use_initial_prompt,
            )
            try:
                self.model = WhisperModel(
                    self.model_size,
                    device=device,
                    compute_type=self.compute_type,
                )
            except Exception:
                if device == "cuda":
                    _log.warning("CUDA load failed — falling back to CPU")
                    self.model = WhisperModel(
                        self.model_size,
                        device="cpu",
                        compute_type=self.compute_type,
                    )
                    device = "cpu"
                else:
                    raise
            model_load_ms = (time.perf_counter() - t0) * 1000
            _log.info("TranscriberWorker model ready  device=%s  load_time=%.0fms", device, model_load_ms)
            self.status_update.emit(f"Model loaded ({model_load_ms:.0f}ms)")

            self.running = True

            while self.running:
                try:
                    try:
                        chunk = self.audio_queue.get(timeout=0.1)
                        if self._chunk_start_time is None:
                            self._chunk_start_time = datetime.now()
                        self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
                    except Empty:
                        continue

                    # --- Decide whether to transcribe now ---
                    min_samples = int(self.sample_rate * self.min_audio_s)
                    silence_samples = int(self.sample_rate * self.silence_frames_s)

                    buffer_full = len(self.audio_buffer) >= self.buffer_samples

                    # Only count *new* audio beyond the kept overlap toward the
                    # minimum, so the overlap portion alone cannot immediately
                    # re-trigger a silence-based transcription.
                    overlap_samples_kept = int(self.sample_rate * self._overlap_s)
                    new_samples_in_buffer = len(self.audio_buffer) - overlap_samples_kept

                    silence_triggered = False
                    if not buffer_full and new_samples_in_buffer >= min_samples:
                        tail = (
                            self.audio_buffer[-silence_samples:]
                            if len(self.audio_buffer) >= silence_samples
                            else self.audio_buffer
                        )
                        rms = float(np.sqrt(np.mean(tail ** 2)))
                        silence_triggered = rms < self.silence_threshold

                    if buffer_full or silence_triggered:
                        trigger = "buffer_full" if buffer_full else "silence"
                        audio_to_transcribe = self.audio_buffer.copy()
                        chunk_start_time = self._chunk_start_time
                        chunk_end_time = datetime.now()

                        if buffer_full:
                            # Keep a 1.0s overlap so boundary phrases aren't re-emitted
                            overlap_samples = int(self.sample_rate * 1.0)
                            self._overlap_s = overlap_samples / self.sample_rate
                            self.audio_buffer = self.audio_buffer[-overlap_samples:]
                            # Overlap carries over into next chunk — record its start now
                            self._chunk_start_time = datetime.now()
                        else:
                            # Clean silence boundary — no overlap needed
                            self._overlap_s = 0.0
                            self.audio_buffer = np.array([], dtype=np.float32)
                            self._chunk_start_time = None

                        audio_duration_s = len(audio_to_transcribe) / self.sample_rate

                        _log.info(
                            "Transcribing  trigger=%s  audio_duration=%.2fs  "
                            "overlap=%.2fs  queue_backlog=%d",
                            trigger,
                            audio_duration_s,
                            self._overlap_s,
                            self.audio_queue.qsize(),
                        )

                        t_sent = datetime.now()
                        t_start = time.perf_counter()
                        segments, info = self.model.transcribe(
                            audio_to_transcribe,
                            language=self.language,
                            beam_size=self.beam_size,
                            vad_filter=self.vad_filter,
                            # Provide previous transcript as context so Whisper
                            # can carry over vocabulary, style and proper nouns.
                            initial_prompt=(self.context_prompt or None) if self.use_initial_prompt else None,
                        )

                        # Iterate the generator — actual decoding happens here
                        transcribed_text = ""
                        for segment in segments:
                            # Skip segments that lie ENTIRELY within the overlap
                            # window — they were already emitted in the previous
                            # chunk.  Segments that CROSS the boundary (end >
                            # overlap_s) are kept in full: with initial_prompt
                            # context Whisper won't repeat the already-said part,
                            # and trimming them was causing the first word of new
                            # speech to be lost.
                            if segment.end <= self._overlap_s:
                                continue
                            transcribed_text += segment.text.strip() + " "

                        t_end = time.perf_counter()
                        t_received = datetime.now()
                        transcribe_ms = (t_end - t_start) * 1000

                        # --- Performance health check: audio queue backlog ---
                        queue_size = self.audio_queue.qsize()
                        if queue_size >= QUEUE_BACKLOG_CRIT:
                            self.performance_alert.emit({
                                "component": "audio_queue",
                                "level": HEALTH_CRITICAL,
                                "detail": f"Audio queue backlog: {queue_size} chunks",
                            })
                        elif queue_size >= QUEUE_BACKLOG_WARN:
                            self.performance_alert.emit({
                                "component": "audio_queue",
                                "level": HEALTH_WARN,
                                "detail": f"Audio queue backlog: {queue_size} chunks",
                            })
                        else:
                            self.performance_alert.emit({
                                "component": "audio_queue",
                                "level": HEALTH_OK,
                                "detail": f"Queue: {queue_size} chunks",
                            })

                        if transcribed_text.strip():
                            _log.info(
                                "Transcription result  words=%d  rtf=%.3f  latency=%.0fms  "
                                "text=%r",
                                len(transcribed_text.split()),
                                transcribe_ms / 1000 / audio_duration_s if audio_duration_s > 0 else 0,
                                transcribe_ms,
                                transcribed_text.strip()[:120],
                            )
                            # Update rolling context (keep last N words)
                            combined = (self.context_prompt + " " + transcribed_text).split()
                            self.context_prompt = " ".join(combined[-self._max_prompt_words:])

                            self.new_text.emit(
                                transcribed_text.strip(),
                                chunk_start_time.strftime("%H:%M:%S.%f")[:-3] if chunk_start_time else "",
                                chunk_end_time.strftime("%H:%M:%S.%f")[:-3],
                                t_received.strftime("%H:%M:%S.%f")[:-3],
                            )

                            word_count = len(transcribed_text.split())
                            rtf = transcribe_ms / 1000 / audio_duration_s if audio_duration_s > 0 else 0
                            wps = word_count / (transcribe_ms / 1000) if transcribe_ms > 0 else 0

                            self.stats_updated.emit(
                                {
                                    "rtf": rtf,
                                    "wps": wps,
                                    "latency_ms": transcribe_ms,
                                    "audio_s": audio_duration_s,
                                    "words": word_count,
                                    "model_load_ms": model_load_ms,
                                    "chunk_count": 1,
                                }
                            )

                            # --- Performance health check: Whisper RTF ---
                            if rtf >= WHISPER_RTF_CRIT:
                                self.performance_alert.emit({
                                    "component": "whisper",
                                    "level": HEALTH_CRITICAL,
                                    "detail": f"Whisper RTF {rtf:.2f} (>{WHISPER_RTF_CRIT}) — too slow, consider a smaller model",
                                })
                            elif rtf >= WHISPER_RTF_WARN:
                                self.performance_alert.emit({
                                    "component": "whisper",
                                    "level": HEALTH_WARN,
                                    "detail": f"Whisper RTF {rtf:.2f} — approaching real-time limit",
                                })
                            else:
                                self.performance_alert.emit({
                                    "component": "whisper",
                                    "level": HEALTH_OK,
                                    "detail": f"Whisper RTF {rtf:.2f}",
                                })

                        else:
                            _log.info(
                                "Transcription empty (no speech detected)  "
                                "audio_duration=%.2fs  trigger=%s",
                                audio_duration_s, trigger,
                            )

                except Exception as e:
                    error_msg = f"Transcription error: {str(e)}"
                    _log.error("TranscriberWorker chunk error: %s", e, exc_info=True)
                    self.error_occurred.emit(error_msg)
                    print(error_msg)

        except Exception as e:
            error_msg = f"Model loading error: {str(e)}"
            _log.error("TranscriberWorker model load error: %s", e, exc_info=True)
            self.status_update.emit("Error loading model")
            self.error_occurred.emit(error_msg)
            print(error_msg)

    def stop(self):
        """Stop the transcription loop."""
        _log.info("TranscriberWorker stopping")
        self.running = False

        # Transcribe any remaining audio in buffer
        if self.model is not None and len(self.audio_buffer) > self.sample_rate:
            _log.info("TranscriberWorker flushing final buffer  samples=%d", len(self.audio_buffer))
            try:
                segments, _ = self.model.transcribe(
                    self.audio_buffer,
                    language=self.language,
                    beam_size=self.beam_size,
                    vad_filter=self.vad_filter,
                    initial_prompt=(self.context_prompt or None) if self.use_initial_prompt else None,
                )
                transcribed_text = ""
                for segment in segments:
                    transcribed_text += segment.text.strip() + " "
                if transcribed_text.strip():
                    _log.info("Final buffer text: %r", transcribed_text.strip()[:120])
                    now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    self.new_text.emit(transcribed_text.strip(), now, now, now)
            except Exception as e:
                _log.error("Error transcribing final buffer: %s", e, exc_info=True)
                print(f"Error transcribing final buffer: {e}")

        self.audio_buffer = np.array([], dtype=np.float32)
        self.context_prompt = ""
        self._overlap_s = 0.0
        self._chunk_start_time = None

    def set_language(self, language: str):
        """Update the language setting."""
        _log.info("TranscriberWorker language changed to %r", language)
        self.language = language if language != "auto" else None
        self.context_prompt = ""  # reset context when language changes
        self._overlap_s = 0.0
        self._chunk_start_time = None


class TranslationWorker(QThread):
    """
    Worker thread that translates text to English using offline Argos Translate.
    """

    new_translation = Signal(str)
    error_occurred = Signal(str)
    status_update = Signal(str)

    def __init__(self, source_language: str = "auto"):
        """
        Initialize the translation worker.

        Args:
            source_language: Source language code (e.g., 'es', 'fr', 'de', 'ja')
        """
        super().__init__()
        self.source_language = source_language if source_language not in ["en", "auto"] else None
        self.text_queue = Queue()
        self.running = False
        self._from_lang = None
        self._to_lang = None

    def run(self):
        """Main translation loop."""
        try:
            # Initialize translator if source language is not English
            if self.source_language:
                _log.info("TranslationWorker initialising  source=%s", self.source_language)
                # Update and get available packages
                argostranslate.package.update_package_index()
                available_packages = argostranslate.package.get_available_packages()

                # Find package for source_language -> en
                package_to_install = None
                for pkg in available_packages:
                    if pkg.from_code == self.source_language and pkg.to_code == "en":
                        package_to_install = pkg
                        break

                # Install package if found and not already installed
                if package_to_install:
                    installed_packages = argostranslate.package.get_installed_packages()
                    already_installed = any(
                        pkg.from_code == self.source_language and pkg.to_code == "en" for pkg in installed_packages
                    )

                    if not already_installed:
                        _log.info("Downloading translation package %s→en", self.source_language)
                        self.status_update.emit(f"Downloading {self.source_language}→en translation model...")
                        argostranslate.package.install_from_path(package_to_install.download())
                        _log.info("Translation package installed %s→en", self.source_language)
                        self.status_update.emit("Translation model ready")
                    else:
                        _log.info("Translation package already installed %s→en", self.source_language)

                    # Cache the language objects for reuse in the loop
                    installed_langs = argostranslate.translate.get_installed_languages()
                    self._from_lang = next(
                        (lang for lang in installed_langs if lang.code == self.source_language), None
                    )
                    self._to_lang = next(
                        (lang for lang in installed_langs if lang.code == "en"), None
                    )
                    _log.info(
                        "TranslationWorker ready  from_lang=%s  to_lang=%s",
                        self._from_lang, self._to_lang,
                    )
                else:
                    error_msg = f"Translation package not available for {self.source_language} → en"
                    _log.error("%s", error_msg)
                    self.error_occurred.emit(error_msg)
                    print(error_msg)
                    return

            self.running = True

            while self.running:
                try:
                    # Get text to translate from queue
                    text = self.text_queue.get(timeout=0.1)

                    # Translate if translator is initialized
                    if self.source_language and text.strip():
                        try:
                            if self._from_lang and self._to_lang:
                                translation = self._from_lang.get_translation(self._to_lang)
                                if translation:
                                    translated = translation.translate(text)
                                    if translated and translated.strip():
                                        _log.info(
                                            "Translation  in=%r  out=%r",
                                            text[:80], translated[:80],
                                        )
                                        self.new_translation.emit(translated.strip())
                        except Exception as e:
                            error_msg = f"Translation error: {str(e)}"
                            _log.error("TranslationWorker translate error: %s", e, exc_info=True)
                            self.error_occurred.emit(error_msg)
                            print(error_msg)

                except Empty:
                    continue
                except Exception as e:
                    error_msg = f"Translation queue error: {str(e)}"
                    _log.error("TranslationWorker queue error: %s", e, exc_info=True)
                    self.error_occurred.emit(error_msg)
                    print(error_msg)

        except Exception as e:
            error_msg = f"Translation worker error: {str(e)}"
            _log.error("TranslationWorker fatal error: %s", e, exc_info=True)
            self.error_occurred.emit(error_msg)
            print(error_msg)

    def stop(self):
        """Stop the translation loop."""
        _log.info("TranslationWorker stopping")
        self.running = False

    def add_text(self, text: str):
        """Add text to the translation queue."""
        if self.source_language and text.strip():
            self.text_queue.put(text)


ANALYSIS_SYSTEM_PROMPT = """\
You are a meeting analyst assistant. You receive a live transcript of an ongoing meeting.
Your job is to extract and highlight the most important points from the meeting content.

You will receive:
1. Your previous analysis (if any) — treat it as your running notes to UPDATE.
2. The latest transcript window (last ~15 minutes of the meeting).

Output format — use this exact structure (Markdown):

## Key Points
- Bullet each important point or decision discussed so far.

## Action Items
- Bullet any tasks, assignments, or follow-ups mentioned (include owner if stated).

## Open Questions
- Bullet any unresolved questions or topics that need further discussion.

Rules:
- Be concise — one sentence per bullet.
- Merge duplicates; keep the latest/most complete version.
- Preserve important points from your previous analysis even if they are no longer in the transcript window.
- Remove items from Open Questions if they have been resolved in newer transcript.
- If the transcript is too short or empty, say "Waiting for more content…"
- Do NOT invent information. Only report what is in the transcript or your previous analysis.
- Return ONLY the formatted output above, no extra commentary.
"""


class LLMAnalysisWorker(QThread):
    """
    Worker thread that periodically sends the last N minutes of transcript
    plus the previous analysis to a DeepSeek LLM for incremental updates.
    """

    new_analysis = Signal(str)       # full analysis text (replaces previous)
    error_occurred = Signal(str)
    status_update = Signal(str)
    # Emits dict with component health: {"component": str, "level": str, "detail": str}
    performance_alert = Signal(dict)
    # Emits cumulative token usage: {"prompt_tokens": int, "completion_tokens": int,
    #                                "total_tokens": int, "estimated_cost_usd": float}
    token_stats_updated = Signal(dict)

    def __init__(self, interval_s: int = 30, window_minutes: int = 15):
        """
        Args:
            interval_s: How often (seconds) to send text to the LLM.
            window_minutes: How many minutes of transcript to include.
        """
        super().__init__()
        self.interval_s = interval_s
        self.window_minutes = window_minutes
        self.running = False

        # Timestamped transcript entries: list of (datetime, text)
        self._lock = threading.Lock()
        self._entries: list = []
        self._last_entry_count: int = 0  # avoid re-sending identical data
        self._previous_analysis: str = ""  # last LLM response

        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        if not api_key:
            _log.warning("DEEPSEEK_API_KEY not set in .env — LLM calls will fail")
            print("WARNING: DEEPSEEK_API_KEY not set in .env")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        # Cumulative token counters for the current session
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0

    # ---- public API (called from GUI thread) ----

    def add_text(self, text: str):
        """Append new timestamped transcription text (thread-safe)."""
        with self._lock:
            self._entries.append((datetime.now(), text.strip()))

    def clear(self):
        """Clear accumulated transcript, previous analysis, and token counters."""
        with self._lock:
            self._entries.clear()
            self._last_entry_count = 0
            self._previous_analysis = ""
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

    # ---- thread loop ----

    def _get_recent_transcript(self) -> str:
        """Return transcript text from the last `window_minutes` minutes."""
        cutoff = datetime.now() - timedelta(minutes=self.window_minutes)
        with self._lock:
            recent = [text for ts, text in self._entries if ts >= cutoff]
        return " ".join(recent).strip()

    def run(self):
        self.running = True
        _log.info(
            "LLMAnalysisWorker started  interval=%ds  window=%dmin",
            self.interval_s, self.window_minutes,
        )
        self.status_update.emit("LLM analysis active")

        while self.running:
            # Sleep in small increments so we can stop promptly
            for _ in range(self.interval_s * 10):
                if not self.running:
                    return
                self.msleep(100)

            # Check if there's new data
            with self._lock:
                current_count = len(self._entries)
            if current_count == 0 or current_count == self._last_entry_count:
                _log.info("LLM skipping — no new transcript entries (total=%d)", current_count)
                continue  # nothing new

            self._last_entry_count = current_count

            transcript_window = self._get_recent_transcript()
            if not transcript_window:
                continue

            # Build user message with previous analysis + recent transcript
            parts = []
            if self._previous_analysis:
                parts.append(
                    f"Your previous analysis:\n\n{self._previous_analysis}\n\n---\n"
                )
            parts.append(
                f"Latest transcript (last ~{self.window_minutes} min):\n\n{transcript_window}"
            )
            user_message = "\n".join(parts)

            try:
                self.status_update.emit("Analyzing meeting…")

                _llm_logger.info("=" * 72)
                _llm_logger.info("REQUEST  model=deepseek-chat  temperature=0.3  max_tokens=1024")
                _llm_logger.info("[SYSTEM PROMPT]\n%s", ANALYSIS_SYSTEM_PROMPT)
                _llm_logger.info("[USER MESSAGE]\n%s", user_message)

                llm_t0 = time.perf_counter()
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=0.3,
                    max_tokens=1024,
                )
                llm_elapsed_s = time.perf_counter() - llm_t0
                analysis = response.choices[0].message.content.strip()

                _llm_logger.info(
                    "RESPONSE  elapsed=%.2fs  finish_reason=%s",
                    llm_elapsed_s,
                    response.choices[0].finish_reason,
                )
                if response.usage:
                    _llm_logger.info(
                        "TOKENS  prompt=%d  completion=%d  total=%d",
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens,
                        response.usage.total_tokens,
                    )
                _llm_logger.info("[RESPONSE CONTENT]\n%s", analysis)

                self._previous_analysis = analysis
                self.new_analysis.emit(analysis)
                self.status_update.emit(f"LLM analysis updated ({llm_elapsed_s:.1f}s)")

                # Track and emit token usage
                usage = response.usage
                if usage:
                    self._total_prompt_tokens += usage.prompt_tokens
                    self._total_completion_tokens += usage.completion_tokens
                    total = self._total_prompt_tokens + self._total_completion_tokens
                    cost = (
                        self._total_prompt_tokens * DEEPSEEK_INPUT_PRICE_PER_1M / 1_000_000
                        + self._total_completion_tokens * DEEPSEEK_OUTPUT_PRICE_PER_1M / 1_000_000
                    )
                    self.token_stats_updated.emit({
                        "prompt_tokens": self._total_prompt_tokens,
                        "completion_tokens": self._total_completion_tokens,
                        "total_tokens": total,
                        "estimated_cost_usd": cost,
                    })

                # --- Performance health check: LLM response time ---
                if llm_elapsed_s >= LLM_RESPONSE_CRIT_S:
                    self.performance_alert.emit({
                        "component": "llm",
                        "level": HEALTH_CRITICAL,
                        "detail": f"LLM response took {llm_elapsed_s:.1f}s (>{LLM_RESPONSE_CRIT_S}s)",
                    })
                elif llm_elapsed_s >= LLM_RESPONSE_WARN_S:
                    self.performance_alert.emit({
                        "component": "llm",
                        "level": HEALTH_WARN,
                        "detail": f"LLM response took {llm_elapsed_s:.1f}s",
                    })
                else:
                    self.performance_alert.emit({
                        "component": "llm",
                        "level": HEALTH_OK,
                        "detail": f"LLM response {llm_elapsed_s:.1f}s",
                    })
            except Exception as e:
                error_msg = f"LLM analysis error: {e}"
                _llm_logger.error("LLM call failed: %s", e, exc_info=True)
                self.error_occurred.emit(error_msg)
                self.performance_alert.emit({
                    "component": "llm",
                    "level": HEALTH_CRITICAL,
                    "detail": f"LLM error: {e}",
                })
                print(error_msg)

    def stop(self):
        """Stop the analysis loop."""
        self.running = False
