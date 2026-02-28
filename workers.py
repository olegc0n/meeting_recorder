"""
Worker threads for audio capture, transcription, and LLM analysis.
"""

import os
import time
from datetime import datetime
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
            self.error_occurred.emit(error_msg)
            print(error_msg)

    def stop(self):
        """Stop the recording loop."""
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
            try:
                self.model = WhisperModel(
                    self.model_size,
                    device=device,
                    compute_type=self.compute_type,
                )
            except Exception:
                if device == "cuda":
                    self.model = WhisperModel(
                        self.model_size,
                        device="cpu",
                        compute_type=self.compute_type,
                    )
                    device = "cpu"
                else:
                    raise
            model_load_ms = (time.perf_counter() - t0) * 1000
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

                        t_sent = datetime.now()
                        t_start = time.perf_counter()
                        segments, info = self.model.transcribe(
                            audio_to_transcribe,
                            language=self.language,
                            beam_size=self.beam_size,
                            vad_filter=self.vad_filter,
                            # Provide previous transcript as context so Whisper
                            # can carry over vocabulary, style and proper nouns.
                            initial_prompt=self.context_prompt or None if self.use_initial_prompt else None,
                        )

                        # Iterate the generator — actual decoding happens here
                        transcribed_text = ""
                        for segment in segments:
                            # Skip segments that START within the overlap window;
                            # their audio was already transcribed in the previous
                            # chunk. Using segment.start (not segment.end) avoids
                            # passing through segments that merely overlap the
                            # boundary, which caused phrase duplication.
                            if segment.start < self._overlap_s:
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

                except Exception as e:
                    error_msg = f"Transcription error: {str(e)}"
                    self.error_occurred.emit(error_msg)
                    print(error_msg)

        except Exception as e:
            error_msg = f"Model loading error: {str(e)}"
            self.status_update.emit("Error loading model")
            self.error_occurred.emit(error_msg)
            print(error_msg)

    def stop(self):
        """Stop the transcription loop."""
        self.running = False

        # Transcribe any remaining audio in buffer
        if self.model is not None and len(self.audio_buffer) > self.sample_rate:
            try:
                segments, _ = self.model.transcribe(
                    self.audio_buffer,
                    language=self.language,
                    beam_size=self.beam_size,
                    vad_filter=self.vad_filter,
                    initial_prompt=self.context_prompt or None if self.use_initial_prompt else None,
                )
                transcribed_text = ""
                for segment in segments:
                    transcribed_text += segment.text.strip() + " "
                if transcribed_text.strip():
                    self.new_text.emit(transcribed_text.strip())
            except Exception as e:
                print(f"Error transcribing final buffer: {e}")

        self.audio_buffer = np.array([], dtype=np.float32)
        self.context_prompt = ""
        self._overlap_s = 0.0
        self._chunk_start_time = None

    def set_language(self, language: str):
        """Update the language setting."""
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
        self.installed_translation = None

    def run(self):
        """Main translation loop."""
        try:
            # Initialize translator if source language is not English
            if self.source_language:
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
                        self.status_update.emit(f"Downloading {self.source_language}→en translation model...")
                        argostranslate.package.install_from_path(package_to_install.download())
                        self.status_update.emit("Translation model ready")

                    # Get the installed translation
                    self.installed_translation = argostranslate.translate.get_installed_languages()
                else:
                    error_msg = f"Translation package not available for {self.source_language} → en"
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
                            from_lang = next(
                                (
                                    lang
                                    for lang in argostranslate.translate.get_installed_languages()
                                    if lang.code == self.source_language
                                ),
                                None,
                            )
                            to_lang = next(
                                (
                                    lang
                                    for lang in argostranslate.translate.get_installed_languages()
                                    if lang.code == "en"
                                ),
                                None,
                            )

                            if from_lang and to_lang:
                                translation = from_lang.get_translation(to_lang)
                                if translation:
                                    translated = translation.translate(text)
                                    if translated and translated.strip():
                                        self.new_translation.emit(translated.strip())
                        except Exception as e:
                            error_msg = f"Translation error: {str(e)}"
                            self.error_occurred.emit(error_msg)
                            print(error_msg)

                except Empty:
                    continue
                except Exception as e:
                    error_msg = f"Translation queue error: {str(e)}"
                    self.error_occurred.emit(error_msg)
                    print(error_msg)

        except Exception as e:
            error_msg = f"Translation worker error: {str(e)}"
            self.error_occurred.emit(error_msg)
            print(error_msg)

    def stop(self):
        """Stop the translation loop."""
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
        self._lock = __import__("threading").Lock()
        self._entries: list = []
        self._last_entry_count: int = 0  # avoid re-sending identical data
        self._previous_analysis: str = ""  # last LLM response

        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        if not api_key:
            print("WARNING: DEEPSEEK_API_KEY not set in .env")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    # ---- public API (called from GUI thread) ----

    def add_text(self, text: str):
        """Append new timestamped transcription text (thread-safe)."""
        with self._lock:
            self._entries.append((datetime.now(), text.strip()))

    def clear(self):
        """Clear accumulated transcript and previous analysis."""
        with self._lock:
            self._entries.clear()
            self._last_entry_count = 0
            self._previous_analysis = ""

    # ---- thread loop ----

    def _get_recent_transcript(self) -> str:
        """Return transcript text from the last `window_minutes` minutes."""
        cutoff = datetime.now() - __import__("datetime").timedelta(minutes=self.window_minutes)
        with self._lock:
            recent = [text for ts, text in self._entries if ts >= cutoff]
        return " ".join(recent).strip()

    def run(self):
        self.running = True
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
                self._previous_analysis = analysis
                self.new_analysis.emit(analysis)
                self.status_update.emit(f"LLM analysis updated ({llm_elapsed_s:.1f}s)")

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
