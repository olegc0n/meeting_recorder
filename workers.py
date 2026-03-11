"""
Worker threads for audio capture, transcription, and LLM analysis.
"""

import os
import time
import threading
import logging
import logging.handlers
import wave
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
#  Debug flag — set to True to save every audio chunk as a WAV file so you
#  can play them back and verify what is being recorded.  Files are written
#  to  debug_chunks/  next to this script, named chunk_001.wav, chunk_002.wav …
# ---------------------------------------------------------------------------
DEBUG_SAVE_CHUNKS: bool = True
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

    Accumulates raw samples until a silence boundary is detected (or the
    maximum chunk duration is reached), then prepends an overlap window from
    the previous chunk and pushes a ready-to-transcribe tuple
    ``(audio: np.ndarray, overlap_s: float, chunk_start: datetime)``
    to the queue.  TranscriberWorker can therefore skip all buffer management
    and silence detection.
    """

    error_occurred = Signal(str)

    def __init__(
        self,
        device_id: str,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        min_duration_s: float = 7.0,
        silence_threshold: float = 0.02,
        silence_frames_s: float = 0.65,
        overlap_s: float = 1.0,
        max_duration_s: float = 30.0,
    ):
        """
        Args:
            device_id: Loopback microphone device ID.
            sample_rate: Audio sample rate (should match Whisper's expectation: 16 000 Hz).
            chunk_size: Raw frames read per mic.record() call.
            min_duration_s: Minimum seconds of audio to accumulate before a
                silence trigger is allowed.
            silence_threshold: RMS energy below which a tail window is
                considered silent.
            silence_frames_s: Length (seconds) of the tail window used for the
                silence check.
            overlap_s: Minimum seconds from the end that the overlap must cover.
                The actual cut point is walked backward from that position until
                a silent window is found, so the overlap always starts at a
                natural speech boundary.
            max_duration_s: Hard upper limit on chunk duration; forces a flush
                even if no silence is detected.
        """
        super().__init__()
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.min_duration_s = min_duration_s
        self.silence_threshold = silence_threshold
        self.silence_frames_s = silence_frames_s
        self.overlap_s = overlap_s
        self.max_duration_s = max_duration_s
        self.audio_queue = Queue()
        self.running = False
        self.microphone = None
        self._debug_chunk_index: int = 0

    def _find_silence(
        self,
        audio: np.ndarray,
        search_start: int,
        search_end: int,
        frame_ms: int = 30,
        min_silent_frames: int = 3,
    ) -> Optional[int]:
        """
        Scan audio[search_start:search_end] in fixed-size frames and return the
        sample index at the START of the most recent (closest to search_end)
        run of at least *min_silent_frames* consecutive silent frames.

        A frame is silent when its RMS is below self.silence_threshold.
        Returns None if no qualifying run is found.

        Using per-frame analysis (30 ms each) is far more robust than a single
        RMS over a long window: one loud transient no longer masks silence.
        Requiring consecutive silent frames avoids false triggers on brief gaps.
        """
        frame_size = int(self.sample_rate * frame_ms / 1000)
        if frame_size == 0 or search_end - search_start < frame_size:
            return None

        # Compute RMS for every full frame in the range
        frames: list[tuple[int, bool]] = []
        pos = search_start
        while pos + frame_size <= search_end:
            rms = float(np.sqrt(np.mean(audio[pos : pos + frame_size] ** 2)))
            frames.append((pos, rms < self.silence_threshold))
            pos += frame_size

        # Walk forward keeping track of the most recent qualifying silent run
        last_valid_start: Optional[int] = None
        run_start: Optional[int] = None
        run_len = 0

        for frame_pos, is_silent in frames:
            if is_silent:
                if run_start is None:
                    run_start = frame_pos
                run_len += 1
                if run_len >= min_silent_frames:
                    last_valid_start = run_start
            else:
                run_start = None
                run_len = 0

        return last_valid_start

    def _save_debug_chunk(self, full_chunk: np.ndarray, overlap: np.ndarray, new_audio: np.ndarray) -> None:
        """Save three WAV files per chunk for debugging:
        - chunk_NNN_full.wav    — what Whisper receives (overlap + new audio)
        - chunk_NNN_overlap.wav — the overlap piece prepended from the previous chunk
        - chunk_NNN_new.wav     — the freshly recorded audio (no overlap)
        """
        debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_chunks")
        os.makedirs(debug_dir, exist_ok=True)
        self._debug_chunk_index += 1
        idx = self._debug_chunk_index

        def _write_wav(path: str, audio: np.ndarray) -> None:
            pcm = np.clip(audio, -1.0, 1.0)
            pcm_int16 = (pcm * 32767).astype(np.int16)
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(pcm_int16.tobytes())

        parts = {
            "full":    full_chunk,
            "overlap": overlap,
            "new":     new_audio,
        }
        for label, audio in parts.items():
            if len(audio) == 0:
                continue
            path = os.path.join(debug_dir, f"chunk_{idx:03d}_{label}.wav")
            _write_wav(path, audio)
            _log.debug(
                "DEBUG_SAVE_CHUNKS  saved %s  duration=%.2fs",
                path, len(audio) / self.sample_rate,
            )

    def run(self):
        """Main recording loop."""
        try:
            self.microphone = sc.get_microphone(self.device_id, include_loopback=True)
            self.running = True
            _log.info(
                "AudioWorker started  device_id=%s  sample_rate=%d  chunk_size=%d  "
                "min_duration=%.1fs  silence_threshold=%.4f  overlap=%.1fs  max_duration=%.1fs",
                self.device_id, self.sample_rate, self.chunk_size,
                self.min_duration_s, self.silence_threshold, self.overlap_s, self.max_duration_s,
            )

            audio_buffer = np.array([], dtype=np.float32)
            prev_overlap = np.array([], dtype=np.float32)
            chunk_start_time: Optional[datetime] = None

            min_samples = int(self.sample_rate * self.min_duration_s)
            silence_samples = int(self.sample_rate * self.silence_frames_s)
            overlap_samples = int(self.sample_rate * self.overlap_s)
            max_samples = int(self.sample_rate * self.max_duration_s)

            with self.microphone.recorder(samplerate=self.sample_rate) as mic:
                while self.running:
                    audio_data = mic.record(numframes=self.chunk_size)

                    if len(audio_data.shape) > 1:
                        audio_data = np.mean(audio_data, axis=1)
                    audio_data = audio_data.astype(np.float32)

                    if chunk_start_time is None:
                        chunk_start_time = datetime.now()
                    audio_buffer = np.concatenate([audio_buffer, audio_data])

                    # --- Decide whether to flush ---
                    buffer_full = len(audio_buffer) >= max_samples

                    silence_triggered = False
                    if not buffer_full and len(audio_buffer) >= min_samples:
                        # Check the last silence_frames_s of audio using frame-level
                        # analysis.  A single-RMS check on the whole tail is fooled by
                        # one loud sample; per-frame consecutive detection is robust.
                        tail_end = len(audio_buffer)
                        tail_start = max(0, tail_end - silence_samples)
                        silence_triggered = (
                            self._find_silence(audio_buffer, tail_start, tail_end) is not None
                        )

                    if buffer_full or silence_triggered:
                        trigger = "buffer_full" if buffer_full else "silence"
                        actual_overlap_s = len(prev_overlap) / self.sample_rate

                        # Prepend overlap from the previous chunk for Whisper context
                        full_chunk = (
                            np.concatenate([prev_overlap, audio_buffer])
                            if len(prev_overlap) > 0
                            else audio_buffer.copy()
                        )

                        _log.info(
                            "AudioWorker chunk ready  trigger=%s  buffer=%.2fs  overlap=%.2fs",
                            trigger,
                            len(audio_buffer) / self.sample_rate,
                            actual_overlap_s,
                        )

                        # Determine overlap for the next chunk: search for a silence
                        # boundary in a limited window just *before* the 1s mark.
                        # Max look-back is 4 s so the overlap stays bounded.
                        buf_len = len(audio_buffer)
                        fallback_start = max(0, buf_len - overlap_samples)
                        lookback_start = max(0, fallback_start - int(self.sample_rate * 4.0))
                        silence_pos = self._find_silence(
                            audio_buffer, lookback_start, fallback_start
                        )
                        if silence_pos is not None:
                            overlap_start = silence_pos
                            _log.debug(
                                "AudioWorker overlap cut at silence  pos=%.2fs  from_end=%.2fs",
                                silence_pos / self.sample_rate,
                                (buf_len - silence_pos) / self.sample_rate,
                            )
                        else:
                            overlap_start = fallback_start
                        prev_overlap = audio_buffer[overlap_start:].copy()

                        if DEBUG_SAVE_CHUNKS:
                            self._save_debug_chunk(full_chunk, prev_overlap, audio_buffer)

                        if self.running:
                            self.audio_queue.put((full_chunk, actual_overlap_s, chunk_start_time))

                        audio_buffer = np.array([], dtype=np.float32)
                        chunk_start_time = None

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
        self.vad_filter = vad_filter
        self.use_initial_prompt = use_initial_prompt
        self.model: Optional[WhisperModel] = None
        self.running = False

        self.sample_rate = 16000

        # Rolling context fed back to Whisper as initial_prompt for better continuity
        self.context_prompt: str = ""
        self._max_prompt_words: int = 30

    @staticmethod
    def _normalize_word(w: str) -> str:
        """Lowercase and strip punctuation for fuzzy word comparison."""
        return "".join(c for c in w.lower() if c.isalnum())

    def _strip_overlap_prefix(self, text: str) -> str:
        """
        Remove words from the START of *text* that duplicate the TAIL of
        self.context_prompt.

        Whisper sometimes re-transcribes the last word(s) of the overlap even
        after the timestamp filter, because a segment straddles the boundary.
        This catches that case at the word level using punctuation-insensitive
        comparison.
        """
        if not self.context_prompt or not text:
            return text

        ctx_words = self.context_prompt.split()
        txt_words = text.split()
        if not ctx_words or not txt_words:
            return text

        # Cap search to avoid false positives on common short words
        max_check = min(len(ctx_words), len(txt_words), 15)

        for length in range(max_check, 0, -1):
            ctx_tail = [self._normalize_word(w) for w in ctx_words[-length:]]
            txt_head = [self._normalize_word(w) for w in txt_words[:length]]
            # Ignore empty tokens (pure-punctuation words normalize to "")
            if any(t == "" for t in ctx_tail + txt_head):
                continue
            if ctx_tail == txt_head:
                removed = " ".join(txt_words[:length])
                result = " ".join(txt_words[length:]).strip()
                _log.info(
                    "Dedup stripped %d word(s) from transcription start: %r",
                    length, removed,
                )
                return result

        return text

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
                "beam_size=%d  vad=%s  initial_prompt=%s",
                self.model_size, device, self.compute_type,
                self.beam_size, self.vad_filter, self.use_initial_prompt,
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
                        item = self.audio_queue.get(timeout=0.1)
                        audio_to_transcribe, overlap_s, chunk_start_time = item
                    except Empty:
                        continue

                    chunk_end_time = datetime.now()
                    audio_duration_s = len(audio_to_transcribe) / self.sample_rate
                    initial_prompt=(self.context_prompt or None) if self.use_initial_prompt else None

                    _log.info(
                        f"Transcribing  audio_duration={audio_duration_s:.2f}s  overlap={overlap_s:.2f}s  initial_prompt={initial_prompt}  queue_backlog={self.audio_queue.qsize()}",
                    )

                    t_start = time.perf_counter()
                    segments, info = self.model.transcribe(
                        audio_to_transcribe,
                        language=self.language,
                        beam_size=self.beam_size,
                        vad_filter=self.vad_filter,
                        # Provide previous transcript as context so Whisper
                        # can carry over vocabulary, style and proper nouns.
                        initial_prompt=initial_prompt,
                    )

                    # Iterate the generator — actual decoding happens here
                    transcribed_text = ""
                    for segment in segments:
                        # Skip segments that lie ENTIRELY within the overlap
                        # window prepended by AudioWorker — they were already
                        # emitted as part of the previous chunk.
                        if segment.end <= overlap_s:
                            continue
                        transcribed_text += segment.text.strip() + " "

                    # A segment that straddles the overlap boundary is kept in
                    # full (its start < overlap_s but end > overlap_s), so its
                    # first words may duplicate the tail of context_prompt.
                    # Strip any such prefix now.
                    transcribed_text = self._strip_overlap_prefix(transcribed_text.strip())

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
                        combined = transcribed_text.split()
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
                            "Transcription empty (no speech detected)  audio_duration=%.2fs",
                            audio_duration_s,
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
        self.context_prompt = ""

    def set_language(self, language: str):
        """Update the language setting."""
        _log.info("TranscriberWorker language changed to %r", language)
        self.language = language if language != "auto" else None
        self.context_prompt = ""  # reset context when language changes


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
