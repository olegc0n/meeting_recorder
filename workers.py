"""
Worker threads for audio capture and transcription.
"""

import time
import soundcard as sc
import numpy as np
from queue import Queue, Empty
from PySide6.QtCore import QThread, Signal
from faster_whisper import WhisperModel
from typing import Optional
import argostranslate.package
import argostranslate.translate


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


class TranscriberWorker(QThread):
    """
    Worker thread that transcribes audio using faster-whisper.
    Reads audio chunks from a queue and emits transcribed text.
    """

    new_text = Signal(str)
    status_update = Signal(str)
    error_occurred = Signal(str)
    stats_updated = Signal(dict)

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
        self.model: Optional[WhisperModel] = None
        self.running = False

        self.sample_rate = 16000
        self.buffer_samples = int(self.sample_rate * buffer_duration)
        self.audio_buffer = np.array([], dtype=np.float32)

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
                        self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
                    except Empty:
                        continue

                    if len(self.audio_buffer) >= self.buffer_samples:
                        audio_to_transcribe = self.audio_buffer.copy()
                        overlap_samples = int(self.sample_rate * 0.5)
                        self.audio_buffer = self.audio_buffer[-overlap_samples:]

                        audio_duration_s = len(audio_to_transcribe) / self.sample_rate

                        t_start = time.perf_counter()
                        segments, info = self.model.transcribe(
                            audio_to_transcribe,
                            language=self.language,
                            beam_size=self.beam_size,
                            vad_filter=self.vad_filter,
                        )
                        t_end = time.perf_counter()
                        transcribe_ms = (t_end - t_start) * 1000

                        transcribed_text = ""
                        for segment in segments:
                            transcribed_text += segment.text.strip() + " "

                        if transcribed_text.strip():
                            self.new_text.emit(transcribed_text.strip())

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
                )
                transcribed_text = ""
                for segment in segments:
                    transcribed_text += segment.text.strip() + " "
                if transcribed_text.strip():
                    self.new_text.emit(transcribed_text.strip())
            except Exception as e:
                print(f"Error transcribing final buffer: {e}")

        self.audio_buffer = np.array([], dtype=np.float32)

    def set_language(self, language: str):
        """Update the language setting."""
        self.language = language if language != "auto" else None


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
