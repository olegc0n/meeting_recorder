"""
AI Meeting Transcriber - Main GUI Application
"""
import sys
from datetime import datetime
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QTextEdit, QStatusBar, QMessageBox,
    QSplitter, QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox, QFormLayout,
)
from PySide6.QtCore import Qt
from workers import AudioWorker, TranscriberWorker
from utils import get_output_devices, find_loopback_for_speaker
from config import load_transcription_config, save_transcription_config


class MeetingTranscriberWindow(QMainWindow):
    """Main window for the AI Meeting Transcriber application."""
    
    def __init__(self):
        super().__init__()
        self.audio_worker: AudioWorker = None
        self.transcriber_worker: TranscriberWorker = None
        self.is_recording = False
        self.current_loopback_id = None
        self.stats_total = {"words": 0, "audio_s": 0, "latency_ms": 0, "chunk_count": 0}

        self.init_ui()
        self.load_devices()
        self.load_settings()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("AI Meeting Transcriber")
        self.setGeometry(100, 100, 1000, 650)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Audio Source Selection
        audio_label = QLabel("Select Audio Source (Speaker):")
        main_layout.addWidget(audio_label)
        
        self.device_combo = QComboBox()
        self.device_combo.currentIndexChanged.connect(self.on_device_changed)
        main_layout.addWidget(self.device_combo)
        
        # Language Selection
        language_label = QLabel("Select Language:")
        main_layout.addWidget(language_label)
        
        self.language_combo = QComboBox()
        self.language_combo.addItems(["en", "es", "fr", "de", "ja", "auto"])
        self.language_combo.setCurrentText("en")
        main_layout.addWidget(self.language_combo)
        
        # Horizontal splitter: Log (left) | Settings + Statistics (right)
        splitter = QSplitter(Qt.Horizontal)

        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.addWidget(QLabel("Transcription Log:"))
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setPlaceholderText("Transcribed text will appear here...")
        log_layout.addWidget(self.text_edit)
        splitter.addWidget(log_widget)

        right_splitter = QSplitter(Qt.Vertical)
        settings_group = QGroupBox("Settings")
        settings_layout = QFormLayout(settings_group)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium"])
        self.model_combo.setToolTip("Larger = better accuracy, slower")
        settings_layout.addRow("Model:", self.model_combo)
        self.device_combo_settings = QComboBox()
        self.device_combo_settings.addItems(["cpu", "cuda", "auto"])
        self.device_combo_settings.setToolTip("cuda requires GPU")
        settings_layout.addRow("Device:", self.device_combo_settings)
        self.compute_combo = QComboBox()
        self.compute_combo.addItems(["int8", "float16", "float32"])
        self.compute_combo.setToolTip("int8 for CPU, float16 for GPU")
        settings_layout.addRow("Compute:", self.compute_combo)
        self.beam_spin = QSpinBox()
        self.beam_spin.setRange(1, 10)
        self.beam_spin.setValue(5)
        self.beam_spin.setToolTip("1=fastest, 5-7=balanced")
        settings_layout.addRow("Beam size:", self.beam_spin)
        self.buffer_spin = QDoubleSpinBox()
        self.buffer_spin.setRange(1.0, 5.0)
        self.buffer_spin.setSingleStep(0.5)
        self.buffer_spin.setValue(3.0)
        self.buffer_spin.setSuffix(" s")
        self.buffer_spin.setToolTip("Shorter = lower latency")
        settings_layout.addRow("Buffer:", self.buffer_spin)
        self.vad_check = QCheckBox("Enable VAD filter")
        self.vad_check.setChecked(True)
        self.vad_check.setToolTip("Off = faster, more noise")
        settings_layout.addRow("", self.vad_check)
        right_splitter.addWidget(settings_group)

        stats_group = QGroupBox("Statistics")
        stats_layout = QFormLayout(stats_group)
        self.rtf_label = QLabel("—")
        self.rtf_label.setToolTip(self.tr("real-time factor for the last chunk (< 1 = faster than real-time)"))
        self.wps_label = QLabel("—")
        self.wps_label.setToolTip(self.tr("words per second for the last chunk"))
        self.latency_label = QLabel("—")
        self.latency_label.setToolTip(self.tr("ms to transcribe last chunk"))
        self.avg_rtf_label = QLabel("—")
        self.total_words_label = QLabel("—")
        self.chunks_label = QLabel("—")
        self.model_load_label = QLabel("—")
        stats_layout.addRow("RTF (last):", self.rtf_label)
        stats_layout.addRow("WPS (last):", self.wps_label)
        stats_layout.addRow("Latency (last):", self.latency_label)
        stats_layout.addRow("Avg RTF:", self.avg_rtf_label)
        stats_layout.addRow("Total words:", self.total_words_label)
        stats_layout.addRow("Chunks:", self.chunks_label)
        stats_layout.addRow("Model load:", self.model_load_label)
        right_splitter.addWidget(stats_group)
        right_splitter.setSizes([200, 200])

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(right_splitter)
        splitter.addWidget(right_widget)
        splitter.setSizes([500, 280])
        main_layout.addWidget(splitter)
        
        # Control Buttons
        button_layout = QHBoxLayout()
        
        self.start_stop_btn = QPushButton("START RECORDING")
        self.start_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.start_stop_btn.clicked.connect(self.toggle_recording)
        button_layout.addWidget(self.start_stop_btn)
        
        self.clear_btn = QPushButton("CLEAR LOG")
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.clear_btn.clicked.connect(self.clear_log)
        button_layout.addWidget(self.clear_btn)
        
        main_layout.addLayout(button_layout)
        
        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready (Model not loaded)")

        self.model_combo.currentTextChanged.connect(self.save_settings)
        self.device_combo_settings.currentTextChanged.connect(self.save_settings)
        self.compute_combo.currentTextChanged.connect(self.save_settings)
        self.beam_spin.valueChanged.connect(self.save_settings)
        self.buffer_spin.valueChanged.connect(self.save_settings)
        self.vad_check.toggled.connect(self.save_settings)

    def load_settings(self):
        cfg = load_transcription_config()
        self.model_combo.blockSignals(True)
        self.device_combo_settings.blockSignals(True)
        self.compute_combo.blockSignals(True)
        self.beam_spin.blockSignals(True)
        self.buffer_spin.blockSignals(True)
        self.vad_check.blockSignals(True)
        self.model_combo.setCurrentText(cfg["model_size"])
        self.device_combo_settings.setCurrentText(cfg["device"])
        self.compute_combo.setCurrentText(cfg["compute_type"])
        self.beam_spin.setValue(cfg["beam_size"])
        self.buffer_spin.setValue(cfg["buffer_duration"])
        self.vad_check.setChecked(cfg["vad_filter"])
        self.model_combo.blockSignals(False)
        self.device_combo_settings.blockSignals(False)
        self.compute_combo.blockSignals(False)
        self.beam_spin.blockSignals(False)
        self.buffer_spin.blockSignals(False)
        self.vad_check.blockSignals(False)

    def save_settings(self):
        save_transcription_config(self.get_transcription_config())

    def get_transcription_config(self) -> dict:
        return {
            "model_size": self.model_combo.currentText(),
            "device": self.device_combo_settings.currentText(),
            "compute_type": self.compute_combo.currentText(),
            "beam_size": self.beam_spin.value(),
            "buffer_duration": self.buffer_spin.value(),
            "vad_filter": self.vad_check.isChecked(),
        }

    def reset_stats(self):
        self.stats_total = {"words": 0, "audio_s": 0, "latency_ms": 0, "chunk_count": 0}
        self.rtf_label.setText("—")
        self.wps_label.setText("—")
        self.latency_label.setText("—")
        self.avg_rtf_label.setText("—")
        self.total_words_label.setText("—")
        self.chunks_label.setText("—")
        self.model_load_label.setText("—")

    def on_stats_updated(self, stats: dict):
        self.stats_total["words"] += stats.get("words", 0)
        self.stats_total["audio_s"] += stats.get("audio_s", 0)
        self.stats_total["latency_ms"] += stats.get("latency_ms", 0)
        self.stats_total["chunk_count"] += stats.get("chunk_count", 0)
        self.rtf_label.setText(f"{stats.get('rtf', 0):.3f}")
        self.wps_label.setText(f"{stats.get('wps', 0):.1f}")
        self.latency_label.setText(f"{stats.get('latency_ms', 0):.0f} ms")
        n = self.stats_total["chunk_count"]
        if n > 0:
            avg = (self.stats_total["latency_ms"] / 1000) / self.stats_total["audio_s"] if self.stats_total["audio_s"] > 0 else 0
            self.avg_rtf_label.setText(f"{avg:.3f}")
        self.total_words_label.setText(str(self.stats_total["words"]))
        self.chunks_label.setText(str(self.stats_total["chunk_count"]))
        self.model_load_label.setText(f"{stats.get('model_load_ms', 0):.0f} ms")

    def set_settings_enabled(self, enabled: bool):
        for w in (self.model_combo, self.device_combo_settings, self.compute_combo,
                  self.beam_spin, self.buffer_spin, self.vad_check):
            w.setEnabled(enabled)

    def load_devices(self):
        """Load available output devices into the combo box."""
        self.device_combo.clear()
        devices = get_output_devices()
        
        if not devices:
            self.device_combo.addItem("No devices found")
            QMessageBox.warning(
                self,
                "No Devices",
                "No audio output devices found. Please check your audio settings."
            )
            return
        
        for device in devices:
            self.device_combo.addItem(device['name'], device['id'])
    
    def on_device_changed(self, index: int):
        """Handle device selection change."""
        if index >= 0:
            device_id = self.device_combo.currentData()
            device_name = self.device_combo.currentText()
            
            # Find corresponding loopback device
            loopback = find_loopback_for_speaker(device_name, device_id)
            
            if loopback:
                self.current_loopback_id = loopback[1]
                self.status_bar.showMessage(f"Selected: {device_name} -> Loopback: {loopback[0]}")
            else:
                self.current_loopback_id = None
                self.status_bar.showMessage(f"Warning: No loopback found for {device_name}")
                QMessageBox.warning(
                    self,
                    "Loopback Not Found",
                    f"Could not find a loopback microphone for '{device_name}'.\n"
                    "Recording may not work correctly."
                )
    
    def toggle_recording(self):
        """Start or stop recording."""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start audio recording and transcription."""
        if not self.current_loopback_id:
            QMessageBox.warning(
                self,
                "No Device Selected",
                "Please select an audio source first."
            )
            return
        
        try:
            language = self.language_combo.currentText()
            cfg = self.get_transcription_config()

            self.audio_worker = AudioWorker(self.current_loopback_id)
            self.audio_worker.error_occurred.connect(self.on_audio_error)
            self.audio_worker.start()

            self.transcriber_worker = TranscriberWorker(
                self.audio_worker.get_queue(),
                language=language,
                model_size=cfg["model_size"],
                device=cfg["device"],
                compute_type=cfg["compute_type"],
                beam_size=cfg["beam_size"],
                buffer_duration=cfg["buffer_duration"],
                vad_filter=cfg["vad_filter"],
            )
            self.transcriber_worker.new_text.connect(self.on_new_transcription)
            self.transcriber_worker.status_update.connect(self.on_status_update)
            self.transcriber_worker.error_occurred.connect(self.on_transcription_error)
            self.transcriber_worker.stats_updated.connect(self.on_stats_updated)
            self.transcriber_worker.start()

            self.reset_stats()
            
            # Update UI
            self.is_recording = True
            self.start_stop_btn.setText("STOP RECORDING")
            self.start_stop_btn.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #da190b;
                }
            """)
            self.device_combo.setEnabled(False)
            self.language_combo.setEnabled(False)
            self.set_settings_enabled(False)
            self.status_bar.showMessage("Recording...")
            
            # Add start message to log
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.append_to_log(f"[{timestamp}] Meeting started...")
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Recording Error",
                f"Failed to start recording:\n{str(e)}"
            )
            self.stop_recording()
    
    def stop_recording(self):
        """Stop audio recording and transcription."""
        self.is_recording = False
        
        # Stop workers
        if self.audio_worker:
            self.audio_worker.stop()
            self.audio_worker.wait(3000)  # Wait up to 3 seconds
            if self.audio_worker.isRunning():
                self.audio_worker.terminate()
            self.audio_worker = None
        
        if self.transcriber_worker:
            self.transcriber_worker.stop()
            self.transcriber_worker.wait(3000)  # Wait up to 3 seconds
            if self.transcriber_worker.isRunning():
                self.transcriber_worker.terminate()
            self.transcriber_worker = None
        
        # Update UI
        self.start_stop_btn.setText("START RECORDING")
        self.start_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.device_combo.setEnabled(True)
        self.language_combo.setEnabled(True)
        self.set_settings_enabled(True)
        self.status_bar.showMessage("Stopped")
        
        # Add stop message to log
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.append_to_log(f"[{timestamp}] Recording stopped.\n")
    
    def on_new_transcription(self, text: str):
        """Handle new transcribed text."""
        self.append_to_log(text)
    
    def append_to_log(self, text: str):
        """Append text to the transcription log."""
        self.text_edit.append(text)
        # Auto-scroll to bottom
        scrollbar = self.text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear_log(self):
        """Clear the transcription log."""
        self.text_edit.clear()
    
    def on_status_update(self, status: str):
        """Handle status updates from workers."""
        if not self.is_recording:
            self.status_bar.showMessage(f"Ready ({status})")
        else:
            self.status_bar.showMessage(f"Recording... ({status})")
    
    def on_audio_error(self, error_msg: str):
        """Handle audio worker errors."""
        QMessageBox.critical(self, "Audio Error", error_msg)
        self.stop_recording()
    
    def on_transcription_error(self, error_msg: str):
        """Handle transcription worker errors."""
        QMessageBox.critical(self, "Transcription Error", error_msg)
        # Don't stop recording on transcription errors, just log them
    
    def closeEvent(self, event):
        """Handle window close event - ensure clean shutdown."""
        if self.is_recording:
            self.stop_recording()
        event.accept()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    window = MeetingTranscriberWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
