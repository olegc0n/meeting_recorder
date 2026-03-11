"""
Settings Dialog for AI Meeting Transcriber.
"""

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QFormLayout,
    QFrame,
)
from PySide6.QtCore import Qt, Signal


DIALOG_STYLE = """
QDialog {
    background-color: #1e1e2e;
    border-radius: 12px;
}
QLabel {
    color: #cdd6f4;
    font-size: 13px;
}
QLabel#sectionLabel {
    color: #89b4fa;
    font-size: 14px;
    font-weight: 600;
    padding-top: 8px;
}
QLabel#titleLabel {
    color: #cdd6f4;
    font-size: 18px;
    font-weight: 700;
}
QComboBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 8px;
    padding: 6px 12px;
    min-height: 28px;
    font-size: 13px;
}
QComboBox:hover {
    border-color: #89b4fa;
}
QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #89b4fa;
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 8px;
    selection-background-color: #45475a;
    selection-color: #cdd6f4;
    padding: 4px;
}
QSpinBox, QDoubleSpinBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 8px;
    padding: 6px 12px;
    min-height: 28px;
    font-size: 13px;
}
QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #89b4fa;
}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: #45475a;
    border: none;
    width: 20px;
    border-radius: 4px;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #585b70;
}
QCheckBox {
    color: #cdd6f4;
    font-size: 13px;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 20px;
    height: 20px;
    border-radius: 4px;
    border: 2px solid #45475a;
    background-color: #313244;
}
QCheckBox::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}
QCheckBox::indicator:hover {
    border-color: #89b4fa;
}
QFrame#separator {
    background-color: #45475a;
    max-height: 1px;
}
"""


class SettingsDialog(QDialog):
    """Modal settings dialog."""

    settings_changed = Signal(dict)

    def __init__(self, current_config: dict, parent=None):
        super().__init__(parent)
        self._config = dict(current_config)
        self.setWindowTitle("Settings")
        self.setFixedWidth(420)
        self.setStyleSheet(DIALOG_STYLE)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self._build_ui()
        self._load_config()

    # ------------------------------------------------------------------ build
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        # ---- Header ----
        header = QHBoxLayout()
        title = QLabel("Settings")
        title.setObjectName("titleLabel")
        header.addWidget(title)
        header.addStretch()
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(32, 32)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #a6adc8;
                font-size: 18px;
                border: none;
                border-radius: 16px;
            }
            QPushButton:hover {
                background-color: #45475a;
                color: #f38ba8;
            }
        """)
        close_btn.clicked.connect(self.reject)
        header.addWidget(close_btn)
        layout.addLayout(header)
        layout.addWidget(self._separator())

        # ---- Transcription section ----
        layout.addWidget(self._section_label("Transcription"))

        form1 = QFormLayout()
        form1.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form1.setFormAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form1.setHorizontalSpacing(16)
        form1.setVerticalSpacing(10)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium"])
        self.model_combo.setToolTip("Larger = better accuracy, slower")
        form1.addRow("Model:", self.model_combo)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda", "auto"])
        self.device_combo.setToolTip("cuda requires GPU")
        form1.addRow("Device:", self.device_combo)

        self.compute_combo = QComboBox()
        self.compute_combo.addItems(["int8", "float16", "float32"])
        self.compute_combo.setToolTip("int8 for CPU, float16 for GPU")
        form1.addRow("Compute:", self.compute_combo)

        self.beam_spin = QSpinBox()
        self.beam_spin.setRange(1, 10)
        self.beam_spin.setValue(5)
        self.beam_spin.setToolTip("1 = fastest, 5-7 = balanced")
        form1.addRow("Beam size:", self.beam_spin)

        self.buffer_spin = QDoubleSpinBox()
        self.buffer_spin.setRange(1.0, 5.0)
        self.buffer_spin.setSingleStep(0.5)
        self.buffer_spin.setValue(3.0)
        self.buffer_spin.setSuffix(" s")
        self.buffer_spin.setToolTip("Shorter = lower latency")
        form1.addRow("Buffer:", self.buffer_spin)

        layout.addLayout(form1)

        # ---- Checkboxes ----
        self.vad_check = QCheckBox("Enable VAD filter")
        self.vad_check.setChecked(True)
        self.vad_check.setToolTip("Off = faster, more noise")
        layout.addWidget(self.vad_check)

        self.initial_prompt_check = QCheckBox("Enable initial prompt")
        self.initial_prompt_check.setChecked(True)
        self.initial_prompt_check.setToolTip("Feed previous transcript as context")
        layout.addWidget(self.initial_prompt_check)

        layout.addWidget(self._separator())

        # ---- LLM section ----
        layout.addWidget(self._section_label("LLM Analysis"))

        self.llm_check = QCheckBox("Enable LLM analysis")
        self.llm_check.setChecked(True)
        self.llm_check.setToolTip("Send transcript to DeepSeek for key-point extraction")
        layout.addWidget(self.llm_check)

        form2 = QFormLayout()
        form2.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form2.setHorizontalSpacing(16)
        form2.setVerticalSpacing(10)

        self.llm_interval_spin = QSpinBox()
        self.llm_interval_spin.setRange(10, 300)
        self.llm_interval_spin.setValue(30)
        self.llm_interval_spin.setSuffix(" s")
        self.llm_interval_spin.setToolTip("How often to send transcript to LLM (seconds)")
        form2.addRow("LLM interval:", self.llm_interval_spin)
        layout.addLayout(form2)

        layout.addStretch()
        layout.addWidget(self._separator())

        # ---- Buttons ----
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #45475a;
                color: #cdd6f4;
                border: none;
                border-radius: 8px;
                padding: 8px 24px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #585b70;
            }
        """)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        apply_btn = QPushButton("Apply")
        apply_btn.setCursor(Qt.PointingHandCursor)
        apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                border: none;
                border-radius: 8px;
                padding: 8px 24px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #74c7ec;
            }
        """)
        apply_btn.clicked.connect(self._apply_and_close)
        btn_row.addWidget(apply_btn)

        layout.addLayout(btn_row)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setObjectName("separator")
        line.setFrameShape(QFrame.HLine)
        return line

    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("sectionLabel")
        return lbl

    # ------------------------------------------------------------------ config
    def _load_config(self):
        c = self._config
        self.model_combo.setCurrentText(c.get("model_size", "tiny"))
        self.device_combo.setCurrentText(c.get("device", "cpu"))
        self.compute_combo.setCurrentText(c.get("compute_type", "int8"))
        self.beam_spin.setValue(c.get("beam_size", 5))
        self.buffer_spin.setValue(c.get("buffer_duration", 7.0))
        self.buffer_spin.setRange(7.0, 30.0)
        self.vad_check.setChecked(c.get("vad_filter", True))
        self.initial_prompt_check.setChecked(c.get("use_initial_prompt", True))
        self.llm_check.setChecked(c.get("llm_analysis_enabled", True))
        self.llm_interval_spin.setValue(c.get("llm_analysis_interval", 30))

    def get_config(self) -> dict:
        return {
            "model_size": self.model_combo.currentText(),
            "device": self.device_combo.currentText(),
            "compute_type": self.compute_combo.currentText(),
            "beam_size": self.beam_spin.value(),
            "buffer_duration": self.buffer_spin.value(),
            "vad_filter": self.vad_check.isChecked(),
            "use_initial_prompt": self.initial_prompt_check.isChecked(),
            "llm_analysis_enabled": self.llm_check.isChecked(),
            "llm_analysis_interval": self.llm_interval_spin.value(),
        }

    def _apply_and_close(self):
        self.settings_changed.emit(self.get_config())
        self.accept()
