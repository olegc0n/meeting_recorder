"""
AI Meeting Transcriber - Main GUI Application (Modern UI)
"""

import sys
from datetime import datetime
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QTextEdit,
    QStatusBar,
    QMessageBox,
    QSplitter,
    QFrame,
    QSizePolicy,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont

from workers import (
    AudioWorker, TranscriberWorker, TranslationWorker, LLMAnalysisWorker,
    HEALTH_OK, HEALTH_WARN, HEALTH_CRITICAL,
)
from utils import get_output_devices, find_loopback_for_speaker
from config import load_transcription_config, save_transcription_config, load_audio_source, save_audio_source
from settings_dialog import SettingsDialog
from stats_panel import StatsPanel

# ---------------------------------------------------------------------------
#  Global application stylesheet - Catppuccin Mocha-inspired dark theme
# ---------------------------------------------------------------------------
APP_STYLESHEET = """
/* ---- Base ---- */
QMainWindow {
    background-color: #181825;
}
QWidget {
    font-family: "Segoe UI", "Noto Sans", "Cantarell", sans-serif;
}
QStatusBar {
    background-color: #11111b;
    color: #a6adc8;
    font-size: 12px;
    border-top: 1px solid #313244;
    padding: 2px 8px;
}

/* ---- Cards (QFrame#card) ---- */
QFrame#card {
    background-color: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 12px;
}

/* ---- Labels ---- */
QLabel {
    color: #cdd6f4;
}
QLabel#headerTitle {
    font-size: 20px;
    font-weight: 700;
    color: #cdd6f4;
}
QLabel#headerSubtitle {
    font-size: 12px;
    color: #a6adc8;
}
QLabel#cardTitle {
    font-size: 13px;
    font-weight: 600;
    color: #89b4fa;
}
QLabel#selectorLabel {
    font-size: 12px;
    color: #a6adc8;
    font-weight: 500;
}

/* ---- Combo boxes ---- */
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
QComboBox:disabled {
    background-color: #1e1e2e;
    color: #585b70;
    border-color: #313244;
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
    padding: 4px;
}

/* ---- TextEdits ---- */
QTextEdit {
    background-color: #11111b;
    color: #cdd6f4;
    border: none;
    border-radius: 8px;
    padding: 10px;
    font-size: 13px;
    selection-background-color: #45475a;
    selection-color: #cdd6f4;
}
QTextEdit[readOnly="true"] {
    background-color: #11111b;
}

/* ---- Scrollbars ---- */
QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #45475a;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #585b70;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background: transparent;
    height: 8px;
}
QScrollBar::handle:horizontal {
    background: #45475a;
    border-radius: 4px;
}

/* ---- Splitter ---- */
QSplitter::handle {
    background-color: #313244;
}
QSplitter::handle:horizontal {
    width: 2px;
}
QSplitter::handle:vertical {
    height: 2px;
}

/* ---- Tooltips ---- */
QToolTip {
    background-color: #1e1e2e;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 8px;
    padding: 8px 10px;
    font-size: 12px;
    opacity: 240;
}
"""


# ---------------------------------------------------------------------------
#  Health-indicator widget – coloured dots per monitored component
# ---------------------------------------------------------------------------
_HEALTH_COLORS = {
    HEALTH_OK: "#a6e3a1",       # green
    HEALTH_WARN: "#f9e2af",     # yellow
    HEALTH_CRITICAL: "#f38ba8", # red
}


# Per-component static metadata used to build rich tooltips.
_COMPONENT_META = {
    "whisper": {
        "label": "Whisper (STT)",
        "metric": "Real-Time Factor (RTF)",
        "description": "RTF = transcription time ÷ audio duration.\n"
                        "RTF < 1 means Whisper is faster than real-time.",
        "warn": f"RTF ≥ {0.8:.1f}  → model is slower than ideal",
        "crit": f"RTF ≥ {1.0:.1f}  → model can't keep up with audio (try a smaller model or GPU)",
    },
    "audio_queue": {
        "label": "Audio Queue",
        "metric": "Backlog (buffered chunks)",
        "description": "Number of audio chunks waiting to be transcribed.\n"
                        "High backlog means transcription is falling behind capture.",
        "warn": "Backlog ≥ 20 chunks → processing falling behind",
        "crit": "Backlog ≥ 50 chunks → severe latency, consider smaller model or faster device",
    },
    "llm": {
        "label": "LLM Analysis",
        "metric": "Response time (seconds)",
        "description": "How long the DeepSeek API took to return the meeting analysis.",
        "warn": "Response ≥ 15 s → API is slow",
        "crit": "Response ≥ 30 s → API timeout risk, check network / API key",
    },
}

_LEVEL_LABELS = {
    HEALTH_OK: ("OK", "#a6e3a1"),
    HEALTH_WARN: ("Warning", "#f9e2af"),
    HEALTH_CRITICAL: ("Critical", "#f38ba8"),
}


class _DotLabel(QLabel):
    """A small coloured circle + text label for one component."""

    def __init__(self, component_key: str, parent=None):
        super().__init__(parent)
        self._level = HEALTH_OK
        self._component_key = component_key
        meta = _COMPONENT_META.get(component_key, {})
        self._name = meta.get("label", component_key.replace("_", " ").title())
        self._detail = ""
        self._update_display()
        self.setCursor(Qt.PointingHandCursor)
        # Enable rich-text tooltips
        self.setToolTipDuration(8000)

    def set_health(self, level: str, detail: str = ""):
        if level == self._level and detail == self._detail:
            return
        self._level = level
        self._detail = detail
        self._update_display()

    @property
    def level(self) -> str:
        return self._level

    def _update_display(self):
        color = _HEALTH_COLORS.get(self._level, _HEALTH_COLORS[HEALTH_OK])
        self.setText(f'<span style="color:{color}; font-size:18px;">●</span>'
                     f'<span style="color:#a6adc8; font-size:11px;"> {self._name}</span>')
        self.setToolTip(self._build_tooltip())

    def _build_tooltip(self) -> str:
        meta = _COMPONENT_META.get(self._component_key, {})
        level_text, level_color = _LEVEL_LABELS.get(self._level, ("OK", "#a6e3a1"))

        current_value_row = ""
        if self._detail:
            current_value_row = (
                f'<tr><td style="color:#a6adc8;">Current&nbsp;value:</td>'
                f'<td style="color:#cdd6f4; font-weight:600;">&#160;{self._detail}</td></tr>'
            )

        return (
            f'<html><body style="font-family: sans-serif; font-size: 12px;">'
            f'<b style="font-size:13px; color:#cdd6f4;">{meta.get("label", self._name)}</b><br/>'
            f'<table cellpadding="2" cellspacing="0" style="margin-top:6px;">'
            f'<tr><td style="color:#a6adc8;">Metric:</td>'
            f'<td style="color:#cdd6f4;">&#160;{meta.get("metric", "—")}</td></tr>'
            f'{current_value_row}'
            f'<tr><td style="color:#a6adc8;">Status:</td>'
            f'<td style="color:{level_color}; font-weight:600;">&#160;{level_text}</td></tr>'
            f'</table>'
            f'<hr style="border:0; border-top:1px solid #45475a; margin:6px 0;"/>'
            f'<span style="color:#a6adc8; font-size:11px;">{meta.get("description", "").replace(chr(10), "<br/>")}</span><br/>'
            f'<span style="color:#f9e2af; font-size:11px;">⚠ {meta.get("warn", "")}</span><br/>'
            f'<span style="color:#f38ba8; font-size:11px;">✖ {meta.get("crit", "")}</span>'
            f'</body></html>'
        )


class HealthIndicator(QWidget):
    """Compact row of health dots shown in the top bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(10)

        self.dots: dict[str, _DotLabel] = {}
        for component in ("whisper", "audio_queue", "llm"):
            dot = _DotLabel(component)
            self.dots[component] = dot
            layout.addWidget(dot)

        # Overall summary label (shows worst level in words)
        self.summary_label = QLabel()
        self.summary_label.setStyleSheet("color: #a6adc8; font-size: 11px;")
        layout.addWidget(self.summary_label)
        self._update_summary()

        # Auto-decay timer: if no update arrives within 60s, fall back to OK
        self._decay_timer = QTimer(self)
        self._decay_timer.setInterval(60_000)
        self._decay_timer.timeout.connect(self._decay_to_ok)

    def update_component(self, component: str, level: str, detail: str = ""):
        dot = self.dots.get(component)
        if dot is None:
            return
        dot.set_health(level, detail)
        self._update_summary()
        self._decay_timer.start()  # restart decay countdown

    def reset(self):
        for dot in self.dots.values():
            dot.set_health(HEALTH_OK, "")
        self._update_summary()
        self._decay_timer.stop()

    def _decay_to_ok(self):
        """Reset all components to OK if nothing reported for a while."""
        for dot in self.dots.values():
            dot.set_health(HEALTH_OK, "")
        self._update_summary()

    def _update_summary(self):
        levels = [d.level for d in self.dots.values()]
        if HEALTH_CRITICAL in levels:
            self.summary_label.setText("Performance issue!")
            self.summary_label.setStyleSheet("color: #f38ba8; font-size: 11px; font-weight: 600;")
        elif HEALTH_WARN in levels:
            self.summary_label.setText("Degraded")
            self.summary_label.setStyleSheet("color: #f9e2af; font-size: 11px; font-weight: 600;")
        else:
            self.summary_label.setText("")
            self.summary_label.setStyleSheet("color: #a6adc8; font-size: 11px;")


# ---------------------------------------------------------------------------
#  Helper - icon-style toolbar button
# ---------------------------------------------------------------------------
def _icon_button(text: str, tooltip: str, size: int = 36) -> QPushButton:
    btn = QPushButton(text)
    btn.setToolTip(tooltip)
    btn.setFixedSize(size, size)
    btn.setCursor(Qt.PointingHandCursor)
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: #313244;
            color: #cdd6f4;
            border: none;
            border-radius: {size // 2}px;
            font-size: 16px;
        }}
        QPushButton:hover {{
            background-color: #45475a;
        }}
        QPushButton:disabled {{
            background-color: #1e1e2e;
            color: #585b70;
        }}
    """)
    return btn


def _action_button(text: str, color: str, hover: str, text_color: str = "#ffffff") -> QPushButton:
    btn = QPushButton(text)
    btn.setCursor(Qt.PointingHandCursor)
    btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {color};
            color: {text_color};
            font-weight: 600;
            font-size: 13px;
            padding: 10px 0;
            border: none;
            border-radius: 8px;
        }}
        QPushButton:hover {{
            background-color: {hover};
        }}
    """)
    return btn


def _card(inner_layout=None) -> QFrame:
    """Wrap *inner_layout* in a rounded card frame."""
    frame = QFrame()
    frame.setObjectName("card")
    if inner_layout is not None:
        frame.setLayout(inner_layout)
    return frame


# =========================================================================
#  Main Window
# =========================================================================
class MeetingTranscriberWindow(QMainWindow):
    """Main window for the AI Meeting Transcriber application."""

    def __init__(self):
        super().__init__()
        self.audio_worker: AudioWorker = None
        self.transcriber_worker: TranscriberWorker = None
        self.translation_worker: TranslationWorker = None
        self.llm_worker: LLMAnalysisWorker = None
        self.is_recording = False
        self.current_loopback_id = None
        self.stats_total = {"words": 0, "audio_s": 0, "latency_ms": 0, "chunk_count": 0}

        self._current_config = load_transcription_config()

        self._init_ui()
        self._load_devices()

    # ------------------------------------------------------------------
    #  UI Construction
    # ------------------------------------------------------------------
    def _init_ui(self):
        self.setWindowTitle("AI Meeting Transcriber")
        self.setMinimumSize(960, 620)
        self.resize(1080, 700)

        # Central widget & root layout
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 8)
        root_layout.setSpacing(10)

        # ---- Top bar ----
        root_layout.addLayout(self._build_top_bar())

        # ---- Selector row (device + language) ----
        root_layout.addLayout(self._build_selector_row())

        # ---- Main content area (splitter) ----
        root_layout.addWidget(self._build_content_area(), stretch=1)

        # ---- Bottom action buttons ----
        root_layout.addLayout(self._build_action_buttons())

        # ---- Status bar ----
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - model not loaded")

        # ---- Stats slide-out panel (lives inside central widget) ----
        self.stats_panel = StatsPanel(root)
        self.stats_panel.reposition()

    # ........................ top bar ........................
    def _build_top_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setContentsMargins(4, 0, 4, 0)

        # Branding
        title = QLabel("AI Meeting Transcriber")
        title.setObjectName("headerTitle")
        bar.addWidget(title)

        bar.addStretch()

        # Stats toggle
        self.stats_btn = _icon_button("\U0001F4CA", "Toggle statistics panel")
        self.stats_btn.clicked.connect(self._toggle_stats)
        bar.addWidget(self.stats_btn)

        # Health indicator (performance dots)
        self.health_indicator = HealthIndicator()
        bar.addWidget(self.health_indicator)

        # Settings gear
        self.settings_btn = _icon_button("\u2699", "Open settings")
        self.settings_btn.clicked.connect(self._open_settings)
        bar.addWidget(self.settings_btn)

        return bar

    # ........................ selectors ........................
    def _build_selector_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)

        # Audio source
        src_lbl = QLabel("Audio Source")
        src_lbl.setObjectName("selectorLabel")
        row.addWidget(src_lbl)

        self.device_combo = QComboBox()
        self.device_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        row.addWidget(self.device_combo, stretch=3)

        row.addSpacing(16)

        # Language
        lang_lbl = QLabel("Language")
        lang_lbl.setObjectName("selectorLabel")
        row.addWidget(lang_lbl)

        self.language_combo = QComboBox()
        self.language_combo.addItems(["en", "es", "fr", "de", "ja", "auto"])
        self.language_combo.setCurrentText("en")
        self.language_combo.setFixedWidth(100)
        row.addWidget(self.language_combo)

        return row

    # ........................ content area ........................
    def _build_content_area(self) -> QSplitter:
        # Outer horizontal splitter: left (transcription/translation) | right (LLM)
        self.hsplitter = QSplitter(Qt.Horizontal)

        # -- LEFT: vertical splitter for transcription + translation --
        left_splitter = QSplitter(Qt.Vertical)

        # Transcription card
        t_layout = QVBoxLayout()
        t_layout.setContentsMargins(14, 10, 14, 10)
        t_layout.setSpacing(6)
        t_header = QLabel("Transcription")
        t_header.setObjectName("cardTitle")
        t_layout.addWidget(t_header)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setPlaceholderText("Transcribed text will appear here...")
        t_layout.addWidget(self.text_edit)
        left_splitter.addWidget(_card(t_layout))

        # Translation card
        tr_layout = QVBoxLayout()
        tr_layout.setContentsMargins(14, 10, 14, 10)
        tr_layout.setSpacing(6)
        tr_header = QLabel("Translation (to English)")
        tr_header.setObjectName("cardTitle")
        tr_layout.addWidget(tr_header)
        self.translation_edit = QTextEdit()
        self.translation_edit.setReadOnly(True)
        self.translation_edit.setPlaceholderText("Translations appear when source language is not English...")
        tr_layout.addWidget(self.translation_edit)
        left_splitter.addWidget(_card(tr_layout))

        left_splitter.setSizes([420, 180])
        self.hsplitter.addWidget(left_splitter)

        # -- RIGHT: LLM analysis card --
        a_layout = QVBoxLayout()
        a_layout.setContentsMargins(14, 10, 14, 10)
        a_layout.setSpacing(6)
        a_header = QLabel("LLM Meeting Analysis")
        a_header.setObjectName("cardTitle")
        a_layout.addWidget(a_header)
        self.analysis_edit = QTextEdit()
        self.analysis_edit.setReadOnly(True)
        self.analysis_edit.setPlaceholderText("AI-generated key points will appear here...")
        a_layout.addWidget(self.analysis_edit)
        self.hsplitter.addWidget(_card(a_layout))

        self.hsplitter.setSizes([520, 440])
        return self.hsplitter

    # ........................ action buttons ........................
    def _build_action_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)

        self.start_stop_btn = _action_button(
            "\u25CF  START RECORDING", "#a6e3a1", "#94e2d5", "#1e1e2e"
        )
        self.start_stop_btn.setStyleSheet(self._rec_btn_style(recording=False))
        self.start_stop_btn.clicked.connect(self._toggle_recording)
        row.addWidget(self.start_stop_btn)

        self.clear_btn = _action_button("Clear Log", "#313244", "#45475a", "#cdd6f4")
        self.clear_btn.clicked.connect(self._clear_log)
        row.addWidget(self.clear_btn)

        self.clear_translation_btn = _action_button(
            "Clear Translation", "#313244", "#45475a", "#cdd6f4"
        )
        self.clear_translation_btn.clicked.connect(self._clear_translation)
        row.addWidget(self.clear_translation_btn)

        self.clear_analysis_btn = _action_button(
            "Clear Analysis", "#313244", "#45475a", "#cdd6f4"
        )
        self.clear_analysis_btn.clicked.connect(self._clear_analysis)
        row.addWidget(self.clear_analysis_btn)

        return row

    # ------------------------------------------------------------------
    #  Styling helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _rec_btn_style(recording: bool) -> str:
        if recording:
            return """
                QPushButton {
                    background-color: #f38ba8;
                    color: #1e1e2e;
                    font-weight: 700;
                    font-size: 13px;
                    padding: 10px 0;
                    border: none;
                    border-radius: 8px;
                }
                QPushButton:hover {
                    background-color: #eba0ac;
                }
            """
        return """
            QPushButton {
                background-color: #a6e3a1;
                color: #1e1e2e;
                font-weight: 700;
                font-size: 13px;
                padding: 10px 0;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #94e2d5;
            }
        """

    # ------------------------------------------------------------------
    #  Settings dialog
    # ------------------------------------------------------------------
    def _open_settings(self):
        dlg = SettingsDialog(self._current_config, self)
        dlg.settings_changed.connect(self._apply_settings)
        dlg.exec()

    def _apply_settings(self, cfg: dict):
        self._current_config = cfg
        save_transcription_config(cfg)

    # ------------------------------------------------------------------
    #  Statistics panel
    # ------------------------------------------------------------------
    def _toggle_stats(self):
        self.stats_panel.toggle()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "stats_panel"):
            self.stats_panel.reposition()

    def _reset_stats(self):
        self.stats_total = {"words": 0, "audio_s": 0, "latency_ms": 0, "chunk_count": 0}
        self.stats_panel.reset()

    def _on_llm_token_stats(self, token_stats: dict):
        """Update the statistics panel with LLM token usage and cost."""
        prompt = token_stats.get("prompt_tokens", 0)
        completion = token_stats.get("completion_tokens", 0)
        total = token_stats.get("total_tokens", 0)
        cost = token_stats.get("estimated_cost_usd", 0.0)
        self.stats_panel.update_stats(
            llm_prompt_tokens=f"{prompt:,}",
            llm_completion_tokens=f"{completion:,}",
            llm_total_tokens=f"{total:,}",
            llm_cost=f"${cost:.6f}",
        )

    def _on_stats_updated(self, stats: dict):
        self.stats_total["words"] += stats.get("words", 0)
        self.stats_total["audio_s"] += stats.get("audio_s", 0)
        self.stats_total["latency_ms"] += stats.get("latency_ms", 0)
        self.stats_total["chunk_count"] += stats.get("chunk_count", 0)

        n = self.stats_total["chunk_count"]
        avg = ""
        if n > 0 and self.stats_total["audio_s"] > 0:
            avg = f"{(self.stats_total['latency_ms'] / 1000) / self.stats_total['audio_s']:.3f}"

        self.stats_panel.update_stats(
            rtf=f"{stats.get('rtf', 0):.3f}",
            wps=f"{stats.get('wps', 0):.1f}",
            latency=f"{stats.get('latency_ms', 0):.0f} ms",
            avg_rtf=avg,
            total_words=str(self.stats_total["words"]),
            chunks=str(self.stats_total["chunk_count"]),
            model_load=f"{stats.get('model_load_ms', 0):.0f} ms",
        )

    # ------------------------------------------------------------------
    #  Device handling
    # ------------------------------------------------------------------
    def _load_devices(self):
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        devices = get_output_devices()
        if not devices:
            self.device_combo.blockSignals(False)
            self.device_combo.addItem("No devices found")
            QMessageBox.warning(self, "No Devices", "No audio output devices found.")
            return
        for d in devices:
            self.device_combo.addItem(d["name"], d["id"])

        # Restore last-used audio source
        saved_id = load_audio_source()
        if saved_id:
            for i in range(self.device_combo.count()):
                if self.device_combo.itemData(i) == saved_id:
                    self.device_combo.setCurrentIndex(i)
                    break

        self.device_combo.blockSignals(False)
        # Manually trigger the handler for the current selection
        self._on_device_changed(self.device_combo.currentIndex())

    def _on_device_changed(self, index: int):
        if index < 0:
            return
        device_id = self.device_combo.currentData()
        device_name = self.device_combo.currentText()
        if device_id:
            save_audio_source(device_id)
        loopback = find_loopback_for_speaker(device_name, device_id)
        if loopback:
            self.current_loopback_id = loopback[1]
            self.status_bar.showMessage(f"Selected: {device_name}  ->  Loopback: {loopback[0]}")
        else:
            self.current_loopback_id = None
            self.status_bar.showMessage(f"Warning: No loopback found for {device_name}")
            QMessageBox.warning(
                self,
                "Loopback Not Found",
                f"Could not find a loopback microphone for '{device_name}'.\n"
                "Recording may not work correctly.",
            )

    # ------------------------------------------------------------------
    #  Recording
    # ------------------------------------------------------------------
    def _toggle_recording(self):
        if not self.is_recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        if not self.current_loopback_id:
            QMessageBox.warning(self, "No Device", "Please select an audio source first.")
            return

        try:
            language = self.language_combo.currentText()
            cfg = self._current_config

            self.audio_worker = AudioWorker(self.current_loopback_id)
            self.audio_worker.error_occurred.connect(self._on_audio_error)
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
                use_initial_prompt=cfg["use_initial_prompt"],
            )
            self.transcriber_worker.new_text.connect(self._on_new_transcription)
            self.transcriber_worker.status_update.connect(self._on_status_update)
            self.transcriber_worker.error_occurred.connect(self._on_transcription_error)
            self.transcriber_worker.stats_updated.connect(self._on_stats_updated)
            self.transcriber_worker.performance_alert.connect(self._on_performance_alert)
            self.transcriber_worker.start()

            if language not in ["en", "auto"]:
                self.translation_worker = TranslationWorker(source_language=language)
                self.translation_worker.new_translation.connect(self._on_new_translation)
                self.translation_worker.error_occurred.connect(self._on_translation_error)
                self.translation_worker.status_update.connect(self._on_status_update)
                self.translation_worker.start()
            else:
                self.translation_worker = None

            if cfg.get("llm_analysis_enabled", True):
                self.llm_worker = LLMAnalysisWorker(
                    interval_s=cfg.get("llm_analysis_interval", 30),
                )
                self.llm_worker.new_analysis.connect(self._on_new_analysis)
                self.llm_worker.error_occurred.connect(self._on_llm_error)
                self.llm_worker.status_update.connect(self._on_status_update)
                self.llm_worker.performance_alert.connect(self._on_performance_alert)
                self.llm_worker.token_stats_updated.connect(self._on_llm_token_stats)
                self.llm_worker.start()
            else:
                self.llm_worker = None

            self._reset_stats()
            self.health_indicator.reset()

            # Update UI state
            self.is_recording = True
            self.start_stop_btn.setText("\u25A0  STOP RECORDING")
            self.start_stop_btn.setStyleSheet(self._rec_btn_style(recording=True))
            self.device_combo.setEnabled(False)
            self.language_combo.setEnabled(False)
            self.settings_btn.setEnabled(False)
            self.status_bar.showMessage("Recording...")

            timestamp = datetime.now().strftime("%H:%M:%S")
            self._append_to_log(f"[{timestamp}] Meeting started...")

        except Exception as e:
            QMessageBox.critical(self, "Recording Error", f"Failed to start recording:\n{e}")
            self._stop_recording()

    def _stop_recording(self):
        self.is_recording = False

        for worker, timeout in [
            (self.audio_worker, 3000),
            (self.transcriber_worker, 3000),
            (self.translation_worker, 3000),
            (self.llm_worker, 5000),
        ]:
            if worker:
                worker.stop()
                worker.wait(timeout)
                if worker.isRunning():
                    worker.terminate()

        self.audio_worker = None
        self.transcriber_worker = None
        self.translation_worker = None
        self.llm_worker = None

        self.start_stop_btn.setText("\u25CF  START RECORDING")
        self.start_stop_btn.setStyleSheet(self._rec_btn_style(recording=False))
        self.device_combo.setEnabled(True)
        self.language_combo.setEnabled(True)
        self.settings_btn.setEnabled(True)
        self.status_bar.showMessage("Stopped")

        timestamp = datetime.now().strftime("%H:%M:%S")
        self._append_to_log(f"[{timestamp}] Recording stopped.\n")

    # ------------------------------------------------------------------
    #  Slots
    # ------------------------------------------------------------------
    def _on_new_transcription(self, text: str, chunk_start: str, chunk_end: str, t_received: str):
        self._append_to_log(text, chunk_start, chunk_end, t_received)
        if self.translation_worker and self.translation_worker.isRunning():
            self.translation_worker.add_text(text)
        if self.llm_worker and self.llm_worker.isRunning():
            self.llm_worker.add_text(text)

    def _append_to_log(self, text: str, chunk_start: str = "", chunk_end: str = "", t_received: str = ""):
        if chunk_start and chunk_end and t_received:
            html = (
                f'<span style="color:#585b70; font-size:11px;">'
                f'{chunk_start} – {chunk_end}'
                f'</span><br/>'
                f'<span style="color:#cdd6f4; font-size:14px;">{text}</span>'
            )
        else:
            # System messages (start/stop) — styled as dim italic
            html = f'<span style="color:#585b70; font-size:12px; font-style:italic;">{text}</span>'
        self.text_edit.append(html)
        sb = self.text_edit.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_new_translation(self, text: str):
        self.translation_edit.append(text)
        sb = self.translation_edit.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_new_analysis(self, text: str):
        self.analysis_edit.setMarkdown(text)

    def _clear_log(self):
        self.text_edit.clear()

    def _clear_translation(self):
        self.translation_edit.clear()

    def _clear_analysis(self):
        self.analysis_edit.clear()
        if self.llm_worker:
            self.llm_worker.clear()

    def _on_status_update(self, status: str):
        prefix = "Recording..." if self.is_recording else "Ready"
        self.status_bar.showMessage(f"{prefix}  ({status})")

    def _on_audio_error(self, error_msg: str):
        self.health_indicator.update_component(
            "audio_queue", HEALTH_CRITICAL, f"Audio error: {error_msg}"
        )
        QMessageBox.critical(self, "Audio Error", error_msg)
        self._stop_recording()

    def _on_transcription_error(self, error_msg: str):
        self.health_indicator.update_component(
            "whisper", HEALTH_CRITICAL, f"Transcription error: {error_msg}"
        )
        QMessageBox.critical(self, "Transcription Error", error_msg)

    def _on_translation_error(self, error_msg: str):
        print(f"Translation error: {error_msg}")
        self.status_bar.showMessage(f"Translation error: {error_msg}")

    def _on_llm_error(self, error_msg: str):
        self.health_indicator.update_component(
            "llm", HEALTH_CRITICAL, f"LLM error: {error_msg}"
        )
        print(f"LLM error: {error_msg}")
        self.status_bar.showMessage(f"LLM error: {error_msg}")

    def _on_performance_alert(self, alert: dict):
        """Handle performance health updates from worker threads."""
        component = alert.get("component", "")
        level = alert.get("level", HEALTH_OK)
        detail = alert.get("detail", "")
        self.health_indicator.update_component(component, level, detail)

    # ------------------------------------------------------------------
    #  Lifecycle
    # ------------------------------------------------------------------
    def closeEvent(self, event):
        if self.is_recording:
            self._stop_recording()
        event.accept()


# ======================================================================
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Apply global stylesheet
    app.setStyleSheet(APP_STYLESHEET)

    window = MeetingTranscriberWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
