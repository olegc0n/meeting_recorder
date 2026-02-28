"""
Slide-out Statistics Panel for AI Meeting Transcriber.
"""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFormLayout,
    QFrame,
    QGraphicsOpacityEffect,
)
from PySide6.QtCore import (
    Qt,
    QPropertyAnimation,
    QEasingCurve,
    QParallelAnimationGroup,
    Property,
    QSize,
)
from PySide6.QtGui import QColor


PANEL_STYLE = """
QWidget#statsPanel {
    background-color: rgba(30, 30, 46, 230);
    border-left: 1px solid #313244;
}
QLabel {
    color: #cdd6f4;
    font-size: 13px;
}
QLabel#statsPanelTitle {
    color: #cdd6f4;
    font-size: 15px;
    font-weight: 700;
}
QLabel#statsValue {
    color: #a6e3a1;
    font-size: 14px;
    font-weight: 600;
    font-family: "JetBrains Mono", "Fira Code", monospace;
}
QLabel#statsLabel {
    color: #a6adc8;
    font-size: 12px;
}
QFrame#statsSeparator {
    background-color: #313244;
    max-height: 1px;
}
"""


class StatValue(QWidget):
    """A single statistic: label on top, large value below."""

    def __init__(self, label_text: str, tooltip: str = "", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        lbl = QLabel(label_text)
        lbl.setObjectName("statsLabel")
        lbl.setToolTip(tooltip)
        layout.addWidget(lbl)

        self.value_label = QLabel("—")
        self.value_label.setObjectName("statsValue")
        layout.addWidget(self.value_label)

    def set_value(self, text: str):
        self.value_label.setText(text)


class StatsPanel(QWidget):
    """
    A slide-out panel anchored to the right edge of its parent.
    Call toggle() to animate it in/out.
    """

    PANEL_WIDTH = 220

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("statsPanel")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAutoFillBackground(True)
        self.setStyleSheet(PANEL_STYLE)
        self.setFixedWidth(self.PANEL_WIDTH)
        self._visible = False

        self._build_ui()

        # Start hidden (off-screen to the right)
        self.hide()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        # Header
        header = QHBoxLayout()
        title = QLabel("Statistics")
        title.setObjectName("statsPanelTitle")
        header.addWidget(title)
        header.addStretch()

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(28, 28)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #a6adc8;
                font-size: 16px;
                border: none;
                border-radius: 14px;
            }
            QPushButton:hover {
                background-color: #45475a;
                color: #f38ba8;
            }
        """)
        close_btn.clicked.connect(self.slide_out)
        header.addWidget(close_btn)
        layout.addLayout(header)

        sep = QFrame()
        sep.setObjectName("statsSeparator")
        sep.setFrameShape(QFrame.HLine)
        layout.addWidget(sep)

        # Stats grid (2-col when space allows, but vertical list is cleaner)
        self.rtf_stat = StatValue("RTF (last)", "Real-time factor (< 1 = faster than real-time)")
        layout.addWidget(self.rtf_stat)

        self.wps_stat = StatValue("WPS (last)", "Words per second")
        layout.addWidget(self.wps_stat)

        self.latency_stat = StatValue("Latency (last)", "Milliseconds to transcribe last chunk")
        layout.addWidget(self.latency_stat)

        sep2 = QFrame()
        sep2.setObjectName("statsSeparator")
        sep2.setFrameShape(QFrame.HLine)
        layout.addWidget(sep2)

        self.avg_rtf_stat = StatValue("Average RTF")
        layout.addWidget(self.avg_rtf_stat)

        self.total_words_stat = StatValue("Total Words")
        layout.addWidget(self.total_words_stat)

        self.chunks_stat = StatValue("Chunks")
        layout.addWidget(self.chunks_stat)

        self.model_load_stat = StatValue("Model Load")
        layout.addWidget(self.model_load_stat)

        layout.addStretch()

    # ---- animation helpers ----
    def slide_in(self):
        if self._visible:
            return
        self._visible = True
        self.show()
        self.raise_()
        self._animate_position(visible=True)

    def slide_out(self):
        if not self._visible:
            return
        self._visible = False
        self._animate_position(visible=False)

    def toggle(self):
        if self._visible:
            self.slide_out()
        else:
            self.slide_in()

    @property
    def is_visible(self) -> bool:
        return self._visible

    def _animate_position(self, visible: bool):
        parent = self.parentWidget()
        if parent is None:
            return
        h = parent.height()
        self.setFixedHeight(h)

        end_x = parent.width() - self.PANEL_WIDTH if visible else parent.width()
        start_x = self.x()

        anim = QPropertyAnimation(self, b"pos", self)
        anim.setDuration(250)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        from PySide6.QtCore import QPoint
        anim.setStartValue(QPoint(start_x, 0))
        anim.setEndValue(QPoint(end_x, 0))
        if not visible:
            anim.finished.connect(self.hide)
        anim.start(QPropertyAnimation.DeleteWhenStopped)

    def reposition(self):
        """Call from parent resizeEvent to keep panel anchored."""
        parent = self.parentWidget()
        if parent is None:
            return
        self.setFixedHeight(parent.height())
        if self._visible:
            from PySide6.QtCore import QPoint
            self.move(parent.width() - self.PANEL_WIDTH, 0)
        else:
            from PySide6.QtCore import QPoint
            self.move(parent.width(), 0)

    # ---- public update API ----
    def reset(self):
        for w in (
            self.rtf_stat,
            self.wps_stat,
            self.latency_stat,
            self.avg_rtf_stat,
            self.total_words_stat,
            self.chunks_stat,
            self.model_load_stat,
        ):
            w.set_value("—")

    def update_stats(self, *, rtf="", wps="", latency="", avg_rtf="",
                     total_words="", chunks="", model_load=""):
        if rtf:
            self.rtf_stat.set_value(rtf)
        if wps:
            self.wps_stat.set_value(wps)
        if latency:
            self.latency_stat.set_value(latency)
        if avg_rtf:
            self.avg_rtf_stat.set_value(avg_rtf)
        if total_words:
            self.total_words_stat.set_value(total_words)
        if chunks:
            self.chunks_stat.set_value(chunks)
        if model_load:
            self.model_load_stat.set_value(model_load)
