"""SceneVision GUI — interface for the existing main.py + scene_analysis.py.

Keep this file beside the existing main.py and scene_analysis.py, then run:
    python 2GUI.py

This file is the GUI adapter only. It does not modify the processing modules.
"""
from __future__ import annotations

import math
import sys
import time
import json
from types import SimpleNamespace
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtCore import (
    QEvent, Qt, QThread, pyqtSignal, pyqtProperty, QUrl, QTimer, QSettings,
    QRectF, QPointF, QSize, QPropertyAnimation, QEasingCurve,
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QColor, QPen, QPainterPath
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QLineEdit, QComboBox,
    QDoubleSpinBox, QTextEdit, QMessageBox, QFormLayout,
    QListWidget, QListWidgetItem, QStackedWidget, QFrame,
    QProgressBar, QSlider, QTextBrowser, QProgressDialog,
    QSpinBox, QSizePolicy, QSpacerItem, QDialog, QScrollArea, QAbstractButton,
)

import main as pipeline
from main import (
    load_model, detect_all, make_record, load_detectors, make_analyzer,
    open_video, open_camera, open_rtsp, video_info, live_info,
    LatestFrameReader, iter_frames, save_result,
)


class SwitchControl(QAbstractButton):
    """Переключатель с подписью и анимированным ползунком.

    Подпись показывает ТЕКУЩЕЕ состояние (unchecked_text / checked_text).
    Нажимается только сам трек справа (подпись не кликабельна), если не
    передан label_clickable=True. Поддерживает клавиатуру (Space / Enter),
    светлую и тёмную темы и необязательную иконку на ползунке:
    icon="theme" рисует солнце / луну.
    """

    TRACK_W, TRACK_H = 44, 25
    KNOB_D, KNOB_PAD = 19, 3
    RIGHT_MARGIN = 2
    TEXT_GAP = 10

    # Испускается, когда ползунок доехал до конца (с итоговым состоянием).
    # Нужен для тяжёлых действий вроде смены темы: их запускают уже ПОСЛЕ
    # анимации, иначе перестройка стилей «съедает» кадры и ползунок дёргается.
    settled = pyqtSignal(bool)

    def __init__(self, unchecked_text, checked_text, icon=None,
                 label_clickable=False, parent=None):
        super().__init__(parent)
        self.unchecked_text = unchecked_text
        self.checked_text = checked_text
        self.icon_kind = icon
        self.label_clickable = bool(label_clickable)
        self.theme = "light"
        self._offset = 0.0          # 0.0 — выключен, 1.0 — включён
        self._hover = False
        self._focus_ring = False

        self.setCheckable(True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.ArrowCursor)
        self.setMinimumSize(150, 42)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self._anim = QPropertyAnimation(self, b"offset", self)
        self._anim.setDuration(170)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self.toggled.connect(lambda _checked: self._sync_knob(animated=True))
        self._anim.finished.connect(lambda: self.settled.emit(self.isChecked()))

    # -- анимируемое свойство ------------------------------------------------
    def _get_offset(self):
        return self._offset

    def _set_offset(self, value):
        self._offset = float(value)
        self.update()

    offset = pyqtProperty(float, fget=_get_offset, fset=_set_offset)

    def _sync_knob(self, animated):
        target = 1.0 if self.isChecked() else 0.0
        self._anim.stop()
        if animated and self.isVisible():
            self._anim.setStartValue(self._offset)
            self._anim.setEndValue(target)
            self._anim.start()
        else:
            self._set_offset(target)

    def setChecked(self, checked):
        # Программная установка (в т.ч. при заблокированных сигналах) не
        # вызывает toggled, поэтому синхронизируем ползунок вручную.
        super().setChecked(checked)
        self._sync_knob(animated=not self.signalsBlocked())

    def showEvent(self, event):
        super().showEvent(event)
        self._sync_knob(animated=False)

    # -- тема и геометрия -------------------------------------------------------
    def set_theme(self, theme):
        self.theme = theme if theme in ("light", "dark") else "light"
        self.update()

    def sizeHint(self):
        fm = self.fontMetrics()
        text_w = max(fm.horizontalAdvance(self.unchecked_text),
                     fm.horizontalAdvance(self.checked_text))
        width = text_w + self.TEXT_GAP + self.TRACK_W + self.RIGHT_MARGIN + 4
        return QSize(max(164, width), 42)

    def minimumSizeHint(self):
        return self.sizeHint()

    def _track_rect(self):
        return QRectF(
            self.width() - self.TRACK_W - self.RIGHT_MARGIN,
            (self.height() - self.TRACK_H) / 2.0,
            self.TRACK_W, self.TRACK_H,
        )

    def hitButton(self, pos):
        """Qt сам вызывает это при нажатии/отпускании мыши: клик засчитывается
        только по треку (с небольшим запасом), а не по подписи."""
        if self.label_clickable:
            return self.rect().contains(pos)
        return self._track_rect().adjusted(-4, -6, 4, 6).contains(QPointF(pos))

    # -- события ------------------------------------------------------------------
    def mouseMoveEvent(self, event):
        over = self.isEnabled() and self.hitButton(event.pos())
        self.setCursor(Qt.PointingHandCursor if over else Qt.ArrowCursor)
        if over != self._hover:
            self._hover = over
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self._hover = False
        self.setCursor(Qt.ArrowCursor)
        self.update()
        super().leaveEvent(event)

    def keyPressEvent(self, event):
        # Space обрабатывает сам QAbstractButton; добавляем Enter / Return.
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not event.isAutoRepeat():
            self.click()
            event.accept()
            return
        super().keyPressEvent(event)

    def focusInEvent(self, event):
        self._focus_ring = event.reason() in (
            Qt.TabFocusReason, Qt.BacktabFocusReason, Qt.ShortcutFocusReason)
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event):
        self._focus_ring = False
        super().focusOutEvent(event)
        self.update()

    # -- отрисовка ----------------------------------------------------------------
    @staticmethod
    def _mix(c1, c2, t):
        return QColor(
            int(c1.red() + (c2.red() - c1.red()) * t),
            int(c1.green() + (c2.green() - c1.green()) * t),
            int(c1.blue() + (c2.blue() - c1.blue()) * t),
        )

    def _draw_icon(self, painter, center, dark_icon):
        if self.icon_kind != "theme":
            return
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        if dark_icon:   # луна
            moon = QPainterPath()
            moon.addEllipse(center, 5.2, 5.2)
            cut = QPainterPath()
            cut.addEllipse(QPointF(center.x() + 2.8, center.y() - 2.0), 4.4, 4.4)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#3B4A66"))
            painter.drawPath(moon.subtracted(cut))
        else:           # солнце
            amber = QColor("#F2A93B")
            painter.setPen(Qt.NoPen)
            painter.setBrush(amber)
            painter.drawEllipse(center, 3.2, 3.2)
            pen = QPen(amber, 1.3)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            for i in range(8):
                angle = i * math.pi / 4
                dx, dy = math.cos(angle), math.sin(angle)
                painter.drawLine(
                    QPointF(center.x() + dx * 5.0, center.y() + dy * 5.0),
                    QPointF(center.x() + dx * 6.8, center.y() + dy * 6.8),
                )
        painter.restore()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        enabled = self.isEnabled()
        dark = self.theme == "dark"
        t = self._offset

        # Подпись: показывает текущее состояние.
        text = self.checked_text if self.isChecked() else self.unchecked_text
        fg = QColor(("#E7ECF5" if dark else "#20384E") if enabled else "#8A98A8")
        painter.setPen(fg)
        painter.setFont(self.font())
        text_rect = self.rect().adjusted(
            0, 0, -(self.TRACK_W + self.RIGHT_MARGIN + self.TEXT_GAP), 0)
        painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, text)

        # Трек: цвет плавно перетекает между «выкл» и «вкл».
        off_color = QColor("#3A4A60" if dark else "#B9C7D5")
        on_color = QColor("#3478F6" if dark else "#1769AA")
        if self._hover and enabled:
            off_color = off_color.lighter(108)
            on_color = on_color.lighter(108)
        track_color = self._mix(off_color, on_color, t)
        if not enabled:
            track_color.setAlpha(110)
        track = self._track_rect()
        radius = track.height() / 2.0

        painter.setPen(Qt.NoPen)
        painter.setBrush(track_color)
        painter.drawRoundedRect(track, radius, radius)

        if self._focus_ring and enabled:
            ring = QColor(on_color)
            ring.setAlpha(150)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(ring, 2))
            painter.drawRoundedRect(track.adjusted(-3, -3, 3, 3), radius + 3, radius + 3)
            painter.setPen(Qt.NoPen)

        # Ползунок.
        d = float(self.KNOB_D)
        knob_x = track.x() + self.KNOB_PAD + (track.width() - d - 2 * self.KNOB_PAD) * t
        knob_y = track.y() + (track.height() - d) / 2.0
        painter.setBrush(QColor(0, 0, 0, 38))               # мягкая тень
        painter.drawEllipse(QRectF(knob_x, knob_y + 1.0, d, d))
        painter.setBrush(QColor("#FFFFFF" if enabled else "#F1F4F8"))
        painter.drawEllipse(QRectF(knob_x, knob_y, d, d))
        self._draw_icon(painter, QPointF(knob_x + d / 2.0, knob_y + d / 2.0), self.isChecked())
        painter.end()


# ---------------------------------------------------------------------------
# Visual language
# ---------------------------------------------------------------------------

CLASS_RU = {
    "person": "человек", "bicycle": "велосипед", "car": "машина",
    "motorcycle": "мотоцикл", "airplane": "самолёт", "bus": "автобус",
    "train": "поезд", "truck": "грузовик", "boat": "лодка",
    "traffic light": "светофор", "fire hydrant": "гидрант",
    "stop sign": "знак стоп", "parking meter": "паркомат",
    "bench": "скамейка", "bird": "птица", "cat": "кошка",
    "dog": "собака", "horse": "лошадь", "sheep": "овца",
    "cow": "корова", "elephant": "слон", "bear": "медведь",
    "zebra": "зебра", "giraffe": "жираф", "backpack": "рюкзак",
    "umbrella": "зонт", "handbag": "сумка", "tie": "галстук",
    "suitcase": "чемодан", "frisbee": "фрисби", "skis": "лыжи",
    "snowboard": "сноуборд", "sports ball": "мяч", "kite": "воздушный змей",
    "baseball bat": "бейсбольная бита", "baseball glove": "бейсбольная перчатка",
    "skateboard": "скейтборд", "surfboard": "доска для сёрфинга",
    "tennis racket": "теннисная ракетка", "bottle": "бутылка",
    "wine glass": "бокал", "cup": "чашка", "fork": "вилка",
    "knife": "нож", "spoon": "ложка", "bowl": "миска",
    "banana": "банан", "apple": "яблоко", "sandwich": "сэндвич",
    "orange": "апельсин", "broccoli": "брокколи", "carrot": "морковь",
    "hot dog": "хот-дог", "pizza": "пицца", "donut": "пончик",
    "cake": "торт", "chair": "стул", "couch": "диван",
    "potted plant": "растение", "bed": "кровать", "dining table": "стол",
    "toilet": "туалет", "tv": "телевизор", "laptop": "ноутбук",
    "mouse": "мышь", "remote": "пульт", "keyboard": "клавиатура",
    "cell phone": "телефон", "microwave": "микроволновка", "oven": "духовка",
    "toaster": "тостер", "sink": "раковина", "refrigerator": "холодильник",
    "book": "книга", "clock": "часы", "vase": "ваза",
    "scissors": "ножницы", "teddy bear": "мишка", "hair drier": "фен",
    "toothbrush": "зубная щётка",
}

APP_STYLE = r"""
* {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 13px;
    color: #E7ECF5;
}
QMainWindow, QWidget#root, QWidget#page {
    background: #0B1018;
}
QFrame#sidebar {
    background: #0E1420;
    border-right: 1px solid #222D3D;
}
QFrame#brandMark {
    background: #3478F6;
    border: none;
    border-radius: 10px;
}
QLabel#brandTitle { font-size: 17px; font-weight: 700; color: #F4F7FC; }
QLabel#brandCaption { font-size: 9px; font-weight: 700; color: #70809A; letter-spacing: 1px; }
QLabel#eyebrow { font-size: 10px; font-weight: 700; color: #6F85A7; letter-spacing: 1.4px; }
QLabel#title { font-size: 27px; font-weight: 700; color: #F4F7FC; }
QLabel#subtitle { font-size: 13px; color: #91A0B5; }
QLabel#muted { color: #8291A8; }
QLabel#smallMuted { color: #78869B; font-size: 11px; }
QLabel#sectionTitle { font-size: 14px; font-weight: 650; color: #EDF2FA; }
QLabel#metricValue { font-size: 22px; font-weight: 700; color: #F5F8FD; }
QLabel#metricLabel { color: #8493A9; font-size: 11px; }
QLabel#pill {
    background: #172B47; color: #90BCFF; border: 1px solid #234675;
    border-radius: 10px; padding: 5px 9px; font-size: 10px; font-weight: 700;
}
QLabel#pillGreen {
    background: #132D2B; color: #67D9C3; border: 1px solid #205047;
    border-radius: 10px; padding: 5px 9px; font-size: 10px; font-weight: 700;
}
QLabel#pillAmber {
    background: #302718; color: #E9BC71; border: 1px solid #604A27;
    border-radius: 10px; padding: 5px 9px; font-size: 10px; font-weight: 700;
}
QFrame#card, QFrame#subCard {
    background: #111925; border: 1px solid #253144; border-radius: 12px;
}
QFrame#subCard { background: #0F1723; }
QFrame#dropZone {
    background: #0F1929; border: 1px dashed #365276; border-radius: 12px;
}
QFrame#dropZone[active="true"] { background: #14233A; border: 1px dashed #4E91FF; }
QPushButton {
    background: #1A2433; border: 1px solid #2B394D; border-radius: 8px;
    padding: 9px 14px; color: #D9E2F0; font-weight: 600;
}
QPushButton:hover { background: #243247; border-color: #405675; }
QPushButton:pressed { background: #172131; }
QPushButton:disabled { background: #151B25; color: #586579; border-color: #222B39; }
QPushButton#primaryBtn {
    background: #3478F6; color: white; border: 1px solid #3478F6;
    padding: 11px 18px; font-weight: 700;
}
QPushButton#primaryBtn:hover { background: #4688FF; border-color: #4688FF; }
QPushButton#primaryBtn:pressed { background: #2C67D7; }
QPushButton#navBtn {
    text-align: left; padding: 12px 13px; border: 1px solid transparent;
    border-radius: 8px; background: transparent; color: #93A0B4; font-weight: 600;
}
QPushButton#navBtn:hover { background: #151F2D; color: #E8EEF8; }
QPushButton#navBtn:checked {
    background: #182A43; color: #8BB9FF; border: 1px solid #25446E;
}
QPushButton#dangerBtn {
    background: #2A1A20; color: #F4A4A8; border: 1px solid #553039;
}
QPushButton#dangerBtn:hover { background: #3A2029; }
QPushButton#cleanVideoBtn {
    background: #121D2B; color: #B9C7D9; border: 1px solid #304057;
    padding: 10px 13px; font-weight: 650;
}
QPushButton#cleanVideoBtn:hover { background: #19283A; border-color: #47617E; }
QPushButton#cleanVideoBtn:checked {
    background: #0E2C2A; color: #6EE7C8; border: 1px solid #168A77;
}
QPushButton#cleanVideoBtn:checked:hover { background: #123B36; border-color: #23B49A; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    min-height: 24px;
    padding: 8px 10px;
    background: #0B121D; border: 1px solid #2A374A; border-radius: 7px;
    padding: 9px 10px; color: #E8EEF7; selection-background-color: #3478F6;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #4789F8;
}
QLineEdit::placeholder { color: #65758D; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView {
    background: #111925; color: #E7ECF5; border: 1px solid #2A374A;
    selection-background-color: #234776; outline: 0;
}
QTextEdit, QTextBrowser, QListWidget {
    background: #0B121D; border: 1px solid #263347; border-radius: 8px;
    color: #DDE6F3; padding: 8px; selection-background-color: #254B7D;
}
QTextBrowser { padding: 10px; }
QListWidget { padding: 5px; }
QListWidget::item { padding: 10px 9px; border-radius: 6px; }
QListWidget::item:hover { background: #162235; }
QListWidget::item:selected { background: #1A3150; color: #EAF2FF; }
QProgressBar {
    background: #202B3B; border: none; border-radius: 5px;
    height: 9px; text-align: center; color: transparent;
}
QProgressBar::chunk { background: #3478F6; border-radius: 5px; }
QSlider { background: transparent; }
QSlider::groove:horizontal { height: 5px; background: #29364A; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #3478F6; border-radius: 2px; }
QSlider::handle:horizontal {
    background: #E8F0FF; border: 2px solid #3478F6; width: 13px;
    height: 13px; margin: -5px 0; border-radius: 7px;
}
QScrollBar:vertical { background: transparent; width: 8px; margin: 0; }
QScrollBar::handle:vertical { background: #33435A; min-height: 24px; border-radius: 4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QStatusBar { background: #0E1420; color: #8796AC; border-top: 1px solid #222D3D; }
QStatusBar::item { border: none; }
QToolTip { background: #192536; color: #F0F4FA; border: 1px solid #3A4D68; padding: 5px; }
"""

# Светлая палитра в духе интерфейсов ELTEX EVI. Видео-превью остаётся тёмным,
# чтобы изображение с камеры не теряло контраст.
LIGHT_APP_STYLE = APP_STYLE
_LIGHT_COLORS = {
    "#E7ECF5": "#202A38", "#0B1018": "#F3F6FA", "#0E1420": "#FFFFFF",
    "#222D3D": "#DCE3EC", "#3478F6": "#1769AA", "#F4F7FC": "#172B3F",
    "#70809A": "#718096", "#6F85A7": "#61758C", "#91A0B5": "#617184",
    "#8291A8": "#64758A", "#78869B": "#718096", "#EDF2FA": "#26384B",
    "#F5F8FD": "#172B3F", "#8493A9": "#718096", "#172B47": "#E7F1FA",
    "#90BCFF": "#1769AA", "#234675": "#C5D9EA", "#132D2B": "#E5F4EF",
    "#67D9C3": "#16785F", "#205047": "#BFE4D8", "#302718": "#FFF4DE",
    "#E9BC71": "#95620B", "#604A27": "#EBD8AE", "#111925": "#FFFFFF",
    "#253144": "#DCE3EC", "#0F1723": "#F8FAFC", "#0F1929": "#F5F9FC",
    "#365276": "#9CB8D0", "#14233A": "#EAF3FA", "#4E91FF": "#1769AA",
    "#1A2433": "#F1F5F9", "#2B394D": "#CFD8E3", "#D9E2F0": "#26384B",
    "#243247": "#E5EDF5", "#405675": "#AFC1D2", "#172131": "#E1E8F0",
    "#151B25": "#E9EEF4", "#586579": "#98A5B5", "#222B39": "#D5DEE8",
    "#4688FF": "#0F5E9C", "#2C67D7": "#124F80", "#93A0B4": "#53677D",
    "#151F2D": "#EAF1F7", "#E8EEF8": "#1D344A", "#182A43": "#E3F0FA",
    "#8BB9FF": "#1769AA", "#25446E": "#C4DCEC", "#2A1A20": "#FCEBEC",
    "#F4A4A8": "#A33D48", "#553039": "#E9C4C8", "#3A2029": "#F7DDE0",
    "#121D2B": "#F7F9FC", "#B9C7D9": "#40566D", "#19283A": "#E9F1F8",
    "#47617E": "#AFC4D7", "#0E2C2A": "#E3F5EF", "#6EE7C8": "#16785F",
    "#168A77": "#A9DCCD", "#123B36": "#D4EEE5", "#23B49A": "#168A77",
    "#0B121D": "#FFFFFF", "#2A374A": "#CBD5E1", "#E8EEF7": "#26384B",
    "#4789F8": "#1769AA", "#65758D": "#8998A9", "#234776": "#D7EAF8",
    "#263347": "#D5DEE8", "#DDE6F3": "#26384B", "#254B7D": "#C9E0F2",
    "#162235": "#EAF1F7", "#1A3150": "#DCECF8", "#EAF2FF": "#1D344A",
    "#202B3B": "#E2E8F0", "#29364A": "#D2DCE7", "#E8F0FF": "#FFFFFF",
    "#33435A": "#B8C5D3", "#8796AC": "#64758A", "#192536": "#FFFFFF",
    "#F0F4FA": "#26384B", "#3A4D68": "#C7D2DE"
}
for _dark, _light in _LIGHT_COLORS.items():
    LIGHT_APP_STYLE = LIGHT_APP_STYLE.replace(_dark, _light)


def panel(object_name: str = "card") -> QFrame:
    frame = QFrame()
    frame.setObjectName(object_name)
    return frame


def label(text: str, object_name: str | None = None, parent=None) -> QLabel:
    widget = QLabel(text, parent)
    widget.setTextFormat(Qt.PlainText)
    if object_name:
        widget.setObjectName(object_name)
    if object_name in ("subtitle", "smallMuted"):
        widget.setWordWrap(True)
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    return widget


def page_heading(kicker: str, title: str, subtitle: str) -> QWidget:
    block = QWidget()
    layout = QVBoxLayout(block)
    layout.setContentsMargins(0, 0, 0, 5)
    layout.setSpacing(7)
    layout.addWidget(label(kicker.upper(), "eyebrow"))
    title_label = label(title, "title")
    title_label.setTextFormat(Qt.PlainText)
    title_label.setFont(QFont("Arial", 27, QFont.Bold))
    title_label.setMinimumHeight(40)
    title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    layout.addWidget(title_label)
    subtitle_label = label(subtitle, "subtitle")
    subtitle_label.setWordWrap(True)
    layout.addWidget(subtitle_label)
    return block


def set_card_layout(card: QFrame, margins=(20, 19, 20, 20), spacing=12) -> QVBoxLayout:
    layout = QVBoxLayout(card)
    layout.setContentsMargins(*margins)
    layout.setSpacing(spacing)
    return layout


def human_size(path: str) -> str:
    try:
        size = Path(path).stat().st_size
    except OSError:
        return "Размер неизвестен"
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if size < 1024 or unit == "ГБ":
            return f"{size:.1f} {unit}" if unit != "Б" else f"{size} {unit}"
        size /= 1024
    return "Размер неизвестен"


def get_font(font_size: int):
    font_paths = [
        "arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "DejaVuSans.ttf",
    ]
    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, font_size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def draw_objects_ru(frame, objects):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font_size = max(14, int(frame.shape[0] * 0.025))
    font = get_font(font_size)
    for obj in objects:
        x1, y1, x2, y2 = map(int, obj["bbox"])
        x1 = max(0, x1)
        y1 = max(0, y1)
        draw.rectangle([(x1, y1), (x2, y2)], outline=(51, 220, 159), width=2)
        class_name = CLASS_RU.get(obj["class"], obj["class"])
        text = f"{class_name} {obj['confidence']:.2f}"
        bounds = draw.textbbox((0, 0), text, font=font)
        text_w = bounds[2] - bounds[0]
        text_h = bounds[3] - bounds[1]
        top = max(0, y1 - text_h - 7)
        draw.rectangle([(x1, top), (x1 + text_w + 10, y1)], fill=(51, 220, 159))
        draw.text((x1 + 5, top), text, font=font, fill=(7, 18, 22))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def frame_to_pixmap(frame, target_size):
    if frame is None:
        return QPixmap()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    image = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(image).scaled(
        target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
    )


# ---------------------------------------------------------------------------
# Background model loading
# ---------------------------------------------------------------------------

class ModelLoadThread(QThread):
    loaded = pyqtSignal(object, str)
    failed = pyqtSignal(str)

    def __init__(self, weights: str, parent=None, accident_weights="", road_weights=""):
        super().__init__(parent)
        self.weights = weights
        self.accident_weights = accident_weights.strip() or None
        self.road_weights = road_weights.strip() or None

    def run(self):
        try:
            args = SimpleNamespace(
                weights=self.weights, accident_weights=self.accident_weights,
                road_weights=self.road_weights, accident_conf=0.4, road_conf=0.4,
                all_coco=False,
            )
            self.loaded.emit(load_detectors(args), self.weights)
        except Exception as exc:
            self.failed.emit(str(exc))


# ---------------------------------------------------------------------------
# Video processing worker
# ---------------------------------------------------------------------------

class VideoThread(QThread):
    frame_ready = pyqtSignal(object)
    record_ready = pyqtSignal(dict, object)
    progress_updated = pyqtSignal(int, str)
    stage_changed = pyqtSignal(int)
    failed = pyqtSignal(str)

    def __init__(self, source_type, source, detectors, conf=0.4, step=1.0,
                 weights="yolov8n.pt", accident_weights="", road_weights="", parent=None):
        super().__init__(parent)
        self.source_type, self.source = source_type, source
        self.detectors, self.conf, self.step = detectors, float(conf), max(float(step), 0.05)
        self.weights = weights
        self.accident_weights = accident_weights or None
        self.road_weights = road_weights or None
        self._running = False
        self.result_json = ""
        self.llm_json = ""
        self.analysis_data = {}

    def run(self):
        self._running = True
        cap = None
        try:
            self.stage_changed.emit(0)
            if self.source_type != "file":
                raise RuntimeError("В этой версии GUI интегрирован анализ видеофайлов. Камеру и RTSP подключим следующим этапом.")
            cap = open_video(self.source)
            info = video_info(cap)
            self.stage_changed.emit(1)
            analyzer_args = SimpleNamespace(no_analysis=False, lang="ru", track_gap=2.5)
            analyzer = make_analyzer(analyzer_args)
            records, totals = [], Counter()
            duration = info.get("duration_sec") or 0
            total_steps = max(1, math.ceil(duration / self.step))
            for index, timestamp, frame, analyze in iter_frames(cap, info["fps"], self.step):
                if not self._running or self.isInterruptionRequested():
                    break
                if not analyze:
                    continue
                objects, events, _ = detect_all(self.detectors, frame, self.conf)
                record = make_record(index, timestamp, objects, events)
                if analyzer is not None:
                    record = analyzer.process(record, size=(info["width"], info["height"]))
                records.append(record)
                totals.update(record.get("summary", {}))
                annotated = pipeline.draw_objects(frame.copy(), objects, events)
                self.record_ready.emit(record, annotated.copy())
                self.frame_ready.emit(annotated)
                progress = min(99, int(len(records) * 100 / total_steps))
                summary = ", ".join(f"{CLASS_RU.get(k, k)}: {v}" for k, v in record.get("summary", {}).items()) or "объектов нет"
                self.progress_updated.emit(progress, f"{timestamp:07.2f} с  ·  {summary}")
                self.stage_changed.emit(2)
                self.msleep(5)
            cap.release()
            cap = None
            if records:
                output = str(Path(self.source).with_name(Path(self.source).stem + "_result.json"))
                args = SimpleNamespace(
                    output=output, step=self.step, weights=self.weights,
                    conf=self.conf, accident_weights=self.accident_weights,
                    accident_conf=self.detectors.accident_conf,
                    road_weights=self.road_weights,
                    road_conf=self.detectors.road_conf, lang="ru", digest_every=5.0,
                    digest_events=120,
                )
                self.stage_changed.emit(3)
                save_result(args, Path(self.source).name, "file", info, records, totals, analyzer)
                self.result_json = output
                self.llm_json = str(Path(output).with_name(Path(output).stem + "_llm.json"))
                try:
                    self.analysis_data = json.loads(Path(output).read_text(encoding="utf-8"))
                except Exception:
                    self.analysis_data = {}
                self.progress_updated.emit(100, f"Готово. JSON: {Path(output).name}")
        except Exception as exc:
            self.failed.emit(str(exc))
            self.progress_updated.emit(0, f"Ошибка: {exc}")
        finally:
            if cap is not None:
                cap.release()

    def stop(self):
        self._running = False
        self.requestInterruption()
        self.wait(10000)


class ExportThread(QThread):
    progress = pyqtSignal(int)
    export_completed = pyqtSignal(str)
    error = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, video_path, records, output_path, fps, total_frames, parent=None, draw_boxes=True):
        super().__init__(parent)
        self.video_path = video_path
        self.records = records
        self.output_path = output_path
        self.fps = fps if fps and math.isfinite(fps) else 25.0
        self.total_frames = max(1, int(total_frames or 1))
        self.draw_boxes = bool(draw_boxes)

    def run(self):
        cap = None
        out = None
        completed = False
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError("Не удалось повторно открыть исходное видео.")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width <= 0 or height <= 0:
                raise RuntimeError("Не удалось определить размер исходного видео.")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))
            if not out.isOpened():
                raise RuntimeError("Не удалось создать выходной видеофайл.")

            record_map = {r["frame_index"]: r.get("objects", []) for r in self.records}
            last_objects = []
            frame_idx = 0
            while cap.isOpened():
                if self.isInterruptionRequested():
                    self.cancelled.emit()
                    return
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx in record_map:
                    last_objects = record_map[frame_idx]
                if self.draw_boxes and last_objects:
                    frame = draw_objects_ru(frame, last_objects)
                out.write(frame)
                frame_idx += 1
                if frame_idx % 10 == 0:
                    self.progress.emit(min(100, int(frame_idx * 100 / self.total_frames)))
            self.progress.emit(100)
            completed = True
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()
            if not completed and self.isInterruptionRequested():
                try:
                    Path(self.output_path).unlink(missing_ok=True)
                except OSError:
                    pass
        if completed:
            self.export_completed.emit(self.output_path)


# ---------------------------------------------------------------------------
# Screen: New analysis
# ---------------------------------------------------------------------------

class PageNewAnalysis(QWidget):
    start_analysis = pyqtSignal(dict)
    SUPPORTED_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

    def __init__(self):
        super().__init__()
        self.file_path = ""
        self.setAcceptDrops(True)
        root = QVBoxLayout(self)
        root.setContentsMargins(34, 30, 34, 26)
        root.setSpacing(22)
        root.addWidget(page_heading(
            "Рабочая область / 01",
            "Новый анализ",
            "Загрузите дорожную запись и задайте параметры обработки.",
        ))

        source_row = QHBoxLayout()
        source_row.setSpacing(18)

        self.drop_zone = panel("dropZone")
        self.drop_zone.setProperty("active", False)
        self.drop_zone.setMinimumHeight(280)
        self.drop_zone.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        drop_layout = QVBoxLayout(self.drop_zone)
        drop_layout.setContentsMargins(28, 28, 28, 28)
        drop_layout.setSpacing(12)
        drop_layout.setAlignment(Qt.AlignCenter)

        self.upload_mark = QLabel("↑")
        self.upload_mark.setAlignment(Qt.AlignCenter)
        self.upload_mark.setFixedSize(56, 56)
        self.upload_mark.setStyleSheet(
            "background:#172A43; border:1px solid #294B76; border-radius:16px; "
            "color:#82B5FF; font-size:28px; font-weight:600;"
        )
        drop_layout.addWidget(self.upload_mark, alignment=Qt.AlignCenter)

        upload_title = label("Перетащите видео сюда", "sectionTitle")
        upload_title.setAlignment(Qt.AlignCenter)
        upload_title.setWordWrap(False)
        upload_title.setFixedHeight(26)
        drop_layout.addWidget(upload_title, alignment=Qt.AlignCenter)

        desc = label("или выберите файл на компьютере", "subtitle")
        desc.setAlignment(Qt.AlignCenter)
        desc.setWordWrap(False)
        desc.setFixedHeight(28)
        desc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        drop_layout.addWidget(desc, alignment=Qt.AlignCenter)

        formats = label("MP4  ·  AVI  ·  MOV  ·  MKV", "smallMuted")
        formats.setAlignment(Qt.AlignCenter)
        formats.setWordWrap(False)
        formats.setFixedHeight(22)
        formats.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        drop_layout.addWidget(formats, alignment=Qt.AlignCenter)
        self.btn_browse = QPushButton("Выбрать видеофайл")
        self.btn_browse.setObjectName("primaryBtn")
        self.btn_browse.setCursor(Qt.PointingHandCursor)
        self.btn_browse.clicked.connect(self._browse_file)
        drop_layout.addWidget(self.btn_browse, alignment=Qt.AlignCenter)
        source_row.addWidget(self.drop_zone, 6)

        side_stack = QVBoxLayout()
        side_stack.setSpacing(18)
        self.file_card = panel()
        file_layout = set_card_layout(self.file_card)
        top = QHBoxLayout()
        top.addWidget(label("Выбранный файл", "sectionTitle"))
        top.addStretch()
        self.file_pill = label("НЕТ ФАЙЛА", "pill")
        top.addWidget(self.file_pill)
        file_layout.addLayout(top)

        file_info = QHBoxLayout()
        file_info.setSpacing(12)
        self.file_icon = QLabel("MP4")
        self.file_icon.setAlignment(Qt.AlignCenter)
        self.file_icon.setFixedSize(50, 50)
        self.file_icon.setStyleSheet(
            "background:#19263A; border:1px solid #2B405E; border-radius:10px; "
            "color:#A6C8FF; font-size:12px; font-weight:800;"
        )
        file_info.addWidget(self.file_icon, alignment=Qt.AlignTop)
        name_col = QVBoxLayout()
        self.lbl_filename = label("Файл не выбран", "sectionTitle")
        self.lbl_filename.setWordWrap(True)
        self.lbl_filemeta = label("Здесь появятся сведения о файле", "smallMuted")
        self.lbl_filemeta.setWordWrap(True)
        name_col.addWidget(self.lbl_filename)
        name_col.addWidget(self.lbl_filemeta)
        name_col.addStretch()
        file_info.addLayout(name_col, 1)
        file_layout.addLayout(file_info)
        self.btn_remove = QPushButton("Убрать выбранный файл")
        self.btn_remove.clicked.connect(self._remove_file)
        self.btn_remove.setEnabled(False)
        file_layout.addWidget(self.btn_remove)
        side_stack.addWidget(self.file_card)

        # Compact processing-settings card: keep the content grouped at the top
        # instead of letting the widgets spread out across the full column.
        params_card = panel()
        params_card.setMinimumWidth(0)
        params_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        params_card.setMinimumHeight(205)
        params_card.setMaximumHeight(235)
        params_card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        params_layout = set_card_layout(
            params_card, margins=(18, 16, 18, 16), spacing=8
        )
        params_layout.setAlignment(Qt.AlignTop)
        params_layout.addWidget(label("Параметры обработки", "sectionTitle"))
        params_layout.addWidget(label(
            "Выберите частоту анализа кадров.", "smallMuted"
        ))

        step_row = QHBoxLayout()
        step_row.setContentsMargins(0, 4, 0, 2)
        step_row.setSpacing(12)
        step_label = label("Шаг анализа", "subtitle")
        step_row.addWidget(step_label)
        step_row.addStretch(1)

        self.combo_step = QComboBox()
        self.combo_step.addItems(["0.5 секунды", "1 секунда", "2 секунды", "5 секунд"])
        self.combo_step.setCurrentIndex(1)
        self.combo_step.setMinimumWidth(145)
        self.combo_step.setMaximumWidth(170)
        self.combo_step.setMinimumHeight(39)
        self.combo_step.setToolTip(
            "Как часто выбирать кадр для детекции. Меньший интервал даёт больше деталей, но увеличивает время анализа."
        )
        step_row.addWidget(self.combo_step)
        params_layout.addLayout(step_row)

        self.step_hint = label("Баланс подробности и скорости обработки.", "smallMuted")
        self.step_hint.setWordWrap(True)
        params_layout.addWidget(self.step_hint)
        self.combo_step.currentIndexChanged.connect(self._update_step_hint)
        self._update_step_hint(self.combo_step.currentIndex())

        side_stack.addWidget(params_card)
        side_stack.addStretch(1)
        source_row.addLayout(side_stack, 5)
        root.addLayout(source_row, 1)

        bottom = QHBoxLayout()
        info_note = label("Этап 1  ·  Детекция объектов и формирование данных", "smallMuted")
        bottom.addWidget(info_note)
        bottom.addStretch()
        self.btn_start = QPushButton("Запустить анализ  →")
        self.btn_start.setObjectName("primaryBtn")
        self.btn_start.setMinimumWidth(190)
        self.btn_start.setMinimumHeight(44)
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self._on_start)
        bottom.addWidget(self.btn_start)
        root.addLayout(bottom)

    def set_theme(self, theme):
        light = theme == "light"
        if light:
            self.upload_mark.setStyleSheet(
                "background:#E7F1FA; border:1px solid #BFD5E8; border-radius:16px; "
                "color:#006FB3; font-size:28px; font-weight:600;"
            )
            self.file_icon.setStyleSheet(
                "background:#EAF2F8; border:1px solid #CBD9E6; border-radius:10px; "
                "color:#24506F; font-size:12px; font-weight:800;"
            )
        else:
            self.upload_mark.setStyleSheet(
                "background:#172A43; border:1px solid #294B76; border-radius:16px; "
                "color:#82B5FF; font-size:28px; font-weight:600;"
            )
            self.file_icon.setStyleSheet(
                "background:#19263A; border:1px solid #2B405E; border-radius:10px; "
                "color:#A6C8FF; font-size:12px; font-weight:800;"
            )

    def _update_step_hint(self, index=None):
        """Show a short, useful explanation for the selected sampling interval."""
        hints = {
            0: "0,5 с — больше кадров и деталей, но обработка займёт больше времени.",
            1: "1 с — сбалансированный вариант для обычного анализа.",
            2: "2 с — быстрее, но короткие события могут остаться между кадрами.",
            3: "5 с — самый быстрый режим; подходит для обзорного анализа длинных видео.",
        }
        if hasattr(self, "step_hint"):
            self.step_hint.setText(hints.get(self.combo_step.currentIndex(), hints[1]))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.drop_zone.setProperty("active", True)
            self.drop_zone.style().unpolish(self.drop_zone)
            self.drop_zone.style().polish(self.drop_zone)

    def dragLeaveEvent(self, event):
        self.drop_zone.setProperty("active", False)
        self.drop_zone.style().unpolish(self.drop_zone)
        self.drop_zone.style().polish(self.drop_zone)

    def dropEvent(self, event):
        self.drop_zone.setProperty("active", False)
        self.drop_zone.style().unpolish(self.drop_zone)
        self.drop_zone.style().polish(self.drop_zone)
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(self.SUPPORTED_EXTENSIONS):
                self._set_file(path)
                event.acceptProposedAction()
                return
        QMessageBox.warning(self, "Формат файла", "Выберите видео в формате MP4, AVI, MOV или MKV.")

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите видео", "", "Видео (*.mp4 *.avi *.mov *.mkv)"
        )
        if path:
            self._set_file(path)

    def _set_file(self, path: str):
        if not Path(path).is_file():
            QMessageBox.warning(self, "Файл недоступен", "Не удалось найти выбранный файл.")
            return
        self.file_path = path
        suffix = Path(path).suffix.upper().replace(".", "") or "VIDEO"
        self.file_icon.setText(suffix[:4])
        self.lbl_filename.setText(Path(path).name)
        self.lbl_filemeta.setText(human_size(path))
        self.file_pill.setText("ГОТОВ К АНАЛИЗУ")
        self.file_pill.setObjectName("pillGreen")
        self.file_pill.style().unpolish(self.file_pill)
        self.file_pill.style().polish(self.file_pill)
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            info = video_info(cap)
            duration = info.get("duration_sec")
            details = f"{info['width']} × {info['height']}"
            if info.get("fps"):
                details += f"  ·  {info['fps']:.1f} FPS"
            if duration is not None:
                details += f"  ·  {int(duration // 60):02d}:{int(duration % 60):02d}"
            self.lbl_filemeta.setText(f"{human_size(path)}  ·  {details}")
        cap.release()
        self.btn_remove.setEnabled(True)
        self.btn_start.setEnabled(True)

    def _remove_file(self):
        self.file_path = ""
        self.lbl_filename.setText("Файл не выбран")
        self.lbl_filemeta.setText("Здесь появятся сведения о файле")
        self.file_icon.setText("MP4")
        self.file_pill.setText("НЕТ ФАЙЛА")
        self.file_pill.setObjectName("pill")
        self.file_pill.style().unpolish(self.file_pill)
        self.file_pill.style().polish(self.file_pill)
        self.btn_remove.setEnabled(False)
        self.btn_start.setEnabled(False)

    def _on_start(self):
        if not self.file_path:
            QMessageBox.warning(self, "Видео не выбрано", "Сначала выберите видеофайл.")
            return
        step_value = float(self.combo_step.currentText().split()[0])
        self.start_analysis.emit({
            "source_type": "file",
            "source": self.file_path,
            "step": step_value,
        })


# ---------------------------------------------------------------------------
# Screen: Processing
# ---------------------------------------------------------------------------

class PageProcessing(QWidget):
    stop_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        # Фиксированная рабочая область без прокрутки страницы.
        # Высота распределяется между прогрессом, предпросмотром, этапами и журналом.
        root = QVBoxLayout(self)
        root.setContentsMargins(26, 18, 26, 14)
        root.setSpacing(12)
        heading_row = QHBoxLayout()
        heading_row.addWidget(page_heading(
            "Рабочая область / 02", "Анализ видео",
            "Следите за ходом обработки и качеством детекции в реальном времени.",
        ))
        heading_row.addStretch()
        self.status_pill = label("ВЫПОЛНЯЕТСЯ", "pill")
        heading_row.addWidget(self.status_pill, alignment=Qt.AlignTop)
        root.addLayout(heading_row)

        progress_card = panel()
        progress_layout = set_card_layout(progress_card, margins=(16, 11, 16, 11), spacing=7)
        top = QHBoxLayout()
        top.addWidget(label("Ход обработки", "sectionTitle"))
        top.addStretch()
        self.lbl_percent = label("0%", "metricValue")
        top.addWidget(self.lbl_percent)
        progress_layout.addLayout(top)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        progress_layout.addWidget(self.progress_bar)
        self.lbl_status = label("Подготовка видео к анализу…", "smallMuted")
        progress_layout.addWidget(self.lbl_status)
        root.addWidget(progress_card)

        body = QHBoxLayout()
        body.setSpacing(14)
        video_card = panel()
        video_layout = set_card_layout(video_card)
        video_header = QHBoxLayout()
        video_header.addWidget(label("Предпросмотр", "sectionTitle"))
        video_header.addStretch()
        video_header.addWidget(label("LIVE PREVIEW", "pillGreen"))
        video_layout.addLayout(video_header)
        self.video_label = QLabel("Ожидание первого кадра")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(260, 180)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet(
            "background:#070B11; border:1px solid #1C2737; border-radius:8px; color:#627188;"
        )
        video_layout.addWidget(self.video_label, 1)
        video_footer = QHBoxLayout()
        video_footer.addWidget(label("Рамки отображают текущие результаты детекции.", "smallMuted"))
        video_footer.addStretch()
        video_layout.addLayout(video_footer)
        body.addWidget(video_card, 7)

        right_col = QVBoxLayout()
        right_col.setSpacing(12)
        stages_card = panel()
        stages_card.setMinimumWidth(0)
        stages_card.setFixedHeight(304)
        stages_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        stages_layout = set_card_layout(stages_card, margins=(16, 12, 16, 10), spacing=3)
        stages_layout.addWidget(label("Этапы обработки", "sectionTitle"))
        self.stage_labels = []
        stage_data = [
            ("01", "Чтение видео", "Получение кадров из источника"),
            ("02", "Детекция объектов", "Анализ кадров моделью YOLO"),
            ("03", "Структурирование", "Подготовка записей результатов"),
            ("04", "Подготовка данных для LLM", "Формирование digest для Qwen2.5"),
        ]
        for index, (number, title, subtitle) in enumerate(stage_data):
            stage_row = QFrame()
            stage_row.setObjectName("stageRow")
            # Отделители остаются тонкими и нейтральными; цвет активного этапа
            # задаётся по его номеру и тексту ниже.
            stage_row.setStyleSheet(
                "QFrame#stageRow { background:transparent; border:none; "
                + ("border-bottom:1px solid #D6E0EA;" if index < len(stage_data) - 1 else "")
                + "}"
            )
            row_layout = QHBoxLayout(stage_row)
            row_layout.setContentsMargins(0, 3, 0, 3)
            row_layout.setSpacing(14)
            stage_row.setFixedHeight(56)
            stage_row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            number_label = QLabel(number)
            number_label.setAlignment(Qt.AlignCenter)
            number_label.setFixedSize(40, 40)
            text_widget = QWidget()
            text_widget.setStyleSheet("background:transparent; border:none;")
            text_col = QVBoxLayout(text_widget)
            text_col.setContentsMargins(0, 0, 0, 0)
            text_col.setSpacing(4)
            title_label = QLabel(title)
            title_label.setFixedHeight(18)
            title_label.setWordWrap(False)
            title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            title_label.setStyleSheet("background:transparent; border:none; color:#DCE5F2; font-size:13px; font-weight:700;")
            subtitle_label = QLabel(subtitle)
            subtitle_label.setFixedHeight(16)
            subtitle_label.setWordWrap(False)
            subtitle_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            subtitle_label.setStyleSheet("background:transparent; border:none; color:#8797AE; font-size:11px;")
            text_col.addWidget(title_label)
            text_col.addWidget(subtitle_label)
            row_layout.addWidget(number_label, 0, Qt.AlignVCenter)
            row_layout.addWidget(text_widget, 1, Qt.AlignVCenter)
            stages_layout.addWidget(stage_row)
            self.stage_labels.append({
                "row": stage_row,
                "number": number_label,
                "title": title_label,
                "subtitle": subtitle_label,
                "description": subtitle,
                "index": index,
            })
        self._stage_index = -1
        self.current_theme = "light"
        self._last_stage_states = ["pending", "pending", "pending", "planned"]
        self._render_stage_states(self._last_stage_states)
        right_col.addWidget(stages_card)

        log_card = panel()
        log_card.setMinimumHeight(0)
        log_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        log_layout = set_card_layout(log_card, margins=(15, 13, 15, 12), spacing=8)
        log_header = QHBoxLayout()
        log_header.addWidget(label("Журнал событий", "sectionTitle"))
        log_header.addStretch()
        log_header.addWidget(label("LOCAL", "pill"))
        log_layout.addLayout(log_header)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.document().setMaximumBlockCount(250)
        self.log_edit.setMinimumHeight(70)
        self.log_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.log_edit.setPlaceholderText("Сообщения обработки появятся здесь.")
        log_layout.addWidget(self.log_edit, 1)
        right_col.addWidget(log_card, 1)
        right_col.setStretch(0, 0)
        right_col.setStretch(1, 1)
        body.addLayout(right_col, 5)
        root.addLayout(body, 1)

        bottom = QHBoxLayout()
        bottom.addWidget(label("Не закрывайте приложение до завершения обработки.", "smallMuted"))
        bottom.addStretch()
        self.btn_stop = QPushButton("Остановить анализ")
        self.btn_stop.setObjectName("dangerBtn")
        self.btn_stop.clicked.connect(self.stop_clicked.emit)
        bottom.addWidget(self.btn_stop)
        root.addLayout(bottom)

    def set_theme(self, theme):
        self.current_theme = theme
        light = theme == "light"
        text = "#20384E" if light else "#DCE5F2"
        muted = "#657B92" if light else "#8797AE"
        border = "#D6E0EA" if light else "#253144"
        preview_bg = "#F4F7FA" if light else "#070B11"
        preview_fg = "#52677D" if light else "#AAB6C5"
        page_bg = "#F1F5F9" if light else "#0B1018"
        self.setStyleSheet(f"QWidget {{ background:{page_bg}; color:{text}; }}")
        self.video_label.setStyleSheet(
            f"background:{preview_bg}; border:1px solid {border}; "
            f"border-radius:8px; color:{preview_fg};"
        )
        self.log_edit.setStyleSheet(
            f"background:{'#FFFFFF' if light else '#0B121D'}; "
            f"color:{text}; border:1px solid {border}; border-radius:8px; padding:8px;"
        )
        for item in self.stage_labels:
            item["row"].setStyleSheet(
                f"QFrame#stageRow {{ background:transparent; border:none; "
                f"border-bottom:1px solid {border}; }}"
            )
        self._render_stage_states(self._last_stage_states)

    def _render_stage_states(self, states):
        """Render pending / active / completed / planned pipeline states."""
        self._last_stage_states = list(states)
        light = getattr(self, "current_theme", "light") == "light"
        styles = {
            "pending": {
                "badge": ("background:#EEF3F8; border:1px solid #C9D5E2; border-radius:11px; color:#52677D; font-size:12px; font-weight:750;" if light else "background:#192335; border:1px solid #2A3A52; border-radius:11px; color:#8FA2BC; font-size:12px; font-weight:750;"),
                "title": ("background:transparent; border:none; color:#29445D; font-size:12px; font-weight:650;" if light else "background:transparent; border:none; color:#CAD5E5; font-size:12px; font-weight:650;"),
                "subtitle": ("background:transparent; border:none; color:#657B92; font-size:10px;" if light else "background:transparent; border:none; color:#8494AA; font-size:10px;"),
                "text": None,
            },
            "active": {
                "badge": "background:#1B4C82; border:1px solid #65A4FF; border-radius:11px; color:#FFFFFF; font-size:12px; font-weight:800;",
                "title": ("background:transparent; border:none; color:#075A91; font-size:12px; font-weight:750;" if light else "background:transparent; border:none; color:#F3F7FF; font-size:12px; font-weight:750;"),
                "subtitle": ("background:transparent; border:none; color:#0877B9; font-size:10px; font-weight:650;" if light else "background:transparent; border:none; color:#91BEFF; font-size:10px; font-weight:650;"),
                "text": "Выполняется…",
            },
            "done": {
                "badge": ("background:#DDF5EE; border:1px solid #36A98A; border-radius:11px; color:#08765D; font-size:12px; font-weight:800;" if light else "background:#123B32; border:1px solid #28A987; border-radius:11px; color:#82F0D0; font-size:12px; font-weight:800;"),
                "title": ("background:transparent; border:none; color:#176B58; font-size:12px; font-weight:700;" if light else "background:transparent; border:none; color:#DDFBF2; font-size:12px; font-weight:700;"),
                "subtitle": ("background:transparent; border:none; color:#16866C; font-size:10px;" if light else "background:transparent; border:none; color:#6ECBB2; font-size:10px;"),
                "text": "Завершено",
            },
            "planned": {
                "badge": ("background:#F0F3F7; border:1px solid #D1DAE4; border-radius:11px; color:#60758B; font-size:12px; font-weight:700;" if light else "background:#151D29; border:1px solid #2A3546; border-radius:11px; color:#687990; font-size:12px; font-weight:700;"),
                "title": ("background:transparent; border:none; color:#526A81; font-size:12px; font-weight:650;" if light else "background:transparent; border:none; color:#8795A9; font-size:12px; font-weight:650;"),
                "subtitle": ("background:transparent; border:none; color:#6C8095; font-size:10px;" if light else "background:transparent; border:none; color:#66768C; font-size:10px;"),
                "text": "JSON для LLM подготовлен",
            },
            "error": {
                "badge": ("background:#FCE8E9; border:1px solid #D85B68; border-radius:11px; color:#9B2632; font-size:12px; font-weight:800;" if light else "background:#3C2027; border:1px solid #D85B68; border-radius:11px; color:#FFC0C6; font-size:12px; font-weight:800;"),
                "title": ("background:transparent; border:none; color:#9B2632; font-size:12px; font-weight:750;" if light else "background:transparent; border:none; color:#FFDDE0; font-size:12px; font-weight:750;"),
                "subtitle": ("background:transparent; border:none; color:#B33D49; font-size:10px; font-weight:650;" if light else "background:transparent; border:none; color:#F29CA5; font-size:10px; font-weight:650;"),
                "text": "Ошибка этапа",
            },
        }
        for item, state in zip(self.stage_labels, states):
            st = styles.get(state, styles["pending"])
            item["number"].setStyleSheet(st["badge"])
            item["title"].setStyleSheet(st["title"])
            item["subtitle"].setStyleSheet(st["subtitle"])
            custom = st["text"]
            if state == "pending":
                item["subtitle"].setText(item["description"])
            elif state == "active":
                item["subtitle"].setText(custom)
            elif state == "done":
                item["subtitle"].setText(custom)
            elif state == "planned":
                item["subtitle"].setText(custom)
            elif state == "error":
                item["subtitle"].setText(custom)

    def set_active_stage(self, index):
        """Mark earlier pipeline steps done and highlight the active one."""
        self._stage_index = max(0, min(int(index), 3))
        states = []
        for i in range(4):
            if i < self._stage_index:
                states.append("done")
            elif i == self._stage_index:
                states.append("active")
            else:
                states.append("pending")
        self._render_stage_states(states)

    def complete_stages(self):
        self._stage_index = 3
        self._render_stage_states(["done", "done", "done", "done"])

    def fail_stage(self):
        states = []
        current = max(0, min(self._stage_index, 3))
        for i in range(4):
            if i < current:
                states.append("done")
            elif i == current:
                states.append("error")
            else:
                states.append("pending")
        self._render_stage_states(states)

    def update_progress(self, value, message):
        self.progress_bar.setValue(max(0, min(100, value)))
        self.lbl_percent.setText(f"{value}%" if value else "•••")
        if message:
            self.log_edit.append(message)
            self.lbl_status.setText(message)
            if message.startswith("Ошибка:"):
                self.status_pill.setText("ОШИБКА")
                self.status_pill.setObjectName("pillAmber")
                self.status_pill.style().unpolish(self.status_pill)
                self.status_pill.style().polish(self.status_pill)

    def update_frame(self, frame):
        self.video_label.setPixmap(frame_to_pixmap(frame, self.video_label.size()))

    def reset(self):
        self.progress_bar.setValue(0)
        self.lbl_percent.setText("0%")
        self.lbl_status.setText("Подготовка видео к анализу…")
        self.status_pill.setText("ВЫПОЛНЯЕТСЯ")
        self.status_pill.setObjectName("pill")
        self.status_pill.style().unpolish(self.status_pill)
        self.status_pill.style().polish(self.status_pill)
        self.log_edit.clear()
        self.video_label.setText("Ожидание первого кадра")
        self.video_label.setPixmap(QPixmap())
        self._stage_index = -1
        self._render_stage_states(["pending", "pending", "pending", "planned"])


# ---------------------------------------------------------------------------
# Screen: Results
# ---------------------------------------------------------------------------

class PageResult(QWidget):
    def __init__(self):
        super().__init__()
        self.records = []
        self.video_path = ""
        self.model = None
        self.conf = 0.4
        self.cap = None
        self.is_playing = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._next_frame)
        self.total_frames = 0
        self.fps = 30.0
        self.current_frame_idx = 0
        self._current_raw_frame = None
        self.export_thread = None
        self.progress_dialog = None
        self._last_results_signature = None
        self.show_clean_video = False

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 20, 28, 18)
        root.setSpacing(12)
        heading_row = QHBoxLayout()
        heading_row.addWidget(page_heading(
            "Рабочая область / 03", "Результаты анализа",
            "Просматривайте детекции, переходите к таймкоду и находите нужные фрагменты.",
        ))
        heading_row.addStretch()
        self.btn_export = QPushButton("Экспорт видео")
        self.btn_export.setObjectName("primaryBtn")
        self.btn_export.clicked.connect(self._export_video)

        self.btn_properties = QPushButton("Свойства")
        self.btn_properties.setToolTip("Открыть сведения об анализе в отдельном окне")
        self.btn_properties.clicked.connect(self._show_properties)

        self.btn_clean_video = SwitchControl("С рамками", "Без рамок")
        self.btn_clean_video.setToolTip("Переключить отображение рамок детекции")
        self.btn_clean_video.toggled.connect(self._toggle_clean_video)

        # Крупные действия в заголовке: свойства слева, экспорт справа.
        for action_button in (self.btn_export, self.btn_properties):
            action_button.setFixedSize(164, 48)
        heading_row.addWidget(self.btn_properties, alignment=Qt.AlignVCenter)
        heading_row.addWidget(self.btn_export, alignment=Qt.AlignVCenter)
        root.addLayout(heading_row)

        metrics = QHBoxLayout()
        metrics.setSpacing(12)
        self.metric_frames = self._metric_card("КАДРЫ С АНАЛИЗОМ", "—", "Отдельные моменты видео")
        self.metric_detections = self._metric_card("ОБНАРУЖЕНИЯ", "—", "Сумма детекций по кадрам")
        self.metric_step = self._metric_card("ШАГ АНАЛИЗА", "—", "Интервал между проверками")
        self.metric_source = self._metric_card("ИСТОЧНИК", "—", "Текущая запись")
        for metric in (self.metric_frames, self.metric_detections, self.metric_step, self.metric_source):
            metrics.addWidget(metric, 1)

        self.properties_dialog = QDialog(self)
        self.properties_dialog.setWindowTitle("Свойства анализа")
        self.properties_dialog.setMinimumWidth(620)
        self.properties_dialog.resize(760, 260)
        properties_layout = QVBoxLayout(self.properties_dialog)
        properties_layout.setContentsMargins(20, 20, 20, 20)
        properties_layout.setSpacing(12)
        properties_layout.addWidget(label("Свойства текущего анализа", "sectionTitle"))
        properties_layout.addLayout(metrics)
        self.properties_dialog.setModal(False)
        self.current_theme = "light"

        content = QHBoxLayout()
        content.setSpacing(14)
        left = QVBoxLayout()
        left.setSpacing(10)

        video_card = panel()
        video_layout = set_card_layout(video_card, margins=(14, 12, 14, 12), spacing=8)
        video_head = QHBoxLayout()
        video_head.addWidget(label("Видео", "sectionTitle"))
        video_head.addStretch()
        self.btn_clean_video.setFixedSize(164, 42)
        video_head.addWidget(self.btn_clean_video, alignment=Qt.AlignVCenter)
        video_head.addSpacing(16)
        self.btn_fullscreen = QPushButton("⛶")
        self.btn_fullscreen.setFixedSize(38, 34)
        self.btn_fullscreen.setToolTip("Открыть видео на весь экран")
        self.btn_fullscreen.clicked.connect(self._open_fullscreen)
        video_head.addWidget(self.btn_fullscreen)
        video_layout.addLayout(video_head)
        self.result_video = QLabel("Выберите видео и завершите анализ")
        # Give the analyzed video more room; the label expands with the result pane.
        self.result_video.setMinimumSize(320, 220)
        self.result_video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.result_video.setAlignment(Qt.AlignCenter)
        self.result_video.setStyleSheet(
            "background:#070B11; border:1px solid #1D2838; border-radius:8px; color:#68788F;"
        )
        self.fullscreen_dialog = QDialog(self)
        self.fullscreen_dialog.setWindowTitle("SceneVision — просмотр видео")
        self.fullscreen_dialog.setMinimumSize(640, 420)
        self.fullscreen_dialog.setStyleSheet("QDialog { background:#070B11; }")
        fullscreen_layout = QVBoxLayout(self.fullscreen_dialog)
        fullscreen_layout.setContentsMargins(12, 12, 12, 12)
        fullscreen_layout.setSpacing(8)
        fullscreen_toolbar = QHBoxLayout()
        fullscreen_toolbar.addWidget(QLabel("Полноэкранный просмотр"))
        fullscreen_toolbar.addStretch()
        self.btn_exit_fullscreen = QPushButton("✕  Закрыть")
        self.btn_exit_fullscreen.clicked.connect(self.fullscreen_dialog.close)
        fullscreen_toolbar.addWidget(self.btn_exit_fullscreen)
        fullscreen_layout.addLayout(fullscreen_toolbar)
        self.fullscreen_video = QLabel("Видео появится после загрузки результата")
        self.fullscreen_video.setAlignment(Qt.AlignCenter)
        self.fullscreen_video.setMinimumSize(0, 0)
        self.fullscreen_video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.fullscreen_video.setStyleSheet("background:#000000; color:#AAB6C5; border:0;")
        fullscreen_layout.addWidget(self.fullscreen_video, 1)

        fullscreen_playback = QHBoxLayout()
        self.fullscreen_btn_play = QPushButton("▶")
        self.fullscreen_btn_play.setFixedSize(42, 36)
        self.fullscreen_btn_play.setToolTip("Воспроизвести / пауза")
        self.fullscreen_btn_play.clicked.connect(self._toggle_play)
        self.fullscreen_slider = QSlider(Qt.Horizontal)
        self.fullscreen_slider.setRange(0, 0)
        self.fullscreen_slider.sliderMoved.connect(self._slider_moved)
        self.fullscreen_slider.sliderPressed.connect(self._slider_pressed)
        self.fullscreen_slider.sliderReleased.connect(self._slider_released)
        self.fullscreen_time_label = QLabel("00:00 / 00:00")
        self.fullscreen_time_label.setMinimumWidth(105)
        self.fullscreen_time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        fullscreen_playback.addWidget(self.fullscreen_btn_play)
        fullscreen_playback.addWidget(self.fullscreen_slider, 1)
        fullscreen_playback.addWidget(self.fullscreen_time_label)
        fullscreen_layout.addLayout(fullscreen_playback)

        video_layout.addWidget(self.result_video, 1)

        playback = QHBoxLayout()
        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedSize(38, 36)
        self.btn_play.setToolTip("Воспроизвести / пауза")
        self.btn_play.clicked.connect(self._toggle_play)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self._slider_moved)
        self.slider.sliderPressed.connect(self._slider_pressed)
        self.slider.sliderReleased.connect(self._slider_released)
        self.lbl_time = label("00:00 / 00:00", "smallMuted")
        playback.addWidget(self.btn_play)
        playback.addWidget(self.slider, 1)
        playback.addWidget(self.lbl_time)
        video_layout.addLayout(playback)
        # Playback controls remain below the video viewport; no note is overlaid on the image.
        playback.setContentsMargins(0, 3, 0, 0)
        left.addWidget(video_card, 9)

        search_card = panel()
        search_layout = set_card_layout(search_card, margins=(16, 14, 16, 14), spacing=10)
        search_layout.addWidget(label("Поиск по детекциям", "sectionTitle"))
        search_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Например: машина, человек, светофор…")
        self.search_input.returnPressed.connect(self._do_search)
        self.search_input.textChanged.connect(self._do_search)
        self.btn_find = QPushButton("Найти")
        self.btn_find.setObjectName("primaryBtn")
        self.btn_find.clicked.connect(self._do_search)
        search_row.addWidget(self.search_input, 1)
        search_row.addWidget(self.btn_find)
        search_layout.addLayout(search_row)
        self.search_list = QListWidget()
        # Keep search useful while giving the video viewport priority.
        self.search_list.setMinimumHeight(95)
        self.search_list.setMaximumHeight(155)
        self.search_list.itemClicked.connect(self._on_search_clicked)
        search_layout.addWidget(self.search_list)
        left.addWidget(search_card, 2)
        # Prefer the video column, but retain enough width for the summary panel.
        content.addLayout(left, 8)

        right_card = panel()
        right_layout = set_card_layout(right_card, margins=(18, 16, 18, 16), spacing=12)
        title_row = QHBoxLayout()
        title_row.addWidget(label("Сводка по кадрам", "sectionTitle"))
        title_row.addStretch()
        title_row.addWidget(label("ЭТАП 1", "pill"))
        right_layout.addLayout(title_row)
        future_card = panel("subCard")
        future_card.setMinimumWidth(0)
        future_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        future_layout = QVBoxLayout(future_card)
        future_layout.setContentsMargins(13, 12, 13, 12)
        future_layout.setSpacing(5)
        future_layout.addWidget(label("Текстовое описание сцены", "sectionTitle"))
        future_layout.addWidget(label("Появится после подключения Qwen2.5. Сейчас отображаются фактические сводки детектора.", "smallMuted"))
        right_layout.addWidget(future_card)
        self.desc_text = QTextBrowser()
        self.desc_text.setOpenLinks(False)
        self.desc_text.anchorClicked.connect(self._on_desc_link_clicked)
        self.desc_text.setPlaceholderText("После анализа здесь появятся временные отметки и сводки детекций.")
        right_layout.addWidget(self.desc_text, 1)
        legend = QHBoxLayout()
        legend.addWidget(label("Нажмите на таймкод, чтобы перейти к кадру.", "smallMuted"))
        right_layout.addLayout(legend)
        content.addWidget(right_card, 3)
        root.addLayout(content, 1)

    def _metric_card(self, title, value, hint):
        card = panel()
        layout = set_card_layout(card, margins=(15, 13, 15, 14), spacing=5)
        layout.addWidget(label(title, "metricLabel"))
        value_label = label(value, "metricValue")
        layout.addWidget(value_label)
        layout.addWidget(label(hint, "smallMuted"))
        card.value_label = value_label
        return card

    def update_results(self, records, last_frame, video_path, model, conf, analysis_data=None):
        signature = (video_path, len(records), records[-1].get("frame_index") if records else None)
        if signature == self._last_results_signature and self.cap and self.cap.isOpened():
            return
        self._last_results_signature = signature
        self.records = list(records)
        self.analysis_data = analysis_data or {}
        self.btn_clean_video.blockSignals(True)
        self.btn_clean_video.setChecked(False)
        self.btn_clean_video.blockSignals(False)
        self.show_clean_video = False
        self.video_path = video_path
        self._current_raw_frame = None
        self.model = model
        self.conf = conf
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.cap = cv2.VideoCapture(self.video_path) if self.video_path else None
        if not self.cap or not self.cap.isOpened():
            self.result_video.setText("Не удалось открыть видео для просмотра")
            self.result_video.setPixmap(QPixmap())
            self.total_frames = 0
            self.fps = 30.0
        else:
            self.total_frames = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
            raw_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps = raw_fps if raw_fps and math.isfinite(raw_fps) and raw_fps > 0 else 30.0
            self.slider.setRange(0, self.total_frames - 1)
            self.fullscreen_slider.setRange(0, self.total_frames - 1)
            self._seek_and_show(0)

        detections = sum(sum(rec.get("summary", {}).values()) for rec in self.records)
        step_value = "—"
        if len(self.records) > 1:
            # This is the observed time between stored records, not an inferred model metric.
            delta = self.records[1].get("time_sec", 0) - self.records[0].get("time_sec", 0)
            step_value = f"{delta:g} с"
        self.metric_frames.value_label.setText(str(len(self.records)))
        self.metric_detections.value_label.setText(str(detections))
        self.metric_step.value_label.setText(step_value)
        self.metric_source.value_label.setText(Path(video_path).name[:18] + ("…" if len(Path(video_path).name) > 18 else "") if video_path else "—")
        self._populate_summary()
        self._do_search()

    def set_theme(self, theme: str):
        self.current_theme = theme if theme in ("light", "dark") else "light"
        self.btn_clean_video.set_theme(self.current_theme)
        if self.current_theme == "light":
            bg, fg, muted, border, link = "#FFFFFF", "#26384B", "#64758A", "#DCE3EC", "#1769AA"
        else:
            bg, fg, muted, border, link = "#0B121D", "#DDE6F3", "#7D8CA2", "#243044", "#80B2FF"
        self.desc_text.setStyleSheet(
            f"QTextBrowser {{ background:{bg}; color:{fg}; border:1px solid {border}; "
            "border-radius:8px; padding:10px; }}"
        )
        self._summary_colors = (fg, muted, border, link)
        if self.records:
            self._populate_summary()

    def _show_properties(self):
        self.properties_dialog.show()
        self.properties_dialog.raise_()
        self.properties_dialog.activateWindow()

    def _populate_summary(self):
        self.desc_text.clear()
        rows = []
        fg, muted, border, link_color = getattr(
            self, "_summary_colors", ("#26384B", "#64758A", "#DCE3EC", "#1769AA")
        )
        scene = self.analysis_data.get("scene", {})
        if scene.get("text"):
            rows.append(f"<h3 style='color:{link_color};'>Общая сводка сцены</h3>")
            rows.append(f"<p style='color:{fg};'>{scene['text']}</p>")
        timeline = self.analysis_data.get("timeline", [])
        if timeline:
            rows.append(f"<h3 style='color:{link_color};'>События на видео</h3>")
            for event in timeline:
                t = float(event.get("t", 0))
                frame_idx = int(t * self.fps)
                timecode = event.get("timecode", f"{t:.1f} с")
                desc = event.get("description", event.get("type", "Событие"))
                rows.append(
                    f"<div style='padding:10px 4px;border-bottom:1px solid {border};'>"
                    f"<a href='{frame_idx}' style='color:{link_color};text-decoration:none;font-weight:700;'>{timecode}</a>"
                    f"<br><span style='color:{fg};'>{desc}</span></div>"
                )
            self.desc_text.setHtml("".join(rows))
            return
        for rec in self.records:
            timestamp = float(rec.get("time_sec", 0))
            frame_idx = int(rec.get("frame_index", 0))
            summary = rec.get("summary", {})
            link = (
                f"<a href='{frame_idx}' style='color:{link_color}; text-decoration:none; "
                f"font-weight:700;'>{timestamp:.1f} с</a>"
            )
            if summary:
                items = [f"{CLASS_RU.get(name, name)}: {count}" for name, count in summary.items()]
                text = ", ".join(items)
                rows.append(
                    f"<div style='padding:10px 4px; border-bottom:1px solid {border};'>"
                    f"{link}<br><span style='color:{fg};'>{text}</span></div>"
                )
            else:
                rows.append(
                    f"<div style='padding:10px 4px; border-bottom:1px solid {border};'>"
                    f"{link}<br><span style='color:{muted};'>Объектов не обнаружено</span></div>"
                )
        if not rows:
            rows = [f"<p style='color:{muted};'>Нет записей для отображения.</p>"]
        self.desc_text.setHtml("".join(rows))

    def _toggle_clean_video(self, checked: bool):
        """Toggle the preview between the source video and the annotated view."""
        self.show_clean_video = bool(checked)
        self._redraw_current_frame()

    def _redraw_current_frame(self):
        """Перерисовывает уже прочитанный кадр (в окне и на весь экран) без seek:
        не сбивает воспроизведение и не читает видео заново."""
        if self._current_raw_frame is None:
            return
        self.result_video.setPixmap(self._pixmap_for_frame(
            self._current_raw_frame, self.current_frame_idx,
            self.result_video.contentsRect().size(),
        ))
        self._refresh_fullscreen_frame()

    def _pixmap_for_frame(self, frame, frame_idx, target_size=None):
        """Рисует исходный кадр сразу под размер нужной области просмотра."""
        display_frame = frame if self.show_clean_video else self._draw_boxes_from_records(frame.copy(), frame_idx)
        return frame_to_pixmap(display_frame, target_size or self.result_video.size())

    def _refresh_fullscreen_frame(self):
        if self._current_raw_frame is None or not self.fullscreen_dialog.isVisible():
            return
        target = self.fullscreen_video.contentsRect().size()
        if target.width() <= 0 or target.height() <= 0:
            return
        pixmap = self._pixmap_for_frame(
            self._current_raw_frame, self.current_frame_idx, target
        )
        self.fullscreen_video.setPixmap(pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "result_video") and self._current_raw_frame is not None:
            self.result_video.setPixmap(
                self._pixmap_for_frame(self._current_raw_frame, self.current_frame_idx,
                                       self.result_video.contentsRect().size())
            )
        if hasattr(self, "fullscreen_dialog") and self.fullscreen_dialog.isVisible():
            QTimer.singleShot(0, self._refresh_fullscreen_frame)

    def _open_fullscreen(self):
        """Открывает текущий кадр, заново отрисованный под размер полноэкранного окна."""
        self.fullscreen_dialog.showFullScreen()
        self.fullscreen_dialog.raise_()
        self.fullscreen_dialog.activateWindow()
        QTimer.singleShot(120, self._refresh_fullscreen_frame)

    def _toggle_play(self):
        if not self.cap or not self.cap.isOpened():
            return
        if self.is_playing:
            self.timer.stop()
            self.btn_play.setText("▶")
            self.fullscreen_btn_play.setText("▶")
            self.is_playing = False
        else:
            self.timer.start(max(1, int(1000 / self.fps)))
            self.btn_play.setText("Ⅱ")
            self.fullscreen_btn_play.setText("Ⅱ")
            self.is_playing = True

    def _next_frame(self):
        if not self.cap or not self.cap.isOpened():
            return
        ok, frame = self.cap.read()
        if not ok:
            self.timer.stop()
            self.btn_play.setText("▶")
            self.fullscreen_btn_play.setText("▶")
            self.is_playing = False
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame_idx = 0
            self._seek_and_show(0)
            return
        self.current_frame_idx = max(0, int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        for timeline in (self.slider, self.fullscreen_slider):
            timeline.blockSignals(True)
            timeline.setValue(self.current_frame_idx)
            timeline.blockSignals(False)
        self._update_time_label()
        self._current_raw_frame = frame.copy()
        self.result_video.setPixmap(self._pixmap_for_frame(frame, self.current_frame_idx))
        self._refresh_fullscreen_frame()

    def _slider_pressed(self):
        self._was_playing = self.is_playing
        if self.is_playing:
            self.timer.stop()

    def _slider_released(self):
        if getattr(self, "_was_playing", False):
            self.timer.start(max(1, int(1000 / self.fps)))

    def _slider_moved(self, value):
        self._seek_and_show(value)

    def _update_time_label(self):
        current = int(self.current_frame_idx / self.fps) if self.fps > 0 else 0
        total = int(self.total_frames / self.fps) if self.fps > 0 else 0
        value = f"{current // 60:02d}:{current % 60:02d} / {total // 60:02d}:{total % 60:02d}"
        self.lbl_time.setText(value)
        self.fullscreen_time_label.setText(value)

    def _draw_boxes_from_records(self, frame, frame_idx):
        relevant = None
        for rec in self.records:
            if rec.get("frame_index", 0) <= frame_idx:
                relevant = rec
            else:
                break
        if relevant and "objects" in relevant:
            return draw_objects_ru(frame, relevant["objects"])
        return frame

    @staticmethod
    def _normalize_search_text(value):
        """Нормализует текст для поиска: регистр, ё/е и лишние пробелы."""
        return " ".join(str(value or "").lower().replace("ё", "е").split())

    @classmethod
    def _search_terms(cls, query):
        """Возвращает варианты запроса, учитывая русские и английские названия."""
        query = cls._normalize_search_text(query)
        aliases = {
            "машина": ("машина", "автомобиль", "легковой автомобиль", "car"),
            "машины": ("машина", "автомобиль", "легковой автомобиль", "car"),
            "авто": ("авто", "автомобиль", "машина", "car"),
            "автомобиль": ("автомобиль", "машина", "car"),
            "автомобили": ("автомобиль", "машина", "car"),
            "человек": ("человек", "люди", "person"),
            "люди": ("человек", "люди", "person"),
            "мотоцикл": ("мотоцикл", "motorcycle"),
            "грузовик": ("грузовик", "truck"),
            "автобус": ("автобус", "bus"),
            "светофор": ("светофор", "traffic light"),
            "велосипед": ("велосипед", "велосипедист", "bicycle", "cyclist"),
            "дтп": ("дтп", "авария", "столкновение", "accident"),
            "авария": ("авария", "дтп", "столкновение", "accident"),
        }
        return aliases.get(query, (query,))

    def _do_search(self, *_args):
        """Ищет по тем же текстовым событиям, которые показаны в правой сводке,
        а также по исходным детекциям. Все результаты ведут к нужному кадру.
        """
        query = self._normalize_search_text(self.search_input.text())
        self.search_list.clear()
        if not query:
            return

        terms = self._search_terms(query)
        matches = []

        def matches_query(*values):
            haystack = self._normalize_search_text(" ".join(
                str(value) for value in values if value is not None
            ))
            return any(term in haystack for term in terms)

        # 1. Основной источник — timeline. Это те же события, что отображаются
        # справа в блоке «События на видео».
        timeline = self.analysis_data.get("timeline", []) or []
        for event in timeline:
            description = event.get("description", event.get("type", "Событие"))
            event_type = event.get("type", "")
            if matches_query(description, event_type):
                t = float(event.get("t", 0) or 0)
                frame_idx = int(round(t * self.fps)) if self.fps > 0 else 0
                time_label = event.get("timecode") or f"{t:.1f} с"
                text = f"{time_label}  ·  {description}"
                matches.append((t, 0, text, frame_idx))

        # 2. Поиск по сводкам отдельных кадров. Он полезен для запросов вроде
        # «машина», даже если в timeline для неё не сформировалось событие.
        for rec in self.records:
            summary = rec.get("summary", {}) or {}
            summary_parts = []
            for class_name, count in summary.items():
                ru_name = CLASS_RU.get(str(class_name).lower(), str(class_name))
                summary_parts.append(f"{ru_name} {class_name} {count}")
            if matches_query(*summary_parts):
                t = float(rec.get("time_sec", 0) or 0)
                translated = [
                    f"{CLASS_RU.get(key, key)}: {count}"
                    for key, count in summary.items()
                ]
                text = f"{t:05.1f} с  ·  Детекции: " + (
                    ", ".join(translated) if translated else "объекты не обнаружены"
                )
                frame_idx = int(rec.get("frame_index", round(t * self.fps)) or 0)
                matches.append((t, 1, text, frame_idx))

        # Сводка сцены не привязана к одному событию, поэтому открывает начало.
        scene = self.analysis_data.get("scene", {}) or {}
        scene_text = scene.get("text", "")
        if scene_text and matches_query(scene_text):
            matches.append((0.0, 2, f"00:00  ·  Общая сводка: {scene_text}", 0))

        # Убираем точные дубли и сортируем результаты по времени.
        unique = {}
        for entry in matches:
            unique[(entry[3], entry[2])] = entry
        matches = sorted(unique.values(), key=lambda entry: (entry[0], entry[1]))

        if not matches:
            self.search_list.addItem(QListWidgetItem("Совпадений не найдено"))
            return

        for _t, _kind, item_text, frame_idx in matches:
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, frame_idx)
            self.search_list.addItem(item)

    def _on_search_clicked(self, item):
        frame_idx = item.data(Qt.UserRole)
        if frame_idx is not None:
            self._seek_and_show(int(frame_idx))

    def _on_desc_link_clicked(self, url: QUrl):
        try:
            frame_idx = int(url.toString())
        except ValueError:
            return
        self._seek_and_show(frame_idx)

    def _seek_and_show(self, frame_idx):
        if not self.cap or not self.cap.isOpened():
            return
        frame_idx = max(0, min(int(frame_idx), max(0, self.total_frames - 1)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok:
            return
        self.current_frame_idx = frame_idx
        for timeline in (self.slider, self.fullscreen_slider):
            timeline.blockSignals(True)
            timeline.setValue(frame_idx)
            timeline.blockSignals(False)
        self._update_time_label()
        self._current_raw_frame = frame.copy()
        self.result_video.setPixmap(self._pixmap_for_frame(frame, frame_idx))
        self._refresh_fullscreen_frame()

    def _export_video(self):
        if not self.video_path or not self.records:
            QMessageBox.warning(self, "Экспорт", "Сначала выполните анализ видео.")
            return
        export_suffix = "_clean" if self.show_clean_video else "_detected"
        dialog_title = "Сохранить видео без рамок" if self.show_clean_video else "Экспорт видео с рамками"
        default_name = str(Path(self.video_path).with_name(f"{Path(self.video_path).stem}{export_suffix}.mp4"))
        output_path, _ = QFileDialog.getSaveFileName(
            self, dialog_title, default_name, "Видео MP4 (*.mp4)"
        )
        if not output_path:
            return
        if not output_path.lower().endswith(".mp4"):
            output_path += ".mp4"
        self.progress_dialog = QProgressDialog("Подготовка видео…", "Отмена", 0, 100, self)
        self.progress_dialog.setWindowTitle("Экспорт")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)
        self.export_thread = ExportThread(
            self.video_path, self.records, output_path, self.fps, self.total_frames,
            self, draw_boxes=not self.show_clean_video,
        )
        self.export_thread.progress.connect(self.progress_dialog.setValue)
        self.export_thread.export_completed.connect(self._on_export_finished)
        self.export_thread.error.connect(self._on_export_error)
        self.export_thread.cancelled.connect(self._on_export_cancelled)
        self.progress_dialog.canceled.connect(self.export_thread.requestInterruption)
        self.btn_export.setEnabled(False)
        self.export_thread.finished.connect(lambda: self.btn_export.setEnabled(True))
        self.export_thread.start()
        self.progress_dialog.show()

    def _on_export_finished(self, path):
        if self.progress_dialog:
            self.progress_dialog.close()
        QMessageBox.information(self, "Экспорт завершён", f"Видео сохранено:\n{path}")

    def _on_export_error(self, message):
        if self.progress_dialog:
            self.progress_dialog.close()
        QMessageBox.critical(self, "Ошибка экспорта", f"Не удалось сохранить видео:\n{message}")

    def _on_export_cancelled(self):
        if self.progress_dialog:
            self.progress_dialog.close()
        self.status_tip = "Экспорт отменён."

    def stop_playback(self):
        self.timer.stop()
        self.btn_play.setText("▶")
        self.is_playing = False

    def close_video(self):
        self.stop_playback()
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ---------------------------------------------------------------------------
# Screen: Settings
# ---------------------------------------------------------------------------

class PageSettings(QWidget):
    apply_requested = pyqtSignal(str, float, str, str)
    theme_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        scroll_area = QScrollArea()
        self.scroll_area = scroll_area
        self.scroll_content = None
        scroll_area.setObjectName("settingsScroll")
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_content = QWidget()
        self.scroll_content = scroll_content
        scroll_content.setObjectName("settingsContent")
        scroll_content.setMinimumWidth(0)
        scroll_content.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        root = QVBoxLayout(scroll_content)
        root.setContentsMargins(34, 30, 34, 26)
        root.setSpacing(21)
        root.setSizeConstraint(QVBoxLayout.SetNoConstraint)
        scroll_area.setWidget(scroll_content)
        outer_layout.addWidget(scroll_area)
        root.addWidget(page_heading(
            "Рабочая область / 04", "Настройки",
            "Управляйте внешним видом приложения, моделями и параметрами обработки.",
        ))

        appearance_card = panel()
        appearance_card.setMinimumWidth(0)
        appearance_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        appearance_layout = set_card_layout(appearance_card, margins=(18, 15, 18, 15), spacing=8)
        appearance_layout.addWidget(label("Внешний вид", "sectionTitle"))
        appearance_description = label(
            "Выберите оформление рабочего пространства. Настройка сохранится после перезапуска.", "subtitle"
        )
        appearance_description.setWordWrap(True)
        appearance_layout.addWidget(appearance_description)
        appearance_row = QHBoxLayout()
        appearance_row.addWidget(label("Тема интерфейса", "muted"))
        appearance_row.addStretch(1)
        self.theme_switch = SwitchControl("Светлая", "Тёмная", icon="theme")
        self.theme_switch.setToolTip("Переключить светлую или тёмную тему")
        appearance_row.addWidget(self.theme_switch)
        appearance_layout.addLayout(appearance_row)
        # Тему меняем не в момент клика, а когда ползунок уже доехал: так
        # анимация идёт плавно, как у переключателя рамок. singleShot(0) даёт
        # Qt отрисовать финальный кадр до тяжёлой перестройки стилей.
        self.theme_switch.settled.connect(
            lambda checked: QTimer.singleShot(
                0, lambda: self.theme_changed.emit("dark" if checked else "light")
            )
        )
        root.addWidget(appearance_card)

        model_card = panel()
        model_card.setMinimumWidth(0)
        model_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        model_layout = set_card_layout(model_card)
        model_header = QHBoxLayout()
        model_header.addWidget(label("Модель компьютерного зрения", "sectionTitle"))
        model_header.addStretch()
        self.model_status = label("YOLO", "pill")
        model_header.addWidget(self.model_status)
        model_layout.addLayout(model_header)
        model_description = label(
            "Основная YOLO обнаруживает объекты. При необходимости можно подключить отдельные веса ДТП и дорожных знаков.", "subtitle"
        )
        model_description.setWordWrap(True)
        model_layout.addWidget(model_description)
        weights_row = QHBoxLayout()
        default_weights = str(Path(__file__).resolve().with_name("road_all.pt")) if Path(__file__).resolve().with_name("road_all.pt").is_file() else "yolov8n.pt"
        self.weights_edit = QLineEdit(default_weights)
        self.weights_edit.setPlaceholderText("Путь к файлу весов .pt")
        self.weights_edit.setMinimumHeight(42)
        self.weights_edit.setMinimumWidth(0)
        self.weights_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_weights = QPushButton("Обзор…")
        self.btn_weights.setMinimumHeight(42)
        self.btn_weights.clicked.connect(self._browse_weights)
        weights_row.addWidget(self.weights_edit, 1)
        weights_row.addWidget(self.btn_weights)
        model_layout.addLayout(weights_row)
        model_layout.addWidget(label("Дополнительные модели (необязательно)", "sectionTitle"))
        self.accident_weights_edit = QLineEdit()
        self.accident_weights_edit.setPlaceholderText("Путь к весам модели ДТП (.pt)")
        self.accident_weights_edit.setMinimumHeight(42)
        self.accident_weights_edit.setMinimumWidth(0)
        self.accident_weights_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.road_weights_edit = QLineEdit()
        self.road_weights_edit.setPlaceholderText("Путь к весам дорожных знаков и переходов (.pt)")
        self.road_weights_edit.setMinimumHeight(42)
        self.road_weights_edit.setMinimumWidth(0)
        self.road_weights_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        model_layout.addWidget(self.accident_weights_edit)
        model_layout.addWidget(self.road_weights_edit)
        root.addWidget(model_card)

        params_card = panel()
        params_layout = set_card_layout(params_card)
        params_layout.addWidget(label("Параметры детекции", "sectionTitle"))
        params_layout.addWidget(label(
            "Порог уверенности применяется к результатам текущей модели; более высокий порог отсекает менее уверенные детекции.",
            "subtitle",
        ))
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setSpacing(12)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.conf_combo = QComboBox()
        self.conf_combo.setObjectName("confidenceCombo")
        self.conf_combo.setMinimumWidth(150)
        self.conf_combo.setMaximumWidth(190)
        self.conf_combo.setMaxVisibleItems(10)
        self.conf_combo.setCursor(Qt.PointingHandCursor)
        self.conf_combo.setToolTip("Минимальная уверенность детекции: от 0,05 до 0,95")
        for value in range(5, 100, 5):
            numeric = value / 100.0
            self.conf_combo.addItem(f"{numeric:.2f}".replace(".", ","), numeric)
        self.conf_combo.setCurrentIndex(self.conf_combo.findData(0.4))
        form.addRow("Порог уверенности", self.conf_combo)
        params_layout.addLayout(form)
        root.addWidget(params_card)

        future_card = panel("subCard")
        future_card.setMinimumWidth(0)
        future_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        future_layout = QHBoxLayout(future_card)
        future_layout.setContentsMargins(18, 17, 18, 17)
        future_layout.setSpacing(14)
        future_mark = QLabel("LLM")
        future_mark.setAlignment(Qt.AlignCenter)
        future_mark.setFixedSize(46, 46)
        future_mark.setStyleSheet(
            "background:#19263A; border:1px solid #2B405E; border-radius:10px; "
            "color:#A6C8FF; font-size:12px; font-weight:800;"
        )
        future_layout.addWidget(future_mark, alignment=Qt.AlignTop)
        future_text = QVBoxLayout()
        future_text.addWidget(label("Генерация текстового описания", "sectionTitle"))
        future_text.addWidget(label(
            "На этом этапе формируется result_llm.json для последующей передачи в Qwen2.5. Сам вызов языковой модели пока не подключён.",
            "subtitle",
        ))
        future_layout.addLayout(future_text, 1)
        future_layout.addWidget(label("В РАЗРАБОТКЕ", "pillAmber"), alignment=Qt.AlignTop)
        root.addWidget(future_card)
        root.addStretch(1)

        bottom = QHBoxLayout()
        bottom.addWidget(label("После применения выбранные веса будут загружены заново.", "smallMuted"))
        bottom.addStretch()
        self.btn_apply = QPushButton("Применить настройки")
        self.btn_apply.setObjectName("primaryBtn")
        self.btn_apply.clicked.connect(self._apply)
        bottom.addWidget(self.btn_apply)
        root.addLayout(bottom)

    def set_theme(self, theme):
        light = theme == "light"
        self.theme_switch.set_theme(theme)
        page_bg = "#F1F5F9" if light else "#0B1018"
        card_bg = "#FFFFFF" if light else "#111925"
        text = "#20384E" if light else "#E7ECF5"
        self.setStyleSheet(
            f"QWidget#settingsContent {{ background:{page_bg}; color:{text}; }}"
            f"QLabel {{ background:transparent; color:{text}; }}"
        )
        self.scroll_area.setStyleSheet(
            f"QScrollArea#settingsScroll {{ background:{page_bg}; border:0; }}"
            f"QScrollArea#settingsScroll QWidget#qt_scrollarea_viewport {{ background:{page_bg}; }}"
        )
        self.scroll_area.viewport().setStyleSheet(f"background:{page_bg};")
        self.scroll_content.setStyleSheet(f"QWidget#settingsContent {{ background:{page_bg}; }}"f"QLabel {{ background:transparent; color:{text}; }}")
        for card in self.scroll_content.findChildren(QFrame):
            if card.objectName() in ("card", "subCard"):
                card.setStyleSheet(
                    f"QFrame#{card.objectName()} {{ background:{card_bg}; "
                    f"border:1px solid {'#D7E1EB' if light else '#253144'}; border-radius:12px; }}"
                )

    def _browse_weights(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите веса YOLO", "", "Веса модели (*.pt);;Все файлы (*.*)")
        if path:
            self.weights_edit.setText(path)

    def _apply(self):
        weights = self.weights_edit.text().strip()
        if not weights:
            QMessageBox.warning(self, "Путь к модели", "Укажите путь к файлу весов или имя модели.")
            return
        if Path(weights).suffix.lower() == ".pt" and (Path(weights).is_absolute() or "/" in weights or "\\" in weights):
            if not Path(weights).is_file():
                QMessageBox.warning(self, "Модель не найдена", "Проверьте путь к файлу весов.")
                return
        self.apply_requested.emit(
            weights, float(self.conf_combo.currentData()),
            self.accident_weights_edit.text().strip(), self.road_weights_edit.text().strip(),
        )

    def set_loading(self, loading: bool):
        self.btn_apply.setEnabled(not loading)
        self.btn_weights.setEnabled(not loading)
        self.weights_edit.setEnabled(not loading)
        self.accident_weights_edit.setEnabled(not loading)
        self.road_weights_edit.setEnabled(not loading)
        self.model_status.setText("ЗАГРУЗКА…" if loading else "НАСТРОЙКА")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SceneVision — анализ видео")
        self.resize(1440, 900)
        self.setMinimumSize(1180, 760)
        self.settings = QSettings("SceneVision", "Desktop")
        self.current_theme = self.settings.value("theme", "light")
        self.setStyleSheet(LIGHT_APP_STYLE if self.current_theme == "light" else APP_STYLE)

        self.model = None
        self.model_thread = None
        self.video_thread = None
        self.last_frame = None
        self.all_records = []
        self.current_video_path = ""
        self.current_conf = 0.4
        self.analysis_data = {}
        unified_weights = Path(__file__).resolve().with_name("road_all.pt")
        self.model_weights = str(unified_weights) if unified_weights.is_file() else "yolov8n.pt"
        self.accident_weights = ""
        self.road_weights = ""
        self.loaded_accident_weights = ""
        self.loaded_road_weights = ""
        self._analysis_error = ""

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(238)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(17, 21, 17, 18)
        side.setSpacing(7)

        brand_row = QHBoxLayout()
        brand_row.setSpacing(11)
        mark = QFrame()
        mark.setObjectName("brandMark")
        mark.setFixedSize(39, 39)
        mark_layout = QVBoxLayout(mark)
        mark_layout.setContentsMargins(0, 0, 0, 0)
        mark_label = QLabel("E")
        mark_label.setAlignment(Qt.AlignCenter)
        mark_label.setStyleSheet("color:white; font-size:13px; font-weight:800; background:transparent;")
        mark_layout.addWidget(mark_label)
        brand_text = QVBoxLayout()
        brand_text.setSpacing(1)
        brand_text.addWidget(label("ELTEX", "brandTitle"))
        brand_text.addWidget(label("VIDEO ANALYTICS", "brandCaption"))
        brand_row.addWidget(mark)
        brand_row.addLayout(brand_text, 1)
        side.addLayout(brand_row)
        side.addSpacing(30)
        side.addWidget(label("РАБОЧЕЕ ПРОСТРАНСТВО", "eyebrow"))
        side.addSpacing(7)

        self.btn_nav_new = QPushButton("  01     Новый анализ")
        self.btn_nav_new.setObjectName("navBtn")
        self.btn_nav_new.setCheckable(True)
        self.btn_nav_new.setChecked(True)
        self.btn_nav_new.setCursor(Qt.PointingHandCursor)
        self.btn_nav_new.clicked.connect(lambda: self._switch_page(0))
        self.btn_nav_res = QPushButton("  02     Результаты")
        self.btn_nav_res.setObjectName("navBtn")
        self.btn_nav_res.setCheckable(True)
        self.btn_nav_res.setCursor(Qt.PointingHandCursor)
        self.btn_nav_res.clicked.connect(lambda: self._switch_page(1))
        self.btn_nav_set = QPushButton("  03     Настройки")
        self.btn_nav_set.setObjectName("navBtn")
        self.btn_nav_set.setCheckable(True)
        self.btn_nav_set.setCursor(Qt.PointingHandCursor)
        self.btn_nav_set.clicked.connect(lambda: self._switch_page(2))
        for button in (self.btn_nav_new, self.btn_nav_res, self.btn_nav_set):
            button.setMinimumHeight(44)
            side.addWidget(button)
        side.addSpacing(16)

        status_card = panel("subCard")
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(12, 12, 12, 12)
        status_layout.setSpacing(7)
        status_layout.addWidget(label("СОСТОЯНИЕ СИСТЕМЫ", "eyebrow"))
        self.model_indicator = label("Загрузка модели…", "subtitle")
        self.model_indicator.setWordWrap(True)
        status_layout.addWidget(self.model_indicator)
        self.model_state_pill = label("ОЖИДАНИЕ", "pillAmber")
        status_layout.addWidget(self.model_state_pill, alignment=Qt.AlignLeft)
        side.addWidget(status_card)
        side.addStretch(1)

        foot = QFrame()
        foot.setStyleSheet("background:transparent; border:none;")
        foot_layout = QVBoxLayout(foot)
        foot_layout.setContentsMargins(0, 0, 0, 0)
        foot_layout.setSpacing(4)
        foot_layout.addWidget(label("SCENEVISION DESKTOP", "brandCaption"))
        foot_layout.addWidget(label("Прототип · версия 0.2", "smallMuted"))
        side.addWidget(foot)
        layout.addWidget(sidebar)

        self.stack = QStackedWidget()
        self.page_new = PageNewAnalysis()
        self.page_proc = PageProcessing()
        self.page_res = PageResult()
        self.page_set = PageSettings()
        for page in (self.page_new, self.page_proc, self.page_res, self.page_set):
            self.stack.addWidget(page)
        layout.addWidget(self.stack, 1)

        self.page_new.start_analysis.connect(self._start_processing)
        self.page_proc.stop_clicked.connect(self._stop_processing)
        self.page_set.apply_requested.connect(self._apply_model_settings)
        self.page_set.theme_changed.connect(self._apply_theme)
        self.page_set.theme_switch.blockSignals(True)
        self.page_set.theme_switch.setChecked(self.current_theme == "dark")
        self.page_set.theme_switch.blockSignals(False)
        self.page_res.set_theme(self.current_theme)
        self.page_new.set_theme(self.current_theme)
        self.page_proc.set_theme(self.current_theme)
        self.page_set.set_theme(self.current_theme)
        self.statusBar().showMessage("Инициализация приложения…")

        self._load_model(self.model_weights, initial=True)

    def _apply_theme(self, theme):
        if theme not in ("light", "dark"):
            return
        self.current_theme = theme
        self.settings.setValue("theme", theme)
        self.setStyleSheet(LIGHT_APP_STYLE if theme == "light" else APP_STYLE)
        self.page_res.set_theme(theme)
        self.page_new.set_theme(theme)
        self.page_proc.set_theme(theme)
        self.page_set.set_theme(theme)
        self.page_set.theme_switch.blockSignals(True)
        self.page_set.theme_switch.setChecked(theme == "dark")
        self.page_set.theme_switch.blockSignals(False)
        # Цвета текста и фона синхронно обновляются на всех страницах.
        self.statusBar().showMessage(
            "Включена светлая тема" if theme == "light" else "Включена тёмная тема", 2500
        )

    def _set_nav(self, index):
        self.btn_nav_new.setChecked(index == 0)
        self.btn_nav_res.setChecked(index == 1)
        self.btn_nav_set.setChecked(index == 2)

    def _analysis_is_running(self):
        return self.video_thread is not None and self.video_thread.isRunning()

    def _show_processing_page(self, notify=False):
        """Возвращает пользователя к текущему анализу, не сбрасывая его состояние."""
        self.stack.setCurrentWidget(self.page_proc)
        self._set_nav(0)
        self.page_proc.btn_stop.setEnabled(self._analysis_is_running())
        self.statusBar().showMessage(
            "Анализ выполняется — прогресс и журнал обновляются.", 4000
        )

    def _switch_page(self, index):
        # Пока идёт анализ, «Новый анализ» и «Результаты» не должны
        # показывать пустую форму вместо активного процесса.
        if self._analysis_is_running() and index in (0, 1):
            self._show_processing_page(notify=(index == 0))
            return

        if index == 1:
            if not self.all_records or not self.current_video_path:
                self.statusBar().showMessage("Сначала выполните анализ видео.", 3500)
                self._set_nav(0)
                return
            self.page_res.update_results(
                self.all_records, self.last_frame, self.current_video_path,
                self.model, self.current_conf, self.analysis_data,
            )
            self.stack.setCurrentWidget(self.page_res)
            self._set_nav(1)
            self.page_res.stop_playback()
        elif index == 0:
            self.stack.setCurrentWidget(self.page_new)
            self._set_nav(0)
        elif index == 2:
            self.stack.setCurrentIndex(3)
            self._set_nav(2)

    def _load_model(self, weights, initial=False, accident_weights=None, road_weights=None):
        if self.video_thread and self.video_thread.isRunning():
            QMessageBox.warning(self, "Обработка выполняется", "Остановите анализ перед заменой модели.")
            return
        self.page_new.btn_start.setEnabled(False)
        self.page_set.set_loading(True)
        self.model_indicator.setText(f"Загрузка весов: {Path(weights).name}")
        self.model_state_pill.setText("ЗАГРУЗКА…")
        self.model_state_pill.setObjectName("pillAmber")
        self.model_state_pill.style().unpolish(self.model_state_pill)
        self.model_state_pill.style().polish(self.model_state_pill)
        self.statusBar().showMessage(f"Загрузка модели: {weights}")
        accident_weights = accident_weights if accident_weights is not None else self.page_set.accident_weights_edit.text()
        road_weights = road_weights if road_weights is not None else self.page_set.road_weights_edit.text()
        self.model_thread = ModelLoadThread(weights, self, accident_weights, road_weights)
        self.model_thread.loaded.connect(self._on_model_loaded)
        self.model_thread.failed.connect(self._on_model_failed)
        self.model_thread.start()
        self._loading_initial = initial

    def _on_model_loaded(self, model, weights):
        self.model = model
        self.model_weights = weights
        if self.model_thread:
            self.loaded_accident_weights = self.model_thread.accident_weights or ""
            self.loaded_road_weights = self.model_thread.road_weights or ""
        self.accident_weights = self.loaded_accident_weights
        self.road_weights = self.loaded_road_weights
        self.page_new.btn_start.setEnabled(bool(self.page_new.file_path))
        self.page_set.set_loading(False)
        self.page_set.model_status.setText("ЗАГРУЖЕНА")
        self.page_set.model_status.setObjectName("pillGreen")
        self.page_set.model_status.style().unpolish(self.page_set.model_status)
        self.page_set.model_status.style().polish(self.page_set.model_status)
        self.page_set.weights_edit.setText(weights)
        self.model_indicator.setText(f"Модель готова: {Path(weights).name}")
        self.model_state_pill.setText("ГОТОВА")
        self.model_state_pill.setObjectName("pillGreen")
        self.model_state_pill.style().unpolish(self.model_state_pill)
        self.model_state_pill.style().polish(self.model_state_pill)
        self.statusBar().showMessage("Модель готова к работе", 5000)

    def _on_model_failed(self, message):
        self.page_set.set_loading(False)
        self.page_new.btn_start.setEnabled(False)
        self.page_set.model_status.setText("ОШИБКА")
        self.page_set.model_status.setObjectName("pillAmber")
        self.page_set.model_status.style().unpolish(self.page_set.model_status)
        self.page_set.model_status.style().polish(self.page_set.model_status)
        self.model_indicator.setText("Не удалось загрузить модель")
        self.model_state_pill.setText("ОШИБКА")
        self.model_state_pill.setObjectName("pillAmber")
        self.model_state_pill.style().unpolish(self.model_state_pill)
        self.model_state_pill.style().polish(self.model_state_pill)
        self.statusBar().showMessage("Ошибка загрузки модели")
        QMessageBox.warning(
            self, "Не удалось загрузить модель",
            f"Не удалось загрузить веса:\n{message}\n\n"
            "Проверьте имя модели или укажите доступный файл .pt в настройках.",
        )

    def _apply_model_settings(self, weights, confidence, accident_weights="", road_weights=""):
        self.current_conf = float(confidence)
        self.accident_weights = accident_weights.strip()
        self.road_weights = road_weights.strip()
        if (self.model is not None and weights == self.model_weights
                and self.accident_weights == self.loaded_accident_weights
                and self.road_weights == self.loaded_road_weights):
            self.statusBar().showMessage(
                f"Порог уверенности применён: {self.current_conf:.2f}".replace(".", ","), 4000
            )
            return
        self._load_model(weights, accident_weights=self.accident_weights, road_weights=self.road_weights)

    def _start_processing(self, config):
        if self.model is None:
            QMessageBox.warning(self, "Модель не готова", "Дождитесь загрузки модели или выберите веса в настройках.")
            return
        if self._analysis_is_running():
            self._show_processing_page(notify=True)
            return
        self.all_records.clear()
        self.analysis_data = {}
        self.last_frame = None
        self.current_video_path = config["source"]
        self._analysis_error = ""
        self.page_proc.reset()
        self.page_proc.set_active_stage(0)
        self.stack.setCurrentWidget(self.page_proc)
        self._set_nav(0)
        self.video_thread = VideoThread(
            config["source_type"], config["source"], self.model,
            conf=self.current_conf, step=config["step"],
            weights=self.model_weights,
            accident_weights=self.loaded_accident_weights,
            road_weights=self.loaded_road_weights,
            parent=self,
        )
        self.video_thread.frame_ready.connect(self._on_frame)
        self.video_thread.record_ready.connect(self._on_record)
        self.video_thread.progress_updated.connect(self.page_proc.update_progress)
        self.video_thread.stage_changed.connect(self.page_proc.set_active_stage)
        self.video_thread.failed.connect(self._on_processing_error)
        self.video_thread.finished.connect(self._on_finished)
        self.page_proc.btn_stop.setEnabled(True)
        self.video_thread.start()

    def _on_frame(self, frame):
        self.last_frame = frame.copy()
        self.page_proc.update_frame(frame)

    def _on_record(self, record, frame):
        self.all_records.append(record)

    def _on_processing_error(self, message):
        self._analysis_error = message

    def _on_finished(self):
        self.page_proc.btn_stop.setEnabled(False)
        if self.video_thread:
            self.analysis_data = self.video_thread.analysis_data or {}
            if self.video_thread.result_json:
                self.statusBar().showMessage(
                    f"Сохранены {Path(self.video_thread.result_json).name} и {Path(self.video_thread.llm_json).name}", 7000
                )
        if self.all_records:
            if self._analysis_error:
                self.page_proc.fail_stage()
                self.page_proc.lbl_status.setText(
                    f"Получены частичные результаты · записей: {len(self.all_records)}"
                )
                self.page_proc.status_pill.setText("ЧАСТИЧНО")
                self.page_proc.status_pill.setObjectName("pillAmber")
            else:
                self.page_proc.complete_stages()
                self.page_proc.lbl_status.setText(f"Обработка завершена · кадров с анализом: {len(self.all_records)}")
                self.page_proc.status_pill.setText("ЗАВЕРШЕНО")
                self.page_proc.status_pill.setObjectName("pillGreen")
            self.page_proc.status_pill.style().unpolish(self.page_proc.status_pill)
            self.page_proc.status_pill.style().polish(self.page_proc.status_pill)
            self.page_proc.progress_bar.setValue(100)
            self.page_proc.lbl_percent.setText("100%")
            if self._analysis_error:
                self.statusBar().showMessage("Обработка прервана с ошибкой · доступны частичные результаты", 6000)
                QMessageBox.warning(self, "Частичные результаты", self._analysis_error)
            else:
                self.statusBar().showMessage(f"Анализ завершён · записей: {len(self.all_records)}", 5000)
            self._switch_page(1)
        else:
            if self._analysis_error:
                self.page_proc.fail_stage()
            self.statusBar().showMessage("Анализ завершён без результатов", 5000)
            if self._analysis_error:
                QMessageBox.warning(self, "Не удалось выполнить анализ", self._analysis_error)
            self._set_nav(0)
            self.stack.setCurrentWidget(self.page_new)

    def _stop_processing(self):
        if self.video_thread and self.video_thread.isRunning():
            self.page_proc.lbl_status.setText("Остановка обработки…")
            self.video_thread.stop()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange:
            if self.isVisible() and not self.isMinimized() and self._analysis_is_running():
                QTimer.singleShot(0, lambda: self._show_processing_page(notify=False))

    def closeEvent(self, event):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        if self.model_thread and self.model_thread.isRunning():
            self.model_thread.wait(3000)
        self.page_res.close_video()
        if self.page_res.export_thread and self.page_res.export_thread.isRunning():
            self.page_res.export_thread.requestInterruption()
            self.page_res.export_thread.wait(3000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SceneVision")
    app.setOrganizationName("SceneVision")
    app.setFont(QFont("Helvetica Neue", 10))
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())