"""SceneVision GUI — polished desktop interface for the current prototype.

Place this file in the same directory as main.py and run:
    python scenevision_gui.py

The interface currently uses the existing detector and record format from main.py.
Qwen2.5 generation and the final JSON contract are intentionally not implemented here.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QLineEdit, QComboBox,
    QDoubleSpinBox, QTextEdit, QMessageBox, QFormLayout,
    QListWidget, QListWidgetItem, QStackedWidget, QFrame,
    QProgressBar, QSlider, QTextBrowser, QProgressDialog,
    QSpinBox, QSizePolicy, QSpacerItem,
)

from main import (
    load_model, detect, make_record,
    open_video, open_camera, open_rtsp, video_info,
    LatestFrameReader, iter_frames,
)


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


def panel(object_name: str = "card") -> QFrame:
    frame = QFrame()
    frame.setObjectName(object_name)
    return frame


def label(text: str, object_name: str | None = None, parent=None) -> QLabel:
    widget = QLabel(text, parent)
    if object_name:
        widget.setObjectName(object_name)
    return widget


def page_heading(kicker: str, title: str, subtitle: str) -> QWidget:
    block = QWidget()
    layout = QVBoxLayout(block)
    layout.setContentsMargins(0, 0, 0, 5)
    layout.setSpacing(7)
    layout.addWidget(label(kicker.upper(), "eyebrow"))
    layout.addWidget(label(title, "title"))
    layout.addWidget(label(subtitle, "subtitle"))
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

    def __init__(self, weights: str, parent=None):
        super().__init__(parent)
        self.weights = weights

    def run(self):
        try:
            self.loaded.emit(load_model(self.weights), self.weights)
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

    def __init__(self, source_type, source, model, conf=0.4, step=1.0, parent=None):
        super().__init__(parent)
        self.source_type = source_type
        self.source = source
        self.model = model
        self.conf = conf
        self.step = step
        self._running = False
        self._cap = None
        self._reader = None
        self.error_message = ""

    def run(self):
        self._running = True
        self.stage_changed.emit(0)  # этап 1: открытие и чтение источника
        try:
            if self.source_type == "file":
                self._run_file()
            else:
                self._run_live(self.source_type)
        except Exception as exc:
            self.error_message = str(exc)
            self.failed.emit(self.error_message)
            self.progress_updated.emit(0, f"Ошибка: {exc}")
        finally:
            self._cleanup()

    def _run_file(self):
        cap = open_video(self.source)
        self._cap = cap
        info = video_info(cap)
        self.stage_changed.emit(1)  # источник открыт: начинается детекция
        duration = info.get("duration_sec") or 0
        total_steps = max(1, math.ceil(duration / max(self.step, 0.001)))
        analyzed_count = 0
        try:
            for index, timestamp, frame, analyze in iter_frames(cap, info["fps"], self.step):
                if not self._running or self.isInterruptionRequested():
                    break
                if not analyze:
                    continue
                objects, _ = detect(self.model, frame, self.conf)
                record = make_record(index, timestamp, objects)
                annotated = draw_objects_ru(frame.copy(), objects)
                self.record_ready.emit(record, annotated.copy())
                self.stage_changed.emit(2)  # первая запись результата сформирована
                self.frame_ready.emit(annotated)
                analyzed_count += 1
                progress = min(100, int(analyzed_count * 100 / total_steps))
                summary = ", ".join(
                    f"{CLASS_RU.get(name, name)}: {count}"
                    for name, count in record.get("summary", {}).items()
                ) or "объектов нет"
                self.progress_updated.emit(
                    progress, f"{timestamp:07.2f} с  ·  {summary}"
                )
                self.msleep(8)
        finally:
            cap.release()
            self._cap = None

    def _run_live(self, kind):
        if kind == "camera":
            cap = open_camera(int(self.source))
            reopen = None
        else:
            cap = open_rtsp(self.source)
            reopen = lambda: open_rtsp(self.source)
        self._cap = cap
        self._reader = LatestFrameReader(cap, reopen).start()
        self.stage_changed.emit(1)  # источник открыт: начинается детекция
        self.progress_updated.emit(0, f"Подключено: {kind}")
        last_seq = -1
        next_analysis = 0.0
        start = time.monotonic()
        index = 0
        last_objects = []
        try:
            while self._running and not self.isInterruptionRequested():
                now = time.monotonic()
                snapshot = self._reader.snapshot()
                if snapshot is None:
                    if not self._reader.alive:
                        break
                    self.msleep(20)
                    continue
                frame, seq, received_at = snapshot
                if seq != last_seq and (now - start) >= next_analysis:
                    elapsed = now - start
                    objects, _ = detect(self.model, frame, self.conf)
                    last_objects = objects
                    record = make_record(index, round(elapsed, 2), objects)
                    annotated = draw_objects_ru(frame.copy(), objects)
                    self.record_ready.emit(record, annotated.copy())
                    self.stage_changed.emit(2)  # первая запись результата сформирована
                    self.frame_ready.emit(annotated)
                    index += 1
                    last_seq = seq
                    next_analysis = elapsed + max(self.step, 0.1)
                    summary = ", ".join(
                        f"{CLASS_RU.get(name, name)}: {count}"
                        for name, count in record.get("summary", {}).items()
                    ) or "объектов нет"
                    self.progress_updated.emit(0, f"{elapsed:07.2f} с  ·  {summary}")
                else:
                    self.frame_ready.emit(draw_objects_ru(frame.copy(), last_objects))
                self.msleep(25)
        finally:
            pass

    def stop(self):
        self._running = False
        self.requestInterruption()
        self.wait(4000)

    def _cleanup(self):
        if self._reader:
            self._reader.stop()
            self._reader = None
        if self._cap:
            self._cap.release()
            self._cap = None


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
        self.drop_zone.setMinimumHeight(310)
        self.drop_zone.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        drop_layout = QVBoxLayout(self.drop_zone)
        drop_layout.setContentsMargins(25, 28, 25, 28)
        drop_layout.setSpacing(13)
        drop_layout.setAlignment(Qt.AlignCenter)

        self.upload_mark = QLabel("↑")
        self.upload_mark.setAlignment(Qt.AlignCenter)
        self.upload_mark.setFixedSize(56, 56)
        self.upload_mark.setStyleSheet(
            "background:#172A43; border:1px solid #294B76; border-radius:16px; "
            "color:#82B5FF; font-size:28px; font-weight:600;"
        )
        drop_layout.addWidget(self.upload_mark, alignment=Qt.AlignCenter)
        drop_layout.addWidget(label("Перетащите видео сюда", "sectionTitle"), alignment=Qt.AlignCenter)
        desc = label("или выберите файл на компьютере", "subtitle")
        drop_layout.addWidget(desc, alignment=Qt.AlignCenter)
        formats = label("MP4  ·  AVI  ·  MOV  ·  MKV", "smallMuted")
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
        root = QVBoxLayout(self)
        root.setContentsMargins(34, 30, 34, 26)
        root.setSpacing(20)
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
        progress_layout = set_card_layout(progress_card, margins=(20, 16, 20, 17), spacing=10)
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
        body.setSpacing(18)
        video_card = panel()
        video_layout = set_card_layout(video_card)
        video_header = QHBoxLayout()
        video_header.addWidget(label("Предпросмотр", "sectionTitle"))
        video_header.addStretch()
        video_header.addWidget(label("LIVE PREVIEW", "pillGreen"))
        video_layout.addLayout(video_header)
        self.video_label = QLabel("Ожидание первого кадра")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(460, 330)
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
        right_col.setSpacing(18)
        stages_card = panel()
        stages_card.setMinimumHeight(218)
        stages_layout = set_card_layout(stages_card, margins=(15, 13, 15, 12), spacing=5)
        stages_layout.addWidget(label("Этапы обработки", "sectionTitle"))
        self.stage_labels = []
        stage_data = [
            ("01", "Чтение видео", "Получение кадров из источника"),
            ("02", "Детекция объектов", "Анализ кадров моделью YOLO"),
            ("03", "Структурирование", "Подготовка записей результатов"),
            ("04", "Текстовое описание", "LLM будет подключена позже"),
        ]
        for index, (number, title, subtitle) in enumerate(stage_data):
            stage_row = QFrame()
            stage_row.setObjectName("stageRow")
            # Отделители остаются тонкими и нейтральными; цвет активного этапа
            # задаётся по его номеру и тексту ниже.
            stage_row.setStyleSheet(
                "QFrame#stageRow { background:transparent; border:none; "
                + ("border-bottom:1px solid #253144;" if index < len(stage_data) - 1 else "")
                + "}"
            )
            row_layout = QHBoxLayout(stage_row)
            row_layout.setContentsMargins(0, 2, 0, 4 if index < len(stage_data) - 1 else 2)
            row_layout.setSpacing(10)
            stage_row.setMinimumHeight(38)
            number_label = QLabel(number)
            number_label.setAlignment(Qt.AlignCenter)
            number_label.setFixedSize(34, 34)
            text_widget = QWidget()
            text_widget.setStyleSheet("background:transparent; border:none;")
            text_col = QVBoxLayout(text_widget)
            text_col.setContentsMargins(0, 0, 0, 0)
            text_col.setSpacing(2)
            title_label = QLabel(title)
            title_label.setMinimumHeight(16)
            title_label.setStyleSheet("background:transparent; border:none; color:#DCE5F2; font-size:12px; font-weight:650;")
            subtitle_label = QLabel(subtitle)
            subtitle_label.setWordWrap(False)
            subtitle_label.setMinimumHeight(14)
            subtitle_label.setStyleSheet("background:transparent; border:none; color:#8797AE; font-size:10px;")
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
        self._render_stage_states(["pending", "pending", "pending", "planned"])
        right_col.addWidget(stages_card)

        log_card = panel()
        log_layout = set_card_layout(log_card)
        log_header = QHBoxLayout()
        log_header.addWidget(label("Журнал событий", "sectionTitle"))
        log_header.addStretch()
        log_header.addWidget(label("LOCAL", "pill"))
        log_layout.addLayout(log_header)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.document().setMaximumBlockCount(250)
        self.log_edit.setMinimumHeight(130)
        self.log_edit.setPlaceholderText("Сообщения обработки появятся здесь.")
        log_layout.addWidget(self.log_edit, 1)
        right_col.addWidget(log_card, 1)
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

    def _render_stage_states(self, states):
        """Render pending / active / completed / planned pipeline states."""
        styles = {
            "pending": {
                "badge": "background:#192335; border:1px solid #2A3A52; border-radius:11px; color:#8FA2BC; font-size:12px; font-weight:750;",
                "title": "background:transparent; border:none; color:#CAD5E5; font-size:12px; font-weight:650;",
                "subtitle": "background:transparent; border:none; color:#8494AA; font-size:10px;",
                "text": None,
            },
            "active": {
                "badge": "background:#1B4C82; border:1px solid #65A4FF; border-radius:11px; color:#FFFFFF; font-size:12px; font-weight:800;",
                "title": "background:transparent; border:none; color:#F3F7FF; font-size:12px; font-weight:750;",
                "subtitle": "background:transparent; border:none; color:#91BEFF; font-size:10px; font-weight:650;",
                "text": "Выполняется…",
            },
            "done": {
                "badge": "background:#123B32; border:1px solid #28A987; border-radius:11px; color:#82F0D0; font-size:12px; font-weight:800;",
                "title": "background:transparent; border:none; color:#DDFBF2; font-size:12px; font-weight:700;",
                "subtitle": "background:transparent; border:none; color:#6ECBB2; font-size:10px;",
                "text": "Завершено",
            },
            "planned": {
                "badge": "background:#151D29; border:1px solid #2A3546; border-radius:11px; color:#687990; font-size:12px; font-weight:700;",
                "title": "background:transparent; border:none; color:#8795A9; font-size:12px; font-weight:650;",
                "subtitle": "background:transparent; border:none; color:#66768C; font-size:10px;",
                "text": "LLM пока не подключена",
            },
            "error": {
                "badge": "background:#3C2027; border:1px solid #D85B68; border-radius:11px; color:#FFC0C6; font-size:12px; font-weight:800;",
                "title": "background:transparent; border:none; color:#FFDDE0; font-size:12px; font-weight:750;",
                "subtitle": "background:transparent; border:none; color:#F29CA5; font-size:10px; font-weight:650;",
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
        self._stage_index = max(0, min(int(index), 2))
        states = []
        for i in range(4):
            if i == 3:
                states.append("planned")
            elif i < self._stage_index:
                states.append("done")
            elif i == self._stage_index:
                states.append("active")
            else:
                states.append("pending")
        self._render_stage_states(states)

    def complete_stages(self):
        self._stage_index = 2
        self._render_stage_states(["done", "done", "done", "planned"])

    def fail_stage(self):
        states = []
        current = max(0, min(self._stage_index, 2))
        for i in range(4):
            if i == 3:
                states.append("planned")
            elif i < current:
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
        self.export_thread = None
        self.progress_dialog = None
        self._last_results_signature = None
        self.show_clean_video = False

        root = QVBoxLayout(self)
        root.setContentsMargins(34, 28, 34, 24)
        root.setSpacing(16)
        heading_row = QHBoxLayout()
        heading_row.addWidget(page_heading(
            "Рабочая область / 03", "Результаты анализа",
            "Просматривайте детекции, переходите к таймкоду и находите нужные фрагменты.",
        ))
        heading_row.addStretch()
        self.btn_clean_video = QPushButton("○  Видео без рамок")
        self.btn_clean_video.setObjectName("cleanVideoBtn")
        self.btn_clean_video.setCheckable(True)
        self.btn_clean_video.setToolTip("Переключить просмотр исходного видео без зелёных рамок")
        self.btn_clean_video.toggled.connect(self._toggle_clean_video)
        heading_row.addWidget(self.btn_clean_video, alignment=Qt.AlignVCenter)

        self.btn_export = QPushButton("Экспорт видео")
        self.btn_export.setObjectName("primaryBtn")
        self.btn_export.clicked.connect(self._export_video)
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
        root.addLayout(metrics)

        content = QHBoxLayout()
        content.setSpacing(16)
        left = QVBoxLayout()
        left.setSpacing(16)

        video_card = panel()
        video_layout = set_card_layout(video_card, margins=(16, 15, 16, 15), spacing=12)
        video_head = QHBoxLayout()
        video_head.addWidget(label("Видео", "sectionTitle"))
        video_head.addStretch()
        self.video_state = label("ПРОСМОТР", "pillGreen")
        video_head.addWidget(self.video_state)
        video_layout.addLayout(video_head)
        self.result_video = QLabel("Выберите видео и завершите анализ")
        # Give the analyzed video more room; the label expands with the result pane.
        self.result_video.setMinimumSize(560, 320)
        self.result_video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.result_video.setAlignment(Qt.AlignCenter)
        self.result_video.setStyleSheet(
            "background:#070B11; border:1px solid #1D2838; border-radius:8px; color:#68788F;"
        )
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
        left.addWidget(video_card, 8)

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
        self.search_list.setMinimumHeight(80)
        self.search_list.setMaximumHeight(125)
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
        content.addWidget(right_card, 4)
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

    def update_results(self, records, last_frame, video_path, model, conf):
        signature = (video_path, len(records), records[-1].get("frame_index") if records else None)
        if signature == self._last_results_signature and self.cap and self.cap.isOpened():
            return
        self._last_results_signature = signature
        self.records = list(records)
        self.btn_clean_video.blockSignals(True)
        self.btn_clean_video.setChecked(False)
        self.btn_clean_video.setText("○  Видео без рамок")
        self.btn_clean_video.blockSignals(False)
        self.show_clean_video = False
        self.video_state.setText("С РАМКАМИ")
        self.video_state.setObjectName("pill")
        self.video_state.style().unpolish(self.video_state)
        self.video_state.style().polish(self.video_state)
        self.video_path = video_path
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

    def _populate_summary(self):
        self.desc_text.clear()
        rows = []
        for rec in self.records:
            timestamp = float(rec.get("time_sec", 0))
            frame_idx = int(rec.get("frame_index", 0))
            summary = rec.get("summary", {})
            link = (
                f"<a href='{frame_idx}' style='color:#80B2FF; text-decoration:none; "
                f"font-weight:700;'>{timestamp:.1f} с</a>"
            )
            if summary:
                items = [
                    f"{CLASS_RU.get(name, name)}: {count}"
                    for name, count in summary.items()
                ]
                text = ", ".join(items)
                rows.append(
                    f"<div style='padding:9px 4px; border-bottom:1px solid #243044;'>"
                    f"{link}<br><span style='color:#DCE5F2;'>{text}</span></div>"
                )
            else:
                rows.append(
                    f"<div style='padding:9px 4px; border-bottom:1px solid #243044;'>"
                    f"{link}<br><span style='color:#7D8CA2;'>Объектов не обнаружено</span></div>"
                )
        if not rows:
            rows = ["<p style='color:#7D8CA2;'>Нет записей для отображения.</p>"]
        self.desc_text.setHtml("".join(rows))

    def _toggle_clean_video(self, checked: bool):
        """Toggle the preview between the source video and the annotated view."""
        self.show_clean_video = bool(checked)
        self.btn_clean_video.setText("●  Видео без рамок" if checked else "○  Видео без рамок")
        if checked:
            self.video_state.setText("БЕЗ РАМОК")
            self.video_state.setObjectName("pillGreen")
        else:
            self.video_state.setText("С РАМКАМИ")
            self.video_state.setObjectName("pill")
        self.video_state.style().unpolish(self.video_state)
        self.video_state.style().polish(self.video_state)
        self._seek_and_show(self.current_frame_idx)

    def _pixmap_for_frame(self, frame, frame_idx):
        """Render a frame with or without detection overlays according to the toggle."""
        display_frame = frame if self.show_clean_video else self._draw_boxes_from_records(frame.copy(), frame_idx)
        return frame_to_pixmap(display_frame, self.result_video.size())

    def _toggle_play(self):
        if not self.cap or not self.cap.isOpened():
            return
        if self.is_playing:
            self.timer.stop()
            self.btn_play.setText("▶")
            self.is_playing = False
        else:
            self.timer.start(max(1, int(1000 / self.fps)))
            self.btn_play.setText("Ⅱ")
            self.is_playing = True

    def _next_frame(self):
        if not self.cap or not self.cap.isOpened():
            return
        ok, frame = self.cap.read()
        if not ok:
            self.timer.stop()
            self.btn_play.setText("▶")
            self.is_playing = False
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame_idx = 0
            self._seek_and_show(0)
            return
        self.current_frame_idx = max(0, int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_frame_idx)
        self.slider.blockSignals(False)
        self._update_time_label()
        self.result_video.setPixmap(self._pixmap_for_frame(frame, self.current_frame_idx))

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
        self.lbl_time.setText(
            f"{current // 60:02d}:{current % 60:02d} / {total // 60:02d}:{total % 60:02d}"
        )

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

    def _do_search(self, *_args):
        query = self.search_input.text().strip().lower()
        self.search_list.clear()
        if not query:
            return
        found = 0
        for rec in self.records:
            summary = rec.get("summary", {})
            summary_str = " ".join(str(key).lower() for key in summary.keys())
            ru_summary = " ".join(CLASS_RU.get(key.lower(), key.lower()) for key in summary.keys())
            if query in summary_str or query in ru_summary:
                translated = [f"{CLASS_RU.get(key, key)}: {count}" for key, count in summary.items()]
                item_text = f"{float(rec.get('time_sec', 0)):.1f} с  ·  " + (
                    ", ".join(translated) if translated else "Объектов не обнаружено"
                )
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, int(rec.get("frame_index", 0)))
                self.search_list.addItem(item)
                found += 1
        if not found:
            self.search_list.addItem(QListWidgetItem("Совпадений не найдено"))

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
        self.slider.blockSignals(True)
        self.slider.setValue(frame_idx)
        self.slider.blockSignals(False)
        self._update_time_label()
        self.result_video.setPixmap(self._pixmap_for_frame(frame, frame_idx))

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
    apply_requested = pyqtSignal(str, float)

    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(34, 30, 34, 26)
        root.setSpacing(21)
        root.addWidget(page_heading(
            "Рабочая область / 04", "Настройки",
            "Управляйте моделью детекции и параметрами обработки.",
        ))

        model_card = panel()
        model_layout = set_card_layout(model_card)
        model_header = QHBoxLayout()
        model_header.addWidget(label("Модель компьютерного зрения", "sectionTitle"))
        model_header.addStretch()
        self.model_status = label("YOLO", "pill")
        model_header.addWidget(self.model_status)
        model_layout.addLayout(model_header)
        model_layout.addWidget(label(
            "Укажите веса YOLO. Для текущего прототипа используется одна модель детекции.", "subtitle"
        ))
        weights_row = QHBoxLayout()
        self.weights_edit = QLineEdit("yolov8n.pt")
        self.weights_edit.setPlaceholderText("Путь к файлу весов .pt")
        self.btn_weights = QPushButton("Обзор…")
        self.btn_weights.clicked.connect(self._browse_weights)
        weights_row.addWidget(self.weights_edit, 1)
        weights_row.addWidget(self.btn_weights)
        model_layout.addLayout(weights_row)
        root.addWidget(model_card)

        params_card = panel()
        params_layout = set_card_layout(params_card)
        params_layout.addWidget(label("Параметры детекции", "sectionTitle"))
        params_layout.addWidget(label(
            "Порог уверенности применяется к результатам текущей модели; более высокий порог отсекает менее уверенные детекции.",
            "subtitle",
        ))
        form = QFormLayout()
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
            "Qwen2.5 и выбор языка описания будут подключены на следующем этапе. Сейчас эти параметры не влияют на анализ.",
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
        self.apply_requested.emit(weights, float(self.conf_combo.currentData()))

    def set_loading(self, loading: bool):
        self.btn_apply.setEnabled(not loading)
        self.btn_weights.setEnabled(not loading)
        self.weights_edit.setEnabled(not loading)
        self.model_status.setText("ЗАГРУЗКА…" if loading else "НАСТРОЙКА")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SceneVision — анализ видео")
        self.resize(1440, 920)
        self.setMinimumSize(1180, 760)
        self.setStyleSheet(APP_STYLE)

        self.model = None
        self.model_thread = None
        self.video_thread = None
        self.last_frame = None
        self.all_records = []
        self.current_video_path = ""
        self.current_conf = 0.4
        self.model_weights = "yolov8n.pt"
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
        mark_label = QLabel("SV")
        mark_label.setAlignment(Qt.AlignCenter)
        mark_label.setStyleSheet("color:white; font-size:13px; font-weight:800; background:transparent;")
        mark_layout.addWidget(mark_label)
        brand_text = QVBoxLayout()
        brand_text.setSpacing(1)
        brand_text.addWidget(label("SceneVision", "brandTitle"))
        brand_text.addWidget(label("VIDEO INTELLIGENCE", "brandCaption"))
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
        self.statusBar().showMessage("Инициализация приложения…")

        self._load_model(self.model_weights, initial=True)

    def _set_nav(self, index):
        self.btn_nav_new.setChecked(index == 0)
        self.btn_nav_res.setChecked(index == 1)
        self.btn_nav_set.setChecked(index == 2)

    def _switch_page(self, index):
        if index == 1:
            if not self.all_records or not self.current_video_path:
                self.statusBar().showMessage("Сначала выполните анализ видео.", 3500)
                self._set_nav(0 if self.stack.currentIndex() != 1 else 0)
                return
            self.page_res.update_results(
                self.all_records, self.last_frame, self.current_video_path,
                self.model, self.current_conf,
            )
            self.stack.setCurrentWidget(self.page_res)
            self._set_nav(1)
            self.page_res.stop_playback()
        elif index == 0:
            self.stack.setCurrentWidget(self.page_new)
            self._set_nav(0)
        elif index == 2:
            self.stack.setCurrentWidget(self.page_set)
            self._set_nav(2)

    def _load_model(self, weights, initial=False):
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
        self.model_thread = ModelLoadThread(weights, self)
        self.model_thread.loaded.connect(self._on_model_loaded)
        self.model_thread.failed.connect(self._on_model_failed)
        self.model_thread.start()
        self._loading_initial = initial

    def _on_model_loaded(self, model, weights):
        self.model = model
        self.model_weights = weights
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

    def _apply_model_settings(self, weights, confidence):
        self.current_conf = float(confidence)
        if self.model is not None and weights == self.model_weights:
            self.statusBar().showMessage(
                f"Порог уверенности применён: {self.current_conf:.2f}".replace(".", ","), 4000
            )
            return
        self._load_model(weights)

    def _start_processing(self, config):
        if self.model is None:
            QMessageBox.warning(self, "Модель не готова", "Дождитесь загрузки модели или выберите веса в настройках.")
            return
        if self.video_thread and self.video_thread.isRunning():
            QMessageBox.information(self, "Обработка", "Анализ уже выполняется.")
            return
        self.all_records.clear()
        self.last_frame = None
        self.current_video_path = config["source"]
        self._analysis_error = ""
        self.page_proc.reset()
        self.page_proc.set_active_stage(0)
        self.stack.setCurrentWidget(self.page_proc)
        self._set_nav(0)
        self.video_thread = VideoThread(
            config["source_type"], config["source"], self.model,
            conf=self.current_conf, step=config["step"], parent=self,
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
