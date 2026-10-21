"""
Прототип этапа 1: видео, камера или RTSP-поток -> детекция объектов -> JSON.

Источники (укажите один):
  - видеофайл (MP4, AVI, MOV, MKV);
  - веб-/USB-камера (--camera);
  - IP-камера по RTSP (--rtsp).

Запуск:
  python main.py                          # окно выбора файла
  python main.py video.mp4                # файл
  python main.py video.mp4 --step 0.5 --save-preview preview
  python main.py video.mp4 --save-video result_video.mp4
  python main.py video.mp4 --no-detect    # только проверить загрузку видео

  python main.py --camera                 # камера №0 (встроенная)
  python main.py --camera 1               # камера №1 (внешняя USB)
  python main.py --camera --save-video cam.mp4
  python main.py --camera --no-detect     # только посмотреть картинку камеры

  python main.py --rtsp rtsp://user:pass@192.168.1.10:554/stream
  python main.py --rtsp rtsp://... --save-video rtsp.mp4 --step 0.5

Камера и RTSP: окно закрывается клавишей q или Esc, после этого сохраняется
result.json. По ходу работы каждый проанализированный кадр дописывается в
result.jsonl, так что при аварийном закрытии данные не пропадут.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import cv2

WINDOW = "SceneVision (q / Esc - exit)"
STALL_SEC = 2.0          # столько секунд без новых кадров = «нет сигнала»
MAX_RECONNECTS = 10      # сколько раз подряд пробуем переподключиться
RTSP_TIMEOUT_MS = 8000   # таймаут открытия и чтения RTSP, мс
FIRST_FRAME_TIMEOUT = {"camera": 5.0, "rtsp": 20.0}  # ожидание первого кадра, с


# ---------------------------------------------------------------- видеофайл

def pick_file() -> str:
    """Окно выбора файла (если путь не передан аргументом)."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Выберите видео",
        filetypes=[("Видео", "*.mp4 *.avi *.mov *.mkv"), ("Все файлы", "*.*")],
    )
    root.destroy()
    return path


def open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {path}")
    return cap


def video_info(cap: cv2.VideoCapture) -> dict:
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": round(fps, 2),
        "frame_count": frames,
        "duration_sec": round(frames / fps, 2) if fps else None,
    }


def sample_frames(cap: cv2.VideoCapture, fps: float, step_sec: float):
    """Отдаёт (индекс кадра, время в секундах, кадр) раз в step_sec секунд."""
    step = max(1, int(round(fps * step_sec))) if fps else 1
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            t = index / fps if fps else 0.0
            yield index, round(t, 2), frame
        index += 1


def iter_frames(cap: cv2.VideoCapture, fps: float, step_sec: float):
    """Отдаёт (индекс, время, кадр, анализировать_ли_кадр) для ВСЕХ кадров."""
    step = max(1, int(round(fps * step_sec))) if fps else 1
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield index, round(index / fps, 2) if fps else 0.0, frame, index % step == 0
        index += 1


# ------------------------------------------------- живые источники: открытие

def mask_url(url: str) -> str:
    """Прячет пароль в адресе: rtsp://user:pass@host -> rtsp://user:***@host."""
    return re.sub(r"(://[^:/@]+):[^/]*@", r"\1:***@", url)


def open_camera(index: int, width: int | None = None,
                height: int | None = None) -> cv2.VideoCapture:
    """Открывает веб-/USB-камеру по номеру устройства."""
    # На Windows бэкенд DirectShow открывает камеру быстрее и стабильнее.
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise RuntimeError(
            f"Не удалось открыть камеру №{index}. Проверьте номер устройства, "
            "доступ приложений к камере в настройках Windows и что камеру "
            "не использует другая программа (Zoom, Teams, браузер)."
        )
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def open_rtsp(url: str) -> cv2.VideoCapture:
    """Подключается к IP-камере по RTSP."""
    # RTSP по TCP надёжнее, чем по UDP: нет потерянных пакетов и «рассыпанной»
    # картинки. Переменную нужно задать до первого открытия потока.
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

    cap = None
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_TIMEOUT_MS,
                  cv2.CAP_PROP_READ_TIMEOUT_MSEC, RTSP_TIMEOUT_MS]
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)
        except TypeError:  # старая версия OpenCV без параметров таймаута
            cap = None
    if cap is None:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        raise RuntimeError(
            f"Не удалось подключиться к RTSP-потоку {mask_url(url)}. Проверьте "
            "адрес, логин и пароль, что камера и компьютер в одной сети "
            "(порт 554 не закрыт) и что поток открывается в VLC."
        )
    return cap


def live_info(cap: cv2.VideoCapture) -> dict:
    """У живого источника нет конца: число кадров и длительность неизвестны."""
    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": None,  # fps, который сообщает камера, часто неверный
        "frame_count": None,
        "duration_sec": None,
    }


# ------------------------------------------- живые источники: потоки и запись

class LatestFrameReader:
    """Читает источник в отдельном потоке и хранит только самый свежий кадр.

    Пока YOLO обрабатывает кадр, камера продолжает отдавать новые. Если их
    не вычитывать, они копятся в буфере и картинка отстаёт от реальности.
    Здесь старые кадры просто отбрасываются.

    Если передан reopen (функция, открывающая источник заново), то при потере
    сигнала поток сам переподключается: до MAX_RECONNECTS попыток подряд
    с нарастающей паузой. После успешного кадра счётчик попыток сбрасывается.
    """

    def __init__(self, cap: cv2.VideoCapture, reopen=None,
                 max_reconnects: int = MAX_RECONNECTS):
        self._cap = cap
        self._reopen = reopen
        self._max_reconnects = max_reconnects
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._frame = None
        self._seq = 0
        self._ts = 0.0
        self._attempts = 0
        self.reconnects = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "LatestFrameReader":
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if ok:
                self._attempts = 0
                with self._lock:
                    self._frame = frame
                    self._seq += 1
                    self._ts = time.monotonic()
                continue
            if self._stop.is_set() or not self._reconnect():
                break

    def _reconnect(self) -> bool:
        if self._reopen is None:
            return False
        self._cap.release()
        while self._attempts < self._max_reconnects:
            self._attempts += 1
            self.reconnects += 1
            print(f"Нет сигнала. Переподключение, попытка "
                  f"{self._attempts}/{self._max_reconnects}...")
            if self._stop.wait(min(2 * self._attempts, 10)):
                return False
            try:
                self._cap = self._reopen()
                return True
            except RuntimeError:
                continue
        return False

    def snapshot(self):
        """(копия кадра, номер кадра, время получения) или None, если кадров не было."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy(), self._seq, self._ts

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)
        if not self._thread.is_alive():
            self._cap.release()


class DetectionWorker:
    """Запускает YOLO в отдельном потоке, чтобы окно и запись видео не подвисали.

    Работает по одному кадру за раз: submit() принимает новую задачу, только
    когда результат предыдущей уже забран через poll().
    """

    def __init__(self, model, conf: float, want_plot: bool = False):
        self._model = model
        self._conf = conf
        self._want_plot = want_plot
        self._cv = threading.Condition()
        self._job = None
        self._done = None
        self._busy = False
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "DetectionWorker":
        self._thread.start()
        return self

    def _run(self):
        while True:
            with self._cv:
                while self._job is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                frame, t = self._job
                self._job = None
            try:
                objects, result = detect(self._model, frame, self._conf)
                plot = result.plot() if self._want_plot else None
                out = (t, objects, plot)
            except Exception as exc:  # передаём ошибку в основной поток
                out = exc
            with self._cv:
                self._done = out

    def submit(self, frame, t: float) -> bool:
        with self._cv:
            if self._busy:
                return False
            self._job = (frame, t)
            self._busy = True
            self._cv.notify()
            return True

    def poll(self):
        """Готовый результат (t, objects, plot), исключение или None."""
        with self._cv:
            if self._done is None:
                return None
            done, self._done = self._done, None
            self._busy = False
            return done

    def stop(self):
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._thread.join(timeout=5)


class PacedVideoWriter:
    """Пишет видео с живого источника с правильной скоростью.

    У камеры нет надёжного fps, а кадры приходят неравномерно. Поэтому видео
    пишется с постоянной частотой fps по часам: если нового кадра нет, прошлый
    повторяется. Так длительность файла совпадает с реальным временем съёмки.
    Размер кадра берётся из первого полученного кадра.
    """

    def __init__(self, path: str, fps: float):
        self.path = path
        self.fps = fps
        self._writer = None
        self._size = None
        self._t0 = 0.0
        self._count = 0

    def write(self, frame, now: float):
        """now - время в секундах (любые часы, главное - одни и те же)."""
        if self._writer is None:
            h, w = frame.shape[:2]
            self._size = (w, h)
            self._writer = cv2.VideoWriter(
                self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self._size)
            if not self._writer.isOpened():
                raise RuntimeError(f"Не удалось создать файл видео: {self.path}")
            self._t0 = now
        elif frame.shape[1::-1] != self._size:  # источник сменил разрешение
            frame = cv2.resize(frame, self._size)
        due = int((now - self._t0) * self.fps) + 1
        while self._count < due:
            self._writer.write(frame)
            self._count += 1

    def release(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None


# --------------------------------------------------------------- отрисовка

def draw_objects(frame, objects):
    """Рисует рамки и подписи на кадре (для видео с результатом)."""
    for o in objects:
        x1, y1, x2, y2 = (int(v) for v in o["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label = f"{o['class']} {o['confidence']:.2f}"
        cv2.putText(frame, label, (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1, cv2.LINE_AA)
    return frame


def window_closed() -> bool:
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


# ------------------------------------------------------------ детекция

def load_model(weights: str):
    """Загружает YOLOv8. При первом запуске веса скачаются сами."""
    from ultralytics import YOLO

    return YOLO(weights)


def detect(model, frame, conf: float):
    """Возвращает (список объектов, результат Ultralytics для отрисовки)."""
    result = model(frame, conf=conf, verbose=False)[0]
    objects = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        objects.append(
            {
                "class": result.names[cls_id],
                "confidence": round(float(box.conf[0]), 3),
                "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
            }
        )
    return objects, result


def make_record(index: int, t: float, objects: list) -> dict:
    """Запись об одном проанализированном кадре (одинаковая для всех источников)."""
    return {"frame_index": index, "time_sec": t,
            "summary": dict(Counter(o["class"] for o in objects)),
            "objects": objects}


def save_result(args, source_name, source_type, info, frames_out, totals):
    data = {
        "video": {"file": source_name, "source_type": source_type, **info},
        "settings": {"step_sec": args.step, "weights": args.weights, "conf": args.conf},
        "frames": frames_out,
        "totals": dict(totals),
    }
    Path(args.output).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nГотово. Проанализировано кадров: {len(frames_out)}")
    print(f"Всего найдено: {dict(totals)}")
    print(f"Результат сохранён: {args.output}")


# ------------------------------------------------------------ режим: файл

def run_file(args, cap, info, source_name):
    model = load_model(args.weights)
    preview_dir = Path(args.save_preview) if args.save_preview else None
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_video, fourcc, info["fps"] or 25.0,
                                 (info["width"], info["height"]))

    frames_out = []
    totals = Counter()
    last_objects = []  # рамки с последнего анализируемого кадра (держим до следующего)
    for index, t, frame, analyze in iter_frames(cap, info["fps"], args.step):
        if analyze:
            objects, result = detect(model, frame, args.conf)
            last_objects = objects
            record = make_record(index, t, objects)
            totals.update(record["summary"])
            frames_out.append(record)
            print(f"  t={t:>7.2f} c  объектов: {len(objects):<3} {record['summary']}")
            if preview_dir:
                cv2.imwrite(str(preview_dir / f"frame_{index:06d}.jpg"), result.plot())
        if writer:
            writer.write(draw_objects(frame, last_objects))
    if writer:
        writer.release()
        print(f"Видео с рамками сохранено: {args.save_video}")
    cap.release()

    save_result(args, source_name, "file", info, frames_out, totals)


# ------------------------------------------- режим: камера и RTSP (живой поток)

def run_live(args, model, cap, reopen, info, source_name, source_type):
    """Общий цикл для веб-камеры и RTSP.

    model=None означает режим --no-detect: только показ (и запись видео).
    reopen - функция переоткрытия источника (None = не переподключаться).
    """
    detecting = model is not None
    preview_dir = Path(args.save_preview) if args.save_preview else None
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)

    journal_path = Path(args.output).with_suffix(".jsonl")
    journal = open(journal_path, "w", encoding="utf-8") if detecting else None
    reader = LatestFrameReader(cap, reopen).start()
    worker = DetectionWorker(model, args.conf, bool(preview_dir)).start() \
        if detecting else None
    writer = PacedVideoWriter(args.save_video, args.video_fps) \
        if args.save_video else None

    frames_out = []
    totals = Counter()
    last_objects = []
    index = 0        # порядковый номер завершённого анализа
    next_t = 0.0     # когда делать следующий анализ (сек от старта)
    last_seq = -1    # номер последнего отправленного на анализ кадра
    start = time.monotonic()
    waiting_printed = False
    first_timeout = FIRST_FRAME_TIMEOUT[source_type]

    print("Идёт работа. Для остановки нажмите q или Esc в окне "
          "(или Ctrl+C в консоли).")
    try:
        while True:
            now = time.monotonic()
            elapsed = now - start
            snap = reader.snapshot()
            if snap is None:
                if not reader.alive or elapsed > first_timeout:
                    raise RuntimeError(
                        "Источник не отдаёт кадры. Проверьте подключение.")
                if not waiting_printed:
                    print("Ожидание первого кадра...")
                    waiting_printed = True
                time.sleep(0.02)
                continue

            frame, seq, ts = snap
            if not info["width"]:  # RTSP иногда сообщает размер только с кадром
                info["height"], info["width"] = frame.shape[:2]
            stalled = (now - ts) > STALL_SEC
            if stalled and not reader.alive:
                print("Источник недоступен, переподключение не удалось. "
                      "Работа остановлена.")
                break

            if worker:
                done = worker.poll()
                if isinstance(done, Exception):
                    raise done
                if done is not None:
                    t, objects, plot = done
                    last_objects = objects
                    record = make_record(index, t, objects)
                    totals.update(record["summary"])
                    frames_out.append(record)
                    journal.write(json.dumps(record, ensure_ascii=False) + "\n")
                    journal.flush()  # данные на диске сразу, а не при закрытии
                    print(f"  t={t:>7.2f} c  объектов: {len(objects):<3} "
                          f"{record['summary']}")
                    if preview_dir and plot is not None:
                        cv2.imwrite(str(preview_dir / f"frame_{index:06d}.jpg"), plot)
                    index += 1
                # новый анализ: только по свежему кадру и не чаще, чем раз в step
                if not stalled and seq != last_seq and elapsed >= next_t:
                    if worker.submit(frame, round(elapsed, 2)):
                        next_t = elapsed + args.step  # от «сейчас», без накопления очереди
                        last_seq = seq

            annotated = draw_objects(frame.copy(), last_objects)
            if stalled:
                cv2.putText(annotated, "NO SIGNAL - reconnecting", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            if writer:
                writer.write(annotated, now)  # в файл - без служебной строки

            hud = f"t={elapsed:6.1f}s  analyzed: {index}"
            if writer:
                hud += "  REC"
            cv2.putText(annotated, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 220, 0), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW, annotated)
            if cv2.waitKey(15) & 0xFF in (ord("q"), 27):
                break
            if window_closed():
                break
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")
    finally:
        if worker:
            worker.stop()
        reader.stop()
        if writer:
            writer.release()
            print(f"Видео сохранено: {args.save_video}")
        if journal:
            journal.close()
        cv2.destroyAllWindows()
        if detecting and frames_out:
            info = {**info, "duration_sec": round(time.monotonic() - start, 2),
                    "reconnects": reader.reconnects}
            save_result(args, source_name, source_type, info, frames_out, totals)
            print(f"Журнал по ходу работы: {journal_path}")


# ----------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Прототип: видео, камера или RTSP -> объекты -> JSON")
    parser.add_argument("video", nargs="?", help="путь к видеофайлу")
    parser.add_argument("--camera", nargs="?", const=0, type=int, metavar="N",
                        help="использовать веб-/USB-камеру с номером N "
                             "(без номера: 0, встроенная)")
    parser.add_argument("--cam-width", type=int, help="ширина кадра камеры")
    parser.add_argument("--cam-height", type=int, help="высота кадра камеры")
    parser.add_argument("--rtsp", metavar="URL",
                        help="адрес RTSP-потока, например "
                             "rtsp://user:pass@192.168.1.10:554/stream")
    parser.add_argument("--step", type=float, default=1.0,
                        help="шаг между анализируемыми кадрами, сек (по умолчанию 1.0)")
    parser.add_argument("--weights", default="C:\\Users\\Furia\\runs\\detect\\train\\weights\\accident_dtp.pt",
                        help="веса YOLO (по умолчанию yolov8n.pt, самая лёгкая)")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="порог уверенности детекции (по умолчанию 0.4)")
    parser.add_argument("--output", default="result.json", help="файл с результатом")
    parser.add_argument("--save-preview", metavar="DIR",
                        help="папка для сохранения кадров с рамками")
    parser.add_argument("--save-video", metavar="FILE",
                        help="сохранить видео с рамками (например, result_video.mp4)")
    parser.add_argument("--video-fps", type=float, default=25.0,
                        help="частота кадров сохраняемого видео для камеры/RTSP "
                             "(по умолчанию 25)")
    parser.add_argument("--no-detect", action="store_true",
                        help="только загрузка видео/подключение без детекции")
    args = parser.parse_args()

    chosen = [args.video is not None, args.camera is not None, args.rtsp is not None]
    if sum(chosen) > 1:
        parser.error("укажите один источник: файл, --camera или --rtsp")
    if args.rtsp and not args.rtsp.lower().startswith(("rtsp://", "rtsps://")):
        parser.error("адрес для --rtsp должен начинаться с rtsp://")

    try:
        # ---- камера и RTSP
        if args.camera is not None or args.rtsp:
            # модель грузим до подключения: чтобы соединение не простаивало
            model = None if args.no_detect else load_model(args.weights)
            if args.camera is not None:
                cap = open_camera(args.camera, args.cam_width, args.cam_height)
                reopen, source_type = None, "camera"
                source_name = f"camera:{args.camera}"
                print(f"Камера №{args.camera}")
            else:
                url = args.rtsp
                cap = open_rtsp(url)
                reopen, source_type = (lambda: open_rtsp(url)), "rtsp"
                source_name = mask_url(url)
                print(f"RTSP: {source_name}")
            info = live_info(cap)
            run_live(args, model, cap, reopen, info, source_name, source_type)
            return 0

        # ---- видеофайл
        path = args.video or pick_file()
        if not path:
            print("Файл не выбран.")
            return 1

        cap = open_video(path)
        info = video_info(cap)
        print(f"Видео: {Path(path).name}")
        print(f"  разрешение: {info['width']}x{info['height']}, FPS: {info['fps']}, "
              f"кадров: {info['frame_count']}, длительность: {info['duration_sec']} с")

        if args.no_detect:
            n = sum(1 for _ in sample_frames(cap, info["fps"], args.step))
            print(f"Будет проанализировано кадров: {n}")
            cap.release()
            return 0

        run_file(args, cap, info, Path(path).name)
        return 0
    except RuntimeError as exc:
        print(f"Ошибка: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
