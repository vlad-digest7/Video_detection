"""
Анализ сцены по результатам детекции: превращает «рамки на кадрах» в данные о том,
ЧТО ПРОИСХОДИТ в видео, чтобы LLM могла красочно описать ролик по JSON.

Что добавляется к результату:
  * у каждого объекта: номер трека (один и тот же объект на разных кадрах), положение
    (слева/по центру/справа, вдали/вблизи), размер, движение (стоит/движется, куда,
    приближается ли к камере, скорость);
  * у каждого проанализированного кадра: сводка сцены (сколько машин, сколько из них
    движется и стоит, пешеходы, видимые знаки, плотность потока, ДТП);
  * события по времени (появился, покинул кадр, остановился, тронулся, замечен знак,
    ДТП, возможное столкновение, изменилась плотность потока) с готовым текстовым
    описанием и важностью;
  * треки: краткая «биография» каждого объекта;
  * компактная сводка для LLM (digest), которую можно целиком отдать модели.

Модуль не зависит ни от OpenCV, ни от Ultralytics: работает со списками словарей.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

MOVE_THRESHOLD = 0.02    # скорость в долях ширины кадра в секунду: быстрее = «движется»
FAST_THRESHOLD = 0.12    # быстрее этого = «быстро»
MIN_HITS = 2             # трек считается настоящим, если замечен минимум в стольких кадрах
EDGE = 0.03              # объект «у края кадра», если ближе к нему, чем эта доля
WINDOW_SEC = 2.0         # по какому отрезку времени считается скорость
OVERLAP_IOU = 0.25       # перекрытие рамок двух машин, после которого пишем событие
OVERLAP_COOLDOWN = 3.0   # не повторять событие перекрытия для той же пары чаще, сек
STOP_SPEED_ABRUPT = 0.10  # была быстрее этого и остановилась = «резко остановилась»

VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle", "bicycle"}

RU_LABELS = {"person": "человек", "bicycle": "велосипедист", "motorcycle": "мотоцикл",
             "car": "легковой автомобиль", "bus": "автобус", "truck": "грузовик",
             "traffic light": "светофор", "stop sign": "знак «Stop»"}
EN_LABELS = {"person": "person", "bicycle": "cyclist", "motorcycle": "motorcycle",
             "car": "car", "bus": "bus", "truck": "truck", "traffic light": "traffic light",
             "stop sign": "stop sign"}

TX = {
    "ru": {
        "h": {"left": "слева", "center": "по центру", "right": "справа"},
        "d": {"far": "вдали", "medium": "на среднем расстоянии", "close": "вблизи"},
        "dir": {"left": "влево", "right": "вправо", "toward": "приближаясь к камере",
                "away": "удаляясь от камеры", "stationary": "стоит на месте"},
        "entry": {"left": "с левого края кадра", "right": "с правого края кадра",
                  "top": "с верхнего края кадра", "bottom": "снизу, рядом с камерой",
                  "far": "вдали", "inside": "в средней части кадра"},
        "exit": {"left": "слева", "right": "справа", "top": "вверху", "bottom": "внизу, рядом с камерой",
                 "far": "вдали", "inside": "из средней части кадра"},
        "dens": {"none": "транспорта нет", "light": "низкая", "moderate": "умеренная",
                 "heavy": "высокая"},
        "appear": "{label} №{id} появился {entry}",
        "present": "В начале видео в кадре уже был {label} №{id} ({pos})",
        "part": {"left": "в левой части кадра", "center": "в центральной части кадра",
                 "right": "в правой части кадра"},
        "left_scene": "{label} №{id} покинул кадр ({exit})",
        "stopped": "{label} №{id} остановился ({pos})",
        "stopped_abrupt": "{label} №{id} резко остановился ({pos})",
        "started": "{label} №{id} начал двигаться: {dir} ({pos})",
        "sign_seen": "Замечен {label} ({pos})",
        "light_seen": "Замечен светофор ({pos})",
        "accident": "Модель обнаружила признаки ДТП {span}; максимальная уверенность {conf}; "
                    "{part}",
        "span_range": "с {a} по {b}", "span_at": "в {a}",
        "overlap": "{la} №{a} и {lb} №{b} сильно перекрываются в кадре "
                   "(возможное столкновение или один заслоняет другого)",
        "density": "Плотность транспортного потока изменилась: {old} → {new} (транспортных средств: {n})",
        "jam": "Большинство транспортных средств в кадре стоит на месте "
               "(возможны пробка или остановка потока)",
        "jam_end": "Движение транспорта возобновилось",
        "summary": "В кадре: {parts}. Плотность потока: {dens}.",
        "nothing": "значимых объектов нет",
        "veh": "транспортных средств: {n} (движется {m}, стоит {s})",
        "per": "людей: {n}", "sg": "знаки: {names}", "acc": "обнаружено ДТП",
        "path_static": "оставался {pos}", "path_move": "{a} → {b}",
        "overall": "Видео {dur}. За это время замечено: {parts}.",
        "overall_none": "объектов не обнаружено",
        "cnt_vehicle": "транспортных средств", "cnt_person": "людей",
        "cnt_sign": "знаков", "cnt_traffic_light": "светофоров",
        "notes": [
            "Время дано в секундах от начала видео (time_sec) и в формате ММ:СС (timecode).",
            "Направления («влево», «вправо», «приближаясь к камере») указаны относительно кадра, "
            "а не сторон света.",
            "Скорость измерена в долях ширины кадра в секунду (frame_widths_per_s). Это движение "
            "в кадре, а не реальная скорость в км/ч. Если камера сама движется, неподвижные "
            "объекты могут казаться движущимися, и наоборот.",
            "Расстояние («вдали», «вблизи») оценено по положению объекта в кадре и приблизительно.",
            "Объекты находит нейросеть: она может пропускать мелкие и далёкие объекты и путать "
            "классы. Описывай только то, что есть в данных, не выдумывай деталей.",
            "Названия знаков взяты из модели на английском; переведи их на русский.",
            "События «ДТП» и «перекрытие» это предположения модели, а не доказанные факты. "
            "Формулируй их осторожно («похоже», «возможно»).",
        ],
    },
    "en": {
        "h": {"left": "left", "center": "center", "right": "right"},
        "d": {"far": "far away", "medium": "at mid distance", "close": "close"},
        "dir": {"left": "to the left", "right": "to the right", "toward": "approaching the camera",
                "away": "moving away from the camera", "stationary": "standing still"},
        "entry": {"left": "from the left edge", "right": "from the right edge",
                  "top": "from the top edge", "bottom": "from the bottom, close to the camera",
                  "far": "in the distance", "inside": "in the middle of the frame"},
        "exit": {"left": "left", "right": "right", "top": "top", "bottom": "bottom, near the camera",
                 "far": "into the distance", "inside": "from the middle of the frame"},
        "dens": {"none": "no traffic", "light": "light", "moderate": "moderate", "heavy": "heavy"},
        "appear": "{label} #{id} appeared {entry}",
        "present": "At the start of the video, {label} #{id} was already in view ({pos})",
        "part": {"left": "in the left part of the frame", "center": "in the central part of the frame",
                 "right": "in the right part of the frame"},
        "left_scene": "{label} #{id} left the frame ({exit})",
        "stopped": "{label} #{id} stopped ({pos})",
        "stopped_abrupt": "{label} #{id} stopped abruptly ({pos})",
        "started": "{label} #{id} started moving: {dir} ({pos})",
        "sign_seen": "{label} spotted ({pos})",
        "light_seen": "Traffic light spotted ({pos})",
        "accident": "The model detected signs of an accident {span}; max confidence {conf}; "
                    "{part}",
        "span_range": "from {a} to {b}", "span_at": "at {a}",
        "overlap": "{la} #{a} and {lb} #{b} overlap strongly in the frame "
                   "(possible collision or one hides the other)",
        "density": "Traffic density changed: {old} -> {new} (vehicles: {n})",
        "jam": "Most vehicles in view are standing still (possible jam or stopped flow)",
        "jam_end": "Traffic started moving again",
        "summary": "In view: {parts}. Traffic density: {dens}.",
        "nothing": "nothing significant",
        "veh": "vehicles: {n} ({m} moving, {s} stopped)",
        "per": "people: {n}", "sg": "signs: {names}", "acc": "accident detected",
        "path_static": "stayed {pos}", "path_move": "{a} -> {b}",
        "overall": "Video {dur}. Seen: {parts}.",
        "overall_none": "no objects detected",
        "cnt_vehicle": "vehicles", "cnt_person": "people",
        "cnt_sign": "signs", "cnt_traffic_light": "traffic lights",
        "notes": [
            "Time is given in seconds from the start (time_sec) and as MM:SS (timecode).",
            "Directions ('left', 'right', 'approaching the camera') are relative to the frame, "
            "not compass directions.",
            "Speed is measured in frame widths per second (frame_widths_per_s). It is motion in the "
            "image, not real km/h. If the camera itself moves, static objects may look moving and "
            "vice versa.",
            "Distance ('far', 'close') is estimated from the object's position in the frame.",
            "Objects are found by a neural network that can miss small or distant objects and confuse "
            "classes. Describe only what is in the data and do not invent details.",
            "Sign names come from the model (in English).",
            "'Accident' and 'overlap' events are the model's guesses, not proven facts. "
            "Phrase them carefully ('seems', 'possibly').",
        ],
    },
}


# ------------------------------------------------------------------ мелочи

def timecode(t: float) -> str:
    t = max(0.0, float(t))
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def group_of(name: str) -> str:
    n = name.lower()
    if n.startswith("sign:") or n == "stop sign":
        return "sign"
    if n == "traffic light":
        return "traffic_light"
    if n == "person":
        return "person"
    if n in VEHICLE_CLASSES:
        return "vehicle"
    return "other"


def sign_name(name: str) -> str:
    return name.split(":", 1)[1].strip() if name.lower().startswith("sign:") else name


def iou_xyxy(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Debounced:
    """Значение меняется, только если новое держится несколько отсчётов подряд."""

    def __init__(self, need: int = 2):
        self.need, self.value, self._cand, self._n, self._since = need, None, None, 0, None

    def update(self, new, t):
        """Возвращает (старое, новое, время начала нового) при смене, иначе None."""
        if self.value is None:
            self.value, self._cand, self._n = new, None, 0
            return None
        if new == self.value:
            self._cand, self._n = None, 0
            return None
        if new == self._cand:
            self._n += 1
        else:
            self._cand, self._n, self._since = new, 1, t
        if self._n >= self.need:
            old, self.value = self.value, new
            since, self._cand, self._n = self._since, None, 0
            return old, new, since
        return None


# ------------------------------------------------------------------ трекер

class Track:
    def __init__(self, tid, cls, t, bbox, conf):
        self.id, self.cls, self.group = tid, cls, group_of(cls)
        self.first_t = self.last_t = t
        self.first_bbox = self.last_bbox = list(bbox)
        self.conf_sum, self.hits = conf, 1
        self.hist = [(t, *self._cwh(bbox))]
        self.confirmed = False
        self.state = None            # "moving" | "stopped" (после сглаживания)
        self.max_speed = 0.0
        self.moving_obs = self.speed_obs = 0
        self.speed_sum = 0.0
        self.deb = Debounced(2)

    @staticmethod
    def _cwh(b):
        return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, max(b[2] - b[0], 1.0), max(b[3] - b[1], 1.0))

    def add(self, t, bbox, conf):
        self.last_t, self.last_bbox = t, list(bbox)
        self.conf_sum += conf
        self.hits += 1
        self.hist.append((t, *self._cwh(bbox)))
        if len(self.hist) > 10:
            self.hist.pop(0)

    def predict(self, t):
        cx, cy = self.hist[-1][1], self.hist[-1][2]
        if len(self.hist) >= 2:
            (t0, x0, y0, _, _), (t1, x1, y1, _, _) = self.hist[-2], self.hist[-1]
            dt = max(t1 - t0, 1e-3)
            k = 0.7 * (t - t1)
            return cx + (x1 - x0) / dt * k, cy + (y1 - y0) / dt * k
        return cx, cy

    def motion(self, W, H):
        """Движение по последним WINDOW_SEC секундам или None, если данных мало."""
        t1, x1, y1, w1, h1 = self.hist[-1]
        first = None
        for rec in self.hist[:-1]:
            if t1 - rec[0] <= WINDOW_SEC + 1e-6:
                first = rec
                break
        if first is None:
            return None
        t0, x0, y0, w0, h0 = first
        dt = max(t1 - t0, 1e-3)
        dx, dy = x1 - x0, y1 - y0
        speed = math.hypot(dx, dy) / dt / W
        ls = math.log(max(math.sqrt(w1 * h1), 1.0) / max(math.sqrt(w0 * h0), 1.0))
        moving = speed >= MOVE_THRESHOLD
        nx, ny = dx / W, dy / H
        if not moving:
            direction = "stationary"
        elif abs(nx) >= max(abs(ny), 0.5 * abs(ls)):
            direction = "right" if nx > 0 else "left"
        else:
            direction = "toward" if (ny > 0 or ls > 0) else "away"
        depth = "approaching" if ls > 0.08 else ("receding" if ls < -0.08 else "steady")
        return {"moving": moving, "direction": direction, "speed": speed,
                "speed_px": math.hypot(dx, dy) / dt, "depth_trend": depth}


class SimpleTracker:
    """Присваивает объектам устойчивые номера между проанализированными кадрами."""

    def __init__(self, max_gap=2.5):
        self.max_gap, self.active, self.finished, self._next = max_gap, [], [], 1

    def update(self, t, dets):
        """dets: список словарей с class, bbox, confidence. Возвращает (треки, закрытые)."""
        assigned = [None] * len(dets)
        by_cls = defaultdict(list)
        for i, d in enumerate(dets):
            by_cls[d["class"]].append(i)
        for cls, idxs in by_cls.items():
            cand = [tr for tr in self.active
                    if tr.cls == cls and 0 < t - tr.last_t <= self.max_gap]
            pairs = []
            for tr in cand:
                dt = t - tr.last_t
                px, py = tr.predict(t)
                for i in idxs:
                    cx, cy, w, h = Track._cwh(dets[i]["bbox"])
                    diag = math.hypot(max(w, tr.hist[-1][3]), max(h, tr.hist[-1][4]))
                    dist = math.hypot(cx - px, cy - py) / max(diag, 1.0)
                    if dist <= 0.8 + 1.2 * dt:
                        size_pen = abs(math.log((w * h) / max(tr.hist[-1][3] * tr.hist[-1][4], 1.0)))
                        pairs.append((dist + 0.25 * size_pen, id(tr), tr, i))
            pairs.sort(key=lambda p: p[0])
            used_tr, used_det = set(), set()
            for _, key, tr, i in pairs:
                if key in used_tr or i in used_det:
                    continue
                used_tr.add(key)
                used_det.add(i)
                tr.add(t, dets[i]["bbox"], dets[i]["confidence"])
                assigned[i] = tr
        for i, d in enumerate(dets):
            if assigned[i] is None:
                tr = Track(self._next, d["class"], t, d["bbox"], d["confidence"])
                self._next += 1
                self.active.append(tr)
                assigned[i] = tr
        closed, still = [], []
        for tr in self.active:
            if t - tr.last_t > self.max_gap:
                closed.append(tr)
            else:
                still.append(tr)
        self.active = still
        self.finished += closed
        return assigned, closed


# ------------------------------------------------------------------ анализ

class SceneAnalyzer:
    def __init__(self, lang="ru", track_gap=2.5):
        self.lang = lang if lang in TX else "ru"
        self.tx = TX[self.lang]
        self.labels = RU_LABELS if self.lang == "ru" else EN_LABELS
        self.tracker = SimpleTracker(track_gap)
        self.events, self.samples = [], []
        self._wmax = self._hmax = 1.0
        self._dens, self._flow = Debounced(2), Debounced(2)
        self._acc = None                  # открытый отрезок ДТП
        self._pair_last = {}
        self._signs_seen = {}             # название знака -> первое время
        self._moments_t = []
        self._t_first = None

    # ---- текст
    def label(self, cls):
        g = group_of(cls)
        if g == "sign":
            return (f"знак «{sign_name(cls)}»" if self.lang == "ru" else f"sign '{sign_name(cls)}'")
        return self.labels.get(cls, cls)

    def pos_text(self, h, d):
        return f"{self.tx['h'][h]} {self.tx['d'][d]}"

    def _event(self, t, typ, importance, description, tr=None, **details):
        ev = {"t": round(t, 2), "timecode": timecode(t), "type": typ, "importance": importance,
              "description": description}
        if tr is not None:
            ev.update({"track_id": tr.id, "class": tr.cls, "group": tr.group})
        if details:
            ev["details"] = details
        self.events.append(ev)

    # ---- геометрия
    @staticmethod
    def _position(bbox, W, H):
        cx, y2 = (bbox[0] + bbox[2]) / 2 / W, bbox[3] / H
        h = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "center")
        d = "close" if y2 >= 0.8 else ("medium" if y2 >= 0.55 else "far")
        return h, d

    @staticmethod
    def _edge(bbox, W, H, d):
        if bbox[0] <= EDGE * W:
            return "left"
        if bbox[2] >= (1 - EDGE) * W:
            return "right"
        if bbox[3] >= (1 - EDGE) * H:
            return "bottom"
        if bbox[1] <= EDGE * H:
            return "top"
        return "far" if d == "far" else "inside"

    # ---- главная функция: вызывается для каждого проанализированного кадра
    def process(self, record, size=None):
        objects, accidents = record["objects"], record.get("events") or []
        t = record["time_sec"]
        if self._t_first is None:
            self._t_first = t
        W, H = (size or (0, 0))
        for o in objects:
            self._wmax = max(self._wmax, o["bbox"][2])
            self._hmax = max(self._hmax, o["bbox"][3])
        W, H = (W or self._wmax), (H or self._hmax)

        tracks, closed = self.tracker.update(t, objects)
        for tr in closed:
            self._on_closed(tr, W, H)

        vehicles = moving = stopped = persons = 0
        signs = []
        for o, tr in zip(objects, tracks):
            h, d = self._position(o["bbox"], W, H)
            bx = o["bbox"]
            o["track_id"], o["group"] = tr.id, tr.group
            o["center"] = [round((bx[0] + bx[2]) / 2 / W, 3), round((bx[1] + bx[3]) / 2 / H, 3)]
            bw, bh = (bx[2] - bx[0]) / W, (bx[3] - bx[1]) / H
            o["size"] = {"w": round(bw, 3), "h": round(bh, 3), "area_pct": round(bw * bh * 100, 2)}
            o["position"] = {"horizontal": h, "distance": d}
            o["track_age_s"] = round(t - tr.first_t, 2)
            if not tr.confirmed and tr.hits >= MIN_HITS:
                self._confirm(tr, W, H)
            if tr.group in ("vehicle", "person"):
                m = tr.motion(W, H)
                if m is None:
                    o["motion"] = {"state": "unknown"}
                else:
                    state = "moving" if m["moving"] else "stopped"
                    self._update_state(tr, state, m, t, h, d)
                    o["motion"] = {"state": state, "direction": m["direction"],
                                   "depth_trend": m["depth_trend"],
                                   "speed": {"frame_widths_per_s": round(m["speed"], 3),
                                             "px_per_s": round(m["speed_px"], 1)}}
                    if m["speed"] >= FAST_THRESHOLD:
                        o["motion"]["fast"] = True
                    tr.max_speed = max(tr.max_speed, m["speed"])
                    tr.speed_obs += 1
                    tr.speed_sum += m["speed"]
                    tr.moving_obs += 1 if m["moving"] else 0
                if tr.group == "vehicle":
                    vehicles += 1
                    st = o["motion"].get("state")
                    moving += st == "moving"
                    stopped += st == "stopped"
                else:
                    persons += 1
            elif tr.group == "sign":
                signs.append(sign_name(o["class"]))
            elif tr.group == "traffic_light":
                signs.append("traffic light")

        self._check_overlaps(objects, tracks, t)
        self._check_accident(accidents, t, W, H)

        density = ("none" if vehicles == 0 else "light" if vehicles <= 2
                   else "moderate" if vehicles <= 6 else "heavy")
        known = moving + stopped
        flow = ("unknown" if known == 0 else
                "mostly_stopped" if (known >= 3 and stopped / known >= 0.7) else "free_or_mixed")
        ch = self._dens.update(density, t)
        if ch and {ch[0], ch[1]} != {"none"}:
            self._event(ch[2], "traffic_density_changed", 2, self.tx["density"].format(
                old=self.tx["dens"][ch[0]], new=self.tx["dens"][ch[1]], n=vehicles),
                from_=ch[0], to=ch[1])
        ch = self._flow.update(flow, t)
        if ch:
            if ch[1] == "mostly_stopped":
                self._event(ch[2], "vehicles_mostly_stopped", 3, self.tx["jam"])
            elif ch[0] == "mostly_stopped":
                self._event(ch[2], "traffic_resumed", 2, self.tx["jam_end"])

        parts = []
        if vehicles:
            parts.append(self.tx["veh"].format(n=vehicles, m=moving, s=stopped))
        if persons:
            parts.append(self.tx["per"].format(n=persons))
        if signs:
            parts.append(self.tx["sg"].format(names=", ".join(sorted(set(signs)))))
        if accidents:
            parts.append(self.tx["acc"])
        scene = {"vehicles": {"total": vehicles, "moving": moving, "stopped": stopped},
                 "persons": persons, "signs": sorted(set(signs)), "traffic_density": density,
                 "flow": flow, "accident": bool(accidents),
                 "text": self.tx["summary"].format(parts="; ".join(parts) or self.tx["nothing"],
                                                   dens=self.tx["dens"][density])}
        record["sample"] = len(self.samples)
        record["timecode"] = timecode(t)
        record["scene"] = scene
        self.samples.append({"t": t, **{k: scene[k] for k in
                             ("vehicles", "persons", "signs", "traffic_density", "flow", "accident")}})
        return record

    # ---- треки: подтверждение и состояния
    def _confirm(self, tr, W, H):
        tr.confirmed = True
        h, d = self._position(tr.first_bbox, W, H)
        edge = self._edge(tr.first_bbox, W, H, d)
        label = self.label(tr.cls)
        if tr.group in ("vehicle", "person") and tr.first_t <= (self._t_first or 0) + 1e-6:
            self._event(tr.first_t, "object_present_at_start", 1,
                        self.tx["present"].format(label=label, id=tr.id, pos=self.pos_text(h, d)),
                        tr, position={"horizontal": h, "distance": d})
        elif tr.group in ("vehicle", "person"):
            self._event(tr.first_t, "object_appeared", 3 if tr.group == "person" else 1,
                        self.tx["appear"].format(label=label, id=tr.id, entry=self.tx["entry"][edge]),
                        tr, entered_from=edge, position={"horizontal": h, "distance": d})
        elif tr.group == "sign":
            name = sign_name(tr.cls)
            first = name not in self._signs_seen
            self._signs_seen.setdefault(name, tr.first_t)
            self._event(tr.first_t, "sign_seen", 3 if first else 1,
                        self.tx["sign_seen"].format(label=label, pos=self.pos_text(h, d)), tr,
                        sign=name, position={"horizontal": h, "distance": d})
        elif tr.group == "traffic_light":
            self._event(tr.first_t, "traffic_light_seen", 3,
                        self.tx["light_seen"].format(pos=self.pos_text(h, d)), tr,
                        position={"horizontal": h, "distance": d})

    def _update_state(self, tr, state, m, t, h, d):
        change = tr.deb.update(state, t)
        if tr.state is None:
            tr.state = state
        if not change:
            return
        old, new, since = change
        tr.state = new
        label, pos = self.label(tr.cls), self.pos_text(h, d)
        if new == "stopped":
            prev = tr.max_speed
            abrupt = tr.group == "vehicle" and prev >= STOP_SPEED_ABRUPT
            self._event(since, "object_stopped", 4 if abrupt else 2,
                        self.tx["stopped_abrupt" if abrupt else "stopped"].format(
                            label=label, id=tr.id, pos=pos), tr, abrupt=abrupt)
        else:
            self._event(since, "object_started_moving", 2,
                        self.tx["started"].format(label=label, id=tr.id,
                                                  dir=self.tx["dir"][m["direction"]], pos=pos),
                        tr, direction=m["direction"])

    def _on_closed(self, tr, W, H):
        if not tr.confirmed or tr.group not in ("vehicle", "person"):
            return
        h, d = self._position(tr.last_bbox, W, H)
        edge = self._edge(tr.last_bbox, W, H, d)
        self._event(tr.last_t, "object_left", 1,
                    self.tx["left_scene"].format(label=self.label(tr.cls), id=tr.id,
                                                 exit=self.tx["exit"][edge]), tr, exited_to=edge)

    # ---- пары машин и ДТП
    def _check_overlaps(self, objects, tracks, t):
        veh = [(o, tr) for o, tr in zip(objects, tracks) if tr.group == "vehicle" and tr.hits >= MIN_HITS]
        for i in range(len(veh)):
            for j in range(i + 1, len(veh)):
                (oa, ta), (ob, tb) = veh[i], veh[j]
                if iou_xyxy(oa["bbox"], ob["bbox"]) < OVERLAP_IOU:
                    continue
                key = tuple(sorted((ta.id, tb.id)))
                if t - self._pair_last.get(key, -1e9) < OVERLAP_COOLDOWN:
                    continue
                self._pair_last[key] = t
                self._event(t, "vehicles_overlap", 4, self.tx["overlap"].format(
                    la=self.label(ta.cls), a=ta.id, lb=self.label(tb.cls), b=tb.id),
                    track_ids=list(key), iou=round(iou_xyxy(oa["bbox"], ob["bbox"]), 2))

    def _check_accident(self, accidents, t, W, H):
        if accidents:
            best = max(accidents, key=lambda e: e["confidence"])
            if self._acc is None:
                self._acc = {"from": t, "to": t, "conf": best["confidence"], "bbox": best["bbox"]}
            else:
                self._acc["to"] = t
                if best["confidence"] >= self._acc["conf"]:
                    self._acc["conf"], self._acc["bbox"] = best["confidence"], best["bbox"]
        elif self._acc is not None and t - self._acc["to"] > 2.0:
            self._close_accident(W, H)

    def _close_accident(self, W, H):
        a, self._acc = self._acc, None
        if a is None:
            return
        h, _ = self._position(a["bbox"], W, H)
        span = (self.tx["span_range"].format(a=timecode(a["from"]), b=timecode(a["to"]))
                if a["to"] > a["from"] else self.tx["span_at"].format(a=timecode(a["from"])))
        self._event(a["from"], "accident_detected", 5, self.tx["accident"].format(
            span=span, conf=f"{a['conf']:.2f}", part=self.tx["part"][h]),
            to_sec=round(a["to"], 2), max_confidence=round(a["conf"], 3))

    # ---- итоги
    def finalize(self, video=None, digest_every=5.0, digest_events=120, digest_tracks=40):
        video = video or {}
        W, H = (video.get("width") or self._wmax), (video.get("height") or self._hmax)
        self._close_accident(W, H)
        all_tracks = [tr for tr in self.tracker.finished + self.tracker.active if tr.confirmed]
        events = sorted(self.events, key=lambda e: (e["t"], -e["importance"]))
        for i, e in enumerate(events, 1):
            e["id"] = i
        tracks = [self._track_summary(tr, W, H, tr in self.tracker.active) for tr in
                  sorted(all_tracks, key=lambda x: x.first_t)]
        scene = self._scene_summary(tracks, events, video)
        digest = self._digest(video, scene, events, tracks, digest_every, digest_events, digest_tracks)
        return {"scene": scene, "timeline": events, "tracks": tracks, "digest": digest}

    def _track_summary(self, tr, W, H, visible_at_end):
        h0, d0 = self._position(tr.first_bbox, W, H)
        h1, d1 = self._position(tr.last_bbox, W, H)
        s = {"track_id": tr.id, "class": tr.cls, "group": tr.group, "label": self.label(tr.cls),
             "first_seen": round(tr.first_t, 2), "first_timecode": timecode(tr.first_t),
             "last_seen": round(tr.last_t, 2), "last_timecode": timecode(tr.last_t),
             "duration_sec": round(tr.last_t - tr.first_t, 2), "observations": tr.hits,
             "avg_confidence": round(tr.conf_sum / tr.hits, 3),
             "still_visible_at_end": visible_at_end,
             "start_position": {"horizontal": h0, "distance": d0},
             "end_position": {"horizontal": h1, "distance": d1}}
        if tr.group == "sign":
            s["sign"] = sign_name(tr.cls)
        if tr.group in ("vehicle", "person"):
            total = max(tr.last_t - tr.first_t, 1e-3)
            x0, y0, w0, h_0 = tr.hist[0][1:]
            x1, y1, w1, h_1 = tr.hist[-1][1:]
            net = math.hypot(tr.hist[-1][1] - tr.hist[0][1], tr.hist[-1][2] - tr.hist[0][2]) / total / W
            share = tr.moving_obs / tr.speed_obs if tr.speed_obs else None
            s["motion"] = {
                "overall": ("unknown" if share is None else "mostly_moving" if share >= 0.66
                            else "mostly_stopped" if share <= 0.33 else "mixed"),
                "avg_speed_frame_widths_per_s": round(tr.speed_sum / tr.speed_obs, 3) if tr.speed_obs else None,
                "max_speed_frame_widths_per_s": round(tr.max_speed, 3),
                "net_speed_frame_widths_per_s": round(net, 3)}
            if len(tr.hist) >= 2 and net >= MOVE_THRESHOLD:
                nx, ny = (x1 - x0) / W, (y1 - y0) / H
                ls = math.log(max(math.sqrt(w1 * h_1), 1.0) / max(math.sqrt(w0 * h_0), 1.0))
                s["motion"]["net_direction"] = (
                    ("right" if nx > 0 else "left") if abs(nx) >= max(abs(ny), 0.5 * abs(ls))
                    else ("toward" if (ny > 0 or ls > 0) else "away"))
            s["path"] = (self.tx["path_static"].format(pos=self.pos_text(h1, d1))
                         if (h0, d0) == (h1, d1) else self.tx["path_move"].format(
                             a=self.pos_text(h0, d0), b=self.pos_text(h1, d1)))
        return s

    def _scene_summary(self, tracks, events, video):
        dur = video.get("duration_sec") or (self.samples[-1]["t"] if self.samples else 0.0)
        uniq, by_cls = Counter(), Counter()
        for tr in tracks:
            uniq[tr["group"]] += 1
            by_cls[tr["class"]] += 1
        maxc = {"vehicle": 0, "person": 0}
        busiest = None
        for sm in self.samples:
            maxc["vehicle"] = max(maxc["vehicle"], sm["vehicles"]["total"])
            maxc["person"] = max(maxc["person"], sm["persons"])
            if busiest is None or sm["vehicles"]["total"] > busiest["vehicles"]:
                busiest = {"time_sec": round(sm["t"], 2), "timecode": timecode(sm["t"]),
                           "vehicles": sm["vehicles"]["total"]}
        signs = {}
        for tr in tracks:
            if tr["group"] == "sign":
                e = signs.setdefault(tr["sign"], {"sign": tr["sign"], "first_seen": tr["first_seen"],
                                                  "first_timecode": tr["first_timecode"],
                                                  "last_seen": tr["last_seen"], "times_seen": 0})
                e["last_seen"] = max(e["last_seen"], tr["last_seen"])
                e["times_seen"] += 1
        acc = [{"from": e["timecode"], "from_sec": e["t"],
                "to_sec": e.get("details", {}).get("to_sec", e["t"]),
                "max_confidence": e.get("details", {}).get("max_confidence")}
               for e in events if e["type"] == "accident_detected"]
        counts = {g: uniq[g] for g in ("vehicle", "person", "sign", "traffic_light") if uniq[g]}
        parts = ", ".join(f"{self.tx['cnt_' + g]}: {n}" for g, n in counts.items())
        return {
            "duration_sec": round(dur, 2), "duration_timecode": timecode(dur),
            "samples_analyzed": len(self.samples),
            "unique_objects": dict(counts), "unique_by_class": dict(by_cls),
            "max_simultaneous": maxc, "busiest_moment": busiest,
            "signs_seen": sorted(signs.values(), key=lambda e: e["first_seen"]),
            "accident": {"detected": bool(acc), "segments": acc},
            "density_timeline": self._timeline("traffic_density"),
            "flow_timeline": self._timeline("flow"),
            "text": self.tx["overall"].format(
                dur=timecode(dur), parts=parts or self.tx["overall_none"]),
        }

    def _timeline(self, key):
        out = []
        for sm in self.samples:
            if out and out[-1]["value"] == sm[key]:
                out[-1]["to"] = timecode(sm["t"])
                out[-1]["to_sec"] = round(sm["t"], 2)
            else:
                out.append({"from": timecode(sm["t"]), "from_sec": round(sm["t"], 2),
                            "to": timecode(sm["t"]), "to_sec": round(sm["t"], 2), "value": sm[key]})
        return out

    def _digest(self, video, scene, events, tracks, every, max_events, max_tracks):
        # события: все важные, остальные по порядку, пока не заполнится лимит
        picked = sorted(events, key=lambda e: (-e["importance"], e["t"]))[:max_events]
        picked = sorted(picked, key=lambda e: e["t"])
        moments = []
        if self.samples:
            end = self.samples[-1]["t"]
            a = 0.0
            while a <= end + 1e-6:
                b = a + every
                win = [s for s in self.samples if a <= s["t"] < b]
                if win:
                    dens = Counter(s["traffic_density"] for s in win).most_common(1)[0][0]
                    flow = Counter(s["flow"] for s in win).most_common(1)[0][0]
                    moments.append({
                        "from": timecode(a), "to": timecode(min(b, max(end, a))),
                        "vehicles_max": max(s["vehicles"]["total"] for s in win),
                        "moving_max": max(s["vehicles"]["moving"] for s in win),
                        "stopped_max": max(s["vehicles"]["stopped"] for s in win),
                        "persons_max": max(s["persons"] for s in win),
                        "signs": sorted({n for s in win for n in s["signs"]}),
                        "traffic_density": dens, "flow": flow,
                        "accident": any(s["accident"] for s in win),
                        "timeline_ids": [e["id"] for e in events if a <= e["t"] < b and e["importance"] >= 2]})
                a = b
        important = [tr for tr in tracks if tr["group"] in ("vehicle", "person")]
        important = sorted(important, key=lambda tr: (-tr["duration_sec"], tr["first_seen"]))[:max_tracks]
        compact = [{k: tr[k] for k in ("track_id", "label", "first_timecode", "last_timecode",
                                       "duration_sec", "path", "motion", "still_visible_at_end")
                    if k in tr} for tr in sorted(important, key=lambda tr: tr["first_seen"])]
        return {
            "schema": "scenevision-llm-digest/1",
            "language": self.lang,
            "video": {k: video.get(k) for k in ("file", "source_type", "width", "height", "duration_sec")},
            "how_to_read": self.tx["notes"],
            "scene": {k: scene[k] for k in ("text", "duration_timecode", "unique_objects",
                                            "max_simultaneous", "busiest_moment", "signs_seen",
                                            "accident", "density_timeline", "flow_timeline")},
            "timeline": [{k: e[k] for k in ("id", "timecode", "t", "type", "importance", "description")}
                         for e in picked],
            "moments": moments,
            "main_tracks": compact,
            "timeline_total": len(events), "timeline_in_digest": len(picked),
        }
