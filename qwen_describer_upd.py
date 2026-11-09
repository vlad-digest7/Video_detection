"""
Описание видео языковой моделью Qwen2.5 по сводке result_llm.json.

Модуль не зависит от Qt и OpenCV: тяжёлые библиотеки (torch, transformers) подключаются только
в момент загрузки модели. Основное:

    digest = load_digest("video_result_llm.json")
    describer = QwenDescriber("Qwen/Qwen2.5-1.5B-Instruct", device="auto")
    text = describer.describe(digest, on_text=print)        # печатает текст по мере генерации

Можно запустить и без GUI:
    python qwen_describer.py video_result_llm.json
    python qwen_describer.py video_result_llm.json --model Qwen/Qwen2.5-3B-Instruct --device mps
"""

from __future__ import annotations

import copy
import html
import json
import re
import threading
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_HINTS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]
MAX_DIGEST_CHARS = 14000     # ограничение размера сводки в запросе (примерно 6-8 тысяч токенов)

SYSTEM_RU = """Ты пишешь живое описание видео с дорожной камеры по структурированным данным детекции объектов. \
Пиши по-русски образно, как короткий репортаж очевидца, но строго опираясь на данные.

Правила:
1. Используй только факты из данных. Не выдумывай цвет и марку машин, погоду, время суток, \
город, людей и события, которых в данных нет.
2. События «ДТП» и «перекрытие автомобилей» это предположения нейросети: пиши о них осторожно \
(«похоже», «возможно»), не утверждай как установленный факт.
3. Названия дорожных знаков в данных на английском. Переведи их на русский, используя названия из ПДД РФ, \
когда уверен; иначе опиши словами.
4. Время указывай в формате ММ:СС (например, 00:12) и только то, что есть в данных.
5. Направления даны относительно кадра: «влево» это к левому краю кадра. Скорость в данных это движение \
в кадре, а не км/ч: не называй скорость в километрах в час.
6. Не перечисляй номера объектов подряд; упоминай номер («автомобиль №3») только если это помогает рассказу.
7. Если объектов мало или ничего не происходит, так и скажи. Не придумывай происшествий.

Структура ответа: сначала один абзац с общей картиной (обстановка, плотность движения, что видно), затем \
хронология ключевых моментов (3-8 коротких абзацев, каждый начинается с таймкода), в конце короткий вывод. \
Пиши обычным текстом без заголовков, звёздочек и маркированных списков."""

SYSTEM_EN = """You write a vivid description of a road-camera video from structured object-detection data. \
Write in English in the style of a short eyewitness report, strictly based on the data.

Rules:
1. Use only facts from the data. Do not invent car colors or makes, weather, time of day, location, people or events \
that are not in the data.
2. 'Accident' and 'overlap' events are the neural network's guesses: phrase them carefully ('seems', 'possibly').
3. Times are given as MM:SS; use only times that appear in the data.
4. Directions are relative to the frame. Speed is motion in the image, not km/h: never state km/h.
5. Do not list object numbers one after another.
6. If little happens, say so. Do not invent incidents.

Structure: one paragraph with the overall picture, then the key moments in chronological order (3-8 short paragraphs, \
each starting with a timecode), then a short conclusion. Plain text only, no headings, asterisks or bullet lists."""


# ------------------------------------------------------------------ сводка

def load_digest(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def description_path(llm_json_path) -> Path:
    """Куда сохраняется готовое описание: рядом с result_llm.json."""
    p = Path(llm_json_path)
    stem = p.stem[:-4] if p.stem.endswith("_llm") else p.stem
    return p.with_name(stem + "_description.txt")


def read_cached(llm_json_path):
    """Готовое описание, если оно уже есть и новее сводки (иначе None)."""
    out = description_path(llm_json_path)
    try:
        if out.is_file() and out.stat().st_mtime >= Path(llm_json_path).stat().st_mtime:
            text = out.read_text(encoding="utf-8").strip()
            return text or None
    except OSError:
        pass
    return None


def save_description(llm_json_path, text: str) -> Path:
    out = description_path(llm_json_path)
    out.write_text(text.strip() + "\n", encoding="utf-8")
    return out


def _compact(d) -> str:
    return json.dumps(d, ensure_ascii=False, separators=(",", ":"))


def shrink_digest(digest: dict, max_chars: int = MAX_DIGEST_CHARS) -> dict:
    """Копия сводки, урезанная до max_chars: сначала лишние поля, затем малозначимые события."""
    d = copy.deepcopy(digest)
    d.pop("how_to_read", None)               # правила уже в системной инструкции
    if len(_compact(d)) <= max_chars:
        return d
    steps = [
        lambda: d.__setitem__("main_tracks", d.get("main_tracks", [])[:12]),
        lambda: d.__setitem__("moments", [m for m in d.get("moments", []) if m.get("timeline_ids")
                                          or m.get("accident")]),
        lambda: d.__setitem__("main_tracks", d.get("main_tracks", [])[:4]),
        lambda: d.get("scene", {}).pop("density_timeline", None),
        lambda: d.get("scene", {}).pop("flow_timeline", None),
        lambda: d.__setitem__("main_tracks", []),
    ]
    for step in steps:
        step()
        if len(_compact(d)) <= max_chars:
            return d
    # ещё много: оставляем самые важные события (порядок по времени сохраняется)
    events = d.get("timeline", [])
    keep = len(events)
    while keep > 5 and len(_compact(d)) > max_chars:
        keep = int(keep * 0.8)
        top = sorted(events, key=lambda e: (-e.get("importance", 1), e.get("t", 0)))[:keep]
        d["timeline"] = sorted(top, key=lambda e: e.get("t", 0))
        d["moments"] = d.get("moments", [])[: max(10, keep)]
    return d


def build_messages(digest: dict, lang: str = "ru") -> list:
    system = SYSTEM_RU if lang == "ru" else SYSTEM_EN
    data = _compact(shrink_digest(digest))
    if lang == "ru":
        user = f"Данные о видео (JSON):\n{data}\n\nСоставь описание видео."
    else:
        user = f"Video data (JSON):\n{data}\n\nWrite the video description."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ------------------------------------------------------------------ вывод текста

_TC = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?::(\d{2}))?(?![\d:])")


def clean_text(text: str) -> str:
    """Убирает markdown-разметку, которую маленькие модели иногда добавляют."""
    text = re.sub(r"\*\*|__", "", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*[-*•]\s+", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def timecode_to_seconds(a: str, b: str, c=None) -> float:
    if c is not None:
        return int(a) * 3600 + int(b) * 60 + int(c)
    return int(a) * 60 + int(b)


def description_to_html(text: str, fps: float, fg: str = "#26384B", link: str = "#1769AA") -> str:
    """HTML для QTextBrowser: абзацы, а таймкоды (ММ:СС) становятся ссылками на кадр."""
    fps = fps if fps and fps > 0 else 30.0

    def repl(m):
        seconds = timecode_to_seconds(*m.groups())
        return (f"<a href='{int(seconds * fps)}' style='color:{link};text-decoration:none;"
                f"font-weight:700;'>{m.group(0)}</a>")

    paragraphs = [p.strip() for p in clean_text(text).split("\n\n") if p.strip()]
    out = []
    for p in paragraphs:
        escaped = html.escape(p).replace("\n", "<br>")
        out.append(f"<p style='color:{fg};margin:0 0 9px 0;line-height:140%;'>"
                   f"{_TC.sub(repl, escaped)}</p>")
    return "".join(out)


# ------------------------------------------------------------------ модель

def pick_device(pref: str = "auto") -> str:
    import torch
    pref = (pref or "auto").lower()
    if pref == "cuda" and torch.cuda.is_available():
        return "cuda"
    if pref == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if pref == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype_kwarg(transformers_module) -> str:
    """В новых версиях transformers параметр называется dtype, в старых torch_dtype."""
    try:
        major, minor = (int(x) for x in transformers_module.__version__.split(".")[:2])
        return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    except Exception:
        return "torch_dtype"


class QwenDescriber:
    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "auto"):
        self.model_id = (model_id or DEFAULT_MODEL).strip()
        self.device_pref = device or "auto"
        self.device = None
        self.tokenizer = None
        self.model = None
        self._cancel = threading.Event()

    # ---- состояние
    def matches(self, model_id: str, device: str) -> bool:
        return self.model_id == (model_id or DEFAULT_MODEL).strip() and self.device_pref == (device or "auto")

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def cancel(self):
        self._cancel.set()

    def reset_cancel(self):
        """Сбрасывает отмену ДО запуска нового потока генерации."""
        self._cancel.clear()

    def unload(self):
        self.model = self.tokenizer = None
        try:
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ---- загрузка
    def load(self, status=None):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = pick_device(self.device_pref)
        if self.device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif self.device == "mps":
            dtype = torch.float16
        else:
            dtype = torch.float32
        if status:
            status(f"Загрузка модели {self.model_id} ({self.device}). Первый запуск скачивает файлы модели, это может занять время…")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, **{_dtype_kwarg(transformers): dtype})
        self.model.to(self.device)
        self.model.eval()
        if status:
            status("Модель загружена, идёт генерация описания…")

    # ---- генерация
    def describe(self, digest: dict, lang: str = "ru", max_new_tokens: int = 700,
                 temperature: float = 0.4, on_text=None, status=None) -> str:
        """Возвращает готовый текст; по мере генерации вызывает on_text(кусок). cancel() прерывает."""
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextStreamer

        # Флаг отмены сбрасывается вызывающим кодом непосредственно перед стартом
        # рабочего потока. Здесь его не трогаем: иначе быстрый cancel до входа в
        # describe() мог бы потеряться.
        if not self.loaded:
            self.load(status)
        elif status:
            status("Идёт генерация описания…")
        if self._cancel.is_set():
            return ""

        messages = build_messages(digest, lang)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        chunks = []
        outer = self

        class _Stream(TextStreamer):
            def on_finalized_text(self, text, stream_end=False):
                if text:
                    chunks.append(text)
                    if on_text:
                        on_text(text)

        class _Cancel(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return torch.full((input_ids.shape[0],), outer._cancel.is_set(),
                                  dtype=torch.bool, device=input_ids.device)

        gen = dict(max_new_tokens=int(max_new_tokens), repetition_penalty=1.08,
                   streamer=_Stream(self.tokenizer, skip_prompt=True, skip_special_tokens=True),
                   stopping_criteria=StoppingCriteriaList([_Cancel()]))
        if temperature and temperature > 0:
            gen.update(do_sample=True, temperature=float(temperature), top_p=0.9)
        else:
            gen.update(do_sample=False)
        with torch.no_grad():
            self.model.generate(**inputs, **gen)
        return "".join(chunks).strip()


# ------------------------------------------------------------------ запуск без GUI

def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Описание видео по result_llm.json (Qwen2.5)")
    ap.add_argument("llm_json", help="файл *_llm.json, созданный main.py")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="имя модели на Hugging Face или папка")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    ap.add_argument("--lang", default="ru", choices=("ru", "en"))
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--temperature", type=float, default=0.4)
    args = ap.parse_args()

    digest = load_digest(args.llm_json)
    describer = QwenDescriber(args.model, args.device)
    text = describer.describe(digest, lang=args.lang, max_new_tokens=args.max_tokens,
                              temperature=args.temperature,
                              on_text=lambda t: print(t, end="", flush=True), status=print)
    print()
    out = save_description(args.llm_json, clean_text(text))
    print(f"\nОписание сохранено: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
