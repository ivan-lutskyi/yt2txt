#!/usr/bin/env python3
"""
yt2txt — YouTube → локальний транскрипт (faster-whisper) → Markdown-файл з промптом для ШІ.

Використання:
    python yt2txt.py "https://youtu.be/XXXX"
    python yt2txt.py URL1 URL2 --lang uk --model large-v3
    python yt2txt.py URL --prompt-file my_prompt.md --out D:/transcripts
"""
import argparse
import io
import math
import os
import re
import site
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

# cuBLAS / cuDNN з pip-пакетів nvidia-* у .venv: ctranslate2 шукає їхні DLL через PATH
for _sp in site.getsitepackages():
    for _lib in ("cublas", "cudnn"):
        _bin = Path(_sp) / "nvidia" / _lib / "bin"
        if _bin.is_dir():
            os.environ["PATH"] = f"{_bin}{os.pathsep}{os.environ['PATH']}"

import numpy as np
import yt_dlp
from faster_whisper import WhisperModel
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_BOLD = r"C:\Windows\Fonts\segoeuib.ttf"
FONT_REGULAR = r"C:\Windows\Fonts\segoeui.ttf"
STAMP_COLOR = (22, 163, 74, 235)  # зелений бейдж
COVER_BG = ((10, 15, 30), (22, 32, 58))  # градієнт власної обкладинки: верх, низ

DEFAULT_PROMPT = """\
Ти — аналітик і фактчекер. Вище — автоматичний транскрипт відео з YouTube (метадані на початку файлу).
Транскрипт зроблено розпізнаванням мовлення, тому можливі помилки в іменах, назвах, числах і термінах.
Враховуй це й не будуй висновків на явно спотворених фрагментах; якщо щось виглядає як помилка розпізнавання, так і познач.

Напиши українською аналітичну статтю для публікації на telegra.ph. Читач відео не дивився, тож стаття має бути зрозумілою без нього.

ФОРМАТ (обмеження редактора telegra.ph — дотримуйся суворо):
- Лише два рівні заголовків: ## для розділів, ### для підрозділів. Без #.
- Жодних таблиць, блоків коду, вкладених списків і горизонтальних ліній усередині статті.
- Дозволено: абзаци, **жирний**, *курсив*, посилання, марковані й нумеровані списки (один рівень), цитати (>).
- Короткі абзаци по 2–4 речення.
- Кожен таймкод оформлюй посиланням на момент відео: [12:34](https://youtu.be/ID?t=754) — шаблон є в метаданих, секунди рахуй від таймкоду.
- Пряму мову автора відео подавай цитатою (>) з таймкодом-посиланням.

СТРУКТУРА:
1. Перший рядок — заголовок статті звичайним текстом без розмітки (я вставлю його в поле назви). До 70 символів, інформативний, без клікбейту.
2. *Лід курсивом* (2–3 речення): чиє відео, про що, і головний висновок аналізу.
3. ## Про що відео — головна теза автора і ключові аргументи списком у логічному порядку, з таймкодами.
4. ## Перевірка фактів — лише конкретні перевірні твердження (цифри, дати, події, цитати, наукові твердження). Кожне — пунктом списку у форматі:
   **«Твердження»** ([мм:сс](посилання)) — **оцінка** (підтверджується / частково / не підтверджується / неможливо перевірити). Пояснення 1–2 речення, джерела посиланнями.
   Якщо маєш доступ до пошуку в інтернеті — використай його. Якщо ні — не вигадуй джерел і познач такі пункти як «потребує перевірки».
5. ## Як побудована аргументація — на чому тримаються тези; логічні хиби, підміна понять, вибірковість даних, маніпулятивні прийоми
   (емоційний тиск, апеляція до авторитету, хибна дилема, узагальнення тощо) з прикладами й таймкодами. Окремим підрозділом ### — що в аргументації сильне.
6. ## Чого не прозвучало — важливий контекст і альтернативні погляди.
7. ## Висновок — наскільки відео заслуговує довіри (**N/10** з обґрунтуванням), кому і для чого воно корисне, що з нього варто взяти.

ТОН: неупереджено й предметно, без оцінок особистості автора. Чітко розділяй, що сказав автор, а що — твоя оцінка. Не переказуй транскрипт дослівно.

Після статті додай рядок «===== НЕ ДЛЯ ПУБЛІКАЦІЇ =====» і під ним нотатки для мене як редактора:
місця, де транскрипт схожий на помилку розпізнавання (імена, числа, терміни), що варто перевірити вручну перед публікацією, і чи робилась перевірка з вебпошуком (якщо ні — до якої дати актуальні твої знання).
"""


def fmt_ts(seconds: float) -> str:
    s = int(seconds)
    h, m, s = s // 3600, (s % 3600) // 60, s % 60
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def safe_name(s: str, maxlen: int = 80) -> str:
    s = re.sub(r'[\\/:*?"<>|\r\n\t]+', " ", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    return s[:maxlen].rstrip(" .") or "video"


def download_audio(url: str, workdir: Path):
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(workdir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    files = list(workdir.glob(f"{info['id']}.*"))
    if not files:
        raise RuntimeError("Аудіо не завантажилось")
    return info, files[0]


def draw_badge(img: Image.Image, text: str) -> Image.Image:
    """Ставить у правий верхній кут бейдж «✓ <text>» (унизу превʼю зазвичай назва й логотип каналу)."""
    W = img.width
    font = ImageFont.truetype(FONT_BOLD, round(W * 0.042))
    pad, margin = round(W * 0.018), round(W * 0.025)
    l, t, r, b = font.getbbox(text)
    tw, th = r - l, b - t
    check = round(th * 1.1)  # ширина галочки
    gap = round(pad * 0.8)
    bw, bh = pad + check + gap + tw + pad, th + 2 * pad
    x0, y0 = W - margin - bw, margin

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rounded_rectangle((x0 + 4, y0 + 6, x0 + bw + 4, y0 + bh + 6), radius=bh // 4, fill=(0, 0, 0, 90))  # тінь
    d.rounded_rectangle((x0, y0, x0 + bw, y0 + bh), radius=bh // 4, fill=STAMP_COLOR)
    # галочка малюється лініями, бо в шрифті може не бути символу ✓
    cx, cy = x0 + pad, y0 + pad
    d.line([(cx, cy + th * 0.55), (cx + check * 0.38, cy + th * 0.92), (cx + check, cy + th * 0.08)],
           fill="white", width=max(3, round(th * 0.16)), joint="curve")
    d.text((cx + check + gap - l, cy - t), text, font=font, fill="white")
    return Image.alpha_composite(img.convert("RGBA"), overlay)


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        test = f"{cur} {word}".strip()
        if font.getlength(test) <= max_w or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def thumb_cover(info: dict, out_path: Path, stamp: str):
    """Превʼю відео з бейджем."""
    req = urllib.request.Request(info["thumbnail"], headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        img = Image.open(io.BytesIO(r.read())).convert("RGBA")
    if img.width < 1280:
        img = img.resize((1280, round(img.height * 1280 / img.width)), Image.LANCZOS)
    draw_badge(img, stamp).convert("RGB").save(out_path, "JPEG", quality=92)


def own_cover(info: dict, out_path: Path):
    """Власна обкладинка без чужих зображень: плашка «АНАЛІЗ ВІДЕО», назва відео в лапках,
    лупа з кнопкою play, таймлайн з позначками, канал · дата · тривалість."""
    k = 2  # малюємо вдвічі більше й зменшуємо — так краї згладжені
    W, H = 1280 * k, 720 * k
    accent = STAMP_COLOR[:3]
    top, bottom = COVER_BG
    grad = Image.linear_gradient("L").resize((W, H))
    img = Image.composite(Image.new("RGB", (W, H), bottom), Image.new("RGB", (W, H), top), grad).convert("RGBA")

    # точкова сітка й зелене світіння за лупою
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    for x in range(40 * k, W, 40 * k):
        for y in range(40 * k, H, 40 * k):
            ld.ellipse((x - 2 * k, y - 2 * k, x + 2 * k, y + 2 * k), fill=(255, 255, 255, 16))
    img = Image.alpha_composite(img, layer)
    lx, ly, lr = 1010 * k, 290 * k, 140 * k  # центр і радіус лінзи
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse((lx - 2 * lr, ly - 2 * lr, lx + 2 * lr, ly + 2 * lr), fill=accent + (60,))
    img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(110 * k))).convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")  # на RGB-зображенні режим RGBA змішує напівпрозорі кольори

    # лупа: ручка, лінза, всередині кнопка play
    ring = 22 * k
    c = math.cos(math.radians(45))
    hx0, hy0 = lx + (lr + ring / 2) * c, ly + (lr + ring / 2) * c
    hx1, hy1 = hx0 + 120 * k * c, hy0 + 120 * k * c
    hw = 38 * k
    d.line((hx0, hy0, hx1, hy1), fill=accent, width=hw)
    d.ellipse((hx1 - hw / 2, hy1 - hw / 2, hx1 + hw / 2, hy1 + hw / 2), fill=accent)
    d.ellipse((lx - lr, ly - lr, lx + lr, ly + lr), fill=(255, 255, 255, 14), outline=accent, width=ring)
    p = 58 * k
    d.polygon([(lx - p * 0.6, ly - p), (lx - p * 0.6, ly + p), (lx + p, ly)], fill=(255, 255, 255, 235))

    x = 90 * k
    # плашка «АНАЛІЗ ВІДЕО» і підпис поруч
    pill_font = ImageFont.truetype(FONT_BOLD, 36 * k)
    pl, pt, pr, pb = pill_font.getbbox("АНАЛІЗ ВІДЕО")
    px, py = 24 * k, 14 * k
    y0 = 70 * k
    pw, ph = pr - pl + 2 * px, pb - pt + 2 * py
    d.rounded_rectangle((x, y0, x + pw, y0 + ph), radius=ph // 2, fill=accent)
    d.text((x + px - pl, y0 + py - pt), "АНАЛІЗ ВІДЕО", font=pill_font, fill=(5, 46, 22))
    sub_font = ImageFont.truetype(FONT_REGULAR, 28 * k)
    sub = "фактчек · аргументи · висновки"
    sx = x + pw + 24 * k
    if sx + sub_font.getlength(sub) < lx - lr - 30 * k:
        sl, st, sr, sb = sub_font.getbbox(sub)
        d.text((sx, y0 + (ph - (sb - st)) / 2 - st), sub, font=sub_font, fill=(148, 163, 184))

    # назва відео в лапках: найбільший кегль, при якому вміщується в 4 рядки
    title = f"«{(info.get('title') or 'Без назви').strip()}»"
    max_w = lx - lr - 50 * k - x
    box_top, box_bottom = 170 * k, 530 * k
    for size in range(80, 39, -4):
        font = ImageFont.truetype(FONT_BOLD, size * k)
        lines = wrap_text(title, font, max_w)
        line_h = round(size * k * 1.17)
        if len(lines) <= 4 and len(lines) * line_h <= box_bottom - box_top:
            break
    else:
        lines = lines[:4]
        lines[-1] = lines[-1].rstrip(" .,»") + "…»"
    y = box_top + (box_bottom - box_top - len(lines) * line_h) // 2
    for line in lines:
        d.text((x, y), line, font=font, fill="white")
        y += line_h

    # таймлайн відео з позначками розібраних моментів
    ty, x_end = 580 * k, W - x
    d.rounded_rectangle((x, ty - 3 * k, x_end, ty + 3 * k), radius=3 * k, fill=(255, 255, 255, 45))
    for f in (0.08, 0.27, 0.41, 0.63, 0.79, 0.93):
        mx, r = x + (x_end - x) * f, 10 * k
        d.ellipse((mx - r, ty - r, mx + r, ty + r), fill=accent, outline=top, width=4 * k)

    date = info.get("upload_date") or ""
    date = f"{date[6:]}.{date[4:6]}.{date[:4]}" if len(date) == 8 else ""
    channel = (info.get("channel") or info.get("uploader") or "").strip()
    meta = "  ·  ".join(p for p in ("YouTube", channel, date, fmt_ts(info.get("duration") or 0)) if p)
    d.text((x, 618 * k), meta, font=ImageFont.truetype(FONT_REGULAR, 30 * k), fill=(203, 213, 225))

    img.resize((W // k, H // k), Image.LANCZOS).save(out_path, "JPEG", quality=92)


def load_model(name: str, device: str):
    if device in ("auto", "cuda"):
        try:
            model = WhisperModel(name, device="cuda", compute_type="float16")
            # cuBLAS/cuDNN підвантажуються лише при першому розпізнаванні — перевіряємо одразу
            list(model.transcribe(np.zeros(16000, dtype=np.float32), language="en")[0])
            return model, "cuda"
        except Exception as e:
            if device == "cuda":
                raise
            print(f"  GPU недоступний ({e.__class__.__name__}), працюю на CPU", file=sys.stderr)
    return WhisperModel(name, device="cpu", compute_type="int8"), "cpu"


def print_progress(done: float, duration: float):
    pct = min(100, done / duration * 100)
    print(f"\r  розпізнано {pct:5.1f}%  [{fmt_ts(done)} / {fmt_ts(duration)}]", end="", flush=True)


def transcribe(model: WhisperModel, audio: Path, lang: str | None, duration: float, on_progress=print_progress):
    """on_progress(розпізнано_секунд, тривалість) викликається після кожного сегмента."""
    segments, info = model.transcribe(
        str(audio),
        language=lang,                    # None = автовизначення
        vad_filter=True,                  # вирізає тишу/музику → менше галюцинацій
        beam_size=5,
        condition_on_previous_text=False, # менше «зациклень» на довгих відео
    )
    result = []
    for seg in segments:
        result.append(seg)
        if duration:
            on_progress(seg.end, duration)
    if on_progress is print_progress:
        print()
    return result, info


def to_paragraphs(segments, max_len_sec: float = 60, pause_sec: float = 2.0):
    """Склеює сегменти в абзаци: новий абзац після паузи або кожні ~60 с."""
    paras, cur, start, prev_end = [], [], None, None
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        new_para = cur and (seg.start - prev_end > pause_sec or seg.start - start > max_len_sec)
        if new_para:
            paras.append((start, " ".join(cur)))
            cur, start = [], None
        if start is None:
            start = seg.start
        cur.append(text)
        prev_end = seg.end
    if cur:
        paras.append((start, " ".join(cur)))
    return paras


def build_markdown(info: dict, url: str, paras, lang_info, model_name: str, prompt: str) -> str:
    date = info.get("upload_date") or ""
    if len(date) == 8:
        date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    lines = [
        f"# {info.get('title', 'Без назви')}",
        "",
        f"- **Канал:** {info.get('channel') or info.get('uploader') or '—'}",
        f"- **Дата публікації:** {date or '—'}",
        f"- **Тривалість:** {fmt_ts(info.get('duration') or 0)}",
        f"- **Посилання:** {url}",
        f"- **Посилання на момент відео:** https://youtu.be/{info['id']}?t=<секунди>",
        f"- **Мова розпізнавання:** {lang_info.language} (ймовірність {lang_info.language_probability:.2f})",
        f"- **Модель:** faster-whisper {model_name}",
        "",
        "> Інструкція для ШІ — в кінці файлу.",
        "",
        "---",
        "",
        "## Транскрипт",
        "",
        "<transcript>",
    ]
    for start, text in paras:
        lines += [f"[{fmt_ts(start)}] {text}", ""]
    lines += ["</transcript>", "", "---", "", "## Інструкція для ШІ", "", prompt.strip(), ""]
    return "\n".join(lines)


def process_url(model: WhisperModel, url: str, *, model_name: str, lang: str | None = None,
                prompt: str = DEFAULT_PROMPT, out_dir: Path = Path("transcripts"),
                cover: str = "own", stamp: str = "ПРОАНАЛІЗОВАНО",
                on_progress=print_progress, log=print):
    """Одне відео: завантаження → розпізнавання → .md (+ обкладинка). Повертає (шлях_md, шлях_jpg | None, info)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        info, audio = download_audio(url, Path(tmp))
        log(f"  {info.get('title')}")
        segments, lang_info = transcribe(model, audio, lang, info.get("duration") or 0, on_progress)
    md = build_markdown(info, url, to_paragraphs(segments), lang_info, model_name, prompt)
    date = info.get("upload_date") or "nodate"
    path = out_dir / f"{date}_{safe_name(info.get('title', info['id']))}.md"
    path.write_text(md, encoding="utf-8")
    log(f"  ✓ {path}")

    cover_path = None
    if cover == "own" or (cover == "thumb" and info.get("thumbnail")):
        try:
            cover_path = path.with_suffix(".jpg")
            if cover == "own":
                own_cover(info, cover_path)
            else:
                thumb_cover(info, cover_path, stamp)
            log(f"  ✓ {cover_path}")
        except Exception as e:
            cover_path = None
            log(f"  ✗ обкладинка: {e}")
    return path, cover_path, info


def main():
    ap = argparse.ArgumentParser(description="YouTube → транскрипт + промпт для ШІ")
    ap.add_argument("urls", nargs="+", help="одне або кілька посилань")
    ap.add_argument("--lang", default=None, help="uk / en / ru; без параметра — автовизначення")
    ap.add_argument("--model", default="large-v3",
                    help="large-v3 (найточніша), large-v3-turbo (швидша), medium, small")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", default="transcripts", help="папка для файлів")
    ap.add_argument("--prompt-file", default=None, help="свій промпт замість стандартного")
    ap.add_argument("--stamp", default="ПРОАНАЛІЗОВАНО", help="текст мітки на превʼю (для --cover thumb)")
    ap.add_argument("--cover", default="own", choices=["own", "thumb", "none"],
                    help="own — власна обкладинка (за замовчуванням), thumb — превʼю відео з міткою, none — без обкладинки")
    args = ap.parse_args()

    prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else DEFAULT_PROMPT

    print(f"Завантажую модель {args.model}…")
    model, device = load_model(args.model, args.device)
    print(f"  готово ({device})")

    for url in args.urls:
        t0 = time.time()
        print(f"\n▶ {url}")
        try:
            process_url(model, url, model_name=args.model, lang=args.lang, prompt=prompt,
                        out_dir=Path(args.out), cover=args.cover, stamp=args.stamp)
            print(f"  ({time.time() - t0:.0f} с)")
        except Exception as e:
            print(f"  ✗ помилка: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
