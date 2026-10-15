"""
Графічний інтерфейс yt2txt у браузері (Gradio).

    python gui.py          # http://127.0.0.1:7860, лише з цього ПК
    python gui.py --lan    # доступ з інших пристроїв у локальній мережі (без пароля!)
"""
import argparse
import gc
import os
import time
from pathlib import Path

import gradio as gr

import yt2txt

OUT_DIR = Path(__file__).parent / "transcripts"
LANGS = [("Автовизначення", "auto"), ("Українська", "uk"), ("Англійська", "en"), ("Російська", "ru")]
MODELS = [("large-v3: найточніша", "large-v3"), ("large-v3-turbo: у кілька разів швидша", "large-v3-turbo"),
          ("medium", "medium"), ("small: найшвидша, найменш точна", "small")]
COVERS = [("Власна", "own"), ("Превʼю відео з міткою", "thumb"), ("Без обкладинки", "none")]

# модель вантажиться один раз і лишається в памʼяті між запусками, поки не натиснеш «Звільнити GPU»
_model = {"name": None, "obj": None, "device": None}


def get_model(name: str):
    if _model["name"] != name:
        free_model()
        _model["obj"], _model["device"] = yt2txt.load_model(name, "auto")
        _model["name"] = name
    return _model["obj"], _model["device"]


def free_model():
    _model.update(name=None, obj=None, device=None)
    gc.collect()
    return "Модель вивантажено, памʼять GPU звільнено."


def list_results():
    files = sorted(OUT_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [(p.stem, str(p)) for p in files]


def show_result(md_path: str | None):
    if not md_path or not Path(md_path).exists():
        return None, "", None
    p = Path(md_path)
    cover = p.with_suffix(".jpg")
    files = [str(p)] + ([str(cover)] if cover.exists() else [])
    return (str(cover) if cover.exists() else None), p.read_text(encoding="utf-8"), files


def run(urls_text, lang, model_name, cover, stamp, prompt, progress=gr.Progress()):
    urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
    if not urls:
        raise gr.Error("Встав хоча б одне посилання на відео.")
    log = []

    def say(msg):
        log.append(msg)

    progress(0, desc=f"Завантажую модель {model_name}…")
    say(f"Модель {model_name}…")
    yield "\n".join(log), gr.update()
    model, device = get_model(model_name)
    say(f"  готово ({'GPU' if device == 'cuda' else 'CPU'})")
    yield "\n".join(log), gr.update()

    last = None
    for i, url in enumerate(urls, 1):
        prefix = f"[{i}/{len(urls)}] " if len(urls) > 1 else ""
        say(f"\n▶ {prefix}{url}")
        progress(0, desc=f"{prefix}Завантажую аудіо…")
        yield "\n".join(log), gr.update()

        def on_progress(done, duration, prefix=prefix):
            progress(done / duration,
                     desc=f"{prefix}Розпізнаю {yt2txt.fmt_ts(done)} / {yt2txt.fmt_ts(duration)}")

        t0 = time.time()
        try:
            md_path, _, _ = yt2txt.process_url(
                model, url, model_name=model_name, lang=None if lang == "auto" else lang,
                prompt=prompt or yt2txt.DEFAULT_PROMPT, out_dir=OUT_DIR, cover=cover,
                stamp=stamp or "ПРОАНАЛІЗОВАНО", on_progress=on_progress, log=say)
            say(f"  за {time.time() - t0:.0f} с")
            last = str(md_path)
        except Exception as e:
            say(f"  ✗ помилка: {e}")
        yield "\n".join(log), gr.update()

    say("\nГотово.")
    yield "\n".join(log), gr.update(choices=list_results(), value=last) if last else gr.update()


def open_folder():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.startfile(OUT_DIR)


with gr.Blocks(title="yt2txt") as demo:
    gr.Markdown("# yt2txt\nYouTube → транскрипт → файл для аналізу в Claude")

    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            urls = gr.Textbox(label="Посилання на відео", lines=3,
                              placeholder="https://youtu.be/...\nМожна кілька, по одному в рядок")
            with gr.Row():
                lang = gr.Dropdown(LANGS, value="auto", label="Мова")
                model_name = gr.Dropdown(MODELS, value="large-v3", label="Модель")
            cover = gr.Radio(COVERS, value="own", label="Обкладинка")
            stamp = gr.Textbox(value="ПРОАНАЛІЗОВАНО", label="Текст мітки на превʼю", visible=False)
            with gr.Accordion("Промпт для ШІ", open=False):
                prompt = gr.Textbox(value=yt2txt.DEFAULT_PROMPT, lines=16, show_label=False)
            start = gr.Button("▶ Почати", variant="primary", size="lg")
        with gr.Column(scale=2):
            log = gr.Textbox(label="Хід роботи", lines=14, max_lines=14, autoscroll=True, interactive=False)
            with gr.Row():
                folder_btn = gr.Button("📂 Відкрити папку", size="sm")
                free_btn = gr.Button("Звільнити GPU", size="sm")

    gr.Markdown("## Результати")
    with gr.Row(equal_height=False):
        with gr.Column(scale=2):
            result = gr.Dropdown(list_results(), label="Транскрипт", info="Останні зверху")
            cover_img = gr.Image(label="Обкладинка", type="filepath", interactive=False, height=300)
            files = gr.File(label="Файли", file_count="multiple", interactive=False)
        with gr.Column(scale=3):
            md_text = gr.Textbox(label="Вміст для Claude: скопіюй кнопкою праворуч угорі й встав у чат",
                                 lines=22, max_lines=22, buttons=["copy"], interactive=False)

    cover.change(lambda c: gr.update(visible=c == "thumb"), cover, stamp)
    start.click(run, [urls, lang, model_name, cover, stamp, prompt], [log, result])
    result.change(show_result, result, [cover_img, md_text, files])
    folder_btn.click(open_folder)
    free_btn.click(free_model, outputs=log)
    demo.load(lambda: gr.update(choices=list_results()), outputs=result)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="yt2txt — графічний інтерфейс у браузері")
    ap.add_argument("--lan", action="store_true", help="відкрити доступ з локальної мережі (без пароля)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0" if args.lan else "127.0.0.1",
        server_port=args.port,
        inbrowser=True,
        allowed_paths=[str(OUT_DIR)],
        theme=gr.themes.Default(primary_hue="green"),
    )
