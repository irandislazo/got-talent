import sys, time, traceback, __main__
from pathlib import Path
from . import stages as S

CODE = Path(__file__).parent / "stages_code"
NS = __main__.__dict__          # mismo espacio global que un kernel Jupyter

def _file(n):
    return next(CODE.glob(f"c{n:02d}_*.py"))

def _exec(n):
    p = _file(n)
    exec(compile(p.read_text(encoding="utf-8"), str(p), "exec"), NS)

def _run_stage(n, i, total):
    t = time.time()
    print(f"\n{'═'*60}\n▶ [{i}/{total}] {_file(n).stem}\n{'═'*60}", flush=True)
    try:
        _exec(n)
    except BaseException:
        print(f"\n❌ Falló la etapa {_file(n).stem}:\n{traceback.format_exc()}")
        raise
    print(f"✓ {_file(n).stem} ({time.time()-t:.0f}s)", flush=True)

def run(skip=(), only=None, desde=None, ui=True, extras=True, share=True):
    """Ejecuta setup + UI y la abre SOLO como enlace (sin iframe bajo la celda).
    skip=[...] celdas a saltar · only=[...] solo esas · desde=N empieza en la celda N."""
    seq = list(only) if only else [n for n in S.SETUP if n not in skip]
    if desde is not None and desde in seq: seq = seq[seq.index(desde):]
    for i, n in enumerate(seq, 1):
        _run_stage(n, i, len(seq))
    if ui: return launch_ui(extras=extras, share=share)

def launch_ui(extras=True, share=True):
    import gradio as gr
    paths, orig = [], gr.Blocks.launch
    def _fake(self, *a, **k):                    # captura allowed_paths, no lanza nada
        paths.extend(k.get("allowed_paths") or []); return (self, None, None)
    gr.Blocks.launch = _fake
    try:
        _exec(S.UI)
        if extras:
            for n, var, _t in S.EXTRA_TABS:
                try: _exec(n)
                except BaseException: print(f"⚠ Pestaña extra omitida ({_file(n).stem}):\n{traceback.format_exc()}")
    finally:
        gr.Blocks.launch = orig
    demo = NS["demo"]
    if extras:
        with demo:
            with NS["_main_tabs"]:
                for n, var, title in S.EXTRA_TABS:
                    b = NS.get(var)
                    if b is None: continue
                    try:
                        with gr.Tab(title): b.render()
                    except BaseException:
                        print(f"⚠ No se pudo montar «{title}»:\n{traceback.format_exc()}")
    kw = dict(share=share, show_error=True, inline=False, prevent_thread_lock=True)
    if paths: kw["allowed_paths"] = list(dict.fromkeys(paths))
    _, local, public = demo.queue(max_size=10).launch(**kw)
    url = public or local
    print(f"\n{'█'*60}\n🎬 UI LISTA → abre este enlace en una pestaña nueva:\n   {url}\n{'█'*60}")
    NS["UI_URL"] = url
    return url

def tool(nombre):
    """Ejecuta una celda suelta. Si abre una UI Gradio, solo muestra el enlace (sin iframe)."""
    n = S.TOOLS.get(nombre)
    if n is None: raise KeyError(f"Herramientas: {', '.join(S.TOOLS)}")
    orig = None
    try:
        import gradio as gr
        orig = gr.Blocks.launch
        def _l(self, *a, **k):
            k["inline"] = False
            return orig(self, *a, **k)
        gr.Blocks.launch = _l
    except Exception:
        pass
    try:
        _exec(n)
    finally:
        if orig is not None: gr.Blocks.launch = orig
