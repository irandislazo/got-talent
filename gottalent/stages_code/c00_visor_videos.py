# ═══════════════════════════════════════════════════════════════
# CELDA EXTRA: 🎬 VISOR DE VIDEOS TERMINADOS (sin GPU, 100% autónoma)
#
# - Muestra SOLO los videos finales de la carpeta de un video (proyecto):
#     · Etapa 2 final   (etapa2_dialogos/etapa2_final.mp4)
#     · Etapa 3 final   (etapa3_cancion/cancion_final.mp4)
#     · Etapa 4 final   (etapa4_dialogos/etapa4_final.mp4)
#     · VIDEO FINAL     (VIDEO_FINAL.mp4, el que une todas las etapas)
#   No muestra fragmentos sueltos, clips de 5 s, previews ni temporales.
# - Por defecto abre el video que se está trabajando (el de ui_state.json);
#   si no hay, el más reciente. Puedes elegir cualquier otro del desplegable.
# - Cada video se puede ver y descargar desde la UI.
# - NO usa GPU, NO usa torch, NO depende de ninguna otra celda: solo lee
#   archivos en disco (y ffprobe, opcional, para la duración).
# - NO cierra otros servidores Gradio: se puede dejar abierta junto a la UI principal.
# ═══════════════════════════════════════════════════════════════
import os, json, glob, time, subprocess
import gradio as gr

# ── Rutas (se detectan solas; cambia _VV_ROOT_FORZADO si usas otra) ──
_VV_ROOT_FORZADO = ""   # ej: "/kaggle/working/VideoAutomationProject"
_VV_ROOT_CANDIDATOS = [r for r in [
    _VV_ROOT_FORZADO,
    "/kaggle/working/VideoAutomationProject",
] if r]

def _vv_root():
    for r in _VV_ROOT_CANDIDATOS:
        if os.path.isdir(os.path.join(r, "videos")):
            return r
    return _VV_ROOT_CANDIDATOS[-1]

_VV_ROOT   = _vv_root()
_VV_VIDEOS = os.path.join(_VV_ROOT, "videos")
_VV_STATE  = os.path.join(_VV_ROOT, "ui_state.json")

# Etapa → (clave en estado.json, rutas por defecto relativas a la carpeta del video)
_VV_ETAPAS = [
    ("etapa2", "🎙️ Etapa 2 — Diálogos iniciales", ["etapa2_dialogos/etapa2_final.mp4"]),
    ("etapa3", "🎵 Etapa 3 — Canción con lipsync", ["etapa3_cancion/cancion_final.mp4"]),
    ("etapa4", "🎬 Etapa 4 — Diálogos finales",   ["etapa4_dialogos/etapa4_final.mp4",
                                                   "etapa4_final/etapa4_final.mp4"]),
]
_VV_FINAL_REL = "VIDEO_FINAL.mp4"


def _vv_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _vv_ok(path):
    try:
        return bool(path) and os.path.isfile(path) and os.path.getsize(path) > 10_000
    except Exception:
        return False

def _vv_finales(video_dir):
    """→ {'etapa2': ruta|None, 'etapa3': ..., 'etapa4': ..., 'final': ...}"""
    estado = _vv_json(os.path.join(video_dir, "estado.json"))
    out = {}
    for key, _titulo, rels in _VV_ETAPAS:
        cand = []
        v = (estado.get(key, {}) or {}).get("video_final", "")
        if v:
            cand.append(v)
        cand += [os.path.join(video_dir, r) for r in rels]
        out[key] = next((c for c in cand if _vv_ok(c)), None)
    cand = []
    if estado.get("video_final"):
        cand.append(estado["video_final"])
    cand.append(os.path.join(video_dir, _VV_FINAL_REL))
    out["final"] = next((c for c in cand if _vv_ok(c)), None)
    return out

def _vv_proyectos():
    """Carpetas de video (más recientes primero)."""
    if not os.path.isdir(_VV_VIDEOS):
        return []
    dirs = [d for d in glob.glob(os.path.join(_VV_VIDEOS, "*")) if os.path.isdir(d)]
    def _mt(d):
        try: return os.path.getmtime(d)
        except Exception: return 0
    return sorted(dirs, key=_mt, reverse=True)

def _vv_actual():
    d = _vv_json(_VV_STATE).get("current_video_dir", "")
    return d if d and os.path.isdir(d) else ""

def _vv_choices():
    actual = _vv_actual()
    ch = []
    for d in _vv_proyectos():
        f = _vv_finales(d)
        marcas = " ".join(f"{n}{'✅' if f[k] else '·'}"
                          for k, n in (("etapa2", "E2"), ("etapa3", "E3"),
                                       ("etapa4", "E4"), ("final", "FINAL")))
        estrella = "⭐ " if os.path.realpath(d) == os.path.realpath(actual or "/__none__") else ""
        ch.append((f"{estrella}{os.path.basename(d)}   [{marcas}]", d))
    return ch

def _vv_default(choices):
    if not choices:
        return None
    actual = _vv_actual()
    for _lbl, d in choices:
        if actual and os.path.realpath(d) == os.path.realpath(actual):
            return d
    return choices[0][1]

def _vv_info(path):
    if not path:
        return "⏳ Todavía no existe este video."
    partes = []
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=noprint_wrappers=1:nokey=1", path],
                           capture_output=True, text=True, timeout=15)
        dur = float((r.stdout or "").strip())
        partes.append(f"⏱ {int(dur // 60)}:{int(dur % 60):02d}")
    except Exception:
        pass
    try:
        partes.append(f"💾 {os.path.getsize(path) / 1024 / 1024:.1f} MB")
        partes.append("🕒 " + time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path))))
    except Exception:
        pass
    return " · ".join(partes) + f"\n\n`{path}`"

def _vv_cargar(video_dir):
    """→ [resumen, (video, archivo, info) × 4]  (E2, E3, E4, FINAL)"""
    if not video_dir or not os.path.isdir(video_dir):
        vacio = [None, None, "⏳ Todavía no existe este video."]
        return ["⚠️ No hay ningún video en la carpeta `videos`."] + vacio * 4
    f = _vv_finales(video_dir)
    hechos = sum(1 for v in f.values() if v)
    res = [f"### 📁 {os.path.basename(video_dir)}\n"
           f"Videos terminados: **{hechos}/4**"]
    out = res
    for key in ("etapa2", "etapa3", "etapa4", "final"):
        p = f[key]
        out += [p, p, _vv_info(p)]
    return out

def _vv_refrescar(sel):
    ch = _vv_choices()
    vals = [v for _l, v in ch]
    val = sel if sel in vals else _vv_default(ch)
    return [gr.update(choices=ch, value=val)] + _vv_cargar(val)


_ch0 = _vv_choices()
_v0 = _vv_default(_ch0)

with gr.Blocks(title="🎬 Videos terminados") as demo_visor:
    gr.Markdown("## 🎬 Videos terminados\n"
                "Solo los videos **finales** de cada etapa (sin fragmentos ni clips en proceso). "
                "No usa GPU: puedes verlos y descargarlos sin gastar horas.")
    with gr.Row():
        dd_video = gr.Dropdown(choices=_ch0, value=_v0, label="Video (⭐ = el que estás trabajando)", scale=5)
        btn_ref = gr.Button("🔄 Refrescar", scale=1)
    resumen = gr.Markdown("")

    _comps = []   # orden: video, archivo, info × 4
    for _key, _titulo in (("etapa2", _VV_ETAPAS[0][1]), ("etapa3", _VV_ETAPAS[1][1]),
                          ("etapa4", _VV_ETAPAS[2][1]), ("final", "🏆 VIDEO FINAL (todas las etapas unidas)")):
        with gr.Accordion(_titulo, open=(_key in ("etapa3", "final"))):
            with gr.Row():
                with gr.Column(scale=3):
                    v = gr.Video(label=_titulo, interactive=False, height=360)
                with gr.Column(scale=2):
                    inf = gr.Markdown("")
                    fl = gr.File(label="⬇️ Descargar", interactive=False)
            _comps += [v, fl, inf]

    _outs_cargar = [resumen] + _comps
    dd_video.input(_vv_cargar, [dd_video], _outs_cargar)
    btn_ref.click(_vv_refrescar, [dd_video], [dd_video] + _outs_cargar)
    demo_visor.load(_vv_refrescar, [dd_video], [dd_video] + _outs_cargar)

demo_visor.queue().launch(share=True, prevent_thread_lock=True, show_error=True,
                          allowed_paths=[_VV_ROOT])
