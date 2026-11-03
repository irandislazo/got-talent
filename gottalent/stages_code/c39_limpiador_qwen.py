# ═══════════════════════════════════════════════════════════════
# CELDA EXTRA: 🧹 LIMPIADOR DE MODELOS QWEN-IMAGE (control total del disco)
#
# - Lista cada modelo Qwen-Image descargado en local_models: diffusion (GGUF), encoder
#   Qwen3-VL (GGUF), mmproj y VAE, con su tamaño real.
# - TÚ marcas cuáles borrar; nada se borra sin tu confirmación.
# - Solo toca archivos dentro de LOCAL_MODELS_DIR. NO toca: TTS, LTX, plantillas, videos,
#   sd-cli ni nada de /kaggle/input (que es de solo lectura).
# - Es independiente: no necesita que la UI principal esté abierta (abre su propio enlace).
# - Si borras un modelo que luego se necesita, la Etapa 1 lo vuelve a descargar sola.
# ═══════════════════════════════════════════════════════════════
import os, re, json, shutil
import gradio as gr

# ── Rutas (usa las del notebook si existen; si no, las de siempre) ──
_LM_DIR = globals().get("LOCAL_MODELS_DIR") or "/kaggle/working/VideoAutomationProject/local_models"
_LM_DIR = os.path.realpath(_LM_DIR)
_QDIFF  = globals().get("QWEN_IMAGE_DIFFUSION_QUANTS") or {
    "Q2_K": "qwen_image_2.1-Q2_K.gguf", "Q3_K": "qwen_image_2.1-Q3_K.gguf",
    "Q4_0": "qwen_image_2.1-Q4_0.gguf", "Q4_K": "qwen_image_2.1-Q4_K.gguf",
    "Q5_0": "qwen_image_2.1-Q5_0.gguf", "Q6_K": "qwen_image_2.1-Q6_K.gguf",
    "Q8_0": "qwen_image_2.1-Q8_0.gguf"}
_QENC   = globals().get("QWEN_VL_ENCODERS") or {
    "Q4_K_M": "Qwen3VL-8B-Instruct-Q4_K_M.gguf", "Q8_0": "Qwen3VL-8B-Instruct-Q8_0.gguf",
    "F16": "Qwen3VL-8B-Instruct-F16.gguf"}
_QMM    = globals().get("QWEN_VL_MMPROJ") or {
    "F16": "mmproj-Qwen3VL-8B-Instruct-F16.gguf", "Q8_0": "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"}

_CATS = [  # (clave, título, descripción)
    ("diff", "🎨 Modelo de imagen (diffusion)", "Es el pesado: genera las imágenes A→G."),
    ("enc",  "🧠 Encoder de texto/visión (Qwen3-VL)", "Entiende el prompt y las imágenes de referencia."),
    ("mm",   "👁️ mmproj (visión del encoder)", "Necesario para las ediciones con imágenes de referencia (B→G)."),
    ("vae",  "🖼️ VAE", "Pequeño, pero imprescindible si vas a generar imágenes."),
    ("inc",  "⚠️ Descargas incompletas (.incomplete)", "Restos de descargas cortadas: se pueden borrar sin riesgo."),
]

def _fmt(n: int) -> str:
    return f"{n/1024**3:.2f} GB" if n >= 1024**3 else f"{n/1024**2:.0f} MB"

def _free(path: str) -> float:
    try:
        return shutil.disk_usage(path).free / 1024**3
    except Exception:
        return float("nan")

def _en_uso() -> set:
    # Nombres de archivo configurados en el proyecto activo (solo para avisarte; tú decides).
    usados = set()
    try:
        vd = globals().get("_CUR_VP_DIR") or ""
        pj = os.path.join(vd, "params.json")
        if vd and os.path.exists(pj):
            p = (json.load(open(pj, encoding="utf-8")).get("etapa1") or {})
            for key, tabla in (("diffusion_quant", _QDIFF), ("encoder_quant", _QENC), ("mmproj_quant", _QMM)):
                f = tabla.get(p.get(key, ""))
                if f: usados.add(f)
    except Exception:
        pass
    return usados

def _clasificar(nombre: str):
    n = nombre.lower()
    if n.endswith(".incomplete"):                                   return "inc"
    if n.startswith("qwen_image") and "vae" in n:                   return "vae"
    if n.startswith("mmproj-qwen3vl") and n.endswith(".gguf"):      return "mm"
    if n.startswith("qwen3vl") and n.endswith(".gguf"):             return "enc"
    if n.startswith("qwen_image") and n.endswith(".gguf"):          return "diff"
    return None

def _escanear():
    # → ({cat: [(relpath, nombre, bytes)]}, bytes_otros, bytes_total_dir)
    cat = {k: [] for k, _, _ in _CATS}
    otros = total = 0
    if not os.path.isdir(_LM_DIR):
        return cat, 0, 0
    for root, _, files in os.walk(_LM_DIR):
        for fn in files:
            p = os.path.join(root, fn)
            if os.path.islink(p):
                continue
            try:
                sz = os.path.getsize(p)
            except OSError:
                continue
            total += sz
            c = _clasificar(fn)
            if c:
                cat[c].append((os.path.relpath(p, _LM_DIR), fn, sz))
            else:
                otros += sz
    for k in cat:
        cat[k].sort(key=lambda x: -x[2])
    return cat, otros, total

def _choices(items, usados):
    out = []
    for rel, fn, sz in items:
        tag = "   ⚙️ configurado en el proyecto activo" if fn in usados else ""
        out.append((f"{fn}  ·  {_fmt(sz)}{tag}", rel))
    return out

def _disk_html():
    cat, otros, total = _escanear()
    qsz = sum(sz for k in ("diff", "enc", "mm", "vae") for _, _, sz in cat[k])
    inc = sum(sz for _, _, sz in cat["inc"])
    return (
        "<div style='background:#10192b;border:1px solid #2a3a5c;border-radius:12px;padding:14px;"
        "color:#dbe4ff;font-size:0.95em;line-height:1.7'>"
        f"💾 <b>Libre en /kaggle/working:</b> {_free('/kaggle/working'):.2f} GB &nbsp;·&nbsp; "
        f"<b>/kaggle/tmp:</b> {_free('/kaggle/tmp'):.2f} GB<br>"
        f"📦 <b>local_models total:</b> {_fmt(total)} &nbsp;·&nbsp; "
        f"<b>Qwen-Image (diffusion+encoder+mmproj+VAE):</b> {_fmt(qsz)} &nbsp;·&nbsp; "
        f"<b>Incompletas:</b> {_fmt(inc)}<br>"
        f"🔒 <b>Otros archivos (TTS, etc.) — no se tocan aquí:</b> {_fmt(otros)}"
        "</div>")

def _updates():
    cat, _, _ = _escanear()
    usados = _en_uso()
    return [gr.update(choices=_choices(cat[k], usados), value=[]) for k, _, _ in _CATS]

def _sel_a_rutas(*sels):
    # Revalida CADA ruta contra un escaneo nuevo: solo se puede borrar lo que este escáner reconoce.
    cat, _, _ = _escanear()
    validas = {rel: (fn, sz) for k in cat for rel, fn, sz in cat[k]}
    out = []
    for grupo in sels:
        for rel in (grupo or []):
            if rel in validas:
                out.append((rel, validas[rel][0], validas[rel][1]))
    return out

def ui_calcular(*sels):
    items = _sel_a_rutas(*sels)
    if not items:
        return "Nada marcado."
    return f"Marcados: {len(items)} archivo(s) → se liberarían **{_fmt(sum(i[2] for i in items))}**."

def ui_pedir_confirmacion(*sels):
    items = _sel_a_rutas(*sels)
    if not items:
        return gr.update(visible=False), "ℹ️ No has marcado ningún archivo."
    usados = _en_uso()
    lineas = [f"- `{fn}` — {_fmt(sz)}" + ("  ⚠️ **está configurado en el proyecto activo**" if fn in usados else "")
              for _, fn, sz in items]
    md = ("### ¿Eliminar definitivamente?\n" + "\n".join(lineas) +
          f"\n\n**Total a liberar: {_fmt(sum(i[2] for i in items))}.** Esta acción no se puede deshacer "
          "(para volver a tenerlos habría que descargarlos de nuevo, y eso requiere internet activado).")
    return gr.update(visible=True), md

def ui_cancelar():
    return gr.update(visible=False), "Cancelado. No se borró nada."

def ui_borrar(*sels):
    items = _sel_a_rutas(*sels)
    if not items:
        return ("ℹ️ No había nada válido que borrar.", _disk_html(), gr.update(visible=False), "", *_updates())
    antes = _free("/kaggle/working")
    res, liberado = [], 0
    for rel, fn, sz in items:
        p = os.path.realpath(os.path.join(_LM_DIR, rel))
        if not p.startswith(_LM_DIR + os.sep):             # defensa extra: jamás salir de local_models
            res.append(f"⛔ Ruta fuera de local_models, omitida: {fn}")
            continue
        try:
            os.remove(p)
            liberado += sz
            res.append(f"🗑️ Borrado: {fn}  ({_fmt(sz)})")
            d = os.path.dirname(p)                          # quita la carpeta si quedó vacía (p. ej. vae/)
            while d != _LM_DIR and d.startswith(_LM_DIR + os.sep) and not os.listdir(d):
                os.rmdir(d); d = os.path.dirname(d)
        except Exception as e:
            res.append(f"❌ No se pudo borrar {fn}: {e}")
    res.append("")
    res.append(f"✅ Liberado: {_fmt(liberado)}  ·  Libre ahora en /working: {_free('/kaggle/working'):.2f} GB "
               f"(antes {antes:.2f} GB)")
    try:
        log("[LIMPIADOR QWEN-IMAGE] " + " | ".join(r for r in res if r))
    except Exception:
        pass
    return ("\n".join(res), _disk_html(), gr.update(visible=False), "", *_updates())

def ui_refrescar():
    return (_disk_html(), *_updates(), "Nada marcado.")

# ── Interfaz ────────────────────────────────────────────────
_cat0, _, _ = _escanear()
_usados0 = _en_uso()

with gr.Blocks(title="🧹 Limpiador Qwen-Image") as demo_clean:
    gr.Markdown("## 🧹 Limpiador de modelos Qwen-Image\n"
                "Marca los modelos que quieras eliminar. **Tú decides**: nada se borra hasta que confirmes.")
    disco_html = gr.HTML(_disk_html())
    btn_ref = gr.Button("🔄 Refrescar lista y espacio", variant="secondary")

    grupos = []
    for k, titulo, desc in _CATS:
        with gr.Group():
            gr.Markdown(f"**{titulo}** — {desc}")
            cg = gr.CheckboxGroup(choices=_choices(_cat0[k], _usados0), value=[], label="",
                                  show_label=False)
        grupos.append(cg)

    calc_md = gr.Markdown("Nada marcado.")
    btn_borrar = gr.Button("🗑️ Eliminar los marcados…", variant="stop")
    with gr.Group(visible=False) as confirm_box:
        confirm_md = gr.Markdown("")
        with gr.Row():
            btn_si = gr.Button("✅ Sí, eliminar definitivamente", variant="stop")
            btn_no = gr.Button("↩️ Cancelar", variant="secondary")
    resultado = gr.Textbox(label="Resultado", interactive=False, lines=8)

    for cg in grupos:
        cg.change(ui_calcular, grupos, [calc_md])
    btn_borrar.click(ui_pedir_confirmacion, grupos, [confirm_box, confirm_md])
    btn_no.click(ui_cancelar, [], [confirm_box, confirm_md])
    btn_si.click(ui_borrar, grupos, [resultado, disco_html, confirm_box, confirm_md] + grupos)
    btn_ref.click(ui_refrescar, [], [disco_html] + grupos + [calc_md])

demo_clean.queue().launch(share=True, prevent_thread_lock=True, show_error=True)
