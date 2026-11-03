# ═══════════════════════════════════════════════════════════════
# CELDA 19 (v2 con UI): 🧹 LIMPIEZA DE VIDEOS Y PLANTILLAS — TÚ DECIDES QUÉ BORRAR
#
# - Lista cada VIDEO (proyecto) y cada PLANTILLA con su tamaño real, fecha y estado.
# - Marcas lo que quieras eliminar; nada se borra hasta que confirmes.
# - Solo toca carpetas que cuelgan directamente de /videos y /plantillas.
#   NO toca: sd-cli, qwen-tts, modelos (para eso está el limpiador de Qwen-Image), LTX.
# - ⚠️ Las plantillas guardan TUS imágenes de jueces, voces y reacciones: no se pueden regenerar.
#   Por eso, para borrar una plantilla hay que marcar además una casilla de aviso.
# - Independiente: abre su propio enlace, no necesita la UI principal.
# ═══════════════════════════════════════════════════════════════
import os, re, json, shutil, time
import gradio as gr

_OUT   = globals().get("OUTPUT_DIR") or "/kaggle/working/VideoAutomationProject"
_VID   = os.path.realpath(globals().get("VIDEOS_DIR") or f"{_OUT}/videos")
_TPL   = os.path.realpath(globals().get("PLANTILLAS_DIR") or f"{_OUT}/plantillas")
_UIST  = os.path.join(_OUT, "ui_state.json")

def _fmt(n: int) -> str:
    return f"{n/1024**3:.2f} GB" if n >= 1024**3 else f"{n/1024**2:.0f} MB"

def _free(path: str) -> float:
    try:    return shutil.disk_usage(path).free / 1024**3
    except Exception: return float("nan")

def _dir_size(path: str) -> int:
    tot = 0
    for root, _, files in os.walk(path):
        for fn in files:
            fp = os.path.join(root, fn)
            if not os.path.islink(fp):
                try: tot += os.path.getsize(fp)
                except OSError: pass
    return tot

def _activos():
    # (video_dir_activo, plantilla_activa) del notebook; si no existen, de ui_state.json
    v = globals().get("_CUR_VP_DIR") or ""
    t = globals().get("_CUR_TPL") or ""
    if not (v and t):
        try:
            st = json.load(open(_UIST, encoding="utf-8"))
            v = v or st.get("current_video_dir", "")
            t = t or st.get("current_template_slug", "")
        except Exception:
            pass
    return (os.path.realpath(v) if v else ""), t

def _subdirs(base: str):
    if not os.path.isdir(base):
        return []
    return [d for d in sorted(os.listdir(base)) if os.path.isdir(os.path.join(base, d)) and not d.startswith(".")]

def _escanear():
    # → (videos, plantillas): listas de dict(nombre, ruta, bytes, mtime, extra)
    vids, tpls = [], []
    for d in _subdirs(_VID):
        p = os.path.join(_VID, d)
        vids.append({"nombre": d, "ruta": p, "bytes": _dir_size(p), "mtime": os.path.getmtime(p),
                     "final": os.path.exists(os.path.join(p, "VIDEO_FINAL.mp4"))})
    for d in _subdirs(_TPL):
        p = os.path.join(_TPL, d)
        nombre = d
        try:
            nombre = json.load(open(os.path.join(p, "plantilla.json"), encoding="utf-8")).get("nombre", d)
        except Exception:
            pass
        tpls.append({"nombre": d, "visible": nombre, "ruta": p, "bytes": _dir_size(p), "mtime": os.path.getmtime(p)})
    vids.sort(key=lambda x: -x["mtime"])
    tpls.sort(key=lambda x: -x["bytes"])
    return vids, tpls

def _choices_v(vids):
    av, _ = _activos()
    out = []
    for v in vids:
        f = time.strftime("%Y-%m-%d %H:%M", time.localtime(v["mtime"]))
        tags = ("  🎞️ con VIDEO_FINAL" if v["final"] else "") + ("  🟢 VIDEO ACTIVO" if os.path.realpath(v["ruta"]) == av else "")
        out.append((f"{v['nombre']}  ·  {_fmt(v['bytes'])}  ·  {f}{tags}", v["nombre"]))
    return out

def _choices_t(tpls):
    _, at = _activos()
    out = []
    for t in tpls:
        tag = "  🟢 PLANTILLA ACTIVA" if t["nombre"] == at else ""
        etiqueta = t["nombre"] if t["visible"] == t["nombre"] else f"{t['visible']} ({t['nombre']})"
        out.append((f"{etiqueta}  ·  {_fmt(t['bytes'])}{tag}", t["nombre"]))
    return out

def _disk_html():
    vids, tpls = _escanear()
    return ("<div style='background:#10192b;border:1px solid #2a3a5c;border-radius:12px;padding:14px;"
            "color:#dbe4ff;font-size:0.95em;line-height:1.7'>"
            f"💾 <b>Libre en /kaggle/working:</b> {_free('/kaggle/working'):.2f} GB &nbsp;·&nbsp; "
            f"<b>/kaggle/tmp:</b> {_free('/kaggle/tmp'):.2f} GB<br>"
            f"🎬 <b>Videos:</b> {len(vids)} ({_fmt(sum(v['bytes'] for v in vids))}) &nbsp;·&nbsp; "
            f"📁 <b>Plantillas:</b> {len(tpls)} ({_fmt(sum(t['bytes'] for t in tpls))})"
            "</div>")

def _updates():
    vids, tpls = _escanear()
    return gr.update(choices=_choices_v(vids), value=[]), gr.update(choices=_choices_t(tpls), value=[])

def _validar(sel_v, sel_t):
    # Revalida contra un escaneo nuevo: solo se borra lo que este escáner reconoce bajo /videos y /plantillas.
    vids, tpls = _escanear()
    mv = {v["nombre"]: v for v in vids}
    mt = {t["nombre"]: t for t in tpls}
    return ([("video", mv[n]) for n in (sel_v or []) if n in mv],
            [("plantilla", mt[n]) for n in (sel_t or []) if n in mt])

def ui_calcular(sel_v, sel_t):
    v, t = _validar(sel_v, sel_t)
    if not (v or t):
        return "Nada marcado."
    tot = sum(x["bytes"] for _, x in v + t)
    return f"Marcados: {len(v)} video(s) y {len(t)} plantilla(s) → se liberarían **{_fmt(tot)}**."

def ui_pedir_confirmacion(sel_v, sel_t):
    v, t = _validar(sel_v, sel_t)
    if not (v or t):
        return gr.update(visible=False), "ℹ️ No has marcado nada.", gr.update(visible=False, value=False)
    av, at = _activos()
    lineas = []
    for _, x in v:
        act = "  ⚠️ **es el video activo**" if os.path.realpath(x["ruta"]) == av else ""
        lineas.append(f"- 🎬 `{x['nombre']}` — {_fmt(x['bytes'])}{act}")
    for _, x in t:
        act = "  ⚠️ **es la plantilla activa**" if x["nombre"] == at else ""
        lineas.append(f"- 📁 `{x['nombre']}` — {_fmt(x['bytes'])}{act}")
    md = ("### ¿Eliminar definitivamente?\n" + "\n".join(lineas) +
          f"\n\n**Total a liberar: {_fmt(sum(x['bytes'] for _, x in v + t))}.** No se puede deshacer.")
    if t:
        md += ("\n\n🚨 **Las plantillas contienen tus imágenes de jueces, voces y reacciones y NO se pueden "
               "regenerar.** Marca la casilla de abajo para confirmar que quieres borrarlas.")
    return gr.update(visible=True), md, gr.update(visible=bool(t), value=False)

def ui_cancelar():
    return gr.update(visible=False), "Cancelado. No se borró nada.", gr.update(visible=False, value=False)

def _limpiar_ui_state(borrados_v, borrados_t):
    try:
        st = json.load(open(_UIST, encoding="utf-8"))
    except Exception:
        return
    ch = False
    if st.get("current_video_dir") and os.path.realpath(st["current_video_dir"]) in borrados_v:
        st["current_video_dir"] = ""; ch = True
    if st.get("current_template_slug") in borrados_t:
        st["current_template_slug"] = ""; ch = True
    if ch:
        json.dump(st, open(_UIST, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def ui_borrar(sel_v, sel_t, ack_plantillas):
    v, t = _validar(sel_v, sel_t)
    if t and not ack_plantillas:
        return ("⛔ Para borrar plantillas marca la casilla de aviso. No se borró nada.",
                _disk_html(), gr.update(visible=True), gr.update(), *[gr.update(), gr.update()],
                gr.update(visible=True))
    if not (v or t):
        return ("ℹ️ No había nada válido que borrar.", _disk_html(), gr.update(visible=False), "",
                *_updates(), gr.update(visible=False, value=False))
    antes = _free("/kaggle/working")
    res, liberado, bv, bt = [], 0, set(), set()
    for tipo, x in v + t:
        base = _VID if tipo == "video" else _TPL
        p = os.path.realpath(x["ruta"])
        if os.path.dirname(p) != base:                      # defensa: solo carpetas hijas directas
            res.append(f"⛔ Ruta no permitida, omitida: {x['nombre']}")
            continue
        try:
            shutil.rmtree(p)
            liberado += x["bytes"]
            (bv if tipo == "video" else bt).add(p if tipo == "video" else x["nombre"])
            res.append(f"🗑️ Borrado ({tipo}): {x['nombre']}  ({_fmt(x['bytes'])})")
        except Exception as e:
            res.append(f"❌ No se pudo borrar {x['nombre']}: {e}")
    _limpiar_ui_state(bv, bt)
    res += ["", f"✅ Liberado: {_fmt(liberado)}  ·  Libre ahora en /working: {_free('/kaggle/working'):.2f} GB "
                f"(antes {antes:.2f} GB)"]
    if bv or bt:
        res.append("ℹ️ Si borraste el video/plantilla activo, en la UI principal pulsa «📋 Listar» y elige otro (o crea uno nuevo).")
    try: log("[LIMPIEZA UI] " + " | ".join(r for r in res if r))
    except Exception: pass
    return ("\n".join(res), _disk_html(), gr.update(visible=False), "", *_updates(),
            gr.update(visible=False, value=False))

def ui_refrescar():
    return (_disk_html(), *_updates(), "Nada marcado.")

# ── Interfaz ────────────────────────────────────────────────
_v0, _t0 = _escanear()
with gr.Blocks(title="🧹 Limpieza de videos y plantillas") as demo_limpieza:
    gr.Markdown("## 🧹 Limpieza de videos y plantillas\n"
                "Marca lo que quieras eliminar. **Tú decides**: nada se borra hasta que confirmes.")
    disco_html = gr.HTML(_disk_html())
    btn_ref = gr.Button("🔄 Refrescar lista y espacio", variant="secondary")

    with gr.Group():
        gr.Markdown("**🎬 Videos (proyectos)** — se borra la carpeta completa del video: imágenes, audios, clips y VIDEO_FINAL.")
        cg_v = gr.CheckboxGroup(choices=_choices_v(_v0), value=[], label="", show_label=False)
    with gr.Group():
        gr.Markdown("**📁 Plantillas** — ⚠️ contienen tus imágenes de jueces, voces y reacciones (no se regeneran).")
        cg_t = gr.CheckboxGroup(choices=_choices_t(_t0), value=[], label="", show_label=False)

    calc_md = gr.Markdown("Nada marcado.")
    btn_borrar = gr.Button("🗑️ Eliminar los marcados…", variant="stop")
    with gr.Group(visible=False) as confirm_box:
        confirm_md = gr.Markdown("")
        ack = gr.Checkbox(label="Entiendo que las plantillas marcadas se borran para siempre", value=False, visible=False)
        with gr.Row():
            btn_si = gr.Button("✅ Sí, eliminar definitivamente", variant="stop")
            btn_no = gr.Button("↩️ Cancelar", variant="secondary")
    resultado = gr.Textbox(label="Resultado", interactive=False, lines=8)

    cg_v.change(ui_calcular, [cg_v, cg_t], [calc_md])
    cg_t.change(ui_calcular, [cg_v, cg_t], [calc_md])
    btn_borrar.click(ui_pedir_confirmacion, [cg_v, cg_t], [confirm_box, confirm_md, ack])
    btn_no.click(ui_cancelar, [], [confirm_box, confirm_md, ack])
    btn_si.click(ui_borrar, [cg_v, cg_t, ack], [resultado, disco_html, confirm_box, confirm_md, cg_v, cg_t, ack])
    btn_ref.click(ui_refrescar, [], [disco_html, cg_v, cg_t, calc_md])

demo_limpieza.queue().launch(share=True, prevent_thread_lock=True, show_error=True)
