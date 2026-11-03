# ═══════════════════════════════════════════════════════════════
# CELDA EXTRA A: 📤 EXPORTAR UN VIDEO (para continuarlo en otra cuenta de Kaggle)
#
# - Empaqueta UN video (carpeta completa en /videos) en un único .zip.
# - Los assets de la PLANTILLA no se exportan: solo va la estructura vacía + plantilla.json
#   + la lista de archivos que debes volver a subir en la otra cuenta.
# - Opcional: voz del concursante (pequeña) y excluir la carpeta tmp.
# - Solo LEE tu proyecto. No modifica nada. Todas las funciones llevan prefijo _xp_ / xp_
#   para no pisar nombres de otras celdas.
# ═══════════════════════════════════════════════════════════════
import os, json, time, shutil, zipfile
import gradio as gr

_XP_OUT   = globals().get("OUTPUT_DIR") or "/kaggle/working/VideoAutomationProject"
_XP_VID   = os.path.realpath(globals().get("VIDEOS_DIR") or f"{_XP_OUT}/videos")
_XP_TPL   = os.path.realpath(globals().get("PLANTILLAS_DIR") or f"{_XP_OUT}/plantillas")
_XP_UIST  = os.path.join(_XP_OUT, "ui_state.json")
_XP_VOICE_DIR   = os.path.join(_XP_OUT, "voice_refs")
_XP_VOICE_STATE = os.path.join(_XP_OUT, "contestant_voice_state.json")
_XP_VOICE_FILES = ["contestant_ref.wav", "contestant_ref.txt", "contestant_preview.wav",
                   "design_as_ref.wav", "design_as_ref.txt"]
_XP_SUBDIRS = list((globals().get("PLANTILLA_SUBDIRS") or {
    "a": "imagenes/jueces", "b": "imagenes/escenario_lejano", "c": "imagenes/escenario_cercano",
    "d": "voces/jueces", "e": "video/reacciones_primera_mitad", "f": "video/reacciones_segunda_mitad",
}).values())
_XP_DESTINOS = ["/kaggle/tmp/exportaciones", "/kaggle/working/exportaciones"]
_XP_SIN_COMPRIMIR = (".mp4", ".wav", ".mp3", ".png", ".jpg", ".jpeg", ".webp", ".mov", ".zip", ".gz", ".safetensors")
for _d in _XP_DESTINOS:
    os.makedirs(_d, exist_ok=True)

def _xp_fmt(n):
    return f"{n/1024**3:.2f} GB" if n >= 1024**3 else f"{n/1024**2:.1f} MB"

def _xp_json(p, default=None):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _xp_dir_size(path):
    tot = 0
    for root, _, files in os.walk(path):
        for fn in files:
            fp = os.path.join(root, fn)
            if not os.path.islink(fp):
                try: tot += os.path.getsize(fp)
                except OSError: pass
    return tot

def _xp_videos():
    st = _xp_json(_XP_UIST, {}) or {}
    activo = os.path.realpath(st.get("current_video_dir", "")) if st.get("current_video_dir") else ""
    out = []
    if os.path.isdir(_XP_VID):
        for d in sorted(os.listdir(_XP_VID)):
            p = os.path.join(_XP_VID, d)
            if d.startswith(".") or not os.path.isdir(p):
                continue
            est = _xp_json(os.path.join(p, "estado.json"), {}) or {}
            out.append({"nombre": d, "ruta": p, "bytes": _xp_dir_size(p), "mtime": os.path.getmtime(p),
                        "tpl": est.get("plantilla_slug", ""), "activo": os.path.realpath(p) == activo})
    out.sort(key=lambda x: -x["mtime"])
    return out

def _xp_choices():
    ch = []
    for v in _xp_videos():
        f = time.strftime("%Y-%m-%d %H:%M", time.localtime(v["mtime"]))
        ch.append((f"{v['nombre']}  ·  {_xp_fmt(v['bytes'])}  ·  {f}" + ("  🟢 ACTIVO" if v["activo"] else ""), v["nombre"]))
    return ch

def _xp_plan(nombre, excluir_tmp, incl_voz):
    vdir = os.path.realpath(os.path.join(_XP_VID, nombre))
    if os.path.dirname(vdir) != _XP_VID or not os.path.isdir(vdir):
        raise RuntimeError(f"Video no válido: {nombre}")
    skip = {"tmp"} if excluir_tmp else set()
    archivos = []
    for root, dirs, files in os.walk(vdir):
        if root == vdir:
            dirs[:] = [d for d in dirs if d not in skip]
        dirs.sort()
        for fn in sorted(files):
            fp = os.path.join(root, fn)
            if os.path.islink(fp):
                continue
            rel = os.path.relpath(fp, vdir).replace(os.sep, "/")
            archivos.append((fp, f"video/{nombre}/{rel}", os.path.getsize(fp)))
    voz = []
    if incl_voz:
        if os.path.isfile(_XP_VOICE_STATE):
            voz.append((_XP_VOICE_STATE, "voz_concursante/contestant_voice_state.json", os.path.getsize(_XP_VOICE_STATE)))
        for fn in _XP_VOICE_FILES:
            p = os.path.join(_XP_VOICE_DIR, fn)
            if os.path.isfile(p):
                voz.append((p, f"voz_concursante/voice_refs/{fn}", os.path.getsize(p)))
    est = _xp_json(os.path.join(vdir, "estado.json"), {}) or {}
    slug = est.get("plantilla_slug", "") or ""
    plantilla = None
    if slug:
        base = os.path.realpath(os.path.join(_XP_TPL, slug))
        if os.path.dirname(base) == _XP_TPL and os.path.isdir(base):
            omit = []
            for root, _, files in os.walk(base):
                for fn in sorted(files):
                    fp = os.path.join(root, fn)
                    rel = os.path.relpath(fp, base).replace(os.sep, "/")
                    if rel == "plantilla.json" or os.path.islink(fp):
                        continue
                    omit.append({"ruta": rel, "bytes": os.path.getsize(fp)})
            plantilla = {"slug": slug,
                         "meta": _xp_json(os.path.join(base, "plantilla.json"), {}) or {},
                         "subdirs": _XP_SUBDIRS, "archivos_omitidos": omit}
    return vdir, archivos, voz, est, slug, plantilla

def xp_resumen(nombre, excluir_tmp, incl_voz):
    if not nombre:
        return "Elige un video."
    try:
        vdir, archivos, voz, est, slug, pl = _xp_plan(nombre, excluir_tmp, incl_voz)
    except Exception as e:
        return f"❌ {e}"
    tot = sum(s for *_, s in archivos + voz)
    txt = f"📦 Se exportarán **{len(archivos)}** archivos del video (**{_xp_fmt(tot)}** en total)"
    if voz: txt += f" + voz del concursante ({len(voz)} archivos)"
    if pl:
        om = sum(o["bytes"] for o in pl["archivos_omitidos"])
        txt += (f"\n\n📁 Plantilla **{slug}**: va VACÍA (solo estructura + plantilla.json). "
                f"No se incluyen {len(pl['archivos_omitidos'])} archivos ({_xp_fmt(om)}).")
    elif slug:
        txt += f"\n\n⚠️ La plantilla «{slug}» de este video ya no existe aquí; no se puede incluir su estructura."
    else:
        txt += "\n\n⚠️ Este video no tiene plantilla registrada en estado.json."
    libre = shutil.disk_usage("/kaggle/tmp").free
    txt += f"\n\n💾 Libre en /kaggle/tmp: {_xp_fmt(libre)}"
    return txt

def xp_exportar(nombre, excluir_tmp, incl_voz, destino, progress=gr.Progress()):
    if not nombre:
        return "⚠️ Elige un video.", None
    if destino not in _XP_DESTINOS:
        destino = _XP_DESTINOS[0]
    try:
        vdir, archivos, voz, est, slug, plantilla = _xp_plan(nombre, excluir_tmp, incl_voz)
    except Exception as e:
        return f"❌ {e}", None
    todos = archivos + voz
    total = sum(s for *_, s in todos) or 1
    os.makedirs(destino, exist_ok=True)
    libre = shutil.disk_usage(destino).free
    if libre < total * 1.02 + 200 * 1024**2:
        return (f"⛔ Espacio insuficiente en {destino}: necesitas ~{_xp_fmt(total)} y hay {_xp_fmt(libre)} libres. "
                "Prueba con el otro destino o excluye tmp."), None
    manifest = {
        "formato": 1, "video": nombre, "plantilla_slug": slug, "personaje": est.get("personaje", ""),
        "output_dir": _XP_OUT, "exportado_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "excluyo_tmp": bool(excluir_tmp), "incluye_voz": bool(voz), "n_archivos": len(archivos),
        "plantilla": plantilla,
    }
    zip_final = os.path.join(destino, f"{nombre}.zip")
    tmp = zip_final + ".part"
    errores, hecho, t0 = [], 0, time.time()
    try:
        with zipfile.ZipFile(tmp, "w", allowZip64=True) as zf:
            zf.writestr("EXPORT_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2),
                        compress_type=zipfile.ZIP_DEFLATED)
            for fp, arc, size in todos:
                comp = zipfile.ZIP_STORED if fp.lower().endswith(_XP_SIN_COMPRIMIR) else zipfile.ZIP_DEFLATED
                try:
                    zf.write(fp, arc, compress_type=comp)
                except OSError as e:
                    errores.append(f"{arc}: {e}")
                hecho += size
                progress(min(hecho / total, 1.0), desc=f"{_xp_fmt(hecho)} / {_xp_fmt(total)}")
        os.replace(tmp, zip_final)
    except Exception as e:
        try: os.remove(tmp)
        except OSError: pass
        return f"❌ Falló la exportación: {e}", None
    msg = [f"✅ Exportado en {time.time()-t0:.0f}s: {zip_final}  ({_xp_fmt(os.path.getsize(zip_final))})"]
    if plantilla and plantilla["archivos_omitidos"]:
        msg.append(f"📁 Plantilla «{slug}» sin assets: en la otra cuenta tendrás que volver a subir "
                   f"{len(plantilla['archivos_omitidos'])} archivos (la importación te los listará).")
    if errores:
        msg.append("⚠️ Archivos que no se pudieron leer (¿se estaban escribiendo?):\n  " + "\n  ".join(errores[:10]))
    msg.append("➡️ Descárgalo con el componente de abajo, o súbelo como Dataset de Kaggle en la otra cuenta "
               "(Kaggle descomprime los .zip solos; la celda de importación entiende carpeta o .zip).")
    try: log("[EXPORT] " + " | ".join(msg))
    except Exception: pass
    return "\n".join(msg), zip_final

def xp_refrescar(excluir_tmp, incl_voz):
    ch = _xp_choices()
    val = ch[0][1] if ch else None
    return gr.update(choices=ch, value=val), xp_resumen(val, excluir_tmp, incl_voz)

_ch0 = _xp_choices()
_v0 = _ch0[0][1] if _ch0 else None
with gr.Blocks(title="📤 Exportar video") as demo_exportar:
    gr.Markdown("## 📤 Exportar un video para continuarlo en otra cuenta\n"
                "Los assets de la plantilla **no** se exportan (los vuelves a subir en la otra cuenta).")
    with gr.Row():
        dd_video = gr.Dropdown(choices=_ch0, value=_v0, label="Video a exportar", scale=4)
        btn_ref = gr.Button("🔄 Refrescar", scale=1)
    with gr.Row():
        chk_tmp = gr.Checkbox(value=True, label="Excluir carpeta tmp (temporales, se regeneran)")
        chk_voz = gr.Checkbox(value=True, label="Incluir voz del concursante (pequeña)")
    dd_dest = gr.Dropdown(choices=_XP_DESTINOS, value=_XP_DESTINOS[0],
                          label="Dónde guardar el .zip (/kaggle/tmp = más espacio, no persiste · /kaggle/working = persiste como output)")
    resumen = gr.Markdown(xp_resumen(_v0, True, True))
    btn_exp = gr.Button("📤 Exportar", variant="primary")
    salida = gr.Textbox(label="Resultado", interactive=False, lines=6)
    archivo = gr.File(label="Descargar .zip", interactive=False)

    for comp in (dd_video, chk_tmp, chk_voz):
        comp.change(xp_resumen, [dd_video, chk_tmp, chk_voz], [resumen])
    btn_ref.click(xp_refrescar, [chk_tmp, chk_voz], [dd_video, resumen])
    btn_exp.click(xp_exportar, [dd_video, chk_tmp, chk_voz, dd_dest], [salida, archivo])

demo_exportar.queue().launch(share=True, prevent_thread_lock=True, show_error=True, allowed_paths=_XP_DESTINOS)