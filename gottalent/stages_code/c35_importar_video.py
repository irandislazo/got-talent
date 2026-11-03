# ═══════════════════════════════════════════════════════════════
# CELDA EXTRA B: 📥 IMPORTAR UN VIDEO exportado desde otra cuenta de Kaggle
#
# - Acepta: un .zip (subido aquí, o detectado en /kaggle/input y otras rutas) o una CARPETA
#   (por ejemplo un Dataset de Kaggle donde subiste el .zip, que Kaggle descomprime solo).
# - Restaura el video en /videos, recrea la estructura VACÍA de la plantilla si no existe aquí
#   (nunca toca una plantilla existente) y te lista los assets que debes volver a subir.
# - Opcional: restaura la voz del concursante y deja el video como activo.
# - No modifica ninguna celda del proyecto. Funciones con prefijo _ip_ / ip_.
# ═══════════════════════════════════════════════════════════════
import os, re, json, time, shutil, zipfile
import gradio as gr

_IP_OUT   = globals().get("OUTPUT_DIR") or "/kaggle/working/VideoAutomationProject"
_IP_VID   = os.path.realpath(globals().get("VIDEOS_DIR") or f"{_IP_OUT}/videos")
_IP_TPL   = os.path.realpath(globals().get("PLANTILLAS_DIR") or f"{_IP_OUT}/plantillas")
_IP_UIST  = os.path.join(_IP_OUT, "ui_state.json")
_IP_VOICE_DIR   = os.path.join(_IP_OUT, "voice_refs")
_IP_VOICE_STATE = os.path.join(_IP_OUT, "contestant_voice_state.json")
_IP_VOICE_FILES = ["contestant_ref.wav", "contestant_ref.txt", "contestant_preview.wav",
                   "design_as_ref.wav", "design_as_ref.txt"]
_IP_VIDEO_SUBDIRS = ["etapa1_imagenes", "etapa2_dialogos", "etapa3_cancion", "etapa4_final",
                     "tmp", "comfy_input", "comfy_output"]
_IP_MANIFEST = "EXPORT_MANIFEST.json"
_IP_BUSCAR = [("/kaggle/input", 5), ("/kaggle/tmp/exportaciones", 2),
              ("/kaggle/working/exportaciones", 2), ("/kaggle/working", 0)]

def _ip_fmt(n):
    return f"{n/1024**3:.2f} GB" if n >= 1024**3 else f"{n/1024**2:.1f} MB"

def _ip_nombre_ok(n):
    return bool(n) and n not in (".", "..") and not n.startswith(".") and not re.search(r"[/\\]", n)

def _ip_escanear():
    out, vistos = [], set()
    for root, maxd in _IP_BUSCAR:
        if not os.path.isdir(root):
            continue
        d0 = root.rstrip("/").count("/")
        for cur, dirs, files in os.walk(root):
            depth = cur.rstrip("/").count("/") - d0
            if _IP_MANIFEST in files:
                if cur not in vistos:
                    vistos.add(cur)
                    try: v = json.load(open(os.path.join(cur, _IP_MANIFEST), encoding="utf-8")).get("video", "?")
                    except Exception: v = "?"
                    out.append((f"📁 {v}  ←  {cur}", cur))
                dirs[:] = []
                continue
            for fn in files:
                if fn.lower().endswith(".zip"):
                    p = os.path.join(cur, fn)
                    if p not in vistos:
                        vistos.add(p)
                        try: sz = _ip_fmt(os.path.getsize(p))
                        except OSError: sz = "?"
                        out.append((f"📦 {p}  ·  {sz}", p))
            if depth >= maxd:
                dirs[:] = []
    return out

def _ip_origen(upload, sel, manual):
    p = None
    if upload:
        p = upload if isinstance(upload, str) else getattr(upload, "name", None)
    if not p:
        p = (manual or "").strip().strip('"').strip("'")
    if not p:
        p = sel or ""
    return p

def _ip_abrir(path):
    # → (tipo, manifest, handle). tipo: "zip" (handle=(ZipFile, prefijo)) | "dir" (handle=raíz)
    if not path or not os.path.exists(path):
        raise RuntimeError("Elige, escribe o sube un .zip / carpeta válidos.")
    if os.path.isdir(path):
        raiz = None
        if os.path.exists(os.path.join(path, _IP_MANIFEST)):
            raiz = path
        else:
            d0 = path.rstrip("/").count("/")
            for cur, dirs, files in os.walk(path):
                if _IP_MANIFEST in files:
                    raiz = cur; break
                if cur.rstrip("/").count("/") - d0 >= 3:
                    dirs[:] = []
        if not raiz:
            raise RuntimeError(f"No encuentro {_IP_MANIFEST} dentro de esa carpeta (¿es un export de esta celda?).")
        return "dir", json.load(open(os.path.join(raiz, _IP_MANIFEST), encoding="utf-8")), raiz
    if not zipfile.is_zipfile(path):
        raise RuntimeError("El archivo no es un .zip válido.")
    zf = zipfile.ZipFile(path)
    cand = [n for n in zf.namelist() if n.endswith(_IP_MANIFEST) and n.count("/") <= 1]
    if not cand:
        zf.close()
        raise RuntimeError(f"El .zip no contiene {_IP_MANIFEST} (¿es un export de esta celda?).")
    mf = min(cand, key=len)
    return "zip", json.loads(zf.read(mf).decode("utf-8")), (zf, mf[:-len(_IP_MANIFEST)])

def _ip_validar(m):
    if m.get("formato") != 1:
        raise RuntimeError("Formato de export desconocido.")
    if not _ip_nombre_ok(m.get("video", "")):
        raise RuntimeError("Nombre de video no válido en el manifiesto.")
    if m.get("plantilla_slug") and not _ip_nombre_ok(m["plantilla_slug"]):
        raise RuntimeError("Nombre de plantilla no válido en el manifiesto.")

def ip_inspeccionar(upload, sel, manual):
    h = None
    try:
        tipo, m, h = _ip_abrir(_ip_origen(upload, sel, manual))
        _ip_validar(m)
    except Exception as e:
        return f"❌ {e}"
    finally:
        try:
            if isinstance(h, tuple): h[0].close()
        except Exception:
            pass
    pl = m.get("plantilla") or {}
    om = pl.get("archivos_omitidos", [])
    ya = os.path.isdir(os.path.join(_IP_VID, m["video"]))
    return (f"🎬 Video: {m['video']}   ·   🎭 Personaje: {m.get('personaje','?')}\n"
            f"📁 Plantilla: {m.get('plantilla_slug') or '—'}"
            f"{'  (ya existe aquí ✅)' if m.get('plantilla_slug') and os.path.isdir(os.path.join(_IP_TPL, m['plantilla_slug'])) else '  (no existe aquí: se creará vacía)'}\n"
            f"🗂️ {m.get('n_archivos','?')} archivos · exportado {m.get('exportado_utc','?')} · "
            f"voz incluida: {'sí' if m.get('incluye_voz') else 'no'}\n"
            f"📋 Assets de plantilla que tendrás que subir: {len(om)}\n"
            f"{'⚠️ Ya existe un video con ese nombre aquí (marca «Reemplazar» para sobrescribirlo).' if ya else ''}")

def _ip_reescribir_json(carpeta, viejo, nuevo):
    if not viejo or viejo == nuevo:
        return 0
    n = 0
    for root, _, files in os.walk(carpeta):
        for fn in files:
            if fn.lower().endswith(".json"):
                p = os.path.join(root, fn)
                try:
                    if os.path.getsize(p) > 64 * 1024**2: continue
                    t = open(p, encoding="utf-8").read()
                    if viejo in t:
                        open(p, "w", encoding="utf-8").write(t.replace(viejo, nuevo)); n += 1
                except Exception:
                    pass
    return n

def _ip_copiar(fsrc, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as o:
        shutil.copyfileobj(fsrc, o, 8 * 1024 * 1024)

def ip_importar(upload, sel, manual, reemplazar, restaurar_voz, activar, progress=gr.Progress()):
    h = None
    stage = None
    try:
        tipo, m, h = _ip_abrir(_ip_origen(upload, sel, manual))
        _ip_validar(m)
        nombre, slug = m["video"], m.get("plantilla_slug", "") or ""
        dest = os.path.join(_IP_VID, nombre)
        if os.path.exists(dest) and not reemplazar:
            return f"⛔ Ya existe el video «{nombre}» aquí. Marca «Reemplazar si ya existe» si quieres sobrescribirlo. No se hizo nada."
        os.makedirs(_IP_VID, exist_ok=True)
        stage = os.path.join(_IP_VID, f".import_{nombre}")
        if os.path.exists(stage):
            shutil.rmtree(stage)
        os.makedirs(stage)
        stage_real = os.path.realpath(stage)

        # ── lista de archivos a copiar ──
        if tipo == "zip":
            zf, pref = h
            pv = f"{pref}video/{nombre}/"
            items = [(i.filename[len(pv):], i.file_size, i) for i in zf.infolist()
                     if i.filename.startswith(pv) and not i.is_dir()]
        else:
            base_v = os.path.join(h, "video", nombre)
            if not os.path.isdir(base_v):
                raise RuntimeError(f"No encuentro la carpeta video/{nombre} en el origen.")
            items = []
            for root, _, files in os.walk(base_v):
                for fn in files:
                    fp = os.path.join(root, fn)
                    items.append((os.path.relpath(fp, base_v).replace(os.sep, "/"), os.path.getsize(fp), fp))
        if not items:
            raise RuntimeError("El export no contiene archivos del video.")
        total = sum(s for _, s, _ in items) or 1
        libre = shutil.disk_usage(_IP_VID).free
        if libre < total * 1.02 + 200 * 1024**2:
            raise RuntimeError(f"Espacio insuficiente en /kaggle/working: necesitas ~{_ip_fmt(total)} y hay {_ip_fmt(libre)}.")

        # ── copiar con protección de rutas ──
        hecho = 0
        for rel, size, src in items:
            dst = os.path.realpath(os.path.join(stage_real, rel))
            if not dst.startswith(stage_real + os.sep):
                continue
            if tipo == "zip":
                with h[0].open(src) as fs:
                    _ip_copiar(fs, dst)
            else:
                with open(src, "rb") as fs:
                    _ip_copiar(fs, dst)
            hecho += size
            progress(min(hecho / total, 1.0) * 0.95, desc=f"{_ip_fmt(hecho)} / {_ip_fmt(total)}")
        for sd in _IP_VIDEO_SUBDIRS:
            os.makedirs(os.path.join(stage_real, sd), exist_ok=True)
        viejo_out = m.get("output_dir", "")
        reesc = _ip_reescribir_json(stage_real, viejo_out, _IP_OUT)

        # ── intercambio atómico ──
        if os.path.exists(dest):
            shutil.rmtree(dest)
        os.replace(stage_real, dest)
        stage = None
        msg = [f"✅ Video importado: {dest}  ({_ip_fmt(total)})"]
        if reesc:
            msg.append(f"🔧 Rutas ajustadas en {reesc} archivo(s) .json.")

        # ── plantilla: solo se crea vacía si NO existe; nunca se toca una existente ──
        pl = m.get("plantilla") or {}
        if slug:
            base_t = os.path.join(_IP_TPL, slug)
            omit = pl.get("archivos_omitidos", [])
            if not os.path.isdir(base_t):
                for sub in (pl.get("subdirs") or []):
                    if _ip_nombre_ok(sub.split("/")[0]) and ".." not in sub.split("/"):
                        os.makedirs(os.path.join(base_t, sub), exist_ok=True)
                os.makedirs(base_t, exist_ok=True)
                meta = dict(pl.get("meta") or {})
                meta.setdefault("nombre", slug); meta["slug"] = slug
                with open(os.path.join(base_t, "plantilla.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                msg.append(f"📁 Plantilla «{slug}» creada VACÍA (estructura + plantilla.json).")
                faltan = omit
            else:
                msg.append(f"📁 Plantilla «{slug}» ya existía: no se tocó.")
                faltan = [o for o in omit if not os.path.exists(os.path.join(base_t, o["ruta"]))]
            if faltan:
                msg.append(f"📋 Vuelve a subir estos {len(faltan)} assets a la plantilla «{slug}» "
                           f"(carpeta {base_t}):")
                msg += [f"   · {o['ruta']}  ({_ip_fmt(o['bytes'])})" for o in faltan[:80]]
                if len(faltan) > 80: msg.append(f"   … y {len(faltan)-80} más.")
            else:
                msg.append("📋 La plantilla ya tiene todos los assets del original ✅")
        else:
            msg.append("ℹ️ El export no tenía plantilla asociada.")

        # ── voz del concursante ──
        if restaurar_voz and m.get("incluye_voz"):
            n_voz = 0
            os.makedirs(_IP_VOICE_DIR, exist_ok=True)
            destinos = {"contestant_voice_state.json": _IP_VOICE_STATE}
            destinos.update({f"voice_refs/{fn}": os.path.join(_IP_VOICE_DIR, fn) for fn in _IP_VOICE_FILES})
            for rel, dst in destinos.items():
                try:
                    if tipo == "zip":
                        nm = f"{h[1]}voz_concursante/{rel}"
                        if nm not in h[0].namelist(): continue
                        with h[0].open(nm) as fs: _ip_copiar(fs, dst)
                    else:
                        sp = os.path.join(h, "voz_concursante", rel)
                        if not os.path.isfile(sp): continue
                        with open(sp, "rb") as fs: _ip_copiar(fs, dst)
                    n_voz += 1
                except Exception as e:
                    msg.append(f"⚠️ Voz: no pude restaurar {rel}: {e}")
            if viejo_out and viejo_out != _IP_OUT and os.path.isfile(_IP_VOICE_STATE):
                t = open(_IP_VOICE_STATE, encoding="utf-8").read()
                open(_IP_VOICE_STATE, "w", encoding="utf-8").write(t.replace(viejo_out, _IP_OUT))
            try:  # sincroniza el estado en memoria sin reasignar la variable
                st = json.load(open(_IP_VOICE_STATE, encoding="utf-8"))
                cur = globals().get("CONTESTANT_VOICE_STATE")
                if isinstance(cur, dict):
                    cur.clear(); cur.update(st)
            except Exception:
                pass
            msg.append(f"🎙️ Voz del concursante restaurada ({n_voz} archivos; sobrescribe la voz que hubiera aquí).")
        elif m.get("incluye_voz"):
            msg.append("🎙️ El export incluía la voz del concursante pero no la restauraste.")

        # ── dejar como video activo ──
        if activar:
            try:
                st = json.load(open(_IP_UIST, encoding="utf-8")) if os.path.exists(_IP_UIST) else {}
            except Exception:
                st = {}
            st["current_video_dir"] = dest
            if slug: st["current_template_slug"] = slug
            json.dump(st, open(_IP_UIST, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            for k, v in (("_CUR_VP_DIR", dest), ("_CUR_TPL", slug)):
                if k in globals() and v: globals()[k] = v
            ui = globals().get("_UI_STATE")
            if isinstance(ui, dict):
                ui["current_video_dir"] = dest
                if slug: ui["current_template_slug"] = slug
            msg.append("🟢 Marcado como video activo. En la UI principal pulsa «📋 Listar» para verlo.")
        progress(1.0, desc="Listo")
        try: log("[IMPORT] " + " | ".join(x for x in msg if x))
        except Exception: pass
        return "\n".join(msg)
    except Exception as e:
        return f"❌ No se importó nada: {e}"
    finally:
        try:
            if h is not None and isinstance(h, tuple): h[0].close()
        except Exception:
            pass
        if stage and os.path.isdir(stage):
            shutil.rmtree(stage, ignore_errors=True)

def ip_refrescar():
    return gr.update(choices=_ip_escanear(), value=None)

with gr.Blocks(title="📥 Importar video") as demo_importar:
    gr.Markdown("## 📥 Importar un video exportado desde otra cuenta\n"
                "Elige **una** de las tres formas de indicar el origen (si hay varias, manda: subida → ruta escrita → lista).")
    with gr.Row():
        dd_origen = gr.Dropdown(choices=_ip_escanear(), value=None, allow_custom_value=True, scale=4,
                                label="Detectado en /kaggle/input y rutas habituales (.zip o carpeta)")
        btn_buscar = gr.Button("🔄 Buscar de nuevo", scale=1)
    tb_manual = gr.Textbox(label="…o escribe la ruta de un .zip o de una carpeta", placeholder="/kaggle/input/mi-dataset/...")
    f_subida = gr.File(label="…o sube aquí el .zip (para archivos muy grandes es más rápido usar un Dataset)",
                       file_types=[".zip"], type="filepath")
    with gr.Row():
        chk_reemp = gr.Checkbox(value=False, label="Reemplazar si ya existe un video con ese nombre (borra el existente)")
        chk_voz = gr.Checkbox(value=True, label="Restaurar voz del concursante (sobrescribe la actual)")
        chk_act = gr.Checkbox(value=True, label="Dejarlo como video activo")
    with gr.Row():
        btn_ver = gr.Button("🔍 Ver qué contiene", variant="secondary")
        btn_imp = gr.Button("📥 Importar", variant="primary")
    salida = gr.Textbox(label="Resultado", interactive=False, lines=14)

    btn_buscar.click(ip_refrescar, [], [dd_origen])
    btn_ver.click(ip_inspeccionar, [f_subida, dd_origen, tb_manual], [salida])
    btn_imp.click(ip_importar, [f_subida, dd_origen, tb_manual, chk_reemp, chk_voz, chk_act], [salida])

demo_importar.queue().launch(share=True, prevent_thread_lock=True, show_error=True)