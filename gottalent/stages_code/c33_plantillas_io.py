# ═══════════════════════════════════════════════════════════════
# CELDA EXTRA C: 📦 EXPORTAR / IMPORTAR PLANTILLAS (todo en una UI con 2 pestañas)
#
# - Exportar: empaqueta una plantilla COMPLETA (imágenes, voces, reacciones, plantilla.json) en un .zip.
#   Opcional: incluye las voces clonadas de los jueces (voice_refs/jueces, archivos pequeños).
# - Importar: desde .zip (subido, detectado o por ruta) o carpeta (p. ej. Dataset de Kaggle).
#   Si ya existe una plantilla con el mismo nombre eliges: no tocar / sobrescribir / fusionar.
# - Exportar solo LEE tu proyecto. Funciones con prefijo _tp_ / tp_ para no pisar otras celdas.
# ═══════════════════════════════════════════════════════════════
import os, re, json, time, shutil, zipfile
import gradio as gr

_TP_OUT  = globals().get("OUTPUT_DIR") or "/kaggle/working/VideoAutomationProject"
_TP_TPL  = os.path.realpath(globals().get("PLANTILLAS_DIR") or f"{_TP_OUT}/plantillas")
_TP_JUECES_DIR = os.path.join(_TP_OUT, "voice_refs", "jueces")
_TP_SUBDIRS = list((globals().get("PLANTILLA_SUBDIRS") or {
    "a": "imagenes/jueces", "b": "imagenes/escenario_lejano", "c": "imagenes/escenario_cercano",
    "d": "voces/jueces", "e": "video/reacciones_primera_mitad", "f": "video/reacciones_segunda_mitad",
}).values())
_TP_MANIFEST = "EXPORT_PLANTILLA_MANIFEST.json"
_TP_DESTINOS = ["/kaggle/tmp/exportaciones", "/kaggle/working/exportaciones"]
_TP_SIN_COMPRIMIR = (".mp4", ".wav", ".mp3", ".png", ".jpg", ".jpeg", ".webp", ".mov", ".zip", ".gz")
_TP_JUEZ_RE = re.compile(r"^juez_\d+_.+\.(wav|txt)$", re.I)
_TP_BUSCAR = [("/kaggle/input", 5), ("/kaggle/tmp/exportaciones", 2),
              ("/kaggle/working/exportaciones", 2), ("/kaggle/working", 0)]
_TP_MODOS = ["No tocar si ya existe", "Sobrescribir (reemplaza toda la plantilla)",
             "Fusionar (añade lo que falte, conserva lo existente)"]
for _d in _TP_DESTINOS:
    os.makedirs(_d, exist_ok=True)

def _tp_fmt(n):
    return f"{n/1024**3:.2f} GB" if n >= 1024**3 else f"{n/1024**2:.1f} MB"

def _tp_nombre_ok(n):
    return bool(n) and n not in (".", "..") and not n.startswith(".") and not re.search(r"[/\\]", n)

def _tp_size(path):
    tot = 0
    for root, _, files in os.walk(path):
        for fn in files:
            fp = os.path.join(root, fn)
            if not os.path.islink(fp):
                try: tot += os.path.getsize(fp)
                except OSError: pass
    return tot

def _tp_copiar(fsrc, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as o:
        shutil.copyfileobj(fsrc, o, 8 * 1024 * 1024)

# ───────────────────────── EXPORTAR ─────────────────────────
def _tp_lista():
    out = []
    if os.path.isdir(_TP_TPL):
        for slug in sorted(os.listdir(_TP_TPL)):
            p = os.path.join(_TP_TPL, slug)
            mp = os.path.join(p, "plantilla.json")
            if slug.startswith(".") or not os.path.isfile(mp):
                continue
            try: nombre = json.load(open(mp, encoding="utf-8")).get("nombre", slug)
            except Exception: nombre = slug
            out.append((f"{nombre}  ({slug})  ·  {_tp_fmt(_tp_size(p))}", slug))
    return out

def _tp_plan(slug, incl_jueces):
    base = os.path.realpath(os.path.join(_TP_TPL, slug))
    if os.path.dirname(base) != _TP_TPL or not os.path.isdir(base):
        raise RuntimeError(f"Plantilla no válida: {slug}")
    files = []
    for root, dirs, fs in os.walk(base):
        dirs.sort()
        for fn in sorted(fs):
            fp = os.path.join(root, fn)
            if os.path.islink(fp):
                continue
            rel = os.path.relpath(fp, base).replace(os.sep, "/")
            files.append((fp, f"plantilla/{slug}/{rel}", os.path.getsize(fp)))
    jueces = []
    if incl_jueces and os.path.isdir(_TP_JUECES_DIR):
        for fn in sorted(os.listdir(_TP_JUECES_DIR)):
            fp = os.path.join(_TP_JUECES_DIR, fn)
            if _TP_JUEZ_RE.match(fn) and os.path.isfile(fp):
                jueces.append((fp, f"voces_jueces/{fn}", os.path.getsize(fp)))
    return base, files, jueces

def tp_resumen(slug, incl_jueces):
    if not slug:
        return "Elige una plantilla."
    try:
        base, files, jueces = _tp_plan(slug, incl_jueces)
    except Exception as e:
        return f"❌ {e}"
    tot = sum(s for *_, s in files + jueces)
    t = f"📦 Se exportarán **{len(files)}** archivos de la plantilla (**{_tp_fmt(tot)}**)"
    if jueces: t += f" + {len(jueces)} archivos de voces clonadas de jueces"
    elif incl_jueces: t += "\n\nℹ️ No hay voces clonadas de jueces en voice_refs/jueces."
    return t + f"\n\n💾 Libre en /kaggle/tmp: {_tp_fmt(shutil.disk_usage('/kaggle/tmp').free)}"

def tp_exportar(slug, incl_jueces, destino, progress=gr.Progress()):
    if not slug:
        return "⚠️ Elige una plantilla.", None
    if destino not in _TP_DESTINOS:
        destino = _TP_DESTINOS[0]
    try:
        base, files, jueces = _tp_plan(slug, incl_jueces)
    except Exception as e:
        return f"❌ {e}", None
    todos = files + jueces
    total = sum(s for *_, s in todos) or 1
    os.makedirs(destino, exist_ok=True)
    libre = shutil.disk_usage(destino).free
    if libre < total * 1.02 + 200 * 1024**2:
        return f"⛔ Espacio insuficiente en {destino}: necesitas ~{_tp_fmt(total)} y hay {_tp_fmt(libre)}.", None
    try: nombre = json.load(open(os.path.join(base, "plantilla.json"), encoding="utf-8")).get("nombre", slug)
    except Exception: nombre = slug
    manifest = {"formato": 1, "tipo": "plantilla", "slug": slug, "nombre": nombre, "output_dir": _TP_OUT,
                "exportado_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "n_archivos": len(files), "bytes": sum(s for *_, s in files),
                "incluye_voces_jueces": bool(jueces)}
    zip_final = os.path.join(destino, f"plantilla_{slug}.zip")
    tmp = zip_final + ".part"
    errores, hecho, t0 = [], 0, time.time()
    try:
        with zipfile.ZipFile(tmp, "w", allowZip64=True) as zf:
            zf.writestr(_TP_MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2),
                        compress_type=zipfile.ZIP_DEFLATED)
            for fp, arc, size in todos:
                comp = zipfile.ZIP_STORED if fp.lower().endswith(_TP_SIN_COMPRIMIR) else zipfile.ZIP_DEFLATED
                try: zf.write(fp, arc, compress_type=comp)
                except OSError as e: errores.append(f"{arc}: {e}")
                hecho += size
                progress(min(hecho / total, 1.0), desc=f"{_tp_fmt(hecho)} / {_tp_fmt(total)}")
        os.replace(tmp, zip_final)
    except Exception as e:
        try: os.remove(tmp)
        except OSError: pass
        return f"❌ Falló la exportación: {e}", None
    msg = [f"✅ Plantilla exportada en {time.time()-t0:.0f}s: {zip_final}  ({_tp_fmt(os.path.getsize(zip_final))})"]
    if errores:
        msg.append("⚠️ No se pudieron leer:\n  " + "\n  ".join(errores[:10]))
    msg.append("➡️ Descárgalo abajo, o súbelo como Dataset de Kaggle en la otra cuenta "
               "(Kaggle lo descomprime solo; la importación acepta .zip o carpeta).")
    try: log("[EXPORT-PLANTILLA] " + " | ".join(msg))
    except Exception: pass
    return "\n".join(msg), zip_final

def tp_refrescar_export(incl_jueces):
    ch = _tp_lista()
    v = ch[0][1] if ch else None
    return gr.update(choices=ch, value=v), tp_resumen(v, incl_jueces)

# ───────────────────────── IMPORTAR ─────────────────────────
def _tp_escanear():
    out, vistos = [], set()
    for root, maxd in _TP_BUSCAR:
        if not os.path.isdir(root):
            continue
        d0 = root.rstrip("/").count("/")
        for cur, dirs, files in os.walk(root):
            depth = cur.rstrip("/").count("/") - d0
            if _TP_MANIFEST in files:
                if cur not in vistos:
                    vistos.add(cur)
                    try: v = json.load(open(os.path.join(cur, _TP_MANIFEST), encoding="utf-8")).get("slug", "?")
                    except Exception: v = "?"
                    out.append((f"📁 plantilla {v}  ←  {cur}", cur))
                dirs[:] = []
                continue
            for fn in files:
                if fn.lower().endswith(".zip"):
                    p = os.path.join(cur, fn)
                    if p not in vistos:
                        vistos.add(p)
                        try: sz = _tp_fmt(os.path.getsize(p))
                        except OSError: sz = "?"
                        out.append((f"📦 {p}  ·  {sz}", p))
            if depth >= maxd:
                dirs[:] = []
    return out

def _tp_origen(upload, sel, manual):
    p = None
    if upload:
        p = upload if isinstance(upload, str) else getattr(upload, "name", None)
    if not p:
        p = (manual or "").strip().strip('"').strip("'")
    return p or sel or ""

def _tp_abrir(path):
    if not path or not os.path.exists(path):
        raise RuntimeError("Elige, escribe o sube un .zip / carpeta válidos.")
    if os.path.isdir(path):
        raiz = path if os.path.exists(os.path.join(path, _TP_MANIFEST)) else None
        if not raiz:
            d0 = path.rstrip("/").count("/")
            for cur, dirs, files in os.walk(path):
                if _TP_MANIFEST in files:
                    raiz = cur; break
                if cur.rstrip("/").count("/") - d0 >= 3:
                    dirs[:] = []
        if not raiz:
            raise RuntimeError(f"No encuentro {_TP_MANIFEST} en esa carpeta (¿es un export de PLANTILLA? "
                               "los exports de video se importan con la otra celda).")
        return "dir", json.load(open(os.path.join(raiz, _TP_MANIFEST), encoding="utf-8")), raiz
    if not zipfile.is_zipfile(path):
        raise RuntimeError("El archivo no es un .zip válido.")
    zf = zipfile.ZipFile(path)
    cand = [n for n in zf.namelist() if n.endswith(_TP_MANIFEST) and n.count("/") <= 1]
    if not cand:
        zf.close()
        raise RuntimeError(f"El .zip no contiene {_TP_MANIFEST} (¿es un export de PLANTILLA? "
                           "los exports de video se importan con la otra celda).")
    mf = min(cand, key=len)
    return "zip", json.loads(zf.read(mf).decode("utf-8")), (zf, mf[:-len(_TP_MANIFEST)])

def _tp_validar(m):
    if m.get("formato") != 1 or m.get("tipo") != "plantilla":
        raise RuntimeError("Este export no es de una plantilla.")
    if not _tp_nombre_ok(m.get("slug", "")):
        raise RuntimeError("Nombre de plantilla no válido en el manifiesto.")

def _tp_cerrar(h):
    try:
        if isinstance(h, tuple): h[0].close()
    except Exception:
        pass

def tp_inspeccionar(upload, sel, manual):
    h = None
    try:
        tipo, m, h = _tp_abrir(_tp_origen(upload, sel, manual))
        _tp_validar(m)
    except Exception as e:
        return f"❌ {e}"
    finally:
        _tp_cerrar(h)
    ya = os.path.isdir(os.path.join(_TP_TPL, m["slug"]))
    return (f"🎭 Plantilla: {m.get('nombre', m['slug'])}  ({m['slug']})\n"
            f"🗂️ {m.get('n_archivos','?')} archivos · {_tp_fmt(m.get('bytes', 0))} · exportada {m.get('exportado_utc','?')}\n"
            f"🎙️ Voces clonadas de jueces incluidas: {'sí' if m.get('incluye_voces_jueces') else 'no'}\n"
            f"{'⚠️ Ya existe una plantilla con ese nombre aquí: elige qué hacer en «Si ya existe».' if ya else '✅ No existe aquí: se creará nueva.'}")

def _tp_reescribir_json(carpeta, viejo, nuevo):
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

def tp_importar(upload, sel, manual, modo, restaurar_jueces, progress=gr.Progress()):
    h = None
    stage = None
    try:
        tipo, m, h = _tp_abrir(_tp_origen(upload, sel, manual))
        _tp_validar(m)
        slug = m["slug"]
        dest = os.path.join(_TP_TPL, slug)
        existe = os.path.isdir(dest)
        if existe and modo == _TP_MODOS[0]:
            return f"⛔ La plantilla «{slug}» ya existe aquí y elegiste «No tocar». No se hizo nada."
        fusionar = existe and modo == _TP_MODOS[2]
        os.makedirs(_TP_TPL, exist_ok=True)

        # ── lista de archivos del export ──
        if tipo == "zip":
            zf, pref = h
            pv = f"{pref}plantilla/{slug}/"
            items = [(i.filename[len(pv):], i.file_size, i) for i in zf.infolist()
                     if i.filename.startswith(pv) and not i.is_dir()]
        else:
            bt = os.path.join(h, "plantilla", slug)
            if not os.path.isdir(bt):
                raise RuntimeError(f"No encuentro la carpeta plantilla/{slug} en el origen.")
            items = []
            for root, _, files in os.walk(bt):
                for fn in files:
                    fp = os.path.join(root, fn)
                    items.append((os.path.relpath(fp, bt).replace(os.sep, "/"), os.path.getsize(fp), fp))
        if not items:
            raise RuntimeError("El export no contiene archivos de la plantilla.")
        if fusionar:
            items = [it for it in items if not os.path.exists(os.path.join(dest, it[0]))]
        total = sum(s for _, s, _ in items) or 1
        libre = shutil.disk_usage(_TP_TPL).free
        if libre < total * 1.02 + 200 * 1024**2:
            raise RuntimeError(f"Espacio insuficiente: necesitas ~{_tp_fmt(total)} y hay {_tp_fmt(libre)}.")

        # ── destino de copia: directo si fusiona, en staging si es nueva/sobrescritura ──
        if fusionar:
            raiz_copia = os.path.realpath(dest)
        else:
            stage = os.path.join(_TP_TPL, f".import_{slug}")
            if os.path.exists(stage): shutil.rmtree(stage)
            os.makedirs(stage)
            raiz_copia = os.path.realpath(stage)
        hecho = 0
        for rel, size, src in items:
            dst = os.path.realpath(os.path.join(raiz_copia, rel))
            if not dst.startswith(raiz_copia + os.sep):
                continue
            if tipo == "zip":
                with h[0].open(src) as fs: _tp_copiar(fs, dst)
            else:
                with open(src, "rb") as fs: _tp_copiar(fs, dst)
            hecho += size
            progress(min(hecho / total, 1.0) * 0.95, desc=f"{_tp_fmt(hecho)} / {_tp_fmt(total)}")
        for sub in _TP_SUBDIRS:
            os.makedirs(os.path.join(raiz_copia, sub), exist_ok=True)
        reesc = 0 if fusionar else _tp_reescribir_json(raiz_copia, m.get("output_dir", ""), _TP_OUT)

        if not fusionar:
            if existe:
                shutil.rmtree(dest)
            os.replace(raiz_copia, dest)
            stage = None
        accion = ("fusionada (solo se añadió lo que faltaba)" if fusionar
                  else "sobrescrita" if existe else "creada")
        msg = [f"✅ Plantilla «{slug}» {accion}: {dest}  ({len(items)} archivos, {_tp_fmt(total)})"]
        if fusionar and not items:
            msg.append("ℹ️ La plantilla existente ya tenía todos los archivos del export.")
        if reesc:
            msg.append(f"🔧 Rutas ajustadas en {reesc} archivo(s) .json.")

        # ── voces clonadas de jueces (globales: voice_refs/jueces) ──
        if m.get("incluye_voces_jueces"):
            if restaurar_jueces:
                os.makedirs(_TP_JUECES_DIR, exist_ok=True)
                n = 0
                if tipo == "zip":
                    pj = f"{h[1]}voces_jueces/"
                    fuentes = [(i.filename[len(pj):], i) for i in h[0].infolist()
                               if i.filename.startswith(pj) and not i.is_dir()]
                else:
                    dj = os.path.join(h, "voces_jueces")
                    fuentes = [(fn, os.path.join(dj, fn)) for fn in (os.listdir(dj) if os.path.isdir(dj) else [])]
                for fn, src in fuentes:
                    if not _TP_JUEZ_RE.match(fn):
                        continue
                    dst = os.path.join(_TP_JUECES_DIR, fn)
                    if tipo == "zip":
                        with h[0].open(src) as fs: _tp_copiar(fs, dst)
                    else:
                        with open(src, "rb") as fs: _tp_copiar(fs, dst)
                    n += 1
                msg.append(f"🎙️ Voces clonadas de jueces restauradas ({n} archivos; sobrescriben las actuales).")
            else:
                msg.append("🎙️ El export incluía voces clonadas de jueces pero no las restauraste.")
        try: log("[IMPORT-PLANTILLA] " + " | ".join(msg))
        except Exception: pass
        progress(1.0, desc="Listo")
        return "\n".join(msg)
    except Exception as e:
        return f"❌ No se importó nada: {e}"
    finally:
        _tp_cerrar(h)
        if stage and os.path.isdir(stage):
            shutil.rmtree(stage, ignore_errors=True)

def tp_refrescar_import():
    return gr.update(choices=_tp_escanear(), value=None)

# ───────────────────────── UI ─────────────────────────
_c0 = _tp_lista()
_s0 = _c0[0][1] if _c0 else None
with gr.Blocks(title="📦 Plantillas: exportar / importar") as demo_plantillas:
    gr.Markdown("## 📦 Plantillas: exportar / importar entre cuentas de Kaggle")
    with gr.Tab("📤 Exportar plantilla"):
        with gr.Row():
            dd_tpl = gr.Dropdown(choices=_c0, value=_s0, label="Plantilla a exportar", scale=4)
            btn_ref = gr.Button("🔄 Refrescar", scale=1)
        chk_jue = gr.Checkbox(value=True, label="Incluir voces clonadas de los jueces (voice_refs/jueces, archivos pequeños)")
        dd_dest = gr.Dropdown(choices=_TP_DESTINOS, value=_TP_DESTINOS[0],
                              label="Dónde guardar el .zip (/kaggle/tmp = más espacio, no persiste · /kaggle/working = persiste como output)")
        resumen = gr.Markdown(tp_resumen(_s0, True))
        btn_exp = gr.Button("📤 Exportar plantilla", variant="primary")
        out_exp = gr.Textbox(label="Resultado", interactive=False, lines=5)
        f_out = gr.File(label="Descargar .zip", interactive=False)
        for comp in (dd_tpl, chk_jue):
            comp.change(tp_resumen, [dd_tpl, chk_jue], [resumen])
        btn_ref.click(tp_refrescar_export, [chk_jue], [dd_tpl, resumen])
        btn_exp.click(tp_exportar, [dd_tpl, chk_jue, dd_dest], [out_exp, f_out])
    with gr.Tab("📥 Importar plantilla"):
        gr.Markdown("Indica el origen de **una** forma (si hay varias manda: subida → ruta escrita → lista).")
        with gr.Row():
            dd_org = gr.Dropdown(choices=_tp_escanear(), value=None, allow_custom_value=True, scale=4,
                                 label="Detectado en /kaggle/input y rutas habituales (.zip o carpeta)")
            btn_bus = gr.Button("🔄 Buscar de nuevo", scale=1)
        tb_man = gr.Textbox(label="…o escribe la ruta de un .zip o de una carpeta", placeholder="/kaggle/input/mi-dataset/...")
        f_sub = gr.File(label="…o sube aquí el .zip (para archivos grandes es más rápido usar un Dataset)",
                        file_types=[".zip"], type="filepath")
        rd_modo = gr.Radio(choices=_TP_MODOS, value=_TP_MODOS[0], label="Si ya existe una plantilla con el mismo nombre")
        chk_rest = gr.Checkbox(value=False,
                               label="Restaurar voces clonadas de jueces (son globales: sobrescriben las que tengas ahora)")
        with gr.Row():
            btn_ver = gr.Button("🔍 Ver qué contiene", variant="secondary")
            btn_imp = gr.Button("📥 Importar plantilla", variant="primary")
        out_imp = gr.Textbox(label="Resultado", interactive=False, lines=8)
        btn_bus.click(tp_refrescar_import, [], [dd_org])
        btn_ver.click(tp_inspeccionar, [f_sub, dd_org, tb_man], [out_imp])
        btn_imp.click(tp_importar, [f_sub, dd_org, tb_man, rd_modo, chk_rest], [out_imp])

demo_plantillas.queue().launch(share=True, prevent_thread_lock=True, show_error=True, allowed_paths=_TP_DESTINOS)