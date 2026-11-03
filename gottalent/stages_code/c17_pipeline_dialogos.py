# ═══════════════════════════════════════════════════════════════
# CELDA 16 (v2) — PIPELINE DE DIÁLOGOS: guion → TTS → lipsync → ensamblaje
#
# Modos de guion (ejecutar_etapa_dialogos):
#   regenerar_guion=True   → xKiro genera un guion NUEVO (el anterior queda
#                            respaldado en estado[etapa]["guion_json_prev"]).
#   usar_guion_actual=True → NUNCA llama a xKiro; usa el guion guardado/editado.
#   ninguno                → reutiliza el guion si existe, si no lo genera.
#
# Garantías:
#   - Audio y clip se reutilizan SOLO si su firma (hablante+texto+voz) coincide:
#     si editas el guion, solo se rehacen los fragmentos que cambiaron.
#   - Todo el TTS se hace primero y Qwen se descarga antes del lipsync (VRAM).
#   - NO se escribe nada en vp.timeline_items (es el timeline de la canción,
#     Etapa 3). El timeline de diálogos va en estado[etapa]["timeline_dialogo"].
# ═══════════════════════════════════════════════════════════════
import hashlib

def _fnum(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

def _norm_hablante(h) -> str:
    h = str(h or "concursante").strip().lower().replace(" ", "_").replace("-", "_")
    h = re.sub(r"[^a-z0-9_]", "", h)
    m = re.match(r"^juez_?(\d+)$", h)
    if m:
        return f"juez_{int(m.group(1))}"
    if h == "concursante":
        return h
    log(f"[Diálogos] ⚠ Hablante desconocido '{h}' → se trata como 'concursante'.")
    return "concursante"

def _read_txt(p: str) -> str:
    try:
        return open(p, "r", encoding="utf-8").read().strip()
    except Exception:
        return ""

def _write_txt(p: str, txt: str) -> None:
    with open(p, "w", encoding="utf-8") as f:
        f.write(txt)

def _rm(*paths) -> None:
    for p in paths:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

def _voice_sig(hablante: str, plantilla_slug: str) -> str:
    try:
        if hablante == "concursante":
            p = CONTESTANT_VOICE_STATE.get("ref_wav_path", "")
            t = CONTESTANT_VOICE_STATE.get("ref_text", "")
        else:
            jid = int(hablante.split("_")[1])
            p = get_juez_ref_wav(plantilla_slug, jid)
            t = get_juez_ref_text(plantilla_slug, jid)
        mt = os.path.getmtime(p) if p and os.path.exists(p) else 0
        return f"{p}|{mt:.0f}|{t}"
    except Exception:
        return ""

def _frag_sig(hablante: str, texto: str, fast_mode: bool, plantilla_slug: str) -> str:
    raw = f"{hablante}|{texto}|{int(bool(fast_mode))}|{_voice_sig(hablante, plantilla_slug)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

def _frag_paths(vp: VideoProject, etapa: str, idx: int, hablante: str) -> dict:
    d = os.path.join(vp.video_dir, f"{etapa}_dialogos", f"frag_{idx:03d}_{hablante}")
    os.makedirs(d, exist_ok=True)
    return {"dir": d,
            "wav": os.path.join(d, "audio.wav"), "wav_sig": os.path.join(d, "audio.sig"),
            "mp4": os.path.join(d, "clip.mp4"),  "mp4_sig": os.path.join(d, "clip.sig"),
            # v7: preview solo-imágenes y marca de "lipsync falló → clip estático"
            "prev": os.path.join(d, "preview.mp4"), "prev_sig": os.path.join(d, "preview.sig"),
            "fb": os.path.join(d, "clip.fallback")}

def _imagen_hablante_segura(vp: VideoProject, plantilla_slug: str, hablante: str, idx: int) -> str:
    img_dir = os.path.join(vp.video_dir, "etapa1_imagenes")
    cerca = os.path.join(img_dir, "concursante_plano_cercano.png")
    close = os.path.join(img_dir, "concursante_closeup.png")
    cand = []
    if hablante == "concursante":
        cand = [close if idx % 3 == 0 else cerca, cerca, close]
    else:
        try:
            tpl = cargar_plantilla(plantilla_slug)
            cand.append(os.path.join(tpl["_base_dir"], PLANTILLA_SUBDIRS["jueces_imagenes"],
                                     f"juez_{hablante.split('_')[1]}.png"))
        except Exception:
            pass
        cand += [cerca, close]
    for p in cand:
        if p and os.path.exists(p):
            return p
    return ""

def _clip_estatico_dialogo(img_path: str, wav: str, out_mp4: str, dur_s: float, fps: int) -> bool:
    # Fallback si el lipsync falla: imagen fija + audio, MISMA resolución que LTX.
    try:
        _rm(out_mp4)
        cmd = ["ffmpeg", "-y", "-loop", "1", "-framerate", str(fps), "-i", img_path, "-i", wav,
               "-t", f"{dur_s:.3f}",
               "-vf", (f"scale={DIALOGO_W}:{DIALOGO_H}:force_original_aspect_ratio=increase,"
                       f"crop={DIALOGO_W}:{DIALOGO_H},fps={fps},format=yuv420p"),
               "-c:v", "libx264", "-c:a", "aac", "-b:a", "192k", "-shortest",
               out_mp4, "-loglevel", "error"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            log(f"[StaticClip] ffmpeg: {(r.stderr or '')[-400:]}")
        return r.returncode == 0 and os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 1000
    except Exception as e:
        log(f"[StaticClip] Error: {e}")
        return False

def _prompt_lipsync_hablante(personaje_key: str, hablante: str) -> str:
    if hablante == "concursante":
        return render_prompt_lipsync_dialogo(personaje_key)
    return ("La persona hablando, sincronización labial natural y muy realista, no modificar los "
            "rasgos y la apariencia de su rostro. Nunca mira a la cámara y mantiene la mirada al "
            "frente siempre, nunca girar la cabeza ni cambia la mirada.")

def generar_guion_con_respaldo(vp: VideoProject, personaje_key: str, etapa: str,
                               historia_base: str = "",
                               model_id: str = DEFAULT_XKIRO_TEXT_MODEL_ID) -> dict:
    # Genera un guion NUEVO con xKiro. Respalda el anterior y, si el nuevo
    # viene vacío/mal formado, RESTAURA el anterior (nunca pierdes un guion bueno).
    prev = (vp.estado.get(etapa, {}) or {}).get("guion_json")
    try:
        data = generar_dialogos_json(vp, personaje_key, etapa,
                                     historia_base=historia_base, model_id=model_id)
        ok = isinstance(data, dict) and bool(data.get("fragmentos"))
    except Exception:
        if prev:
            vp.estado.setdefault(etapa, {})["guion_json"] = prev
            vp.save()
        raise
    if not ok:
        if prev:
            vp.estado.setdefault(etapa, {})["guion_json"] = prev
            vp.save()
        raise RuntimeError(f"xKiro devolvió un guion vacío o mal formado para {etapa}. "
                           f"{'Se conservó el guion anterior.' if prev else ''}")
    if prev:
        vp.estado[etapa]["guion_json_prev"] = prev
        vp.save()
    return data

def _dialogo_audio_fragmento(vp, etapa, idx, hablante, texto, dur_est,
                             plantilla_slug, fast_mode_tts, force=False):
    fp = _frag_paths(vp, etapa, idx, hablante)
    sig = _frag_sig(hablante, texto, fast_mode_tts, plantilla_slug)
    if (not force and os.path.exists(fp["wav"]) and os.path.getsize(fp["wav"]) > 8000
            and _read_txt(fp["wav_sig"]) == sig):
        log(f"  [{idx:03d}] audio vigente (sin cambios) → reutilizado.")
        return fp["wav"], sig
    _rm(fp["wav"], fp["wav_sig"], fp["mp4"], fp["mp4_sig"],
        fp["prev"], fp["prev_sig"], fp["fb"])                    # invalida audio, clip y preview
    wav_tts = tts_fragmento(vp=vp, etapa=etapa, idx=idx, hablante=hablante, texto=texto,
                            dur_target_s=dur_est, plantilla_slug=plantilla_slug,
                            fast_mode=fast_mode_tts)
    shutil.copyfile(wav_tts, fp["wav"])
    _write_txt(fp["wav_sig"], sig)
    return fp["wav"], sig

def _dialogo_clip_fragmento(vp, plantilla_slug, personaje_key, etapa, idx, hablante,
                            ltx_paths, fps, sig, permitir_respaldo=True) -> str:
    # v7: un clip estático de RESPALDO ya no se da por "vigente": lleva la marca clip.fallback
    #     y NO guarda clip.sig, así que la próxima ejecución reintenta el lipsync real.
    #     permitir_respaldo=False (regeneración manual): si LTX falla devuelve "" y no crea nada.
    fp = _frag_paths(vp, etapa, idx, hablante)
    if (os.path.exists(fp["mp4"]) and os.path.getsize(fp["mp4"]) > 10000
            and not os.path.exists(fp["fb"])
            and _read_txt(fp["mp4_sig"]) == sig):
        log(f"  [{idx:03d}] clip vigente (sin cambios) → reutilizado.")
        return fp["mp4"]
    _rm(fp["mp4"], fp["mp4_sig"], fp["fb"])
    real_dur = min(_ffprobe_duration(fp["wav"]) or 5.0, DIALOGO_MAX_AUDIO_S)
    img = _imagen_hablante_segura(vp, plantilla_slug, hablante, idx)
    prompt = _prompt_lipsync_hablante(personaje_key, hablante)
    ok = False
    try:
        if not img:
            raise FileNotFoundError(f"Sin imagen de referencia para {hablante}")
        generar_clip_lipsync_dialogo(vp, img, fp["wav"], prompt, fp["mp4"], ltx_paths,
                                     dur_target_s=real_dur,
                                     seed=random.randint(0, 2**31 - 1), fps=fps)
        ok = os.path.exists(fp["mp4"]) and os.path.getsize(fp["mp4"]) > 10000
    except Exception as e:
        log(f"  [{idx:03d}] ❌ lipsync {hablante}: {e}"
            + (" → clip estático de respaldo." if permitir_respaldo else ""))
    if not ok:
        if not permitir_respaldo:
            _rm(fp["mp4"])
            return ""
        if not img or not _clip_estatico_dialogo(img, fp["wav"], fp["mp4"], real_dur, fps):
            log(f"  [{idx:03d}] ❌ no se pudo crear clip para {hablante}.")
            return ""
        _write_txt(fp["fb"], "estatico")      # marca: NO es lipsync real → se reintentará
        return fp["mp4"]
    _write_txt(fp["mp4_sig"], sig)
    return fp["mp4"]

def _items_validos_guion(guion: dict):
    out = []
    for idx, frag in enumerate((guion or {}).get("fragmentos", []) or []):
        texto = str(frag.get("texto", "")).strip()
        if not texto:
            continue
        out.append({"idx": idx, "hablante": _norm_hablante(frag.get("hablante")),
                    "texto": texto, "emocion": frag.get("emocion", "neutral"),
                    "dur_est": _fnum(frag.get("duracion_estimada_s"), 5.0)})
    return out

def _ensamblar_etapa_dialogos(vp: VideoProject, etapa: str, clips: list, fps: int = 24) -> str:
    etapa_dir = os.path.join(vp.video_dir, f"{etapa}_dialogos")
    concat_list = os.path.join(etapa_dir, "concat_list.txt")
    with open(concat_list, "w", encoding="utf-8") as f:
        for c in clips:
            f.write(f"file '{os.path.abspath(c)}'\n")
    out_final = os.path.join(etapa_dir, f"{etapa}_final.mp4")
    cmd = (f"ffmpeg -y -f concat -safe 0 -i '{concat_list}' "
           f"-c:v libx264 -r {fps} -pix_fmt yuv420p -c:a aac -b:a 192k '{out_final}' -loglevel error")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("FFmpeg concat falló:\n" + (r.stderr or "")[-3000:])
    return out_final

def _registrar_etapa_dialogos(vp, etapa, items, clips_por_idx, out_final):
    # Timeline PROPIO de diálogos (no toca vp.timeline_items de la canción).
    t, tl = 0.0, []
    for it in items:
        c = clips_por_idx.get(it["idx"])
        if not c:
            continue
        d = _ffprobe_duration(c) or it["dur_est"]
        tl.append({"clip_id": f"{etapa}_frag_{it['idx']:03d}_{it['hablante']}", "tipo": "dialogo",
                   "hablante": it["hablante"], "start": round(t, 3), "end": round(t + d, 3),
                   "dur_s": round(d, 3), "path_video": c,
                   "estado": ("completado_fallback"
                              if os.path.exists(_frag_paths(vp, etapa, it["idx"], it["hablante"])["fb"])
                              else "completado"),
                   "meta": {"emocion": it["emocion"], "texto": it["texto"][:80]}})
        t += d
    st = vp.estado.setdefault(etapa, {})
    st["video_final"] = out_final
    st["n_fragmentos"] = len(items)
    st["audio_paths"] = [_frag_paths(vp, etapa, it["idx"], it["hablante"])["wav"] for it in items]
    st["timeline_dialogo"] = tl
    vp.save()

def ejecutar_etapa_dialogos(
    vp: VideoProject,
    plantilla_slug: str,
    personaje_key: str,
    etapa: str,
    ltx_paths: dict,
    historia_base: str = "",
    model_id: str = DEFAULT_XKIRO_TEXT_MODEL_ID,
    fast_mode_tts: bool = False,
    voz_concursante_label: str = "Mujer — Cálida y maternal",   # compat (no usado)
    fps: int = 24,
    seed: int = -1,                                              # compat (no usado)
    regenerar_guion: bool = False,
    usar_guion_actual: bool = False,
    solo_preview: bool = False,      # v7: True → TTS + video SOLO con imágenes (sin LTX ni lipsync)
) -> str:
    assert etapa in ("etapa2", "etapa4"), "etapa debe ser 'etapa2' o 'etapa4'"
    os.makedirs(os.path.join(vp.video_dir, f"{etapa}_dialogos"), exist_ok=True)
    log(f"\n[{etapa.upper()}] Iniciando pipeline de diálogos...")

    # PASO 0 — falla rápido y claro si la voz no está validada
    ref_info = get_contestant_ref_for_tts()
    log(f"[{etapa.upper()}] Voz concursante OK: {os.path.basename(ref_info['ref_wav'])}")

    # PASO 1 — guion
    guion = (vp.estado.get(etapa, {}) or {}).get("guion_json")
    tiene = bool(guion and guion.get("fragmentos"))
    if usar_guion_actual:
        if not tiene:
            raise RuntimeError(f"No hay guion guardado para {etapa}. Pulsa '📝 Generar solo guion' "
                               f"(o 'Ejecutar completa') primero.")
        log(f"[{etapa.upper()}] Usando el guion ACTUAL ({len(guion['fragmentos'])} frags). xKiro NO se llama.")
    elif regenerar_guion or not tiene:
        log(f"[{etapa.upper()}] Generando guion NUEVO con xKiro...")
        guion = generar_guion_con_respaldo(vp, personaje_key, etapa,
                                           historia_base=historia_base, model_id=model_id)
    else:
        log(f"[{etapa.upper()}] Reutilizando guion existente ({len(guion['fragmentos'])} frags).")

    items = _items_validos_guion(guion)
    if not items:
        raise RuntimeError(f"El guion de {etapa} no tiene fragmentos con texto.")
    log(f"[{etapa.upper()}] {len(items)} fragmentos | "
        f"dur. estimada: {_fnum(guion.get('duracion_total_estimada_min')):.1f} min")

    # PASO 1b — transcripción de jueces (no fatal)
    try:
        auto_transcribe_all_judges(plantilla_slug)
    except Exception as e:
        log(f"[{etapa.upper()}] ⚠ Transcripción de jueces omitida: {e}")

    # PASO 2 — TODO el audio primero (Qwen)
    log(f"[{etapa.upper()}] PASO 2: audio de {len(items)} fragmentos...")
    sigs = {}
    for k, it in enumerate(items):
        log(f"[{etapa.upper()}] audio [{k+1}/{len(items)}] {it['hablante']} | «{it['texto'][:70]}»")
        _wav, sigs[it["idx"]] = _dialogo_audio_fragmento(
            vp, etapa, it["idx"], it["hablante"], it["texto"], it["dur_est"],
            plantilla_slug, fast_mode_tts)
    try:                       # liberar VRAM de Qwen antes de LTX
        qwen3_tts.unload()
    except Exception:
        pass

    # v7 — MODO PREVIEW: timeline solo con imágenes + audio para validar el guion hablado.
    #      No usa LTX. Para pasar al lipsync después: «Continuar con lipsync» (reutiliza los audios).
    if solo_preview:
        return _preview_etapa_dialogos(vp, plantilla_slug, etapa, items, sigs, fps)

    # PASO 3 — lipsync por fragmento (LTX)
    log(f"[{etapa.upper()}] PASO 3: lipsync de {len(items)} fragmentos...")
    clips_por_idx = {}
    for k, it in enumerate(items):
        log(f"[{etapa.upper()}] video [{k+1}/{len(items)}] {it['hablante']}")
        c = _dialogo_clip_fragmento(vp, plantilla_slug, personaje_key, etapa,
                                    it["idx"], it["hablante"], ltx_paths, fps, sigs[it["idx"]])
        if c:
            clips_por_idx[it["idx"]] = c
    if not clips_por_idx:
        raise RuntimeError(f"No se generaron clips para {etapa}.")
    if len(clips_por_idx) < len(items):
        log(f"[{etapa.upper()}] ⚠ {len(items)-len(clips_por_idx)} fragmentos sin clip (se omiten).")

    n_fb = sum(1 for it in items
               if os.path.exists(_frag_paths(vp, etapa, it["idx"], it["hablante"])["fb"]))
    if n_fb:
        log(f"[{etapa.upper()}] ⚠ {n_fb} fragmento(s) quedaron con clip ESTÁTICO porque el lipsync LTX "
            f"falló (revisa el log). Usa «🔄 Regenerar lipsync» en el timeline o vuelve a ejecutar la etapa.")

    # PASO 4 — ensamblaje
    clips = [clips_por_idx[it["idx"]] for it in items if it["idx"] in clips_por_idx]
    out_final = _ensamblar_etapa_dialogos(vp, etapa, clips, fps)
    _registrar_etapa_dialogos(vp, etapa, items, clips_por_idx, out_final)
    log(f"[{etapa.upper()}] ✅ COMPLETADA → {out_final} ({_ffprobe_duration(out_final) or 0:.2f}s)")
    return out_final

def regenerar_fragmento_dialogo(
    vp: VideoProject,
    plantilla_slug: str,
    personaje_key: str,
    etapa: str,
    frag_idx: int,
    nuevo_texto: str = "",
    ltx_paths: dict = None,
    voz_concursante_label: str = "Mujer — Cálida y maternal",   # compat (no usado)
    fps: int = 24,
    fast_mode_tts: bool = False,
) -> str:
    # Regenera UN fragmento (audio + lipsync). nuevo_texto vacío = texto actual.
    try:
        guion = (vp.estado.get(etapa, {}) or {}).get("guion_json") or {}
        frags = guion.get("fragmentos", []) or []
        if not frags:
            return f"❌ No hay guion guardado para {etapa}."
        frag_idx = int(frag_idx)
        if not (0 <= frag_idx < len(frags)):
            return f"❌ Índice {frag_idx} fuera de rango (0..{len(frags)-1})."
        if not ltx_paths:
            return "❌ Descarga LTX primero."
        get_contestant_ref_for_tts()

        frag = frags[frag_idx]
        if nuevo_texto and nuevo_texto.strip():
            frag["texto"] = nuevo_texto.strip()
            frag["duracion_estimada_s"] = max(3.0, len(frag["texto"].split()) / 2.5)
            vp.estado[etapa]["guion_json"] = guion
            vp.save()
        texto = str(frag.get("texto", "")).strip()
        if not texto:
            return "❌ El fragmento no tiene texto."
        hablante = _norm_hablante(frag.get("hablante"))
        log(f"[Regen] {etapa} frag {frag_idx} ({hablante}): «{texto[:60]}»")

        _wav, sig = _dialogo_audio_fragmento(vp, etapa, frag_idx, hablante, texto,
                                             _fnum(frag.get("duracion_estimada_s"), 5.0),
                                             plantilla_slug, fast_mode_tts, force=True)
        try:
            qwen3_tts.unload()
        except Exception:
            pass
        clip = _dialogo_clip_fragmento(vp, plantilla_slug, personaje_key, etapa,
                                       frag_idx, hablante, ltx_paths, fps, sig)
        if not clip:
            return f"❌ No se pudo generar el clip del fragmento {frag_idx}."

        # Reensamblar el video de la etapa si todos los clips existen
        items = _items_validos_guion(guion)
        clips_por_idx = {}
        for it in items:
            p = _frag_paths(vp, etapa, it["idx"], it["hablante"])["mp4"]
            if os.path.exists(p) and os.path.getsize(p) > 10000:
                clips_por_idx[it["idx"]] = p
        extra = ""
        if len(clips_por_idx) == len(items):
            out_final = _ensamblar_etapa_dialogos(
                vp, etapa, [clips_por_idx[it["idx"]] for it in items], fps)
            _registrar_etapa_dialogos(vp, etapa, items, clips_por_idx, out_final)
            extra = f"\n🎞️ Video de {etapa} reensamblado."
        else:
            extra = (f"\nℹ️ Faltan {len(items)-len(clips_por_idx)} clips: "
                     f"el video de {etapa} se ensambla al ejecutar la etapa.")
        return (f"✅ Fragmento {frag_idx} regenerado ({hablante}, "
                f"{_ffprobe_duration(_frag_paths(vp, etapa, frag_idx, hablante)['wav']) or 0:.1f}s).{extra}")
    except Exception as e:
        import traceback
        log(f"[Regen] ❌ {e}\n{traceback.format_exc()[-1200:]}")
        return f"❌ Error regenerando fragmento: {e}"


# ═══════════════════════════════════════════════════════════════
# v7 — PREVIEW SOLO IMÁGENES · REGENERAR CLIP LIPSYNC INDIVIDUAL · ESTADO PARA EL TIMELINE
# ═══════════════════════════════════════════════════════════════
def _dialogo_preview_fragmento(vp, plantilla_slug, etapa, idx, hablante, sig, fps) -> str:
    # Clip de PREVIEW de un fragmento: imagen fija del hablante + su audio (misma duración que tendrá el lipsync).
    fp = _frag_paths(vp, etapa, idx, hablante)
    img = _imagen_hablante_segura(vp, plantilla_slug, hablante, idx)
    if not img:
        log(f"  [{idx:03d}] ❌ sin imagen para el preview de {hablante}.")
        return ""
    try:
        img_mt = os.path.getmtime(img)
    except Exception:
        img_mt = 0
    sig_p = f"{sig}|{img}|{img_mt:.0f}"
    if (os.path.exists(fp["prev"]) and os.path.getsize(fp["prev"]) > 1000
            and _read_txt(fp["prev_sig"]) == sig_p):
        return fp["prev"]
    _rm(fp["prev"], fp["prev_sig"])
    real_dur = min(_ffprobe_duration(fp["wav"]) or 5.0, DIALOGO_MAX_AUDIO_S)
    if not _clip_estatico_dialogo(img, fp["wav"], fp["prev"], real_dur, fps):
        log(f"  [{idx:03d}] ❌ no se pudo crear el preview de {hablante}.")
        return ""
    _write_txt(fp["prev_sig"], sig_p)
    return fp["prev"]

def _concat_dialogo_clips(clips: list, concat_list: str, out_mp4: str, fps: int = 24) -> str:
    with open(concat_list, "w", encoding="utf-8") as f:
        for c in clips:
            f.write(f"file '{os.path.abspath(c)}'\n")
    cmd = (f"ffmpeg -y -f concat -safe 0 -i '{concat_list}' "
           f"-c:v libx264 -r {fps} -pix_fmt yuv420p -c:a aac -b:a 192k '{out_mp4}' -loglevel error")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("FFmpeg concat (preview) falló:\n" + (r.stderr or "")[-3000:])
    return out_mp4

def _preview_etapa_dialogos(vp, plantilla_slug, etapa, items, sigs, fps=24) -> str:
    # Ensambla {etapa}_preview.mp4. NO toca estado[etapa]["video_final"] (que es el video con lipsync
    # que usa el render final): el preview es solo para validar el guion en audio.
    log(f"[{etapa.upper()}] PREVIEW: {len(items)} fragmentos con imagen fija + audio (sin LTX)...")
    clips = []
    for k, it in enumerate(items):
        c = _dialogo_preview_fragmento(vp, plantilla_slug, etapa, it["idx"], it["hablante"],
                                       sigs[it["idx"]], fps)
        if c:
            clips.append(c)
    if not clips:
        raise RuntimeError(f"No se pudo crear ningún clip de preview para {etapa}.")
    etapa_dir = os.path.join(vp.video_dir, f"{etapa}_dialogos")
    out = os.path.join(etapa_dir, f"{etapa}_preview.mp4")
    _concat_dialogo_clips(clips, os.path.join(etapa_dir, "concat_preview.txt"), out, fps)
    st = vp.estado.setdefault(etapa, {})
    st["video_preview"] = out
    st["n_fragmentos"] = len(items)
    vp.save()
    log(f"[{etapa.upper()}] ✅ PREVIEW listo → {out} ({_ffprobe_duration(out) or 0:.2f}s). "
        f"Si el guion suena bien: «🎬 Continuar con lipsync».")
    return out

def _reensamblar_si_completo(vp, etapa, guion, fps=24) -> str:
    items = _items_validos_guion(guion)
    clips_por_idx = {}
    for it in items:
        p = _frag_paths(vp, etapa, it["idx"], it["hablante"])["mp4"]
        if os.path.exists(p) and os.path.getsize(p) > 10000:
            clips_por_idx[it["idx"]] = p
    if items and len(clips_por_idx) == len(items):
        out_final = _ensamblar_etapa_dialogos(
            vp, etapa, [clips_por_idx[it["idx"]] for it in items], fps)
        _registrar_etapa_dialogos(vp, etapa, items, clips_por_idx, out_final)
        return f"🎞️ Video de {etapa} reensamblado."
    return (f"ℹ️ Faltan {len(items) - len(clips_por_idx)} clips lipsync: el video de {etapa} "
            f"se ensambla al completar la etapa.")

def regenerar_clip_lipsync_fragmento(vp, plantilla_slug, personaje_key, etapa, frag_idx,
                                     ltx_paths, fps=24, fast_mode_tts=False) -> str:
    # Regenera SOLO el clip lipsync (LTX) de UN fragmento, con seed nueva y REUTILIZANDO su audio
    # (el audio solo se rehace si falta o si el texto del guion cambió).
    # Si LTX falla se conserva el clip anterior.
    try:
        guion = (vp.estado.get(etapa, {}) or {}).get("guion_json") or {}
        frags = guion.get("fragmentos", []) or []
        if not frags:
            return f"❌ No hay guion guardado para {etapa}."
        frag_idx = int(frag_idx)
        if not (0 <= frag_idx < len(frags)):
            return f"❌ Índice {frag_idx} fuera de rango (0..{len(frags)-1})."
        if not ltx_paths:
            return "❌ Descarga LTX primero."
        frag = frags[frag_idx]
        texto = str(frag.get("texto", "")).strip()
        if not texto:
            return "❌ El fragmento no tiene texto."
        hablante = _norm_hablante(frag.get("hablante"))
        fp = _frag_paths(vp, etapa, frag_idx, hablante)

        # ── audio: reutilizar si es vigente; si falta o el texto cambió → rehacerlo ──
        slug = vp.estado.get("plantilla_slug", plantilla_slug)
        wav_ok = os.path.exists(fp["wav"]) and os.path.getsize(fp["wav"]) > 8000
        sig_wav = _read_txt(fp["wav_sig"])
        vigentes = {_frag_sig(hablante, texto, False, slug), _frag_sig(hablante, texto, True, slug)}
        if wav_ok and sig_wav in vigentes:
            sig = sig_wav
        else:
            log(f"[Regen lipsync] {etapa} frag {frag_idx}: audio ausente/desactualizado → se rehace.")
            get_contestant_ref_for_tts()
            _w, sig = _dialogo_audio_fragmento(vp, etapa, frag_idx, hablante, texto,
                                               _fnum(frag.get("duracion_estimada_s"), 5.0),
                                               plantilla_slug, fast_mode_tts, force=True)
            try:
                qwen3_tts.unload()
            except Exception:
                pass

        # ── clip: se conserva el anterior hasta que el nuevo salga bien ──
        prev_mp4 = fp["mp4"] + ".prev"
        had_sig, had_fb = _read_txt(fp["mp4_sig"]), os.path.exists(fp["fb"])
        if os.path.exists(fp["mp4"]):
            shutil.copyfile(fp["mp4"], prev_mp4)
        _rm(fp["mp4"], fp["mp4_sig"], fp["fb"])
        log(f"[Regen lipsync] {etapa} frag {frag_idx} ({hablante}): nuevo clip LTX con seed nueva...")
        clip = _dialogo_clip_fragmento(vp, plantilla_slug, personaje_key, etapa, frag_idx,
                                       hablante, ltx_paths, fps, sig, permitir_respaldo=False)
        if not clip:
            if os.path.exists(prev_mp4):
                shutil.move(prev_mp4, fp["mp4"])
                if had_sig: _write_txt(fp["mp4_sig"], had_sig)
                if had_fb:  _write_txt(fp["fb"], "estatico")
            return ("❌ El lipsync LTX falló; se conservó el clip anterior. "
                    "Revisa la consola (run.log) y reintenta.")
        _rm(prev_mp4)

        extra = _reensamblar_si_completo(vp, etapa, guion, fps)
        dur = _ffprobe_duration(fp["wav"]) or 0
        return f"✅ Lipsync del fragmento {frag_idx} regenerado ({hablante}, {dur:.1f}s).\n{extra}"
    except Exception as e:
        import traceback
        log(f"[Regen lipsync] ❌ {e}\n{traceback.format_exc()[-1200:]}")
        return f"❌ Error regenerando lipsync: {e}"

def estado_fragmentos_dialogo(vp, etapa: str) -> list:
    # Estado de cada fragmento del guion para pintar el timeline de la UI.
    # estado: lipsync | fallback | preview | audio | pendiente
    guion = (vp.estado.get(etapa, {}) or {}).get("guion_json") or {}
    slug = vp.estado.get("plantilla_slug", "")
    out = []
    for it in _items_validos_guion(guion):
        fp = _frag_paths(vp, etapa, it["idx"], it["hablante"])
        wav_ok = os.path.exists(fp["wav"]) and os.path.getsize(fp["wav"]) > 8000
        sig_wav = _read_txt(fp["wav_sig"])
        vigentes = {_frag_sig(it["hablante"], it["texto"], False, slug),
                    _frag_sig(it["hablante"], it["texto"], True, slug)}
        desact = bool(wav_ok and sig_wav and sig_wav not in vigentes)
        clip_ok = os.path.exists(fp["mp4"]) and os.path.getsize(fp["mp4"]) > 10000
        prev_ok = os.path.exists(fp["prev"]) and os.path.getsize(fp["prev"]) > 1000
        if clip_ok and os.path.exists(fp["fb"]):
            estado = "fallback"
        elif clip_ok:
            estado = "lipsync"
        elif prev_ok:
            estado = "preview"
        elif wav_ok:
            estado = "audio"
        else:
            estado = "pendiente"
        out.append({
            "idx": it["idx"], "hablante": it["hablante"], "texto": it["texto"],
            "emocion": it["emocion"], "estado": estado, "desactualizado": desact,
            "dur": (_ffprobe_duration(fp["wav"]) if wav_ok else None),
            "audio": fp["wav"] if wav_ok else "",
            "preview": fp["prev"] if prev_ok else "",
            "clip": fp["mp4"] if clip_ok else "",
        })
    return out


log("CELDA 16 COMPLETADA.")