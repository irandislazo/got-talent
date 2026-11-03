# ═══════════════════════════════════════════════════════════════
# CELDA 8 (v2): ETAPAS 2 y 4 — DIÁLOGOS (xKiro) + AUDIO CLONADO
#
# CAMBIOS vs v1:
#   - TTS → qwen3_tts.synth_clone() para TODO el guion (concursante + jueces)
#   - Auto-transcripción de jueces integrada al inicio de cada etapa
#   - Validación de voz del concursante ANTES de procesar
#   - fit_audio_exact_duration aplicado a cada fragmento
#   - Generación de video lipsync por fragmento (sin cambios)
#   - regenerar_fragmento_dialogo corregido y extendido
#   - publicar_modelos_como_dataset conservado
# ═══════════════════════════════════════════════════════════════
import asyncio
import shlex

from openai import OpenAI

# ── xKiro client (OpenAI compatible) ─────────────────────────

def xkiro_client():
    api_key = os.environ.get("XKIRO_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Falta XKIRO_API_KEY en variables de entorno. "
            "Ponla en Kaggle Secrets → XKIRO_API_KEY."
        )
    return OpenAI(api_key=api_key, base_url=XKIRO_BASE_URL)


def call_xkiro_json(
    system_prompt: str,
    user_prompt: str,
    model_id: str = DEFAULT_XKIRO_TEXT_MODEL_ID,
    fallback_pool: List[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2500,
) -> dict:
    """
    Llama a xKiro y retorna un dict JSON.
    Prueba model_id + fallback_pool ante errores.
    """
    fallback_pool = fallback_pool or XKIRO_FALLBACK_POOL
    client = xkiro_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    last_err = None
    models_to_try = [model_id] + [m for m in fallback_pool if m != model_id]

    for mid in models_to_try:
        for attempt in range(1, 4):
            try:
                resp = client.chat.completions.create(
                    model=mid,
                    messages=messages,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                )
                txt = (resp.choices[0].message.content or "").strip()
                # Extraer JSON aunque venga con markdown ```json ... ```
                txt_clean = re.sub(r"^```(?:json)?", "", txt).rstrip("`").strip()
                data = json.loads(txt_clean)
                log(f"[xKiro] ✅ {mid} (intento {attempt})")
                return data
            except Exception as e:
                last_err = e
                log(f"[xKiro] ⚠ {mid} intento {attempt}: {e}")
                time.sleep(2 * attempt)

    raise RuntimeError(
        f"xKiro falló en todos los modelos. Último error: {last_err}"
    )


# ═══════════════════════════════════════════════════════════════
# SECCIÓN 8.1 — GENERACIÓN DE GUION JSON (Etapas 2 y 4)
# ═══════════════════════════════════════════════════════════════

def generar_dialogos_json(
    vp: VideoProject,
    personaje_key: str,
    etapa: str,
    historia_base: str = "",
    model_id: str = DEFAULT_XKIRO_TEXT_MODEL_ID,
) -> dict:
    """
    Genera el guion JSON para etapa2 o etapa4 vía xKiro.
    Persiste en vp.estado y en disco.
    """
    assert etapa in ("etapa2", "etapa4"), f"Etapa inválida: {etapa}"

    if etapa == "etapa2":
        system = build_dialogo_inicial_system_prompt(
            personaje_key, historia_base=historia_base
        )
        user = (
            "Genera los diálogos iniciales según TODAS las reglas del system prompt. "
            "Devuelve SOLO JSON válido, sin texto adicional, sin markdown."
        )
    else:
        system = build_dialogo_final_system_prompt(personaje_key)
        user = (
            "Genera los diálogos finales según TODAS las reglas del system prompt. "
            "Devuelve SOLO JSON válido, sin texto adicional, sin markdown."
        )

    log(f"[xKiro] Generando guion {etapa} — modelo: {model_id}")
    data = call_xkiro_json(system, user, model_id=model_id)

    vp.estado.setdefault(etapa, {})["guion_json"] = data
    vp.save()
    n_frags = len(data.get("fragmentos", []))
    try:
        dur_total = float(data.get("duracion_total_estimada_min", 0) or 0)
    except Exception:
        dur_total = 0.0
    log(f"[xKiro] Guion {etapa}: {n_frags} fragmentos | ~{dur_total:.1f} min")
    return data


# ═══════════════════════════════════════════════════════════════
# SECCIÓN 8.2 — SÍNTESIS DE AUDIO POR FRAGMENTO
# ═══════════════════════════════════════════════════════════════

# ── Audio de diálogos: duración NATURAL (sin estirar/comprimir) ─────────────
DIALOGO_MAX_AUDIO_S = 15.0      # límite del lipsync LTX
DIALOGO_W, DIALOGO_H = 1152, 704  # = defaults de generar_clip_lipsync  768, 432        848, 480

def _fit_dialog_audio(in_wav: str, out_wav: str, max_s: float = DIALOGO_MAX_AUDIO_S) -> float:
    # Mantiene el audio TTS a su velocidad natural. Solo si supera max_s lo
    # acelera lo mínimo necesario. Seguro con in_wav == out_wav (ffmpeg no
    # puede editar in-place: se escribe a un temporal y se reemplaza).
    d = _ffprobe_duration(in_wav) or 0.0
    if d <= 0:
        raise RuntimeError(f"Duración WAV inválida: {in_wav}")
    if d <= max_s:
        if os.path.abspath(in_wav) != os.path.abspath(out_wav):
            shutil.copyfile(in_wav, out_wav)
        return d
    ratio = d / max_s
    log(f"[TTS] ⚠ Fragmento de {d:.1f}s supera {max_s:.0f}s → acelerado x{ratio:.2f} "
        f"(considera dividir ese fragmento del guion)")
    parts, r = [], ratio
    while r > 2.0:
        parts.append("atempo=2.0")
        r /= 2.0
    parts.append(f"atempo={r:.6f}")
    tmp = out_wav + ".fit.wav"
    cmd = ["ffmpeg", "-y", "-i", in_wav, "-af",
           ",".join(parts) + f",atrim=0:{max_s:.6f},asetpts=N/SR/TB",
           "-acodec", "pcm_s16le", "-ar", "24000", tmp, "-loglevel", "error"]
    rr = subprocess.run(cmd, capture_output=True, text=True)
    if rr.returncode != 0:
        raise RuntimeError(f"FFmpeg falló ajustando audio:\n{(rr.stderr or '')[-1500:]}")
    os.replace(tmp, out_wav)
    return max_s


def tts_fragmento(
    vp: VideoProject,
    etapa: str,
    idx: int,
    hablante: str,
    texto: str,
    dur_target_s: float,
    plantilla_slug: str,
    fast_mode: bool = False,
) -> str:
    """
    Genera audio para UN fragmento de diálogo:
    - Usa synth_fragment_for_dialog() → clonación siempre
    - Normaliza a 24k mono
    - Ajusta a dur_target_s exactos
    Retorna la ruta del WAV final.
    """
    assert etapa in ("etapa2", "etapa4"), f"Etapa inválida: {etapa}"

    out_dir = os.path.join(vp.video_dir, f"{etapa}_dialogos", "audio")
    os.makedirs(out_dir, exist_ok=True)

    safe_hablante = re.sub(r"[^a-z0-9_]", "", hablante)
    raw_wav = os.path.join(out_dir, f"frag_{idx:03d}_{safe_hablante}_raw.wav")
    out_wav = os.path.join(out_dir, f"frag_{idx:03d}_{safe_hablante}.wav")

    # ── Síntesis ──────────────────────────────────────────────
    synth_fragment_for_dialog(
        texto=texto,
        hablante=hablante,
        out_wav=raw_wav,
        plantilla_slug=plantilla_slug,
        fast_mode=fast_mode,
    )

    # ── Normalizar → 24k mono ─────────────────────────────────
    ffmpeg_to_wav_mono_24k(raw_wav, out_wav)

    # ── Duración natural (máx. 15s). dur_target_s es solo orientativo ──
    _fit_dialog_audio(out_wav, out_wav, DIALOGO_MAX_AUDIO_S)

    # ── Verificación ──────────────────────────────────────────
    dur_real = _ffprobe_duration(out_wav) or 0
    log(f"[TTS] Frag {idx:03d} {hablante} | target={dur_target_s:.1f}s | real={dur_real:.1f}s")

    return out_wav


# ═══════════════════════════════════════════════════════════════
# SECCIÓN 8.3 — PIPELINE COMPLETO DE DIÁLOGOS (Etapas 2 y 4)
# ═══════════════════════════════════════════════════════════════

def ejecutar_etapa_dialogos(
    vp: VideoProject,
    plantilla_slug: str,
    personaje_key: str,
    etapa: str,
    ltx_paths: dict,
    historia_base: str = "",
    model_id: str = DEFAULT_XKIRO_TEXT_MODEL_ID,
    fast_mode_tts: bool = False,      # ← AÑADIDO (era el error)
    voz_concursante_label: str = "Mujer — Cálida y maternal",
    regenerar_guion: bool = False,
) -> str:
    """
    Pipeline completo para una etapa de diálogos:
    1. Auto-transcribir jueces (si no está hecho)
    2. Validar voz del concursante
    3. Generar o reutilizar guion JSON
    4. Sintetizar audio por fragmento (clonación)
    5. Generar clips de video lipsync por fragmento
    6. Concatenar video+audio final de la etapa
    Retorna la ruta del video final de la etapa.
    """
    assert etapa in ("etapa2", "etapa4"), f"Etapa inválida: {etapa}"
    log(f"\n[{etapa.upper()}] Iniciando pipeline de diálogos...")

    # ── PASO 1: Auto-transcribir jueces ──────────────────────
    log(f"[{etapa}] Transcribiendo audios de jueces...")
    jueces_transcritos = auto_transcribe_all_judges(plantilla_slug)
    for jid, txt in jueces_transcritos.items():
        if txt:
            log(f"  {jid}: '{txt[:60]}...'")
        else:
            log(f"  {jid}: ⚠ sin audio o transcripción vacía")

    # ── PASO 2: Validar voz del concursante ──────────────────
    ref_info = get_contestant_ref_for_tts()  # Raises si no está validado
    log(f"[{etapa}] Voz concursante OK: {os.path.basename(ref_info['ref_wav'])}")

    # ── PASO 3: Generar o reutilizar guion ───────────────────
    guion_existente = vp.estado.get(etapa, {}).get("guion_json", {})
    if guion_existente and not regenerar_guion:
        log(f"[{etapa}] Reutilizando guion existente ({len(guion_existente.get('fragmentos',[]))} frags).")
        data = guion_existente
    else:
        data = generar_dialogos_json(
            vp, personaje_key, etapa,
            historia_base=historia_base,
            model_id=model_id,
        )

    fragmentos = data.get("fragmentos", [])
    if not fragmentos:
        raise RuntimeError(f"El guion de {etapa} está vacío o mal formado.")

    # ── PASO 4: Sintetizar audio por fragmento ────────────────
    audio_dir = os.path.join(vp.video_dir, f"{etapa}_dialogos", "audio")
    os.makedirs(audio_dir, exist_ok=True)
    audio_paths = []

    log(f"[{etapa}] Sintetizando {len(fragmentos)} fragmentos de audio...")
    for i, frag in enumerate(fragmentos):
        hablante  = frag.get("hablante", "concursante")
        texto     = frag.get("texto", "").strip()
        dur_est   = float(frag.get("duracion_estimada_s", 5.0))

        if not texto:
            log(f"  Frag {i:03d} VACÍO — saltando.")
            continue

        out_wav = tts_fragmento(
            vp=vp,
            etapa=etapa,
            idx=i,
            hablante=hablante,
            texto=texto,
            dur_target_s=dur_est,
            plantilla_slug=plantilla_slug,
            fast_mode=fast_mode_tts,
        )
        audio_paths.append({"idx": i, "hablante": hablante, "wav": out_wav, "dur": dur_est})

    log(f"[{etapa}] ✅ {len(audio_paths)}/{len(fragmentos)} fragmentos sintetizados.")

    # ── PASO 5: Generar video lipsync por fragmento ───────────
    video_dir_etapa = os.path.join(vp.video_dir, f"{etapa}_dialogos", "video")
    os.makedirs(video_dir_etapa, exist_ok=True)

    img_dir = os.path.join(vp.video_dir, "etapa1_imagenes")
    img_concursante_c = os.path.join(img_dir, "concursante_plano_cercano.png")
    img_concursante_d = os.path.join(img_dir, "concursante_closeup.png")
    juez_imgs_dir = os.path.join(
        cargar_plantilla(plantilla_slug)["_base_dir"],
        PLANTILLA_SUBDIRS["jueces_imagenes"]
    )

    video_paths = []
    for item in audio_paths:
        i        = item["idx"]
        hablante = item["hablante"]
        wav      = item["wav"]
        dur      = item["dur"]
        frag     = fragmentos[i]
        emocion  = frag.get("emocion", "neutral")

        # Elegir imagen de referencia según hablante
        if hablante == "concursante":
            # Alternar entre plano cercano y closeup para variedad
            img_ref = img_concursante_d if i % 3 == 0 else img_concursante_c
        else:
            juez_id = hablante.replace("juez_", "")
            img_ref = os.path.join(juez_imgs_dir, f"juez_{juez_id}.png")
            if not os.path.exists(img_ref):
                img_ref = img_concursante_c  # fallback

        out_clip = os.path.join(video_dir_etapa, f"frag_{i:03d}_{hablante}.mp4")

        if os.path.exists(out_clip) and os.path.getsize(out_clip) > 10000:
            log(f"  Frag {i:03d}: video ya existe, reutilizando.")
            video_paths.append(out_clip)
            continue

        try:
            if not os.path.exists(img_ref):
                raise FileNotFoundError(f"Imagen ref no encontrada: {img_ref}")

            # Prompt de lipsync según hablante
            if hablante == "concursante":
                prompt_ls = render_prompt_lipsync_dialogo(personaje_key)
            else:
                juez_num  = int(hablante.split("_")[1]) if "_" in hablante else 1
                prompt_ls = (
                    f"El juez hablando con emoción natural, "
                    f"sincronización labial realista, "
                    f"expresión {emocion}, mirada al concursante."
                )

            clip_path = generar_clip_lipsync_dialogo(
                vp=vp,
                img_ref=img_ref,
                audio_wav=wav,
                prompt=prompt_ls,
                out_mp4=out_clip,
                ltx_paths=ltx_paths,
                dur_s=dur,
            )
            video_paths.append(clip_path)
            log(f"  Frag {i:03d} {hablante}: ✅ {os.path.basename(clip_path)}")

        except Exception as e:
            log(f"  Frag {i:03d} {hablante}: ❌ {e} — generando clip imagen estática.")
            static_clip = _make_static_clip_with_audio(img_ref, wav, out_clip, dur)
            if static_clip:
                video_paths.append(static_clip)

    # ── PASO 6: Concatenar video final de la etapa ───────────
    if not video_paths:
        raise RuntimeError(f"No se generaron clips de video para {etapa}.")

    video_final = os.path.join(vp.video_dir, f"{etapa}_dialogos", f"{etapa}_final.mp4")
    _concatenar_clips(video_paths, video_final)

    vp.estado.setdefault(etapa, {})["video_final"] = video_final
    vp.estado[etapa]["audio_paths"] = [a["wav"] for a in audio_paths]
    vp.estado[etapa]["n_fragmentos"] = len(fragmentos)
    vp.save()

    dur_final = _ffprobe_duration(video_final) or 0
    log(f"[{etapa.upper()}] ✅ Video final: {video_final} ({dur_final:.1f}s)")
    return video_final


def _make_static_clip_with_audio(img_path: str, audio_wav: str, out_mp4: str, dur_s: float) -> str:
    """
    Fallback: crea un clip de video estático (imagen fija + audio) cuando el lipsync falla.
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(out_mp4)), exist_ok=True)
        cmd = (
            f"ffmpeg -y "
            f"-loop 1 -i '{img_path}' "
            f"-i '{audio_wav}' "
            f"-t {dur_s:.3f} "
            f"-vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' "
            f"-c:v libx264 -pix_fmt yuv420p "
            f"-c:a aac -shortest "
            f"'{out_mp4}' -loglevel error"
        )
        r = subprocess.run(cmd, shell=True, capture_output=True)
        if r.returncode == 0 and os.path.exists(out_mp4):
            return out_mp4
    except Exception as e:
        log(f"[StaticClip] Error: {e}")
    return ""


def _concatenar_clips(clip_paths: List[str], out_mp4: str) -> str:
    """Concatena una lista de clips MP4 en un solo archivo."""
    os.makedirs(os.path.dirname(os.path.abspath(out_mp4)), exist_ok=True)
    list_file = out_mp4 + ".concat_list.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for p in clip_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    cmd = (
        f"ffmpeg -y -f concat -safe 0 -i '{list_file}' "
        f"-c:v libx264 -pix_fmt yuv420p -c:a aac "
        f"'{out_mp4}' -loglevel error"
    )
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    try:
        os.remove(list_file)
    except Exception:
        pass
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg concat falló:\n{(r.stderr or '')[-2000:]}")
    return out_mp4


def generar_clip_lipsync_dialogo(
    vp: VideoProject,
    img_ref: str,
    audio_wav: str,
    prompt: str,
    out_mp4: str,
    ltx_paths: dict,
    dur_s: float,
    fps: int = 24,
) -> str:
    """
    Genera un clip de video con lipsync usando LTX-2.5 (Wan2GP).
    Si Wan2GP no está disponible, usa clip estático como fallback.
    """
    try:
        backend = get_ltx_backend()
        if not backend.is_available():
            raise RuntimeError("Backend LTX-2.5 no disponible.")

        result = backend.generate_lipsync(
            image_path=img_ref,
            audio_path=audio_wav,
            prompt=prompt,
            output_path=out_mp4,
            duration_s=dur_s,
            fps=fps,
            ltx_paths=ltx_paths,
        )
        return result
    except Exception as e:
        log(f"[Lipsync] ⚠ LTX falló ({e}). Usando clip estático.")
        return _make_static_clip_with_audio(img_ref, audio_wav, out_mp4, dur_s)


# ═══════════════════════════════════════════════════════════════
# SECCIÓN 8.4 — REGENERAR FRAGMENTO INDIVIDUAL
# ═══════════════════════════════════════════════════════════════

def regenerar_fragmento_dialogo(
    vp: VideoProject,
    plantilla_slug: str,
    personaje_key: str,
    etapa: str,
    frag_idx: int,
    nuevo_texto: str = "",
    ltx_paths: dict = None,
    fast_mode_tts: bool = False,
) -> str:
    """
    Regenera UN fragmento específico del guion (audio + video).
    Si nuevo_texto está vacío, usa el texto original del guion.
    Actualiza el guion JSON y persiste.
    Retorna status_msg.
    """
    guion = (vp.estado.get(etapa, {}) or {}).get("guion_json", {})
    fragmentos = guion.get("fragmentos", [])
    if not (0 <= frag_idx < len(fragmentos)):
        return f"❌ Índice {frag_idx} fuera de rango (0..{len(fragmentos)-1})."

    frag = fragmentos[frag_idx]
    hablante   = frag.get("hablante", "concursante")
    texto_orig = frag.get("texto", "")
    dur_est    = float(frag.get("duracion_estimada_s", 5.0))

    texto_final = nuevo_texto.strip() if nuevo_texto and nuevo_texto.strip() else texto_orig
    if not texto_final:
        return "❌ El fragmento no tiene texto."

    log(f"[Regen] {etapa} frag {frag_idx} ({hablante}): '{texto_final[:60]}...'")

    # Actualizar texto en guion si se proporcionó uno nuevo
    if nuevo_texto and nuevo_texto.strip():
        fragmentos[frag_idx]["texto"] = texto_final
        # Recalcular duración estimada
        dur_nueva = max(3.0, len(texto_final.split()) / 2.5)
        fragmentos[frag_idx]["duracion_estimada_s"] = dur_nueva
        dur_est = dur_nueva
        vp.estado[etapa]["guion_json"]["fragmentos"] = fragmentos
        vp.save()

    try:
        # Regenerar audio
        out_wav = tts_fragmento(
            vp=vp,
            etapa=etapa,
            idx=frag_idx,
            hablante=hablante,
            texto=texto_final,
            dur_target_s=dur_est,
            plantilla_slug=plantilla_slug,
            fast_mode=fast_mode_tts,
        )

        # Regenerar video lipsync
        img_dir = os.path.join(vp.video_dir, "etapa1_imagenes")
        img_concursante_c = os.path.join(img_dir, "concursante_plano_cercano.png")
        img_concursante_d = os.path.join(img_dir, "concursante_closeup.png")
        juez_imgs_dir = os.path.join(
            cargar_plantilla(plantilla_slug)["_base_dir"],
            PLANTILLA_SUBDIRS["jueces_imagenes"]
        )

        if hablante == "concursante":
            img_ref = img_concursante_d if frag_idx % 3 == 0 else img_concursante_c
            prompt_ls = render_prompt_lipsync_dialogo(personaje_key)
        else:
            juez_id = hablante.replace("juez_", "")
            img_ref = os.path.join(juez_imgs_dir, f"juez_{juez_id}.png")
            if not os.path.exists(img_ref):
                img_ref = img_concursante_c
            prompt_ls = (
                f"El juez hablando con emoción natural, "
                f"sincronización labial realista, expresión {frag.get('emocion','neutral')}."
            )

        video_dir_etapa = os.path.join(vp.video_dir, f"{etapa}_dialogos", "video")
        safe_hablante   = re.sub(r"[^a-z0-9_]", "", hablante)
        out_clip        = os.path.join(video_dir_etapa, f"frag_{frag_idx:03d}_{safe_hablante}.mp4")

        generar_clip_lipsync_dialogo(
            vp=vp,
            img_ref=img_ref,
            audio_wav=out_wav,
            prompt=prompt_ls,
            out_mp4=out_clip,
            ltx_paths=ltx_paths or {},
            dur_s=dur_est,
        )

        dur_real = _ffprobe_duration(out_wav) or 0
        return (f"✅ Fragmento {frag_idx} regenerado.\n"
                f"Hablante: {hablante} | Duración: {dur_real:.1f}s\n"
                f"Texto: «{texto_final[:100]}»")

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log(f"[Regen] ❌ {e}\n{tb[-1000:]}")
        return f"❌ Error regenerando fragmento {frag_idx}: {e}"


# ═══════════════════════════════════════════════════════════════
# SECCIÓN 8.5 — RENDER FINAL (Etapas 2+3+4 → video completo)
# ═══════════════════════════════════════════════════════════════

def render_video_final(
    vp: VideoProject,
    include: dict = None,
) -> str:
    """
    Concatena los videos de las etapas en el orden:
    E2 (diálogos) → E3 (canción) → E4 (diálogos finales)
    Retorna la ruta del video final.
    """
    if include is None:
        include = {"etapa2": True, "etapa3": True, "etapa4": True}

    clips = []
    if include.get("etapa2"):
        v2 = vp.estado.get("etapa2", {}).get("video_final", "")
        if v2 and os.path.exists(v2):
            clips.append(v2)
        else:
            log("[Render Final] ⚠ Etapa 2 no disponible.")

    if include.get("etapa3"):
        v3 = vp.estado.get("etapa3", {}).get("video_final", "")
        if v3 and os.path.exists(v3):
            clips.append(v3)
        else:
            log("[Render Final] ⚠ Etapa 3 no disponible.")

    if include.get("etapa4"):
        v4 = vp.estado.get("etapa4", {}).get("video_final", "")
        if v4 and os.path.exists(v4):
            clips.append(v4)
        else:
            log("[Render Final] ⚠ Etapa 4 no disponible.")

    if not clips:
        raise RuntimeError("No hay clips disponibles para el render final.")

    out_final = os.path.join(vp.video_dir, "VIDEO_FINAL.mp4")
    _concatenar_clips(clips, out_final)

    vp.estado["video_final"] = out_final
    vp.save()

    dur = _ffprobe_duration(out_final) or 0
    log(f"[Render Final] ✅ {out_final} ({dur:.1f}s, {len(clips)} etapas)")
    return out_final


# ═══════════════════════════════════════════════════════════════
# SECCIÓN 8.6 — PUBLICAR MODELOS COMO DATASET KAGGLE
# ═══════════════════════════════════════════════════════════════

def publicar_modelos_como_dataset() -> str:
    """Publica LOCAL_MODELS_DIR como Kaggle Dataset para persistencia."""
    return publish_dir_as_kaggle_dataset(
        dataset_slug="audiciones-modelos",
        src_dir=LOCAL_MODELS_DIR,
        title="Audiciones — Modelos persistentes",
        update_existing=True,
    )
# ═══════════════════════════════════════════════════════════════
# SECCIÓN 8.7 — EDITAR GUION MANUALMENTE (NUEVA FUNCIONALIDAD)
# ═══════════════════════════════════════════════════════════════
def guardar_guion_editado(vp: VideoProject, etapa: str, guion_json_str: str) -> str:
    """
    Guarda un guion JSON editado manualmente por el usuario.
    Valida estructura antes de guardar.
    Retorna mensaje de estado.
    """
    try:
        data = json.loads(guion_json_str)
        fragmentos = data.get("fragmentos", [])
        if not fragmentos:
            return "❌ El JSON no contiene 'fragmentos' o está vacío."

        # Validar estructura mínima de cada fragmento
        for i, frag in enumerate(fragmentos):
            if "hablante" not in frag or "texto" not in frag:
                return f"❌ Fragmento {i}: falta 'hablante' o 'texto'."
            if "duracion_estimada_s" not in frag:
                # Auto-calcular duración si falta
                frag["duracion_estimada_s"] = max(3.0, len(frag["texto"].split()) / 2.5)
            if "emocion" not in frag:
                frag["emocion"] = "neutral"

        # Recalcular duración total
        total_s = sum(f.get("duracion_estimada_s", 5.0) for f in fragmentos)
        data["duracion_total_estimada_min"] = round(total_s / 60.0, 2)

        # Guardar en estado del proyecto
        vp.estado.setdefault(etapa, {})["guion_json"] = data
        vp.save()

        n_frags = len(fragmentos)
        return (f"✅ Guion {etapa} guardado correctamente.\n"
                f"   {n_frags} fragmentos | ~{data['duracion_total_estimada_min']:.1f} min\n"
                f"   → Pulsa 'Ejecutar Etapa' para sintetizar con este guion.")

    except json.JSONDecodeError as e:
        return f"❌ JSON inválido: {e}"
    except Exception as e:
        return f"❌ Error guardando guion: {e}"




log("\n✅ CELDA 8 (v2) COMPLETADA")
log("   Pipeline diálogos: clonación consistente para concursante y jueces")
log("   Auto-transcripción: faster-whisper (jueces + concursante)")
log("   Fallback lipsync: clip estático con audio si LTX no disponible")