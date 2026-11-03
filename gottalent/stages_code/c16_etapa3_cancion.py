# ═══════════════════════════════════════════════════════════════
# CELDA 15 (v2 CORREGIDA): ETAPA 3 COMPLETA — CANCIÓN CON LIPSYNC
# FIX: usa generar_clip_lipsync_15s (ahora definido en Celda 14)
# FIX: ltx_paths propagado correctamente
# ═══════════════════════════════════════════════════════════════

# ── FIX post-reinicio: helpers de la Celda 9 (solo se definen si faltan) ──
import os, subprocess

if "cut_audio_segment" not in globals():
    def cut_audio_segment(in_audio: str, out_wav: str, start_s: float, dur_s: float, sr: int = 24000) -> str:
        os.makedirs(os.path.dirname(out_wav), exist_ok=True)
        cmd = f"ffmpeg -y -i '{in_audio}' -ss {start_s:.6f} -t {dur_s:.6f} -ac 1 -ar {sr} -acodec pcm_s16le '{out_wav}'"
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("FFmpeg falló:\n" + (r.stderr or "")[-2000:])
        return out_wav

if "mute_video_audio" not in globals():
    def mute_video_audio(in_video: str, out_video: str) -> str:
        os.makedirs(os.path.dirname(out_video), exist_ok=True)
        cmd = f"ffmpeg -y -i '{in_video}' -c:v copy -an '{out_video}'"
        subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return out_video

if "trim_or_pad_video_to_duration" not in globals():
    def trim_or_pad_video_to_duration(in_video: str, out_video: str, target_s: float, fps: int = 30) -> str:
        """Hace que cada fragmento dure EXACTAMENTE target_s para evitar desfaces."""
        d = _ffprobe_duration(in_video) or 0.0
        os.makedirs(os.path.dirname(out_video), exist_ok=True)
        if d <= 0:
            raise RuntimeError("Duración de video inválida.")
        if d >= target_s:
            cmd = f"ffmpeg -y -i '{in_video}' -t {target_s:.6f} -r {fps} -c:v libx264 -pix_fmt yuv420p -an '{out_video}'"
        else:
            pad = target_s - d
            cmd = (f"ffmpeg -y -i '{in_video}' "
                   f"-vf 'tpad=stop_mode=clone:stop_duration={pad:.6f}' "
                   f"-t {target_s:.6f} -r {fps} -c:v libx264 -pix_fmt yuv420p -an '{out_video}'")
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("FFmpeg falló:\n" + (r.stderr or "")[-2000:])
        return out_video



def etapa3_paso1_generar_clips_5s(
    vp: VideoProject,
    plantilla_slug: str,
    personaje_key: str,
    ltx_paths: dict,
    fps: int = 24,
    seed: int = -1,
) -> List[str]:
    """
    PASO 3.1: Genera 2 clips de 5s por cada plano lejano (E, F, G).
    """
    img_dir   = os.path.join(vp.video_dir, "etapa1_imagenes")
    out_5s_dir = os.path.join(vp.video_dir, "etapa3_cancion", "planos_5s")
    os.makedirs(out_5s_dir, exist_ok=True)

    keys_far = {
        "E": os.path.join(img_dir, "concursante_plano_lejano_1.png"),
        "F": os.path.join(img_dir, "concursante_plano_lejano_2.png"),
        "G": os.path.join(img_dir, "concursante_plano_lejano_3.png"),
    }

    generated = []
    for key, img_path in keys_far.items():
        if not os.path.exists(img_path):
            log(f"[ETAPA 3.1] Imagen {key} no existe, se omite.")
            continue
        for variant in [1, 2]:
            prompt_str, direccion = render_prompt_5s(personaje_key)
            out_mp4 = os.path.join(out_5s_dir, f"far_{key}_v{variant}.mp4")
            if os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 10000:
                log(f"[ETAPA 3.1] {os.path.basename(out_mp4)} ya existe, se omite.")
                generated.append(out_mp4)
                continue
            log(f"[ETAPA 3.1] Generando {os.path.basename(out_mp4)} | dir={direccion}")
            vp.prompts.setdefault("etapa3_5s", {})[f"{key}_v{variant}"] = {
                "prompt": prompt_str,
                "direccion": direccion,
                "imagen": key,
            }
            vp.save()
            try:
                generar_clip_5s(
                    vp, img_path, prompt_str, out_mp4, ltx_paths,
                    seed=seed, fps=fps, width=1152, height=704,
                )
                generated.append(out_mp4)
            except Exception as e:
                log(f"[ETAPA 3.1] ❌ Error generando {key}_v{variant}: {e}")

    vp.estado.setdefault("etapa3", {})["clips_5s"] = generated
    vp.save()
    log(f"[ETAPA 3.1] Clips 5s generados: {len(generated)}")
    return generated


def etapa3_regenerar_clip_5s(
    vp: VideoProject,
    clip_path: str,
    personaje_key: str,
    ltx_paths: dict,
    fps: int = 24,
    seed: int = -1,
) -> str:
    """Regenera UN clip de 5s específico."""
    img_key = None
    for k in ["E", "F", "G"]:
        if k.lower() in os.path.basename(clip_path).lower():
            img_key = k
            break
    if not img_key:
        raise RuntimeError(f"No se pudo determinar la imagen base para {clip_path}")

    idx_map = {"E": 1, "F": 2, "G": 3}
    img_dir = os.path.join(vp.video_dir, "etapa1_imagenes")
    img_path = os.path.join(img_dir, f"concursante_plano_lejano_{idx_map[img_key]}.png")
    if not os.path.exists(img_path):
        raise RuntimeError(f"Imagen base no existe: {img_path}")

    prompt_str, direccion = render_prompt_5s(personaje_key)
    if os.path.exists(clip_path):
        os.remove(clip_path)

    generar_clip_5s(vp, img_path, prompt_str, clip_path, ltx_paths, seed=seed, fps=fps)
    log(f"[ETAPA 3.1 REGEN] Clip regenerado: {clip_path}")
    return clip_path



def _sig_lipsync_cancion(start_s, dur_s, img_path, song_wav, prompt):
    """Firma de un clip lipsync de la canción: si no cambia, el clip se REUTILIZA."""
    import hashlib
    def _mt(p):
        try: return os.path.getmtime(p)
        except Exception: return 0
    raw = f"{start_s:.3f}|{dur_s:.3f}|{img_path}|{_mt(img_path):.0f}|{_mt(song_wav):.0f}|{prompt}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def etapa3_paso3_renderizar_secuencia(
    vp: VideoProject,
    plantilla_slug: str,
    personaje_key: str,
    ltx_paths: dict,
    fps: int = 24,
    forzar_lipsync: bool = False,
) -> str:
    """
    PASO 3.3: Renderiza todos los clips del timeline y ensambla frame-perfect.
    v7: los clips lipsync ya generados y vigentes se REUTILIZAN (no se rehacen en cada render);
        forzar_lipsync=True los rehace todos.
    FIX: usa generar_clip_lipsync_15s (definido en Celda 14 corregida).
    """
    song_wav = vp.estado.get("etapa3", {}).get("song_wav", "")
    if not song_wav or not os.path.exists(song_wav):
        raise RuntimeError("No hay song.wav. Sube el audio de la canción primero.")

    if not vp.timeline_items:
        raise RuntimeError("El timeline está vacío. Ejecuta PASO 3.2 primero.")

    clips_dir = os.path.join(vp.video_dir, "etapa3_cancion", "clips_renderizados")
    os.makedirs(clips_dir, exist_ok=True)

    img_dir  = os.path.join(vp.video_dir, "etapa1_imagenes")
    song_dur = _ffprobe_duration(song_wav) or 0.0

    video_parts = []
    audio_t = 0.0

    prompt_lipsync = vp.prompts.get("etapa3_lipsync", PROMPT_LIPSYNC_CANCION)

    for item in vp.timeline_items:
        cid   = item["clip_id"]
        tipo  = item["tipo"]
        dur   = float(item["dur_s"])
        src   = item.get("path_video", "")
        out   = os.path.join(clips_dir, f"{cid}.mp4")

        if tipo == "plano_lejano_5s":
            if os.path.exists(src) and src.endswith(".mp4"):
                trim_or_pad_video_to_duration(src, out, target_s=dur, fps=fps)
                muted = out + "_m.mp4"
                mute_video_audio(out, muted)
                shutil.move(muted, out)
                item["estado"] = "completado"
            else:
                log(f"[PASO 3.3] AVISO: clip 5s no encontrado: {src}. Usando imagen estática.")
                img_path = os.path.join(img_dir, "concursante_plano_lejano_1.png")
                if os.path.exists(img_path):
                    cmd = (f"ffmpeg -y -loop 1 -i '{img_path}' "
                           f"-t {dur:.6f} -r {fps} -c:v libx264 -pix_fmt yuv420p -an '{out}'")
                    subprocess.run(cmd, shell=True, capture_output=True)
                    item["estado"] = "completado_fallback"

        elif tipo == "reaccion":
            if os.path.exists(src):
                trim_or_pad_video_to_duration(src, out, target_s=dur, fps=fps)
                muted = out + "_m.mp4"
                mute_video_audio(out, muted)
                shutil.move(muted, out)
                item["estado"] = "completado"
            else:
                log(f"[PASO 3.3] Reacción no encontrada: {src}")

        elif tipo == "lipsync_15s":
            seg_wav = os.path.join(clips_dir, f"{cid}_audio.wav")
            safe_dur = min(dur, max(0.1, song_dur - audio_t))
            if safe_dur < 0.5:
                log(f"[PASO 3.3] Audio insuficiente para lipsync en {cid}. Saltando.")
                audio_t += dur
                continue

            img_C = os.path.join(img_dir, "concursante_plano_cercano.png")
            img_D = os.path.join(img_dir, "concursante_closeup.png")
            use_img = (src if src.endswith(".png") and os.path.exists(src)
                      else (img_C if random.random() < 0.5 else img_D))

            sig = _sig_lipsync_cancion(audio_t, safe_dur, use_img, song_wav, prompt_lipsync)
            if (not forzar_lipsync and item.get("estado") == "completado"
                    and item.get("sig") == sig
                    and os.path.exists(out) and os.path.getsize(out) > 10000):
                log(f"[PASO 3.3] {cid} lipsync vigente → reutilizado (no se regenera).")
            else:
                cut_audio_segment(song_wav, seg_wav, start_s=audio_t, dur_s=safe_dur)
                try:
                    # FIX: usa generar_clip_lipsync_15s (ahora definido en Celda 14)
                    generar_clip_lipsync_15s(
                        vp, use_img, seg_wav, prompt_lipsync, out,
                        ltx_paths=ltx_paths,
                        seed=random.randint(0, 2**31 - 1), fps=fps,
                    )
                    item["estado"] = "completado"
                    item["sig"] = sig
                except Exception as e:
                    log(f"[PASO 3.3] ❌ Lipsync {cid} falló: {e}. Fallback imagen estática.")
                    cmd_fb = (f"ffmpeg -y -loop 1 -i '{use_img}' -i '{seg_wav}' "
                              f"-c:v libx264 -r {fps} -pix_fmt yuv420p -c:a aac -b:a 192k "
                              f"-t {safe_dur:.6f} '{out}'")
                    subprocess.run(cmd_fb, shell=True, capture_output=True)
                    item["estado"] = "completado_fallback"
                    item.pop("sig", None)     # v7: un fallback NUNCA se reutiliza (se reintenta)

        audio_t += dur
        if os.path.exists(out):
            video_parts.append(out)

        vp.save()

    if not video_parts:
        raise RuntimeError("No se generaron partes de video para ensamblar.")

    # ── Ensamblaje final ─────────────────────────────────────────
    concat_list = os.path.join(clips_dir, "concat_list.txt")
    with open(concat_list, "w", encoding="utf-8") as f:
        for p in video_parts:
            f.write(f"file '{os.path.abspath(p)}'\n")

    out_video_noaudio = os.path.join(vp.video_dir, "etapa3_cancion", "cancion_noaudio.mp4")
    cmd_concat = (
        f"ffmpeg -y -f concat -safe 0 -i '{concat_list}' "
        f"-c:v libx264 -r {fps} -pix_fmt yuv420p -an '{out_video_noaudio}'"
    )
    r = subprocess.run(cmd_concat, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("FFmpeg concat falló:\n" + (r.stderr or "")[-3000:])

    v_dur = _ffprobe_duration(out_video_noaudio) or 0.0
    a_dur = _ffprobe_duration(song_wav) or 0.0
    log(f"[PASO 3.3] Video: {v_dur:.3f}s | Audio: {a_dur:.3f}s | diff: {abs(v_dur-a_dur)*1000:.1f}ms")

    out_final = os.path.join(vp.video_dir, "etapa3_cancion", "cancion_final.mp4")
    cmd_mix = (
        f"ffmpeg -y -i '{out_video_noaudio}' -i '{song_wav}' "
        f"-c:v copy -c:a aac -b:a 192k -shortest '{out_final}'"
    )
    subprocess.run(cmd_mix, shell=True, capture_output=True, text=True)

    vp.estado["etapa3"]["video_final"] = out_final
    vp.save()
    log(f"[ETAPA 3] ✅ COMPLETADA → {out_final}")
    return out_final



# ═══════════════════════════════════════════════════════════════
# v7 — REGENERAR UN CLIP LIPSYNC DE LA CANCIÓN + FILAS PARA EL TIMELINE DE LA UI
# ═══════════════════════════════════════════════════════════════
def _etapa3_lipsync_pendientes(vp: VideoProject) -> list:
    """clip_id de los lipsync que aún no tienen un clip LTX real vigente."""
    clips_dir = os.path.join(vp.video_dir, "etapa3_cancion", "clips_renderizados")
    pend = []
    for it in (vp.timeline_items or []):
        if it.get("tipo") != "lipsync_15s":
            continue
        out = os.path.join(clips_dir, f"{it['clip_id']}.mp4")
        if not (it.get("estado") == "completado" and it.get("sig")
                and os.path.exists(out) and os.path.getsize(out) > 10000):
            pend.append(it["clip_id"])
    return pend


def etapa3_regenerar_clip_lipsync(vp: VideoProject, plantilla_slug: str, personaje_key: str,
                                  clip_id: str, ltx_paths: dict, fps: int = 24,
                                  reensamblar: bool = True):
    """Regenera UN clip lipsync (LTX) del timeline de la canción.
    - Corta de nuevo su tramo de audio y genera con una seed nueva.
    - Si LTX falla, se conserva el clip anterior (no se sustituye por nada).
    - Reensambla la canción solo si TODOS los lipsync ya están renderizados
      (si no, evita lanzar de golpe los que faltan).
    Devuelve (ruta_clip, reensamblado: bool)."""
    song_wav = vp.estado.get("etapa3", {}).get("song_wav", "")
    if not song_wav or not os.path.exists(song_wav):
        raise RuntimeError("No hay song.wav. Sube el audio de la canción primero.")
    item = next((i for i in (vp.timeline_items or []) if i.get("clip_id") == clip_id), None)
    if item is None:
        raise RuntimeError(f"El clip {clip_id} no está en el timeline actual "
                           f"(¿reconstruiste el timeline? Pulsa 'Actualizar fragmentos').")
    if item.get("tipo") != "lipsync_15s":
        raise RuntimeError(f"{clip_id} no es un clip lipsync ({item.get('tipo')}).")

    clips_dir = os.path.join(vp.video_dir, "etapa3_cancion", "clips_renderizados")
    os.makedirs(clips_dir, exist_ok=True)
    img_dir = os.path.join(vp.video_dir, "etapa1_imagenes")
    song_dur = _ffprobe_duration(song_wav) or 0.0
    start = float(item["start"]); dur = float(item["dur_s"])
    safe_dur = min(dur, max(0.1, song_dur - start))
    if safe_dur < 0.5:
        raise RuntimeError(f"Audio insuficiente para el lipsync {clip_id}.")

    src = item.get("path_video", "") or ""
    use_img = (src if src.endswith(".png") and os.path.exists(src)
               else os.path.join(img_dir, "concursante_plano_cercano.png"))
    if not os.path.exists(use_img):
        raise RuntimeError(f"Falta la imagen base del lipsync: {use_img}")
    prompt_lipsync = vp.prompts.get("etapa3_lipsync", PROMPT_LIPSYNC_CANCION)

    seg_wav = os.path.join(clips_dir, f"{clip_id}_audio.wav")
    out = os.path.join(clips_dir, f"{clip_id}.mp4")
    backup = out + ".prev"
    cut_audio_segment(song_wav, seg_wav, start_s=start, dur_s=safe_dur)
    if os.path.exists(out):
        shutil.copyfile(out, backup)
    log(f"[ETAPA 3 REGEN] {clip_id}: lipsync {start:.1f}s→{start+safe_dur:.1f}s con seed nueva")
    try:
        generar_clip_lipsync_15s(vp, use_img, seg_wav, prompt_lipsync, out,
                                 ltx_paths=ltx_paths,
                                 seed=random.randint(0, 2**31 - 1), fps=fps)
    except Exception:
        if os.path.exists(backup):
            shutil.move(backup, out)          # conserva el clip anterior
        elif os.path.exists(out):
            try: os.remove(out)
            except Exception: pass
        raise
    if os.path.exists(backup):
        try: os.remove(backup)
        except Exception: pass

    item["estado"] = "completado"
    item["sig"] = _sig_lipsync_cancion(start, safe_dur, use_img, song_wav, prompt_lipsync)
    vp.save()

    reens = False
    if reensamblar and not _etapa3_lipsync_pendientes(vp):
        etapa3_paso3_renderizar_secuencia(vp, plantilla_slug, personaje_key, ltx_paths, fps)
        reens = True
    return out, reens


def etapa3_filas_timeline(vp: VideoProject) -> list:
    """Filas (una por clip del timeline) para pintar el timeline por fragmentos en la UI."""
    clips_dir = os.path.join(vp.video_dir, "etapa3_cancion", "clips_renderizados")
    rows = []
    for it in (vp.timeline_items or []):
        out = os.path.join(clips_dir, f"{it['clip_id']}.mp4")
        ok = os.path.exists(out) and os.path.getsize(out) > 10000
        rows.append({
            "clip_id": it["clip_id"], "tipo": it.get("tipo", ""),
            "hablante": it.get("hablante", ""),
            "start": float(it.get("start", 0)), "end": float(it.get("end", 0)),
            "dur": float(it.get("dur_s", 0)),
            "estado": it.get("estado", "pendiente"),
            "video": out if ok else "",
        })
    return rows


log("CELDA 15 (v2 CORREGIDA) COMPLETADA.")