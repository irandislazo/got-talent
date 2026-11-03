# ═══════════════════════════════════════════════════════════════
# CELDA 14 (NUEVA): CLIPS LTX-2.5 — WRAPPERS COMPATIBLES
# Mantiene MISMA interfaz que Celda 18 espera:
#   generar_clip_5s(vp, image_path, prompt, out_mp4, ltx_paths, ...)
#   generar_clip_lipsync(vp, image_path, audio_path, prompt, out_mp4, ...)
#   generar_clip_lipsync_15s(...)
#   generar_clip_lipsync_dialogo(...)
# Internamente usa Wan2GPLTXBackend en vez de ComfyUI   
# ═══════════════════════════════════════════════════════════════
import os, subprocess, shutil, random, time

def _ffprobe_duration(path):
    try:
        cmd = ["ffprobe","-v","error","-show_entries","format=duration",
               "-of","default=noprint_wrappers=1:nokey=1", path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return float((r.stdout or "").strip())
    except: return None

def _to_mp4_exact(in_path, out_path, target_s, fps=24, mute=False):
    d = _ffprobe_duration(in_path) or 0.0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    af = "-an" if mute else "-c:a aac -b:a 192k"
    if d >= target_s:
        cmd = f"ffmpeg -y -i '{in_path}' -t {target_s:.6f} -r {fps} -c:v libx264 -pix_fmt yuv420p {af} '{out_path}'"
    else:
        pad = target_s - d
        cmd = (f"ffmpeg -y -i '{in_path}' -vf 'tpad=stop_mode=clone:stop_duration={pad:.6f}' "
               f"-t {target_s:.6f} -r {fps} -c:v libx264 -pix_fmt yuv420p {af} '{out_path}'")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("FFmpeg _to_mp4_exact falló:\n" + (r.stderr or "")[-2000:])
    if in_path != out_path and os.path.exists(in_path):
        try: os.remove(in_path)
        except: pass
    return out_path

def generar_clip_5s(vp, image_path, prompt, out_mp4, ltx_paths=None,
                    seed=-1, fps=24, width=1152, height=704):
    """Clip 5s I2V usando Wan2GP LTX-2.5. Misma firma que Celda 18 espera."""
    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)
    engine = get_ltx_backend()
    if not engine.is_available():
        raise RuntimeError("Wan2GP no disponible. Verifica Celda 13.")

    num_frames = calc_num_frames(5.0, fps)
    steps = int(ltx_paths.get("_steps", 8)) if ltx_paths else 8
    cfg = float(ltx_paths.get("_cfg", 1.0)) if ltx_paths else 1.0

    log(f"[CLIP 5s] I2V | {width}x{height} | {num_frames}f@{fps}fps | steps={steps}")

    raw = engine.generate(
        prompt=prompt, image_path=image_path,
        num_frames=num_frames, width=width, height=height,
        seed=seed, steps=steps, cfg=cfg, fps=fps,
        input_video_strength=0.8, denoise_strength=1.0,
    )
    _to_mp4_exact(raw, out_mp4, target_s=5.0, fps=fps, mute=True)
    log(f"[CLIP 5s] ✅ → {out_mp4}")
    return out_mp4

def generar_clip_lipsync(vp, image_path, audio_path, prompt, out_mp4,
                         ltx_paths=None, target_s=15.0, seed=-1, fps=24,
                         width=1152, height=704, audio_strength=0.9):  # 864 x 480 
    """Clip lipsync I2V usando Wan2GP LTX-2.5. Audio se muxa post-generación."""
    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)
    engine = get_ltx_backend()
    if not engine.is_available():
        raise RuntimeError("Wan2GP no disponible. Verifica Celda 13.")

    num_frames = calc_num_frames(target_s, fps)
    steps = int(ltx_paths.get("_steps", 8)) if ltx_paths else 8
    cfg = float(ltx_paths.get("_cfg", 1.0)) if ltx_paths else 1.0

    log(f"[LIPSYNC {target_s:.1f}s] I2V | {width}x{height} | {num_frames}f | steps={steps}")

    raw = engine.generate(
    prompt=prompt,
    image_path=image_path,
    audio_path=audio_path,  # NUEVO: entregar audio al backend
    num_frames=num_frames,
    width=width,
    height=height,
    seed=seed,
    steps=steps,
    cfg=cfg,
    fps=fps,
    input_video_strength=0.8,
    denoise_strength=1.0,
)

    tmp_video = out_mp4 + ".tmp_noaudio.mp4"
    _to_mp4_exact(raw, tmp_video, target_s=target_s, fps=fps, mute=True)

    if audio_path and os.path.exists(audio_path):
        cmd_mux = (f"ffmpeg -y -i '{tmp_video}' -i '{audio_path}' "
                   f"-c:v copy -c:a aac -b:a 192k -shortest '{out_mp4}'")
        r_mux = subprocess.run(cmd_mux, shell=True, capture_output=True, text=True)
        if r_mux.returncode != 0:
            log(f"[LIPSYNC] ⚠ Mux falló: {(r_mux.stderr or '')[-500:]}")
            shutil.move(tmp_video, out_mp4)
        else:
            try: os.remove(tmp_video)
            except: pass
    else:
        shutil.move(tmp_video, out_mp4)

    if not os.path.exists(out_mp4) or os.path.getsize(out_mp4) < 10000:
        raise RuntimeError(f"No se generó vídeo lipsync válido: {out_mp4}")
    log(f"[LIPSYNC] ✅ → {out_mp4} ({target_s:.2f}s)")
    return out_mp4

# ═══ ALIAS OBLIGATORIOS — usados por Celdas 15, 16 y 18 ═══
def generar_clip_lipsync_15s(vp, image_path, audio_path, prompt, out_mp4,
                             ltx_paths=None, seed=-1, fps=24, **kwargs):
    """Alias para generar_clip_lipsync con target_s=10.0."""
    return generar_clip_lipsync(vp, image_path, audio_path, prompt, out_mp4,
                                ltx_paths=ltx_paths, target_s=10.0, seed=seed, fps=fps, **kwargs)

def generar_clip_lipsync_dialogo(vp, image_path, audio_path, prompt, out_mp4,
                                 ltx_paths=None, dur_target_s=8.0, seed=-1, fps=24, **kwargs):
    """Alias para generar_clip_lipsync con duración variable."""
    return generar_clip_lipsync(vp, image_path, audio_path, prompt, out_mp4,
                                ltx_paths=ltx_paths, target_s=dur_target_s, seed=seed, fps=fps, **kwargs)

log("CELDA 14 (NUEVA LTX-2.5 WRAPPERS) COMPLETADA.")
log("  ✅ Misma interfaz que Celda 18 espera")
log("  ✅ generar_clip_5s / generar_clip_lipsync / _15s / _dialogo definidos")
log("  ✅ Backend: Wan2GP LTX-2.5 (NO ComfyUI)")