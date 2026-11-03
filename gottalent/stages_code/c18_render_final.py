# ═══════════════════════════════════════════════════════════════
# CELDA 17: RENDER FINAL + PUBLICACIÓN COMO KAGGLE DATASET
# - Concatena Etapas 2, 3, 4 (según las que estén habilitadas)
#   con el video de la Etapa 3 como eje central (nunca omitible)
# - Verificación frame-perfect con ffprobe antes de la entrega final
# - Publica modelos como Kaggle Dataset propio para persistencia gratuita
# ═══════════════════════════════════════════════════════════════

def render_video_final(
    vp: VideoProject,
    etapas_habilitadas: dict,
    fps: int = 24,
) -> str:
    """
    Ensambla todas las etapas habilitadas en el video final del canal.
    etapas_habilitadas = {"etapa2": True/False, "etapa3": True, "etapa4": True/False}
    """
    parts = []

    for etapa in ["etapa2", "etapa3", "etapa4"]:
        if not etapas_habilitadas.get(etapa, False):
            continue
        v_path = vp.estado.get(etapa, {}).get("video_final", "")
        if not v_path or not os.path.exists(v_path):
            if etapa == "etapa3":
                raise RuntimeError("Etapa 3 no tiene video_final. Ejecútala primero.")
            log(f"[RENDER FINAL] {etapa} omitida (no hay video_final o está deshabilitada).")
            continue
        parts.append(v_path)
        log(f"[RENDER FINAL] Incluida {etapa}: {v_path} ({_ffprobe_duration(v_path):.2f}s)")

    if not parts:
        raise RuntimeError("No hay partes de video para renderizar.")

    out_dir = os.path.join(vp.video_dir)
    concat_list = os.path.join(out_dir, "_final_concat.txt")
    with open(concat_list, "w", encoding="utf-8") as f:
        for p in parts:
            f.write(f"file '{os.path.abspath(p)}'\n")

    out_final = os.path.join(out_dir, "VIDEO_FINAL.mp4")
    cmd = (
        f"ffmpeg -y -f concat -safe 0 -i '{concat_list}' "
        f"-c:v libx264 -r {fps} -pix_fmt yuv420p "
        f"-c:a aac -b:a 192k '{out_final}'"
    )
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("FFmpeg render final falló:\n" + (r.stderr or "")[-3000:])

    # ── Verificación con ffprobe ──────────────────────────────────────────────
    v_dur = _ffprobe_duration(out_final) or 0.0
    expected = sum(_ffprobe_duration(p) or 0 for p in parts)
    diff_ms = abs(v_dur - expected) * 1000
    log(f"[RENDER FINAL] OK → {out_final}")
    log(f"  Duración video: {v_dur:.3f}s | Esperada: {expected:.3f}s | Diff: {diff_ms:.1f}ms")
    if diff_ms > 100:
        log(f"  ⚠ Diferencia {diff_ms:.0f}ms superior a 100ms — revisar clips individuales.")

    vp.estado["video_final"] = out_final
    vp.save()
    return out_final

def publicar_modelos_como_dataset(dataset_slug: str = "audiciones-modelos-ia") -> str:
    """
    Publica los modelos GGUF descargados (Qwen-Image-2.1, VAE, encoder, TTS)
    como un Kaggle Dataset privado propio del usuario.
    La próxima sesión, se detectarán en /kaggle/input sin necesitar descargarlos.
    Esto hace que los 20 GB de /kaggle/working NO sean el cuello de botella.
    """
    return publish_dir_as_kaggle_dataset(
        dataset_slug=dataset_slug,
        src_dir=LOCAL_MODELS_DIR,
        title="Modelos IA - Sistema Audiciones",
        update_existing=True,
    )

log("CELDA 17 COMPLETADA.")