def FAR_DUR():
    return float(globals().get("FAR_CLIP_DUR_S", 5.0))   # 2/3/5 s desde la UI (por defecto 5.0 = igual que antes)

# ═══════════════════════════════════════════════════════════════
# CELDA 9 (v2): ETAPA 3 — CANCIÓN CON DISTRIBUCIÓN RÍTMICA
# Reglas de distribución:
#   - NUNCA dos lipsync consecutivos
#   - Máximo 2 reacciones consecutivas
#   - Patrón preferido: Reacción → PlanoLejano5s → Lipsync → PlanoLejano → Reacción
#   - Intro (~12%): solo reacciones + planos lejanos (SIN lipsync)
#   - Audio de canción = pista maestra del render final
#   - Frame-perfect: cada clip ajustado a duración exacta
# ═══════════════════════════════════════════════════════════════
def cut_audio_segment(in_audio: str, out_wav: str, start_s: float, dur_s: float, sr: int = 24000) -> str:
    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    cmd = f"ffmpeg -y -i '{in_audio}' -ss {start_s:.6f} -t {dur_s:.6f} -ac 1 -ar {sr} -acodec pcm_s16le '{out_wav}'"
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("FFmpeg falló:\n" + (r.stderr or "")[-2000:])
    return out_wav

def mute_video_audio(in_video: str, out_video: str) -> str:
    os.makedirs(os.path.dirname(out_video), exist_ok=True)
    cmd = f"ffmpeg -y -i '{in_video}' -c:v copy -an '{out_video}'"
    subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return out_video

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

def build_song_timeline(vp: VideoProject, plantilla_slug: str, personaje_key: str,
                        song_audio_path: str, use_far_planes: List[str],
                        target_fps: int = 30) -> list:
    """
    Construye timeline con distribución rítmica profesional:
      - Intro (~12%): reacciones + planos lejanos ONLY (sin lipsync)
      - Main: ciclo Reacción → PlanoLejano5s → Lipsync → PlanoLejano → Reacción
      - NUNCA dos lipsync consecutivos
      - Máximo 2 reacciones consecutivas (solo cuando no queda tiempo para lipsync)
    """
    tpl = cargar_plantilla(plantilla_slug)
    base = tpl["_base_dir"]

    song_dur = _ffprobe_duration(song_audio_path) or 0.0
    if song_dur <= 5:
        raise RuntimeError("Audio de canción demasiado corto.")

    # ── Cargar assets ──────────────────────────────────────────
    react1_dir = os.path.join(base, PLANTILLA_SUBDIRS["reacciones_primera_mitad"])
    react2_dir = os.path.join(base, PLANTILLA_SUBDIRS["reacciones_segunda_mitad"])
    reacts1 = sorted([os.path.join(react1_dir, f) for f in os.listdir(react1_dir)
                      if f.lower().endswith(".mp4")]) if os.path.isdir(react1_dir) else []
    reacts2 = sorted([os.path.join(react2_dir, f) for f in os.listdir(react2_dir)
                      if f.lower().endswith(".mp4")]) if os.path.isdir(react2_dir) else []

    far5_dir = os.path.join(vp.video_dir, "etapa3_cancion", "planos_5s")
    far5 = sorted([os.path.join(far5_dir, f) for f in os.listdir(far5_dir)
                   if f.lower().endswith(".mp4")]) if os.path.isdir(far5_dir) else []
    if not far5:
        raise RuntimeError("No hay clips 5s generados. Ejecuta PASO 3.1 primero.")

    img_dir = os.path.join(vp.video_dir, "etapa1_imagenes")
    img_C = os.path.join(img_dir, "concursante_plano_cercano.png")
    img_D = os.path.join(img_dir, "concursante_closeup.png")
    if not os.path.exists(img_C) or not os.path.exists(img_D):
        raise RuntimeError("Faltan imágenes C/D para lipsync.")

    # ── Helpers internos ───────────────────────────────────────
    clips = []
    t = 0.0
    clip_idx = 0
    consecutive_reactions = 0   # contador para limitar a max 2
    last_was_lipsync = False    # flag para NUNCA poner 2 lipsync seguidos

    def add(tipo, hablante, dur, path):
        nonlocal clip_idx, clips, t, consecutive_reactions, last_was_lipsync
        clips.append(TimelineClip(
            clip_id=f"c{clip_idx:04d}",
            tipo=tipo,
            hablante=hablante,
            dur_s=float(dur),
            path_video=path,
            estado="pendiente",
            meta={}
        ))
        clip_idx += 1
        t += float(dur)
        # Actualizar contadores de ritmo
        if tipo == "reaccion":
            consecutive_reactions += 1
            last_was_lipsync = False
        elif tipo == "lipsync_15s":
            consecutive_reactions = 0
            last_was_lipsync = True
        else:  # plano_lejano_5s
            consecutive_reactions = 0
            last_was_lipsync = False

    def pick_reaction():
        """Selecciona reacción según mitad de la canción."""
        if reacts2 and (t > song_dur / 2):
            return random.choice(reacts2)
        elif reacts1:
            return random.choice(reacts1)
        return None

    def remaining():
        return song_dur - t

    # ══════════════════════════════════════════════════════════
    # FASE 1: INTRO (~12% de la canción, máx 30s)
    # Solo reacciones + planos lejanos. NINGÚN lipsync.
    # ══════════════════════════════════════════════════════════
    intro_target = min(30.0, song_dur * 0.12)
    while t < intro_target and remaining() > 1.0:
        rv = pick_reaction()
        if rv:
            d = _ffprobe_duration(rv) or 3.0
            if t + d <= intro_target + 2.0:  # tolerancia pequeña
                add("reaccion", f"juez_{random.randint(1,4)}", d, rv)
        add("plano_lejano_5s", "concursante", FAR_DUR(), random.choice(far5))

    # ══════════════════════════════════════════════════════════
    # FASE 2: MAIN LOOP — Distribución rítmica profesional
    # Patrón: Reacción → PlanoLejano5s → Lipsync → PlanoLejano → (repeat)
    # Reglas:
    #   - Nunca 2 lipsync seguidos
    #   - Max 2 reacciones seguidas
    #   - Si no cabe lipsync (15s), rellenar con lejano+reacción
    # ══════════════════════════════════════════════════════════
    while t + 1.0 < song_dur:
        rem = remaining()

        # ── Caso A: Cabe un lipsync completo (≥15s) y NO viene de otro lipsync ──
        if rem >= 10.0 and not last_was_lipsync:
            # Paso 1: Reacción (si no llevamos ya 2 seguidas)
            if consecutive_reactions < 2:
                rv = pick_reaction()
                if rv:
                    d = _ffprobe_duration(rv) or 3.0
                    if t + d + 5.0 + 10.0 <= song_dur + 2.0:  # cabe reacción+lejano+lipsync
                        add("reaccion", f"juez_{random.randint(1,4)}", d, rv)

            # Paso 2: Plano lejano 5s (separador visual antes del lipsync)
            if t + 5.0 + 10.0 <= song_dur + 2.0:
                add("plano_lejano_5s", "concursante", FAR_DUR(), random.choice(far5))

            # Paso 3: Lipsync 15s
            if remaining() >= 10.0:
                add("lipsync_15s", "concursante", 10.0,
                    img_C if random.random() < 0.5 else img_D)

            # Paso 4: Plano lejano 5s (separador visual después del lipsync)
            if remaining() >= 5.0:
                add("plano_lejano_5s", "concursante", FAR_DUR(), random.choice(far5))

        # ── Caso B: No cabe lipsync pero cabe plano lejano + reacción ──
        elif rem >= 5.0:
            add("plano_lejano_5s", "concursante", FAR_DUR(), random.choice(far5))
            # Reacción solo si no llevamos 2 seguidas
            if consecutive_reactions < 2:
                rv = pick_reaction()
                if rv:
                    d = _ffprobe_duration(rv) or 3.0
                    if t + d <= song_dur + 1.0:
                        add("reaccion", f"juez_{random.randint(1,4)}", d, rv)

        # ── Caso C: Solo queda relleno fino (<5s) ──
        else:
            # Permitir hasta 2 reacciones seguidas como cierre
            if consecutive_reactions < 2:
                rv = pick_reaction()
                if rv:
                    d = min(_ffprobe_duration(rv) or 3.0, rem)
                    if d >= 0.5:
                        add("reaccion", f"juez_{random.randint(1,4)}", d, rv)
            # Si aún sobra tiempo, un último plano lejano recortado
            if remaining() >= 1.0:
                add("plano_lejano_5s", "concursante", min(FAR_DUR(), remaining()), random.choice(far5))
            break

        # Seguridad anti-loop infinito
        if len(clips) > 9999:
            log("[TIMELINE] ⚠ Límite de seguridad alcanzado (9999 clips).")
            break

    # ── Ajuste final: recortar último clip para cuadrar con song_dur ──
    total = sum(c.dur_s for c in clips)
    if total > song_dur and clips:
        excess = total - song_dur
        clips[-1].dur_s = max(0.5, clips[-1].dur_s - excess)

    # ── Persistir ──────────────────────────────────────────────
    vp.timeline_items = timeline_to_json(clips)
    vp.save()

    # Log resumen de distribución
    tipo_counts = {}
    for c in clips:
        tipo_counts[c.tipo] = tipo_counts.get(c.tipo, 0) + 1
    log(f"[TIMELINE] Construido: {len(clips)} clips | {song_dur:.1f}s total")
    log(f"[TIMELINE] Distribución: {tipo_counts}")

    # Verificar regla: nunca 2 lipsync seguidos
    for i in range(len(clips) - 1):
        if clips[i].tipo == "lipsync_15s" and clips[i+1].tipo == "lipsync_15s":
            log(f"[TIMELINE] ⚠ VIOLACIÓN: 2 lipsync seguidos en {clips[i].clip_id}/{clips[i+1].clip_id}")


    return vp.timeline_items


def build_song_timeline_lavoz(vp: VideoProject, plantilla_slug: str, personaje_key: str,
                              song_audio_path: str, use_far_planes: List[str],
                              intro_seconds: float = 30.0, target_fps: int = 30) -> list:
    """Timeline específico La Voz. No modifica la función Got Talent.
    La primera mitad intercala lipsync del concursante de 10 s entre reacciones,
    conservando los cuatro giros de jueces y sin insertar dos lipsync consecutivos.
    v3 (distribución realista): nunca el mismo archivo ni el mismo juez dos clips seguidos,
    sin dos planos lejanos seguidos, los archivos se reutilizan solo cuando se agotan los
    subidos, y en la 1ª mitad (tras el intro) máximo 3 clips seguidos sin lipsync.
    """
    tpl = cargar_plantilla(plantilla_slug)
    base = tpl["_base_dir"]
    song_dur = _ffprobe_duration(song_audio_path) or 0.0
    if song_dur <= 5:
        raise RuntimeError("Audio de canción demasiado corto.")
    far_dir = os.path.join(vp.video_dir, "etapa3_cancion", "planos_5s")
    far = sorted(os.path.join(far_dir, f) for f in os.listdir(far_dir)
                 if f.lower().endswith(".mp4")) if os.path.isdir(far_dir) else []
    if not far:
        raise RuntimeError("No hay clips 5s generados. Ejecuta PASO 3.1 primero.")
    img_dir = os.path.join(vp.video_dir, "etapa1_imagenes")
    img_c, img_d = os.path.join(img_dir, "concursante_plano_cercano.png"), os.path.join(img_dir, "concursante_closeup.png")
    if not os.path.exists(img_c) or not os.path.exists(img_d):
        raise RuntimeError("Faltan imágenes C/D para lipsync.")

    root = os.path.join(base, PLANTILLA_SUBDIRS.get("reacciones_lavoz", "video/reacciones_lavoz"))
    def files(j, state):
        d = os.path.join(root, f"juez_{j}", state)
        return sorted(os.path.join(d, f) for f in os.listdir(d)
                      if f.lower().endswith(".mp4")) if os.path.isdir(d) else []
    before = {j: files(j, "antes") for j in range(1,5)}
    turning = {j: files(j, "girando") for j in range(1,5)}
    turned = {j: files(j, "girado") for j in range(1,5)}
    missing = [j for j in range(1,5) if not turning[j]]
    if missing:
        raise RuntimeError("Plantilla La Voz: falta al menos un clip 'girando' para " +
                           ", ".join(f"Juez {j}" for j in missing) + ". Sube un clip por juez.")

    clips, used, t = [], set(), 0.0
    usage = {}   # v3: veces que se usó cada archivo en ESTE timeline
    def add(tipo, judge, dur, path):
        nonlocal t
        real_tipo = "reaccion" if tipo == "reaccion_girando" else tipo
        meta = {"estilo": "la_voz"}
        if tipo == "reaccion_girando": meta["estado_juez"] = "girando"
        clips.append(TimelineClip(clip_id=f"c{len(clips):04d}", tipo=real_tipo,
            hablante=f"juez_{judge}" if judge else "concursante", dur_s=float(dur),
            path_video=path, estado="pendiente", meta=meta))
        t += float(dur)
    def choose(pool, no_repeat=True):
        pool = [x for x in pool if not no_repeat or x not in used]
        if not pool: pool = list(pool0) if False else []
        if not pool: return None
        x = random.choice(pool); used.add(x); return x
    def fresh(pool, recent=4):
        # v3: evita los últimos `recent` clips y prioriza los archivos menos usados.
        pool = list(dict.fromkeys(pool))
        if not pool: return None
        recent_paths = [c.path_video for c in clips[-recent:]]
        cand = ([x for x in pool if x not in recent_paths]
                or [x for x in pool if not clips or x != clips[-1].path_video]
                or pool)
        low = min(usage.get(x, 0) for x in cand)
        x = random.choice([y for y in cand if usage.get(y, 0) == low])
        usage[x] = usage.get(x, 0) + 1
        return x
    def react(j, state):
        pool = before[j] if state == "antes" else turned[j]
        pool = [x for x in pool if x not in used] or pool
        return fresh(pool)
    def dur(path, fallback=3.0):
        return max(0.5, min(_ffprobe_duration(path) or fallback, 5.0))
    def far_clip():
        # v3: sin repetir lejanos hasta agotar los disponibles (y nunca el mismo seguido).
        return fresh(far)
    def far_ok():
        return not clips or clips[-1].tipo != "plano_lejano_5s"
    def last_judge():
        if clips and str(clips[-1].hablante).startswith("juez_"):
            try: return int(str(clips[-1].hablante).split("_")[1])
            except Exception: return 0
        return 0
    def pick_k(ks, state):
        # v3: juez distinto al del clip anterior; prioriza quien aún tiene clips sin usar.
        lj = last_judge()
        cand = [k for k in ks if k != lj] or list(ks)
        if state:
            def _pool(k): return before[k] if state == "antes" else turned[k]
            nuevos = [k for k in cand if any(usage.get(x, 0) == 0 for x in _pool(k))]
            cand = nuevos or cand
        return random.choice(cand)
    def pick_judge_state(el_turned, el_before, p_turned=0.45):
        # Misma lógica de estados que antes (girado/antes), evitando repetir juez seguido.
        lj = last_judge()
        t_ok = [k for k in el_turned if k != lj]
        b_ok = [k for k in el_before if k != lj]
        if t_ok and (not b_ok or random.random() < p_turned):
            return pick_k(t_ok, "girado"), "girado"
        if b_ok:
            return pick_k(b_ok, "antes"), "antes"
        if el_turned and (not el_before or random.random() < p_turned):
            return pick_k(el_turned, "girado"), "girado"
        if el_before:
            return pick_k(el_before, "antes"), "antes"
        return None, None
    def rand_judge():
        lj = last_judge()
        return random.choice([k for k in range(1, 5) if k != lj])
    intro_n = 0   # v3: se fija al terminar el intro (la racha de clips se cuenta desde ahí)
    def run_len():
        # v3: clips seguidos sin lipsync (reacción / giro / lejano) desde el último lipsync.
        n = 0
        for c in reversed(clips[intro_n:]):
            if c.tipo == "lipsync_15s": break
            n += 1
        return n
    turn_len = {j: max([dur(p) for p in turning[j]] or [3.0]) for j in range(1, 5)}
    pending = []  # v3: jueces cuyo giro aún no se ha colocado (se reserva su tiempo)
    def can_lip(extra=0.0):
        # ¿cabe un lipsync de 10 s sin comprometer el tiempo de los giros pendientes?
        return (half - t - extra - 10.0) >= (sum(turn_len[k] for k in pending) + 0.5)
    def need_lip():
        # Racha de 3 sin lipsync, o "última oportunidad": tras otro clip ya no cabría.
        r = run_len()
        return r >= 3 or (r >= 1 and not can_lip(5.0))
    def add_lipsync(max_dur=10.0):
        # Cada fragmento conserva su posición temporal sobre el audio maestro.
        d = min(10.0, max_dur)
        if d >= 9.5 and (not clips or clips[-1].tipo != "lipsync_15s"):
            add("lipsync_15s", None, 10.0, random.choice([img_c, img_d]))
            return True
        return False

    half = song_dur / 2.0
    intro_end = min(max(0.0, float(intro_seconds)), half * 0.55)
    # Intro: solo música, sin voz del concursante.
    # Por tanto, aquí SOLO se muestran reacciones pre-giro; nunca lipsync ni planos lejanos.
    while t < intro_end - 0.25:
        remaining_intro = intro_end - t
        available = [j for j in range(1,5) if before[j]]
        if not available:
            break
        j = pick_k(available, "antes")
        path = react(j, "antes")
        if not path:
            break
        d = min(dur(path), remaining_intro)
        add("reaccion", j, d, path)
    intro_n = len(clips)   # v3

    # Reservar cuatro giros obligatorios, con separación temporal.
    order = random.sample([1,2,3,4], 4)
    pending = list(order)   # v3
    # Objetivos espaciados entre después del intro y antes del final de la primera mitad.
    start = max(t + 0.1, min(intro_end + 3.0, half * 0.22))
    targets = [max(start, half * f) for f in (0.30, 0.48, 0.66, 0.83)]
    targets = [max(targets[i], (targets[i-1] + 3.0) if i else 0) for i in range(4)]
    for idx, j in enumerate(order):
        target = min(targets[idx], half - 2.5)
        # Relleno antes del giro: solo jueces aún no girados o ya girados según estado.
        while t + 1.0 < target:   # v3: no se rellena con mini-clips (<1 s)
            remaining_to_turn = target - t
            # v3: máximo 3 clips seguidos sin lipsync (si cabe sin afectar los giros).
            if need_lip() and can_lip():
                if add_lipsync(10.0):
                    continue
            # Insertar canto sincronizado entre reacciones siempre que quepa sin
            # desplazar el giro reservado. La intro ya garantiza el patrón inicial.
            if (remaining_to_turn >= 12.0 and
                    (not clips or clips[-1].tipo != "lipsync_15s") and
                    random.random() < 0.38):
                if add_lipsync(remaining_to_turn):
                    continue
            use_far_now = (remaining_to_turn > 7 and random.random() < 0.22 and
                           (not clips or clips[-1].tipo != "plano_lejano_5s"))
            if use_far_now:
                add("plano_lejano_5s", None, min(FAR_DUR(), remaining_to_turn), far_clip())
                continue
            eligible_turned = [k for k in order[:idx] if turned[k]]
            eligible_before = [k for k in order[idx:] if before[k] and k != j]  # v3: el que va a girar no aparece justo antes de su giro
            k, state = pick_judge_state(eligible_turned, eligible_before)
            if k is None:
                break
            if k == last_judge() and far_ok() and remaining_to_turn > 5:
                add("plano_lejano_5s", None, min(FAR_DUR(), remaining_to_turn), far_clip())
                continue
            path = react(k, state)
            if not path: break
            d = min(dur(path), remaining_to_turn)
            add("reaccion", k, d, path)
        # v3: el juez que gira no puede ser el del clip anterior: se intercala la reacción de otro juez.
        if last_judge() == j:
            k2, st2 = pick_judge_state([k for k in order[:idx] if turned[k]],
                                       [k for k in order[idx+1:] if before[k]])
            if k2 is not None and k2 != j:
                p2 = react(k2, st2)
                if p2:
                    d2 = dur(p2)
                    if (half - t) >= d2 + sum(turn_len[k] for k in pending) + 0.5:
                        add("reaccion", k2, d2, p2)
        # v3: el giro también cuenta dentro de la racha de 3 sin lipsync.
        if need_lip() and can_lip():
            add_lipsync(10.0)
        # El giro de cada juez aparece exactamente una vez.
        path = random.choice([x for x in turning[j] if x not in used] or turning[j])
        used.add(path)
        if t + dur(path) > half:
            # Prioridad absoluta a los cuatro giros; ajusta duración para no invadir mitad 2.
            d = max(0.5, half - t)
        else:
            d = dur(path)
        add("reaccion_girando", j, d, path)
        pending.remove(j)   # v3

    # Relleno restante de primera mitad, manteniendo estado por juez e intercalando canto.
    while t < half - 2.0:   # v3: no se rellena con mini-clips; el resto (<2 s) lo cubre el inicio de la 2ª mitad
        remaining_half = half - t
        # v3: máximo 3 clips seguidos sin lipsync.
        if run_len() >= 3 and remaining_half >= 10.0 and clips[-1].tipo != "lipsync_15s":
            if add_lipsync(remaining_half):
                continue
        if run_len() >= 3 and remaining_half < 10.0:
            break   # v3: ya van 3 clips y no cabe otro lipsync → el lipsync de la 2ª mitad toma el relevo
        if (remaining_half >= 10.0 and
                (not clips or clips[-1].tipo != "lipsync_15s") and
                (not clips or clips[-1].tipo == "reaccion" or random.random() < 0.35)):
            if add_lipsync(remaining_half):
                continue
        # v3: sin espacio para otro lipsync, un lejano cubre más tiempo con menos clips.
        if remaining_half < 10.0 and remaining_half > 5.0 and run_len() >= 1 and far_ok():
            add("plano_lejano_5s", None, min(FAR_DUR(), remaining_half), far_clip())
            continue
        turned_j = [j for j in range(1,5) if turning[j] and any(c.hablante == f"juez_{j}" and c.meta.get("estado_juez") == "girando" for c in clips)]
        notturned_j = [j for j in range(1,5) if j not in turned_j and before[j]]
        if t < intro_end:
            pool_j = notturned_j
            state = "antes"
        elif turned_j and (not notturned_j or random.random() < 0.5):
            pool_j, state = turned_j, "girado"
        else:
            pool_j, state = notturned_j, "antes"
        if not clips or clips[-1].tipo != "plano_lejano_5s" and random.random() < 0.16 and half-t > 5:
            add("plano_lejano_5s", None, min(FAR_DUR(), half-t), far_clip())
            continue
        if not pool_j:
            add("plano_lejano_5s", None, min(FAR_DUR(), half-t), far_clip())
            continue
        j = pick_k(pool_j, state)
        if j == last_judge() and far_ok() and half - t > 5:   # v3: nunca el mismo juez seguido
            add("plano_lejano_5s", None, min(FAR_DUR(), half-t), far_clip())
            continue
        path = react(j, state)
        if not path: break
        add("reaccion", j, min(dur(path), half-t), path)

    # Mitad 2: conservar la carpeta actual de reacciones ya giradas y patrón de planos/lipsync.
    react2_dir = os.path.join(base, PLANTILLA_SUBDIRS["reacciones_segunda_mitad"])
    reacts2 = sorted(os.path.join(react2_dir, f) for f in os.listdir(react2_dir)
                     if f.lower().endswith(".mp4")) if os.path.isdir(react2_dir) else []
    last_lipsync = False
    while t < song_dur - 0.25:
        rem = song_dur - t
        if rem >= 15 and not last_lipsync:
            budget = 3 - run_len()   # v3: clips permitidos antes del lipsync (racha máx. 3)
            if reacts2 and budget >= 1 and (random.random() < 0.6 or not far_ok()):   # v3: reacción obligatoria si el anterior fue lejano
                p = fresh(reacts2)
                add("reaccion", rand_judge(), min(dur(p), rem), p)
                budget -= 1
            if song_dur-t >= 5.0 and far_ok() and budget >= 1:   # v3: nunca dos lejanos seguidos
                add("plano_lejano_5s", None, min(FAR_DUR(), song_dur-t), far_clip())
            if song_dur-t >= 10.0 and (not clips or clips[-1].tipo != "lipsync_15s"):
                add("lipsync_15s", None, 10.0, random.choice([img_c,img_d])); last_lipsync=True
            if song_dur-t >= 5.0 and far_ok():   # v3
                add("plano_lejano_5s", None, min(FAR_DUR(), song_dur-t), far_clip())
            last_lipsync = False
        else:
            # v3: el cierre alterna reacción / lejano (sin reacciones ni lejanos seguidos).
            if reacts2 and rem > 0.5 and (not far_ok() or clips[-1].tipo != "reaccion"):
                p = fresh(reacts2)
                add("reaccion", rand_judge(), min(dur(p), rem), p)
            elif reacts2 and rem > 0.5:
                add("plano_lejano_5s", None, min(FAR_DUR(), rem), far_clip())
            elif rem > 0.5:
                add("plano_lejano_5s", None, rem, far_clip())
            else: break
        if len(clips) > 9999: break

    # Cuadrar el último fragmento al audio exacto.
    total = sum(c.dur_s for c in clips)
    if clips and total < song_dur and clips[-1].tipo == "plano_lejano_5s":
        clips[-1].dur_s += song_dur - total   # v3: no dejar dos lejanos seguidos
    elif clips and total < song_dur:
        clips.append(TimelineClip(clip_id=f"c{len(clips):04d}", tipo="plano_lejano_5s",
            hablante="concursante", dur_s=song_dur-total, path_video=far_clip(),
            estado="pendiente", meta={"estilo":"la_voz"}))
    elif clips and total > song_dur:
        clips[-1].dur_s = max(0.1, clips[-1].dur_s - (total-song_dur))
    vp.timeline_items = timeline_to_json(clips)
    vp.save()
    lip_count = sum(1 for c in clips if c.tipo == "lipsync_15s")
    first_half_lips = sum(1 for c in clips if c.tipo == "lipsync_15s" and
                          sum(x.dur_s for x in clips[:clips.index(c)]) < half)
    log(f"[TIMELINE LA VOZ] {len(clips)} clips | {song_dur:.1f}s | giros: {order} | lipsync total: {lip_count} | lipsync 1ª mitad: {first_half_lips}")
    return vp.timeline_items

log("CELDA 9 (v2) COMPLETADA — Distribución rítmica profesional.")