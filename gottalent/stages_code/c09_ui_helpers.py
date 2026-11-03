# ═══════════════════════════════════════════════════════════════
# CELDA 10: FUNCIONES AUXILIARES DE UI (SIN LAUNCH — UI real en Celda 18)
# ═══════════════════════════════════════════════════════════════

# --- Estado global UI (persistente por working) ---
GLOBAL_STATE_PATH = os.path.join(OUTPUT_DIR, "ui_state.json")

def load_ui_state():
    if os.path.exists(GLOBAL_STATE_PATH):
        try:
            return json.load(open(GLOBAL_STATE_PATH, "r", encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_ui_state(st: dict):
    json.dump(
        st,
        open(GLOBAL_STATE_PATH, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2
    )

UI_STATE         = load_ui_state()
CURRENT_VIDEO_DIR = UI_STATE.get("current_video_dir", "")
CURRENT_TEMPLATE  = UI_STATE.get("current_template_slug", "")

def ui_set_current(video_dir: str, template_slug: str):
    global CURRENT_VIDEO_DIR, CURRENT_TEMPLATE
    CURRENT_VIDEO_DIR = video_dir
    CURRENT_TEMPLATE  = template_slug
    UI_STATE["current_video_dir"]     = video_dir
    UI_STATE["current_template_slug"] = template_slug
    save_ui_state(UI_STATE)

def get_current_vp() -> VideoProject:
    if not CURRENT_VIDEO_DIR or not os.path.isdir(CURRENT_VIDEO_DIR):
        raise RuntimeError(
            "No hay video activo. Crea un proyecto de video primero."
        )
    return VideoProject(CURRENT_VIDEO_DIR)

# ---------------------------------------------------------------
# Funciones auxiliares de UI reutilizadas por Celda 18
# ---------------------------------------------------------------
def ui_refresh_templates():
    opts = listar_plantillas()
    return gr.update(
        choices=[(n, slug) for (n, slug) in opts],
        value=(
            CURRENT_TEMPLATE if CURRENT_TEMPLATE
            else (opts[0][1] if opts else None)
        ),
    )

def ui_upload_template_assets(
    template_slug, juez_imgs, escenario_cercano,
    escenarios_lejanos, juez_voice_wavs, reacciones_1, reacciones_2
):
    tpl  = cargar_plantilla(template_slug)
    base = tpl["_base_dir"]

    def _copy_files(files, dst_dir, rename_map=None):
        os.makedirs(dst_dir, exist_ok=True)
        out = []
        if not files:
            return out
        if not isinstance(files, list):
            files = [files]
        for i, f in enumerate(files, start=1):
            if not f:
                continue
            src = f if isinstance(f, str) else getattr(f, "name", "")
            if not src or not os.path.exists(src):
                continue
            fn = os.path.basename(src)
            if rename_map and i in rename_map:
                fn = rename_map[i]
            dst = os.path.join(dst_dir, fn)
            shutil.copyfile(src, dst)
            out.append(dst)
        return out

    # Jueces
    jdir = os.path.join(base, PLANTILLA_SUBDIRS["jueces_imagenes"])
    _copy_files(
        juez_imgs, jdir,
        rename_map={1:"juez_1.png", 2:"juez_2.png",
                    3:"juez_3.png", 4:"juez_4.png"}
    )

    # Escenario cercano
    cdir = os.path.join(base, PLANTILLA_SUBDIRS["escenario_cercano"])
    if escenario_cercano:
        src = (escenario_cercano if isinstance(escenario_cercano, str)
               else escenario_cercano.name)
        shutil.copyfile(src, os.path.join(cdir, "plantilla_plano_cercano.png"))

    # Escenarios lejanos
    fdir = os.path.join(base, PLANTILLA_SUBDIRS["escenario_lejano"])
    _copy_files(
        escenarios_lejanos, fdir,
        rename_map={1:"plantilla_plano_lejano_1.png",
                    2:"plantilla_plano_lejano_2.png",
                    3:"plantilla_plano_lejano_3.png"}
    )

    # Voces jueces
    vdir  = os.path.join(base, PLANTILLA_SUBDIRS["voces_jueces"])
    vpaths = _copy_files(
        juez_voice_wavs, vdir,
        rename_map={1:"juez_1.wav", 2:"juez_2.wav",
                    3:"juez_3.wav", 4:"juez_4.wav"}
    )
    for p in vpaths:
        ffmpeg_to_wav_mono_24k(p, p)

    # Reacciones
    r1dir = os.path.join(base, PLANTILLA_SUBDIRS["reacciones_primera_mitad"])
    r2dir = os.path.join(base, PLANTILLA_SUBDIRS["reacciones_segunda_mitad"])
    _copy_files(reacciones_1, r1dir)
    _copy_files(reacciones_2, r2dir)

    # Actualizar meta
    meta = cargar_plantilla(template_slug)
    meta["planos_lejanos_disponibles"] = sum(
        1 for i in [1,2,3]
        if os.path.exists(
            os.path.join(
                base,
                PLANTILLA_SUBDIRS["escenario_lejano"],
                f"plantilla_plano_lejano_{i}.png"
            )
        )
    )
    guardar_plantilla(meta)
    return "Assets cargados/actualizados OK."

def ui_set_song_audio(song_file):
    vp = get_current_vp()
    if not song_file:
        return "No se subió audio."
    src = song_file if isinstance(song_file, str) else song_file.name
    out = os.path.join(vp.video_dir, "etapa3_cancion", "song.wav")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ffmpeg_to_wav_mono_24k(src, out)
    vp.estado.setdefault("etapa3", {})["song_wav"] = out
    vp.save()
    return f"Canción lista: {out} ({_ffprobe_duration(out):.2f}s)"

def ui_build_timeline(tpl, personaje, intro_seconds=30):
    vp   = get_current_vp()
    song = vp.estado.get("etapa3", {}).get("song_wav", "")
    if not song or not os.path.exists(song):
        raise RuntimeError("Sube primero el audio de la canción.")
    estilo = cargar_plantilla(tpl).get("estilo", "got_talent")
    if estilo == "la_voz":
        tl = build_song_timeline_lavoz(vp, tpl, personaje, song_audio_path=song,
                                       use_far_planes=["E","F","G"],
                                       intro_seconds=float(intro_seconds or 30))
    else:
        tl = build_song_timeline(vp, tpl, personaje, song_audio_path=song,
                                 use_far_planes=["E","F","G"])
    html = render_timeline_html(tl)
    return "Timeline construido.", html

# ---------------------------------------------------------------
# SIN demo.launch() — La UI real está en CELDA 18
# ---------------------------------------------------------------
log("CELDA 10 COMPLETADA (funciones auxiliares, sin UI propia).")