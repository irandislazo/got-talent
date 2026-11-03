# ═══════════════════════════════════════════════════════════════
# CELDA 18 (v7): UI GRADIO — ESTILO AIQUEST + VOZ CONCURSANTE
#
# NOVEDADES v7 (todo lo demás intacto):
#   - 🖼️ Preview con imágenes en Etapas 2 y 4 (audio + imagen, sin lipsync) y
#     🎬 «Continuar con lipsync» (reutiliza guion y audios ya validados)
#   - 🧩 Timeline por fragmentos con botón 🔄 «Regenerar lipsync (LTX)» en Etapa 2, 3 y 4
#   - 🚀 AUTOMATIZACIÓN: E2+E4 juntas · TODO con casillas para excluir etapas · solo canción
#   - 🧪 Botón de diagnóstico del entorno LTX (arregla el conflicto de huggingface_hub)
#
# CAMBIOS vs v5:
#   - Nueva sección "🎙️ Voz del Concursante" encima de Etapas 2 y 4:
#       · Subir audio → auto-transcribir → preview clonado → validar
#       · O bien: escribir descripción → VoiceDesign preview → adoptar como ref
#       · La voz adoptada clona consistentemente todo el guion
#   - Estado de voz del concursante visible en panel de progreso
#   - Auto-transcripción de jueces al arrancar Etapa 2/4
#   - Preview de audio de prueba antes de comprometerse con una voz
#   - Botón "Transcribir jueces ahora" para verificar antes de ejecutar
#   - NINGÚN cambio en funcionalidades de imagen o LTX/video
# ═══════════════════════════════════════════════════════════════


# ── HfFolder backward-compatibility patch ───────────────────
try:
    import huggingface_hub.utils as hf_utils
    if not hasattr(hf_utils, 'HfFolder'):
        class HfFolder:
            @staticmethod
            def get_token():
                try:
                    from huggingface_hub import get_token
                    return get_token() or os.environ.get('HF_TOKEN')
                except Exception:
                    return os.environ.get('HF_TOKEN')
            @staticmethod
            def save_token(token):
                pass
        setattr(hf_utils, 'HfFolder', HfFolder)
        if 'huggingface_hub.utils' in sys.modules:
            setattr(sys.modules['huggingface_hub.utils'], 'HfFolder', HfFolder)
        print("✅ HfFolder patch aplicado solo para compatibilidad legacy del proyecto")
    else:
        print(f"✅ HfFolder ya disponible (huggingface_hub {hf_utils.__name__})")
except Exception as e:
    print(f"⚠ HfFolder patch warning: {e}")



import gradio as gr
import sys
import os
import time
import json
import shutil
import re

# ── Estado global ─────────────────────────────────────────────
_UI_STATE    = load_ui_state()
_CUR_VP_DIR  = _UI_STATE.get("current_video_dir", "")
_CUR_TPL     = _UI_STATE.get("current_template_slug", "")
_LTX_PATHS   = globals().get("_LTX_PATHS", {})

def _get_vp() -> VideoProject:
    if not _CUR_VP_DIR or not os.path.isdir(_CUR_VP_DIR):
        raise gr.Error("No hay video activo. Crea un proyecto primero.")
    return VideoProject(_CUR_VP_DIR)

def _set_vp(vp_dir: str, tpl_slug: str):
    global _CUR_VP_DIR, _CUR_TPL
    _CUR_VP_DIR = vp_dir
    _CUR_TPL    = tpl_slug
    _UI_STATE["current_video_dir"]     = vp_dir
    _UI_STATE["current_template_slug"] = tpl_slug
    save_ui_state(_UI_STATE)


# ═══════════════════════════════════════════════════════════════
# HANDLERS ETAPA 1 (sin cambios — imagen/LTX estable)
# ═══════════════════════════════════════════════════════════════

def _etapa1_common_params():
    vp = _get_vp()
    params = vp.params.get("etapa1", {})
    return {
        "vp": vp,
        "tpl": _CUR_TPL,
        "diffusion_quant": params.get("diffusion_quant", "Q8_0"),
        "encoder_quant": params.get("encoder_quant", "Q4_K_M"),
        "mmproj_quant": params.get("mmproj_quant", "Q8_0"),
        "steps_A": int(params.get("steps_A", 30)),
        "steps_edit": int(params.get("steps_edit", 30)),
        "cfg": float(params.get("cfg", 2.0)),
        "seed": int(params.get("seed", -1)),
    }


def ui_etapa1_paso(key, upload_file, personaje, p_reemplazar, edad, vestuario, planos):
    """
    Genera UNA sola imagen de Etapa 1, o usa la subida manualmente.
    key: A, B, C, D, E, F, G
    """
    try:
        cp = _etapa1_common_params()
        vp, tpl = cp["vp"], cp["tpl"]
        if not tpl:
            return f"❌ Selecciona plantilla primero.", None

        img_dir = os.path.join(vp.video_dir, "etapa1_imagenes")
        os.makedirs(img_dir, exist_ok=True)

        fname_map = {
            "A": "concursante_base.png",
            "B": "concursante_cuerpo_completo.png",
            "C": "concursante_plano_cercano.png",
            "D": "concursante_closeup.png",
            "E": "concursante_plano_lejano_1.png",
            "F": "concursante_plano_lejano_2.png",
            "G": "concursante_plano_lejano_3.png",
        }
        out_path = os.path.join(img_dir, fname_map[key])

        # ── UPLOAD MANUAL: si hay archivo, copiar y saltar generación ──
        if upload_file is not None:
            src = upload_file if isinstance(upload_file, str) else getattr(upload_file, "name", "")
            if src and os.path.exists(src):
                shutil.copyfile(src, out_path)
                vp.estado.setdefault("aprobados", {})[f"etapa1_{key}"] = True
                vp.save()
                log(f"[ETAPA1] {key} subida manualmente → {out_path}")
                return f"✅ {key} cargada manualmente.", out_path
            else:
                return f"⚠ Archivo inválido para {key}.", None

        # ── GENERACIÓN CON IA ──
        mp = ensure_qwen_image21_models(cp["diffusion_quant"], cp["encoder_quant"], cp["mmproj_quant"])
        sd = SDQwenImage21(SDCLI_PATH, mp)

        W_9_16, H_9_16 = 832, 1472
        W_16_9, H_16_9 = 1376, 768

        if key == "A":
            p1 = render_prompt_1(personaje, int(edad), str(vestuario))
            sd.generate_text2img(p1, out_path, W_9_16, H_9_16, cp["steps_A"], cp["cfg"], seed=cp["seed"])

        elif key == "B":
            ref_a = os.path.join(img_dir, fname_map["A"])
            if not os.path.exists(ref_a):
                return "❌ Necesitas la imagen A primero.", None
            sd.edit_with_refs(PROMPT_2_CUERPO_COMPLETO, [ref_a], out_path, W_9_16, H_9_16, cp["steps_edit"], cp["cfg"], seed=cp["seed"])

        elif key in ("C", "E", "F", "G"):
            ref_b = os.path.join(img_dir, fname_map["B"])
            if not os.path.exists(ref_b):
                return "❌ Necesitas la imagen B primero.", None
            if key == "C":
                tpl_img = os.path.join(cargar_plantilla(tpl)["_base_dir"], PLANTILLA_SUBDIRS["escenario_cercano"], "plantilla_plano_cercano.png")
            else:
                idx = {"E": 1, "F": 2, "G": 3}[key]
                tpl_img = os.path.join(cargar_plantilla(tpl)["_base_dir"], PLANTILLA_SUBDIRS["escenario_lejano"], f"plantilla_plano_lejano_{idx}.png")
            if not os.path.exists(tpl_img):
                return f"❌ Falta plantilla: {os.path.basename(tpl_img)}", None
            p3 = render_prompt_3(personaje, p_reemplazar)
            sd.edit_with_refs(p3, [ref_b, tpl_img], out_path, W_16_9, H_16_9, cp["steps_edit"], cp["cfg"], seed=cp["seed"])

        elif key == "D":
            ref_c = os.path.join(img_dir, fname_map["C"])
            if not os.path.exists(ref_c):
                return "❌ Necesitas la imagen C primero.", None
            sd.edit_with_refs(PROMPT_4_CLOSEUP, [ref_c], out_path, W_16_9, H_16_9, cp["steps_edit"], cp["cfg"], seed=cp["seed"])

        vp.estado.setdefault("aprobados", {})[f"etapa1_{key}"] = os.path.exists(out_path)
        vp.save()
        log(f"[ETAPA1] {key} generada → {out_path}")
        return f"✅ Imagen {key} generada.", out_path

    except Exception as e:
        return f"❌ Error {key}: {e}", None

def ui_etapa1_continuar_desde(start_key, personaje, p_reemplazar, edad, vestuario, planos):
    keys_order = ["A", "B", "C", "D", "E", "F", "G"]
    try:
        idx_start = keys_order.index(start_key)
    except ValueError:
        return f"❌ Key inválida: {start_key}"
    resultados = []
    for k in keys_order[idx_start:]:
        img_dir = os.path.join(_get_vp().video_dir, "etapa1_imagenes")
        fname_map = {"A":"concursante_base.png","B":"concursante_cuerpo_completo.png",
                     "C":"concursante_plano_cercano.png","D":"concursante_closeup.png",
                     "E":"concursante_plano_lejano_1.png","F":"concursante_plano_lejano_2.png",
                     "G":"concursante_plano_lejano_3.png"}
        existing = os.path.join(img_dir, fname_map[k])
        if os.path.exists(existing) and os.path.getsize(existing) > 5000:
            resultados.append(f"{k}: ✅ ya existe")
            continue
        msg, _ = ui_etapa1_paso(k, None, personaje, p_reemplazar, edad, vestuario, planos)
        resultados.append(f"{k}: {msg}")
        if "❌" in msg:
            break
    return "\n".join(resultados)

def ui_etapa1_todo(personaje, p_reemplazar, edad, vestuario, planos,
                   dq, eq, mq, stA, stE, cfg_v, seed_v):
    if not _CUR_TPL:
        return "❌ Selecciona plantilla.", []
    try:
        vp = _get_vp()
        vp.params.setdefault("etapa1", {}).update({
            "edad": int(edad), "vestuario": str(vestuario),
            "personaje": personaje, "personaje_plantilla": p_reemplazar,
            "planos_lejanos_a_usar": int(planos),
            "diffusion_quant": dq, "encoder_quant": eq, "mmproj_quant": mq,
            "steps_A": int(stA), "steps_edit": int(stE),
            "cfg": float(cfg_v), "seed": int(seed_v),
        })
        vp.save()
        paths = etapa1_generar_imagenes(
            vp, _CUR_TPL, personaje, p_reemplazar,
            edad=int(edad), vestuario=str(vestuario),
            planos_lejanos_a_usar=int(planos),
            diffusion_quant=dq, encoder_quant=eq, mmproj_quant=mq,
            steps_A=int(stA), steps_edit=int(stE),
            cfg=float(cfg_v), seed=int(seed_v),
        )
        gallery = [paths[k] for k in ["A","B","C","D","E","F","G"]
                   if k in paths and os.path.exists(paths[k])]
        return "✅ Etapa 1 completa.", gallery
    except Exception as e:
        return f"❌ Error: {e}", []


# ═══════════════════════════════════════════════════════════════
# HANDLERS VOZ DEL CONCURSANTE (NUEVOS — Celda 18 v6)
# ═══════════════════════════════════════════════════════════════

def ui_upload_contestant_audio(audio_file):
    """Procesa el audio subido: convierte + transcribe + actualiza estado."""
    if audio_file is None:
        return "❌ No se recibió archivo de audio.", "", "", gr.update(interactive=False)
    src = audio_file if isinstance(audio_file, str) else getattr(audio_file, "name", "")
    if not src or not os.path.exists(src):
        return "❌ Archivo inválido.", "", "", gr.update(interactive=False)
    texto, idioma, msg = process_contestant_upload(src)
    return msg, texto, idioma, gr.update(interactive=bool(texto))


def ui_generate_clone_preview(ref_text_box, preview_text_box):
    """Genera preview clonando el audio subido del concursante."""
    ref_text   = ref_text_box or CONTESTANT_VOICE_STATE.get("ref_text", "")
    prev_text  = preview_text_box.strip() if preview_text_box and preview_text_box.strip() else PREVIEW_TEXT_ES
    wav_path, msg = generate_contestant_clone_preview(
        ref_wav=CONTESTANT_REF_WAV,
        ref_text=ref_text,
        preview_text=prev_text,
    )
    return msg, wav_path if wav_path and os.path.exists(wav_path) else None


def ui_generate_design_preview(design_instruct_box, preview_text_box):
    """Genera preview con VoiceDesign desde descripción texto."""
    instruct  = design_instruct_box.strip() if design_instruct_box else ""
    prev_text = preview_text_box.strip() if preview_text_box and preview_text_box.strip() else PREVIEW_TEXT_ES
    if not instruct:
        return "❌ Escribe una descripción de voz.", None
    wav_path, msg = generate_voice_design_preview(instruct, prev_text)
    return msg, wav_path if wav_path and os.path.exists(wav_path) else None


def ui_adopt_design_as_ref(design_instruct_box):
    """Adopta el preview de VoiceDesign como referencia de clonación."""
    instruct = design_instruct_box.strip() if design_instruct_box else ""
    ref_wav, msg = adopt_design_as_reference(instruct)
    return msg, gr.update(interactive=True)


def ui_validate_contestant_voice():
    """Marca la voz como validada."""
    ok, msg = validate_contestant_voice()
    color = "✅" if ok else "❌"
    return msg


def ui_transcribe_judges(tpl_slug):
    """Transcribe los audios de todos los jueces en la plantilla."""
    if not tpl_slug:
        return "❌ Selecciona una plantilla primero."
    resultados = []
    for i in range(1, 5):
        wav = get_juez_ref_wav(tpl_slug, i)
        if not wav or not os.path.exists(wav):
            resultados.append(f"Juez {i}: ⚠ sin audio en plantilla")
            continue
        # Verificar si ya existe transcripción
        existing = get_juez_ref_text(tpl_slug, i)
        if existing:
            resultados.append(f"Juez {i}: ✅ ya transcrito → «{existing[:60]}...»")
            continue
        texto = auto_transcribe_juez(tpl_slug, i)
        if texto:
            resultados.append(f"Juez {i}: ✅ «{texto[:70]}...»")
        else:
            resultados.append(f"Juez {i}: ❌ transcripción fallida")
    return "\n".join(resultados)


def ui_show_voice_state():
    """Muestra el estado actual de la voz del concursante."""
    state = load_contestant_voice_state()
    lines = [
        f"Modo: {state.get('mode') or 'no configurado'}",
        f"Referencia: {os.path.basename(state.get('ref_wav_path','')) or 'ninguna'}",
        f"Transcripción: «{state.get('ref_text','')[:100]}{'...' if len(state.get('ref_text',''))>100 else ''}»",
        f"Diseño: {state.get('design_instruct','')[:80] or 'N/A'}",
        f"✅ VALIDADA: {state.get('validated', False)}",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# HANDLERS CLIPS 5s Y DEMÁS (sin cambios vs v5)
# ═══════════════════════════════════════════════════════════════

def ui_gen_5s_individual(img_key, personaje, seed_v):
    """Genera UN clip de 5s. Retorna (status, gallery_list)."""
    try:
        vp = _get_vp()
        if not _LTX_PATHS:
            return "❌ Descarga primero los modelos LTX.", []

        img_dir = os.path.join(vp.video_dir, "etapa1_imagenes")
        fname_map = {
            "E": "concursante_plano_lejano_1.png",
            "F": "concursante_plano_lejano_2.png",
            "G": "concursante_plano_lejano_3.png",
        }
        if img_key not in fname_map:
            return f"❌ Key inválida: {img_key}", []

        img_path = os.path.join(img_dir, fname_map[img_key])
        if not os.path.exists(img_path):
            return f"❌ Imagen {img_key} no existe. Genera Etapa 1 primero.", []

        out_dir = os.path.join(vp.video_dir, "etapa3_cancion", "planos_5s")
        os.makedirs(out_dir, exist_ok=True)

        prompt_str, direccion = render_prompt_5s(personaje)
        out_mp4 = os.path.join(out_dir, f"far_{img_key}_v1.mp4")

        generar_clip_5s(
            vp, img_path, prompt_str, out_mp4, _LTX_PATHS,
            seed=int(seed_v), fps=24, width=1152, height=704,  # 1280×704, 1152×704, 1920×1088
        )

        vp.estado.setdefault("etapa3", {}).setdefault("clips_5s", [])
        if out_mp4 not in vp.estado["etapa3"]["clips_5s"]:
            vp.estado["etapa3"]["clips_5s"].append(out_mp4)
        vp.save()

        log(f"[5s] {img_key} generado → {out_mp4}")

        # ── Gallery acepta list[str] de imágenes o list[tuple(path,caption)] ──
        # Para video, devolver la imagen de referencia usada como preview
        gallery_items = []
        if os.path.exists(img_path):
            gallery_items.append((img_path, f"Ref {img_key}"))
        # Añadir frame preview del video generado si existe
        preview_png = out_mp4.replace(".mp4", "_preview.png")
        if not os.path.exists(preview_png) and os.path.exists(out_mp4):
            try:
                import subprocess as _sp
                _sp.run(
                    f"ffmpeg -y -i '{out_mp4}' -vframes 1 -q:v 2 '{preview_png}'",
                    shell=True, capture_output=True, timeout=10
                )
            except Exception:
                pass
        if os.path.exists(preview_png):
            sz_kb = os.path.getsize(out_mp4) // 1024
            gallery_items.append((preview_png, f"Frame 1 — {sz_kb}KB"))

        return (f"✅ Clip 5s {img_key} generado ({direccion}) "
                f"→ {os.path.basename(out_mp4)}"), gallery_items

    except Exception as e:
        import traceback
        log(f"[UI gen_5s_{img_key}] ❌ {e}\n{traceback.format_exc()[-1000:]}")
        return f"❌ Error: {e}", []


def ui_gen_5s_todos(personaje, seed_v):
    """Genera todos los clips 5s. Retorna (status, gallery_list)."""
    try:
        vp = _get_vp()
        if not _LTX_PATHS:
            return "❌ Descarga primero los modelos LTX.", []

        clips = etapa3_paso1_generar_clips_5s(
            vp, _CUR_TPL, personaje, _LTX_PATHS, seed=int(seed_v)
        )

        # Gallery: previews de todos los clips generados
        gallery_items = []
        for clip_path in clips:
            preview = clip_path.replace(".mp4", "_preview.png")
            if not os.path.exists(preview) and os.path.exists(clip_path):
                try:
                    import subprocess as _sp
                    _sp.run(
                        f"ffmpeg -y -i '{clip_path}' -vframes 1 -q:v 2 '{preview}'",
                        shell=True, capture_output=True, timeout=10
                    )
                except Exception:
                    pass
            if os.path.exists(preview):
                sz_kb = os.path.getsize(clip_path) // 1024 if os.path.exists(clip_path) else 0
                gallery_items.append((preview, f"{os.path.basename(clip_path)} — {sz_kb}KB"))
            elif os.path.exists(clip_path.replace("far_", "").replace(".mp4", ".png")):
                gallery_items.append(clip_path.replace("far_", "").replace(".mp4", ".png"))

        return f"✅ {len(clips)} clips 5s generados.", gallery_items

    except Exception as e:
        return f"❌ Error: {e}", []


# ═══════════════════════════════════════════════════════════════
# RESTO DE HANDLERS (sin cambios vs v5)
# ═══════════════════════════════════════════════════════════════

def ui_refresh_tpl():
    opts = listar_plantillas()
    choices = [(n, s) for (n, s) in opts]
    val = _CUR_TPL if _CUR_TPL else (choices[0][1] if choices else None)
    return gr.update(choices=choices, value=val)

def ui_crear_plantilla(nombre, estilo="got_talent"):
    if not nombre or not nombre.strip():
        return "❌ Escribe un nombre.", gr.update()
    base = crear_plantilla(nombre.strip())
    meta = cargar_plantilla(os.path.basename(base))
    meta["estilo"] = estilo if estilo in ("got_talent", "la_voz") else "got_talent"
    guardar_plantilla(meta)
    if meta["estilo"] == "la_voz":
        for j in range(1,5):
            for estado in ("antes", "girando", "girado"):
                os.makedirs(os.path.join(base, "video", "reacciones_lavoz", f"juez_{j}", estado), exist_ok=True)
    return f"✅ Plantilla creada: {os.path.basename(base)} ({meta['estilo']})", ui_refresh_tpl()

def ui_subir_assets_juez_independiente(tpl_slug, j1_img, j1_audios, j2_img, j2_audios,
                                        j3_img, j3_audios, j4_img, j4_audios,
                                        esc_cercano, escs_lejanos, reac1, reac2,
                                        lv_j1_a, lv_j1_g, lv_j1_d, lv_j2_a, lv_j2_g, lv_j2_d,
                                        lv_j3_a, lv_j3_g, lv_j3_d, lv_j4_a, lv_j4_g, lv_j4_d):
    if not tpl_slug:
        return "❌ Selecciona plantilla."
    try:
        tpl = cargar_plantilla(tpl_slug)
        base = tpl["_base_dir"]
        resultados = []
        img_dir = os.path.join(base, PLANTILLA_SUBDIRS["jueces_imagenes"])
        voz_dir = os.path.join(base, PLANTILLA_SUBDIRS["voces_jueces"])
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(voz_dir, exist_ok=True)
        for j_num, img_f, aud_fs in [(1,j1_img,j1_audios),(2,j2_img,j2_audios),
                                      (3,j3_img,j3_audios),(4,j4_img,j4_audios)]:
            if img_f is not None:
                src = img_f if isinstance(img_f, str) else getattr(img_f, "name", "")
                if src and os.path.exists(src):
                    shutil.copyfile(src, os.path.join(img_dir, f"juez_{j_num}.png"))
                    resultados.append(f"Juez {j_num}: imagen ✅")
            if aud_fs is not None:
                if not isinstance(aud_fs, list): aud_fs = [aud_fs]
                wavs = []
                for af in aud_fs:
                    if af is None: continue
                    src = af if isinstance(af, str) else getattr(af, "name", "")
                    if src and os.path.exists(src):
                        tmp = os.path.join(voz_dir, f"j{j_num}_tmp_{len(wavs)}.wav")
                        ffmpeg_to_wav_mono_24k(src, tmp)
                        wavs.append(tmp)
                if wavs:
                    dst = os.path.join(voz_dir, f"juez_{j_num}.wav")
                    if len(wavs) == 1:
                        shutil.copyfile(wavs[0], dst)
                    else:
                        cl = os.path.join(voz_dir, f"j{j_num}_c.txt")
                        with open(cl, "w") as f:
                            for w in wavs: f.write(f"file '{os.path.abspath(w)}'\n")
                        subprocess.run(
                            f"ffmpeg -y -f concat -safe 0 -i '{cl}' "
                            f"-ac 1 -ar 24000 -acodec pcm_s16le '{dst}'",
                            shell=True, capture_output=True)
                        try: os.remove(cl)
                        except: pass
                    for w in wavs:
                        try:
                            if w != dst and os.path.exists(w): os.remove(w)
                        except: pass
                    # Invalidar transcripción anterior del juez (si existe)
                    txt_old = os.path.join(JUECES_REF_DIR, f"juez_{j_num}_ref.txt")
                    if os.path.exists(txt_old):
                        try: os.remove(txt_old)
                        except: pass
                    resultados.append(f"Juez {j_num}: {len(wavs)} audio(s) ✅ (transcripción pendiente)")
        cdir = os.path.join(base, PLANTILLA_SUBDIRS["escenario_cercano"])
        os.makedirs(cdir, exist_ok=True)
        if esc_cercano:
            src = esc_cercano if isinstance(esc_cercano, str) else esc_cercano.name
            if src and os.path.exists(src):
                shutil.copyfile(src, os.path.join(cdir, "plantilla_plano_cercano.png"))
                resultados.append("Escenario cercano ✅")
        fdir = os.path.join(base, PLANTILLA_SUBDIRS["escenario_lejano"])
        os.makedirs(fdir, exist_ok=True)
        if escs_lejanos:
            if not isinstance(escs_lejanos, list): escs_lejanos = [escs_lejanos]
            for i, ef in enumerate(escs_lejanos, 1):
                if ef and i <= 3:
                    src = ef if isinstance(ef, str) else getattr(ef, "name", "")
                    if src and os.path.exists(src):
                        shutil.copyfile(src, os.path.join(fdir, f"plantilla_plano_lejano_{i}.png"))
            resultados.append("Escenarios lejanos ✅")
        r1d = os.path.join(base, PLANTILLA_SUBDIRS["reacciones_primera_mitad"])
        r2d = os.path.join(base, PLANTILLA_SUBDIRS["reacciones_segunda_mitad"])
        os.makedirs(r1d, exist_ok=True); os.makedirs(r2d, exist_ok=True)
        for rf_list, rd in [(reac1, r1d), (reac2, r2d)]:
            if rf_list:
                if not isinstance(rf_list, list): rf_list = [rf_list]
                for rf in rf_list:
                    if rf is None: continue
                    src = rf if isinstance(rf, str) else getattr(rf, "name", "")
                    if src and os.path.exists(src):
                        shutil.copyfile(src, os.path.join(rd, os.path.basename(src)))
        # Reacciones específicas La Voz: juez X / antes, girando, girado.
        for j, estado, lista in [
            (1,"antes",lv_j1_a),(1,"girando",lv_j1_g),(1,"girado",lv_j1_d),
            (2,"antes",lv_j2_a),(2,"girando",lv_j2_g),(2,"girado",lv_j2_d),
            (3,"antes",lv_j3_a),(3,"girando",lv_j3_g),(3,"girado",lv_j3_d),
            (4,"antes",lv_j4_a),(4,"girando",lv_j4_g),(4,"girado",lv_j4_d)]:
            rd = os.path.join(base, "video", "reacciones_lavoz", f"juez_{j}", estado)
            os.makedirs(rd, exist_ok=True)
            if lista:
                if not isinstance(lista, list): lista = [lista]
                for rf in lista:
                    if rf is None: continue
                    src = rf if isinstance(rf, str) else getattr(rf, "name", "")
                    if src and os.path.exists(src):
                        shutil.copyfile(src, os.path.join(rd, os.path.basename(src)))
        meta = cargar_plantilla(tpl_slug)
        meta["planos_lejanos_disponibles"] = sum(
            1 for i in [1,2,3] if os.path.exists(
                os.path.join(base, PLANTILLA_SUBDIRS["escenario_lejano"], f"plantilla_plano_lejano_{i}.png")))
        guardar_plantilla(meta)
        return "✅ Assets guardados:\n" + "\n".join(resultados)
    except Exception as e:
        return f"❌ Error: {e}"

def ui_nuevo_video(tpl_slug, personaje):
    if not tpl_slug:
        return "❌ Selecciona plantilla.", "", gr.update()
    try:
        vp = create_new_video_project(tpl_slug, personaje)
        vp.estado["plantilla_slug"] = tpl_slug
        vp.estado["personaje"] = personaje
        vp.estado["creado_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        vp.save()
        _set_vp(vp.video_dir, tpl_slug)
        return f"✅ Video activo: {os.path.basename(vp.video_dir)}", vp.video_dir, ui_refresh_tpl()
    except Exception as e:
        return f"❌ Error: {e}", "", gr.update()


def ui_descargar_ltx(preset_label: str):
    global _LTX_PATHS, _LTX_ENGINE
    try:
        if not preset_label or preset_label not in LTX23_PRESETS:
            return f"❌ Preset inválido: '{preset_label}'"

        log(f"[UI] Descargando modelos LTX — preset: {preset_label}")
        paths = ensure_ltx23_models(preset_label)

        if not paths or not isinstance(paths, dict):
            return "❌ ensure_ltx23_models() no devolvió rutas válidas."

        _LTX_PATHS = paths

        # Motor LTX ahora es Wan2GP (Celda 13 nueva). No crear SDLTXVideo23.
        # Verificar que el backend esté disponible:
        try:
            backend = get_ltx_backend()
            if not backend.is_available():
                log("[UI] ⚠ Backend LTX-2.5 no disponible. Verifica Celda 13.")
        except Exception:
            log("[UI] ⚠ Backend LTX-2.5 no disponible. Verifica Celda 13.")

        p = _LTX_PATHS
        lineas = [
            f"✅ LTX-{globals().get('LTX_VERSION', '2.5')} listo:",
            f"  Preset:      {p.get('_preset', preset_label)}",
            f"  Transformer: {os.path.basename(p.get('transformer', '?'))}",
            f"  Encoder:     {os.path.basename(p.get('encoder', '?'))}",
            f"  Video VAE:   {os.path.basename(p.get('video_vae', '?'))}",
            f"  Audio VAE:   {os.path.basename(p.get('audio_vae', '?'))}",
            f"  Embeddings:  {os.path.basename(p.get('embeddings_connector', '?'))}",
            f"  VRAM est.:   {p.get('_vram_estimado', 0):.0f} GB",
            f"  Steps:       {p.get('_steps', 8)} | CFG: {p.get('_cfg', 1.0)}",
            f"  /tmp libre:  {free_gb('/kaggle/tmp'):.1f} GB",
        ]
        return "\n".join(lineas)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log(f"[UI] Error en ui_descargar_ltx:\n{tb}")
        return f"❌ Error: {e}"

def ui_regenerar_imagen(tpl, personaje, p_reemplazar, img_key, seed_v):
    try:
        vp = _get_vp(); tpl = tpl or _CUR_TPL
        params = vp.params.get("etapa1", {})
        paths = etapa1_generar_imagenes(
            vp, tpl, personaje, p_reemplazar,
            edad=int(params.get("edad", 28)),
            vestuario=params.get("vestuario", "vestido elegante de gala"),
            planos_lejanos_a_usar=int(params.get("planos_lejanos_a_usar", 3)),
            diffusion_quant=params.get("diffusion_quant","Q4_K"),
            encoder_quant=params.get("encoder_quant","Q4_K_M"),
            mmproj_quant=params.get("mmproj_quant","Q8_0"),
            steps_A=int(params.get("steps_A",18)),
            steps_edit=int(params.get("steps_edit",18)),
            cfg=float(params.get("cfg",6.0)), seed=int(seed_v))
        out = paths.get(img_key)
        return f"✅ {img_key} regenerada.", out if out and os.path.exists(out) else None
    except Exception as e:
        return f"❌ Error: {e}", None

def ui_subir_cancion(song_file):
    if not song_file: return "❌ No se subió audio."
    try:
        vp = _get_vp()
        src = song_file if isinstance(song_file, str) else song_file.name
        out = os.path.join(vp.video_dir, "etapa3_cancion", "song.wav")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        ffmpeg_to_wav_mono_24k(src, out)
        vp.estado.setdefault("etapa3", {})["song_wav"] = out
        vp.save()
        dur = _ffprobe_duration(out) or 0
        return f"✅ Canción lista: {out} ({dur:.1f}s)"
    except Exception as e:
        return f"❌ Error: {e}"

def ui_regen_5s(tpl, personaje, clip_path, seed_v):
    try:
        vp = _get_vp(); tpl = tpl or _CUR_TPL
        if not _LTX_PATHS: return "❌ Descarga LTX.", None
        new = etapa3_regenerar_clip_5s(vp, clip_path, personaje, _LTX_PATHS, seed=int(seed_v))
        return "✅ Clip 5s regenerado.", new
    except Exception as e: return f"❌ Error: {e}", None

def ui_build_tl(tpl, personaje, intro_seconds=30):
    try:
        tpl = tpl or _CUR_TPL
        result, html = ui_build_timeline(tpl, personaje, intro_seconds)
        return result, html
    except Exception as e:
        return f"❌ Error: {e}", ""

def ui_renderizar_cancion(tpl, personaje):
    try:
        vp = _get_vp(); tpl = tpl or _CUR_TPL
        if not _LTX_PATHS: return "❌ Descarga LTX.", None
        out = etapa3_paso3_renderizar_secuencia(vp, tpl, personaje, _LTX_PATHS)
        return f"✅ Canción renderizada: {os.path.basename(out)}", out
    except Exception as e:
        return f"❌ Error: {e}", None

def ui_regen_5s(tpl, personaje, clip_path, seed_v):
    try:
        vp = _get_vp(); tpl = tpl or _CUR_TPL
        if not _LTX_PATHS: return "❌ Descarga LTX.", None
        new = etapa3_regenerar_clip_5s(vp, clip_path, personaje, _LTX_PATHS, seed=int(seed_v))
        return "✅ Clip 5s regenerado.", new
    except Exception as e:
        return f"❌ Error: {e}", None

import threading
_GPU_LOCK = threading.Lock()      # v7: evita lanzar dos tareas de GPU a la vez (LTX/Qwen)
_BUSY_MSG = "⏳ Hay otra tarea de GPU en curso (generación o regeneración). Espera a que termine."

def _ui_run_etapa(etapa, tpl, personaje, historia, model_id, fast_mode_tts, modo, solo_preview=False):
    # modo: "nuevo" → guion NUEVO (xKiro) | "actual" → usa el guion guardado/editado |
    #       "auto"  → reutiliza el guion si existe, si no lo genera
    # solo_preview=True → TTS + video SOLO con imágenes (no usa LTX ni lipsync)
    if not _GPU_LOCK.acquire(blocking=False):
        return _BUSY_MSG, None
    try:
        vp = _get_vp(); tpl = tpl or _CUR_TPL
        if not _LTX_PATHS and not solo_preview:
            return "❌ Descarga LTX.", None
        out = ejecutar_etapa_dialogos(
            vp, tpl, personaje, etapa, _LTX_PATHS or {},
            historia_base=historia or "", model_id=model_id,
            fast_mode_tts=bool(fast_mode_tts),
            regenerar_guion=(modo == "nuevo"),
            usar_guion_actual=(modo == "actual"),
            solo_preview=bool(solo_preview),
        )
        origen = {"nuevo": "guion NUEVO", "actual": "guion ACTUAL"}.get(modo, "guion existente o nuevo")
        if solo_preview:
            return (f"✅ {etapa.capitalize()} PREVIEW listo ({origen}): audio + imágenes, sin lipsync.\n"
                    f"Escúchalo; si el guion suena bien pulsa «🎬 Continuar con lipsync».", out)
        return f"✅ {etapa.capitalize()} completada ({origen}).", out
    except Exception as e:
        import traceback
        log(f"[UI {etapa}] ❌ {e}\n{traceback.format_exc()[-1500:]}")
        return f"❌ Error: {e}", None
    finally:
        _GPU_LOCK.release()

def ui_etapa2(tpl, personaje, historia, model_id, fast_mode_tts):
    return _ui_run_etapa("etapa2", tpl, personaje, historia, model_id, fast_mode_tts, "nuevo")

def ui_etapa2_guion_actual(tpl, personaje, historia, model_id, fast_mode_tts):
    return _ui_run_etapa("etapa2", tpl, personaje, historia, model_id, fast_mode_tts, "actual")

def ui_etapa4(tpl, personaje, model_id, fast_mode_tts):
    return _ui_run_etapa("etapa4", tpl, personaje, "", model_id, fast_mode_tts, "nuevo")

def ui_etapa4_guion_actual(tpl, personaje, model_id, fast_mode_tts):
    return _ui_run_etapa("etapa4", tpl, personaje, "", model_id, fast_mode_tts, "actual")

def ui_generar_solo_guion(etapa, tpl, personaje, historia, model_id):
    # Genera el guion (sin TTS ni video) y lo deja cargado en el editor.
    try:
        vp = _get_vp()
        data = generar_guion_con_respaldo(vp, personaje, etapa,
                                          historia_base=historia or "", model_id=model_id)
        n = len(data.get("fragmentos", []))
        dur = _fnum(data.get("duracion_total_estimada_min"))
        msg = (f"✅ Guion {etapa} generado y guardado: {n} fragmentos (~{dur:.1f} min).\n"
               f"Revísalo/edítalo en '✏️ EDITAR GUION MANUALMENTE' y luego pulsa "
               f"'⏯️ Generar con guion actual'.")
        return msg, json.dumps(data, ensure_ascii=False, indent=2), etapa
    except Exception as e:
        import traceback
        log(f"[UI guion {etapa}] ❌ {e}\n{traceback.format_exc()[-1200:]}")
        return f"❌ Error generando guion: {e}", gr.update(), gr.update()

def ui_regen_frag(tpl, personaje, etapa_sel, frag_idx, nuevo_texto):
    try:
        vp = _get_vp(); tpl = tpl or _CUR_TPL
        if not _LTX_PATHS: return "❌ Descarga LTX."
        return regenerar_fragmento_dialogo(
            vp, tpl, personaje, etapa_sel,
            int(frag_idx), nuevo_texto=nuevo_texto,
            ltx_paths=_LTX_PATHS,
        )
    except Exception as e:
        return f"❌ Error: {e}"

def ui_render_final(tpl, etapa2_on, etapa4_on):
    try:
        vp = _get_vp()
        out = render_video_final(vp, {"etapa2": bool(etapa2_on), "etapa3": True, "etapa4": bool(etapa4_on)})
        return f"✅ Video final: {os.path.basename(out)}", out
    except Exception as e:
        return f"❌ Error: {e}", None

def ui_publicar_dataset():
    try: return publicar_modelos_como_dataset()
    except Exception as e: return f"❌ Error: {e}"

def ui_ver_estado():
    try:
        vp = _get_vp()
        voice_st = load_contestant_voice_state()
        lineas = [
            f"Video: {os.path.basename(vp.video_dir)}",
            f"Aprobados: {vp.estado.get('aprobados',{})}",
            f"Timeline: {len(vp.timeline_items)} clips",
            f"",
            f"🎙️ Voz concursante: {voice_st.get('mode') or 'no config'} | validada={voice_st.get('validated',False)}",
            f"   Ref: {os.path.basename(voice_st.get('ref_wav_path','')) or 'ninguna'}",
        ]
        for et in ["etapa2", "etapa3", "etapa4"]:
            lineas.append(f"{et}: {vp.estado.get(et,{}).get('video_final','(sin video)')}")
        lineas.append(f"FINAL: {vp.estado.get('video_final','(sin render)')}")
        lineas.append(f"/working: {free_gb('/kaggle/working'):.2f} GB | /tmp: {free_gb('/kaggle/tmp'):.2f} GB")
        return "\n".join(lineas)
    except Exception as e:
        return f"Error: {e}"

def ui_ver_guion_json():
    try:
        vp = _get_vp()
        g2 = vp.estado.get("etapa2", {}).get("guion_json", {})
        g4 = vp.estado.get("etapa4", {}).get("guion_json", {})
        return (json.dumps(g2, ensure_ascii=False, indent=2) +
                "\n\n--- ETAPA 4 ---\n\n" +
                json.dumps(g4, ensure_ascii=False, indent=2))
    except Exception as e:
        return str(e)

# ── Helpers de reanudación ───────────────────────────────────

def _video_mtime(p):
    try:
        for fn in ["estado.json","timeline.json"]:
            fp = os.path.join(p, fn)
            if os.path.exists(fp): return os.path.getmtime(fp)
        return os.path.getmtime(p)
    except: return time.time()

def _list_video_dirs():
    if not os.path.isdir(VIDEOS_DIR): return []
    dirs = []
    for name in os.listdir(VIDEOS_DIR):
        p = os.path.join(VIDEOS_DIR, name)
        if os.path.isdir(p) and (os.path.exists(os.path.join(p,"estado.json")) or
                                  os.path.exists(os.path.join(p,"timeline.json"))):
            dirs.append(p)
    dirs.sort(key=_video_mtime, reverse=True)
    return dirs

def _video_label(p):
    try: mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(_video_mtime(p)))
    except: mtime = "?"
    return f"{os.path.basename(p)} · {mtime}"

def _gallery_images(vp_dir):
    img_dir = os.path.join(vp_dir, "etapa1_imagenes")
    keys = [("A","concursante_base.png"),("B","concursante_cuerpo_completo.png"),
            ("C","concursante_plano_cercano.png"),("D","concursante_closeup.png"),
            ("E","concursante_plano_lejano_1.png"),("F","concursante_plano_lejano_2.png"),
            ("G","concursante_plano_lejano_3.png")]
    return [(os.path.join(img_dir,fn),lb) for lb,fn in keys
            if os.path.exists(os.path.join(img_dir,fn))]

def _safe_dur(path):
    try:
        d = _ffprobe_duration(path)
        return f"{d:.1f}s" if d else "?"
    except: return "?"

def _progress_text(vp_dir):
    if not vp_dir or not os.path.isdir(vp_dir): return "No hay video activo."
    try: vp = VideoProject(vp_dir)
    except Exception as e: return f"Error: {e}"
    imgs = _gallery_images(vp_dir)
    labels = [x[1] for x in imgs]
    missing = [k for k in ["A","B","C","D","E","F","G"] if k not in labels]
    e2 = vp.estado.get("etapa2",{}) or {}
    g2 = (e2.get("guion_json",{}) or {}).get("fragmentos",[]) or []
    e3 = vp.estado.get("etapa3",{}) or {}
    song = e3.get("song_wav","")
    song_ok = bool(song and os.path.exists(song))
    clips5 = [c for c in (e3.get("clips_5s",[]) or []) if os.path.exists(c)]
    e4 = vp.estado.get("etapa4",{}) or {}
    g4 = (e4.get("guion_json",{}) or {}).get("fragmentos",[]) or []
    fv = vp.estado.get("video_final","")
    # Voz del concursante
    voice_st = load_contestant_voice_state()
    voz_ok = voice_st.get("validated", False)
    voz_info = f"{'✅' if voz_ok else '❌'} {voice_st.get('mode') or 'no configurada'}"
    L = [
        f"📂 {os.path.basename(vp_dir)}",
        f"🎭 {vp.estado.get('personaje','?')} | 📁 {vp.estado.get('plantilla_slug',_CUR_TPL or '?')}",
        f"🎙️ Voz concursante: {voz_info}",
        "",
        f"🖼️ E1: {len(imgs)}/7 | Faltan: {', '.join(missing) if missing else 'ninguna'}",
        f"🎙️ E2: {'✅' if g2 else '❌'} ({len(g2)} frags) | 🎵 E3: canción {'✅' if song_ok else '❌'} | clips5s: {len(clips5)}",
        f"🏆 E4: {'✅' if g4 else '❌'} ({len(g4)} frags) | 🎞️ FINAL: {'✅ '+os.path.basename(fv) if fv and os.path.exists(fv) else '❌'}",
    ]
    try:
        L.append(f"💾 /working: {free_gb('/kaggle/working'):.2f} GB | /tmp: {free_gb('/kaggle/tmp'):.2f} GB")
    except: pass
    return "\n".join(L)

def ui_listar_videos():
    dirs = _list_video_dirs()
    choices = [(_video_label(p),p) for p in dirs]
    val = _CUR_VP_DIR if _CUR_VP_DIR and os.path.isdir(_CUR_VP_DIR) else (dirs[0] if dirs else None)
    if val and all(c[1]!=val for c in choices):
        choices.insert(0, (_video_label(val), val))
    return gr.update(choices=choices, value=val)

def ui_scan_ultimo_video():
    global _CUR_VP_DIR, _CUR_TPL
    dirs = _list_video_dirs()
    choices = [(_video_label(p),p) for p in dirs]
    sel = _CUR_VP_DIR if _CUR_VP_DIR and os.path.isdir(_CUR_VP_DIR) else (dirs[0] if dirs else None)
    if sel and all(c[1]!=sel for c in choices):
        choices.insert(0, (_video_label(sel), sel))
    if sel:
        try:
            vp = VideoProject(sel)
            tpl = vp.estado.get("plantilla_slug") or _CUR_TPL
            if tpl: _CUR_TPL = tpl
        except: pass
        _set_vp(sel, _CUR_TPL or "")
        return _progress_text(sel), _gallery_images(sel), gr.update(choices=choices, value=sel), sel
    return "No hay videos.\n1) Crea plantilla.\n2) Crea video.", [], gr.update(choices=choices, value=None), ""

def ui_cargar_video_seleccionado(video_path):
    global _CUR_VP_DIR, _CUR_TPL
    if not video_path or not os.path.isdir(video_path):
        return "❌ Video no existe.", [], "Ninguno", ""
    try:
        vp = VideoProject(video_path)
        tpl = vp.estado.get("plantilla_slug") or _CUR_TPL
        if tpl: _CUR_TPL = tpl
    except: pass
    _set_vp(video_path, _CUR_TPL or "")
    return _progress_text(video_path), _gallery_images(video_path), os.path.basename(video_path), video_path

def ui_refrescar_progreso():
    if _CUR_VP_DIR and os.path.isdir(_CUR_VP_DIR):
        return _progress_text(_CUR_VP_DIR), _gallery_images(_CUR_VP_DIR)
    return "No hay video activo.", []

def ui_initial_load():
    prog, gal, dd, ruta = ui_scan_ultimo_video()
    return prog, gal, dd, ruta, ui_refresh_tpl()


def ui_guardar_guion_editado(etapa_sel, guion_texto):
    try:
        vp = _get_vp()
        return guardar_guion_editado(vp, etapa_sel, guion_texto)
    except Exception as e:
        return f"❌ Error: {e}"

# ═══════════════════════════════════════════════════════════════
# v7 — NUEVOS HANDLERS: preview con imágenes · timeline por fragmento ·
#      regenerar clip lipsync · automatización por etapas · diagnóstico LTX
# ═══════════════════════════════════════════════════════════════
def _bump(n):
    try:
        return int(n) + 1
    except Exception:
        return 1

def _bump3(a, b, c):
    return _bump(a), _bump(b), _bump(c)

# ── Preview con imágenes (Etapas 2 y 4) ──────────────────────
def ui_etapa2_preview(tpl, personaje, historia, model_id, fast_mode_tts):
    return _ui_run_etapa("etapa2", tpl, personaje, historia, model_id, fast_mode_tts,
                         "auto", solo_preview=True)

def ui_etapa4_preview(tpl, personaje, model_id, fast_mode_tts):
    return _ui_run_etapa("etapa4", tpl, personaje, "", model_id, fast_mode_tts,
                         "auto", solo_preview=True)

def ui_probar_entorno_ltx():
    try:
        if globals().get("LTX_VERSION") == "2.3":
            return ltx23_diag()
        pp = ensure_ltx_env(force=True)
        msg = "✅ Entorno LTX-2.5 OK: el subproceso puede importar transformers + mmgp."
        if pp:
            msg += f"\n   Se usa un huggingface_hub compatible aislado en: {pp}"
        return msg
    except Exception as e:
        return f"❌ {e}"

# ── Filas del timeline por fragmentos ────────────────────────
_ESTADO_DIALOGO = {
    "lipsync":   "🎬 Lipsync LTX ✅",
    "fallback":  "⚠️ Lipsync FALLÓ → clip estático",
    "preview":   "🖼️ Preview (imagen + audio)",
    "audio":     "🔊 Solo audio",
    "pendiente": "⏳ Pendiente",
}

def ui_filas_timeline_dialogo(etapa):
    try:
        vp = _get_vp()
    except Exception:
        return {"msg": "Crea o carga un video para ver el timeline.", "rows": []}
    try:
        filas = estado_fragmentos_dialogo(vp, etapa)
    except Exception as e:
        return {"msg": f"❌ No se pudo leer el timeline: {e}", "rows": []}
    if not filas:
        return {"msg": "Aún no hay guion en esta etapa. Pulsa «📝 Generar solo guion».", "rows": []}
    cuenta = {}
    for f in filas:
        cuenta[f["estado"]] = cuenta.get(f["estado"], 0) + 1
    resumen = " · ".join(f"{_ESTADO_DIALOGO.get(k, k)}: **{v}**" for k, v in cuenta.items())
    rows = []
    for f in filas:
        txt = f["texto"].replace("\n", " ")
        if len(txt) > 170:
            txt = txt[:170] + "…"
        dur = f"{f['dur']:.1f}s" if f.get("dur") else "—"
        extra = " · ✏️ el texto cambió: audio desactualizado" if f["desactualizado"] else ""
        md = (f"**#{f['idx']:02d} · {f['hablante']}** · ⏱ {dur} · "
              f"{_ESTADO_DIALOGO.get(f['estado'], f['estado'])}{extra}\n\n«{txt}»")
        video = f["clip"] or f["preview"] or ""
        rows.append({"idx": f["idx"], "md": md, "video": video,
                     "audio": (f["audio"] if not video else "")})
    return {"msg": f"{len(filas)} fragmentos — {resumen}", "rows": rows}

_TIPO_CLIP = {"reaccion": "😮 reacción", "plano_lejano_5s": "🎥 plano lejano 5s",
              "lipsync_15s": "🎤 lipsync", "dialogo": "💬 diálogo"}
_ESTADO_CLIP = {"pendiente": "⏳ pendiente", "completado": "✅ listo",
                "completado_fallback": "⚠️ lipsync FALLÓ → imagen estática"}

def ui_filas_timeline_cancion():
    try:
        vp = _get_vp()
    except Exception:
        return {"msg": "Crea o carga un video para ver el timeline.", "rows": []}
    try:
        filas = etapa3_filas_timeline(vp)
    except Exception as e:
        return {"msg": f"❌ No se pudo leer el timeline: {e}", "rows": []}
    if not filas:
        return {"msg": "Aún no hay timeline. Pulsa «📊 Construir timeline».", "rows": []}
    n_lip = sum(1 for f in filas if f["tipo"] == "lipsync_15s")
    n_ok = sum(1 for f in filas if f["tipo"] == "lipsync_15s" and f["estado"] == "completado")
    rows = []
    for f in filas:
        es_lip = (f["tipo"] == "lipsync_15s")
        md = (f"**{f['clip_id']} · {_TIPO_CLIP.get(f['tipo'], f['tipo'])}** · "
              f"{f['start']:.1f}s → {f['end']:.1f}s ({f['dur']:.1f}s) · "
              f"{_ESTADO_CLIP.get(f['estado'], f['estado'])}")
        rows.append({"clip_id": f["clip_id"], "md": md, "video": f["video"], "lipsync": es_lip})
    return {"msg": f"{len(filas)} clips en el timeline — lipsync LTX listos: **{n_ok}/{n_lip}**. "
                   f"Los planos de 5s son compartidos: se regeneran en «Regenerar clip 5s específico».",
            "rows": rows}

# ── Regenerar un clip lipsync ────────────────────────────────
def ui_regen_clip_dialogo(etapa, idx, tpl, personaje):
    if not _GPU_LOCK.acquire(blocking=False):
        return _BUSY_MSG
    try:
        vp = _get_vp(); tpl = tpl or _CUR_TPL
        if not _LTX_PATHS:
            return "❌ Descarga LTX."
        return regenerar_clip_lipsync_fragmento(vp, tpl, personaje, etapa, int(idx), _LTX_PATHS)
    except Exception as e:
        import traceback
        log(f"[UI regen lipsync {etapa} #{idx}] ❌ {e}\n{traceback.format_exc()[-1200:]}")
        return f"❌ Error: {e}"
    finally:
        _GPU_LOCK.release()

def ui_regen_clip_cancion(clip_id, tpl, personaje):
    if not _GPU_LOCK.acquire(blocking=False):
        return _BUSY_MSG
    try:
        vp = _get_vp(); tpl = tpl or _CUR_TPL
        if not _LTX_PATHS:
            return "❌ Descarga LTX."
        out, reens = etapa3_regenerar_clip_lipsync(vp, tpl, personaje, clip_id, _LTX_PATHS)
        extra = ("🎞️ Canción reensamblada." if reens else
                 "ℹ️ Aún faltan lipsync por renderizar: pulsa «▶️ Renderizar Etapa 3» para ensamblar.")
        return f"✅ Lipsync {clip_id} regenerado.\n{extra}"
    except Exception as e:
        import traceback
        log(f"[UI regen lipsync canción {clip_id}] ❌ {e}\n{traceback.format_exc()[-1200:]}")
        return f"❌ Error: {e}"
    finally:
        _GPU_LOCK.release()

def _mk_regen_dialogo(etapa, idx):
    def _fn(tpl, personaje):
        return ui_regen_clip_dialogo(etapa, idx, tpl, personaje)
    return _fn

def _mk_regen_cancion(clip_id):
    def _fn(tpl, personaje):
        return ui_regen_clip_cancion(clip_id, tpl, personaje)
    return _fn

# ── Timelines dinámicos (se llaman DENTRO del gr.Blocks) ─────
def _make_dialog_timeline(etapa, tick, status_box, tpl_c, pj_c):
    @gr.render(inputs=[tick])
    def _render_dialog_tl(_t):
        data = ui_filas_timeline_dialogo(etapa)
        gr.Markdown(data["msg"])
        for r in data["rows"]:
            with gr.Row(equal_height=True):
                with gr.Column(scale=5, min_width=260):
                    gr.Markdown(r["md"])
                with gr.Column(scale=3, min_width=220):
                    if r["video"]:
                        gr.Video(value=r["video"], height=150, interactive=False, show_label=False)
                    elif r["audio"]:
                        gr.Audio(value=r["audio"], type="filepath", interactive=False, show_label=False)
                with gr.Column(scale=2, min_width=170):
                    btn = gr.Button("🔄 Regenerar lipsync (LTX)", size="sm", variant="secondary")
                    btn.click(fn=_mk_regen_dialogo(etapa, r["idx"]),
                              inputs=[tpl_c, pj_c], outputs=[status_box]
                    ).then(fn=_bump, inputs=[tick], outputs=[tick])

def _make_song_timeline(tick, status_box, tpl_c, pj_c):
    @gr.render(inputs=[tick])
    def _render_song_tl(_t):
        data = ui_filas_timeline_cancion()
        gr.Markdown(data["msg"])
        for r in data["rows"]:
            if not r["lipsync"]:
                gr.Markdown(r["md"])
                continue
            with gr.Row(equal_height=True):
                with gr.Column(scale=5, min_width=260):
                    gr.Markdown(r["md"])
                with gr.Column(scale=3, min_width=220):
                    if r["video"]:
                        gr.Video(value=r["video"], height=150, interactive=False, show_label=False)
                with gr.Column(scale=2, min_width=170):
                    btn = gr.Button("🔄 Regenerar lipsync (LTX)", size="sm", variant="secondary")
                    btn.click(fn=_mk_regen_cancion(r["clip_id"]),
                              inputs=[tpl_c, pj_c], outputs=[status_box]
                    ).then(fn=_bump, inputs=[tick], outputs=[tick])

# ═══════════════════════════════════════════════════════════════
# v8 — CLIPS 5s LEJANOS: ver · regenerar · subir desde el PC
#   (código NUEVO y aislado: no modifica ninguna función existente)
#   - Lista TODOS los far_*.mp4 de planos_5s con su video.
#   - Regenerar: genera en un archivo temporal FUERA de planos_5s y solo si
#     sale bien reemplaza el clip (si LTX falla se conserva el anterior).
#   - Subir: normaliza el clip del PC (1152x704, 24 fps, sin audio, 5.0 s)
#     para que el render lo trate igual que uno generado por LTX.
# ═══════════════════════════════════════════════════════════════
import tempfile as _tf5

_CLIP5_W, _CLIP5_H, _CLIP5_FPS, _CLIP5_DUR = 1152, 704, 24, 5.0

def _clips5s_dir(vp):
    d = os.path.join(vp.video_dir, "etapa3_cancion", "planos_5s")
    os.makedirs(d, exist_ok=True)
    return d

def _clip5s_key(path):
    """'far_E_v1.mp4' → 'E' (None si el nombre no sigue el patrón far_<E|F|G>_...)."""
    m = re.match(r"far_([EFG])_", os.path.basename(path), re.I)
    return m.group(1).upper() if m else None

def _clip5s_normalizar(src, dst):
    """Convierte un video del PC al formato de los clips LTX y lo deja en dst (atómico)."""
    import subprocess as _sp5
    tmpd = _tf5.mkdtemp(prefix="clip5s_")
    try:
        tmp = os.path.join(tmpd, "norm.mp4")
        vf = (f"scale={_CLIP5_W}:{_CLIP5_H}:force_original_aspect_ratio=increase,"
              f"crop={_CLIP5_W}:{_CLIP5_H},setsar=1,fps={_CLIP5_FPS},"
              f"tpad=stop_mode=clone:stop_duration={_CLIP5_DUR:.3f}")
        r = _sp5.run(["ffmpeg", "-y", "-i", src, "-an", "-vf", vf,
                      "-t", f"{_CLIP5_DUR:.6f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                      "-crf", "18", "-preset", "fast", tmp],
                     capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) < 10000:
            raise RuntimeError("FFmpeg no pudo convertir el clip:\n" + (r.stderr or "")[-800:])
        d = _ffprobe_duration(tmp) or 0.0
        if d < _CLIP5_DUR - 0.2:
            raise RuntimeError(f"El clip convertido dura {d:.2f}s (se esperaban {_CLIP5_DUR}s).")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(tmp, dst)
        return dst
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)

def _clip5s_borrar_preview(clip_path):
    p = clip_path.replace(".mp4", "_preview.png")
    if os.path.exists(p):
        try: os.remove(p)
        except Exception: pass

def ui_filas_clips5s():
    try:
        vp = _get_vp()
    except Exception:
        return {"msg": "Crea o carga un video para ver los clips 5s.", "rows": []}
    try:
        d = os.path.join(vp.video_dir, "etapa3_cancion", "planos_5s")
        files = (sorted(f for f in os.listdir(d) if f.lower().endswith(".mp4"))
                 if os.path.isdir(d) else [])
        rows = []
        for f in files:
            p = os.path.join(d, f)
            dur = _ffprobe_duration(p) or 0.0
            kb = os.path.getsize(p) // 1024
            rows.append({"path": p, "key": _clip5s_key(p),
                         "md": f"**{f}** · {dur:.1f}s · {kb} KB"})
        if not rows:
            return {"msg": "Aún no hay clips 5s. Genéralos arriba o sube los tuyos aquí abajo.", "rows": []}
        return {"msg": f"{len(rows)} clips 5s lejanos. Regenerar o reemplazar un clip conserva su nombre, "
                       f"así que el timeline ya construido lo usa automáticamente. "
                       f"Un clip NUEVO (subido) entra en el próximo «📊 Construir timeline».",
                "rows": rows}
    except Exception as e:
        return {"msg": f"❌ No se pudo leer planos_5s: {e}", "rows": []}

def ui_regen_clip5s_fila(clip_path, personaje, seed_v):
    if not _GPU_LOCK.acquire(blocking=False):
        return _BUSY_MSG
    try:
        vp = _get_vp()
        if not _LTX_PATHS:
            return "❌ Descarga LTX."
        key = _clip5s_key(clip_path)
        if not key:
            return ("❌ Este clip no viene de una imagen E/F/G (nombre sin patrón far_<E|F|G>_...). "
                    "Reemplázalo subiendo otro clip.")
        idx = {"E": 1, "F": 2, "G": 3}[key]
        img_path = os.path.join(vp.video_dir, "etapa1_imagenes", f"concursante_plano_lejano_{idx}.png")
        if not os.path.exists(img_path):
            return f"❌ La imagen base {key} no existe. Genera la Etapa 1 primero."
        try:
            seed = int(seed_v) if seed_v not in (None, "") else -1
        except Exception:
            seed = -1
        prompt_str, direccion = render_prompt_5s(personaje)
        tmpd = _tf5.mkdtemp(prefix="regen5s_")
        try:
            tmp = os.path.join(tmpd, os.path.basename(clip_path))
            generar_clip_5s(vp, img_path, prompt_str, tmp, _LTX_PATHS, seed=seed,
                            fps=_CLIP5_FPS, width=_CLIP5_W, height=_CLIP5_H)
            if not os.path.exists(tmp) or os.path.getsize(tmp) < 10000:
                raise RuntimeError("El clip regenerado no es válido; se conserva el anterior.")
            shutil.move(tmp, clip_path)
        finally:
            shutil.rmtree(tmpd, ignore_errors=True)
        _clip5s_borrar_preview(clip_path)
        log(f"[5s REGEN UI] {os.path.basename(clip_path)} regenerado ({direccion})")
        return f"✅ {os.path.basename(clip_path)} regenerado ({direccion})."
    except Exception as e:
        import traceback
        log(f"[5s REGEN UI] ❌ {e}\n{traceback.format_exc()[-1000:]}")
        return f"❌ Error (se conserva el clip anterior): {e}"
    finally:
        _GPU_LOCK.release()

def ui_reemplazar_clip5s(clip_path, archivo):
    try:
        if not archivo:
            return "⚠️ Selecciona primero un video de tu PC."
        src = archivo if isinstance(archivo, str) else getattr(archivo, "name", str(archivo))
        _get_vp()
        _clip5s_normalizar(src, clip_path)
        _clip5s_borrar_preview(clip_path)
        log(f"[5s SUBIR UI] {os.path.basename(clip_path)} reemplazado por clip propio")
        return f"✅ {os.path.basename(clip_path)} reemplazado por tu clip (convertido a {_CLIP5_W}x{_CLIP5_H}, 5s)."
    except Exception as e:
        return f"❌ Error: {e}"

def ui_agregar_clip5s(key, archivo):
    try:
        if not archivo:
            return "⚠️ Selecciona primero un video de tu PC."
        if key not in ("E", "F", "G"):
            return "❌ Elige E, F o G."
        src = archivo if isinstance(archivo, str) else getattr(archivo, "name", str(archivo))
        vp = _get_vp()
        d = _clips5s_dir(vp)
        n = 1
        while os.path.exists(os.path.join(d, f"far_{key}_u{n}.mp4")):
            n += 1
        dst = os.path.join(d, f"far_{key}_u{n}.mp4")
        _clip5s_normalizar(src, dst)
        vp.estado.setdefault("etapa3", {}).setdefault("clips_5s", [])
        if dst not in vp.estado["etapa3"]["clips_5s"]:
            vp.estado["etapa3"]["clips_5s"].append(dst)
        vp.save()
        log(f"[5s SUBIR UI] clip propio añadido → {dst}")
        return (f"✅ Clip añadido: {os.path.basename(dst)}. "
                f"Pulsa «📊 Construir timeline» para que entre en la distribución.")
    except Exception as e:
        return f"❌ Error: {e}"

def _mk_regen_clip5s(clip_path):
    def _fn(personaje, seed_v):
        return ui_regen_clip5s_fila(clip_path, personaje, seed_v)
    return _fn

def _mk_reemplazar_clip5s(clip_path):
    def _fn(archivo):
        return ui_reemplazar_clip5s(clip_path, archivo)
    return _fn

# ── Generar un clip EXTRA (E+ / F+ / G+) con la misma imagen ──
def ui_gen_5s_extra(img_key, personaje, seed_v):
    """Genera un clip 5s ADICIONAL de la imagen E/F/G: usa el primer far_<K>_v<N>.mp4 libre
    (N desde 2). Nunca sobrescribe ni toca los clips existentes. Retorna (status, gallery)."""
    if not _GPU_LOCK.acquire(blocking=False):
        return _BUSY_MSG, []
    try:
        vp = _get_vp()
        if not _LTX_PATHS:
            return "❌ Descarga primero los modelos LTX.", []
        idx = {"E": 1, "F": 2, "G": 3}.get(img_key)
        if not idx:
            return f"❌ Key inválida: {img_key}", []
        img_path = os.path.join(vp.video_dir, "etapa1_imagenes", f"concursante_plano_lejano_{idx}.png")
        if not os.path.exists(img_path):
            return f"❌ Imagen {img_key} no existe. Genera Etapa 1 primero.", []
        try:
            seed = int(seed_v) if seed_v not in (None, "") else -1
        except Exception:
            seed = -1

        out_dir = _clips5s_dir(vp)
        n = 2
        while os.path.exists(os.path.join(out_dir, f"far_{img_key}_v{n}.mp4")):
            n += 1
        out_mp4 = os.path.join(out_dir, f"far_{img_key}_v{n}.mp4")

        prompt_str, direccion = render_prompt_5s(personaje)
        tmpd = _tf5.mkdtemp(prefix="extra5s_")
        try:
            tmp = os.path.join(tmpd, os.path.basename(out_mp4))
            generar_clip_5s(vp, img_path, prompt_str, tmp, _LTX_PATHS, seed=seed,
                            fps=_CLIP5_FPS, width=_CLIP5_W, height=_CLIP5_H)
            if not os.path.exists(tmp) or os.path.getsize(tmp) < 10000:
                raise RuntimeError("El clip generado no es válido; no se añadió nada.")
            shutil.move(tmp, out_mp4)
        finally:
            shutil.rmtree(tmpd, ignore_errors=True)

        vp.estado.setdefault("etapa3", {}).setdefault("clips_5s", [])
        if out_mp4 not in vp.estado["etapa3"]["clips_5s"]:
            vp.estado["etapa3"]["clips_5s"].append(out_mp4)
        vp.save()
        log(f"[5s EXTRA] {img_key}+ generado → {out_mp4} ({direccion})")

        gallery_items = [(img_path, f"Ref {img_key}")]
        preview_png = out_mp4.replace(".mp4", "_preview.png")
        try:
            import subprocess as _sp5
            _sp5.run(["ffmpeg", "-y", "-i", out_mp4, "-vframes", "1", "-q:v", "2", preview_png],
                     capture_output=True, timeout=15)
        except Exception:
            pass
        if os.path.exists(preview_png):
            gallery_items.append((preview_png, f"{os.path.basename(out_mp4)} — {os.path.getsize(out_mp4)//1024}KB"))
        return (f"✅ Clip extra {img_key}+ generado ({direccion}) → {os.path.basename(out_mp4)}. "
                f"Pulsa «📊 Construir timeline» para que entre en la distribución."), gallery_items
    except Exception as e:
        import traceback
        log(f"[5s EXTRA] ❌ {e}\n{traceback.format_exc()[-1000:]}")
        return f"❌ Error: {e}", []
    finally:
        _GPU_LOCK.release()

def _make_gen5s_extra_fn(key):
    def _fn(personaje, seed_v):
        return ui_gen_5s_extra(key, personaje, seed_v)
    return _fn

# ── Lista dinámica (se llama DENTRO del gr.Blocks, igual que _make_song_timeline) ──
def _make_clips5s_panel(tick, status_box, pj_c, seed_c):
    @gr.render(inputs=[tick])
    def _render_clips5s(_t):
        data = ui_filas_clips5s()
        gr.Markdown(data["msg"])
        for r in data["rows"]:
            with gr.Row(equal_height=True):
                with gr.Column(scale=4, min_width=240):
                    gr.Markdown(r["md"])
                    up = gr.File(label="Reemplazar con un clip de mi PC",
                                 file_types=[".mp4", ".mov", ".webm", ".mkv", ".avi"],
                                 type="filepath")
                with gr.Column(scale=3, min_width=220):
                    gr.Video(value=r["path"], height=150, interactive=False, show_label=False)
                with gr.Column(scale=2, min_width=170):
                    b_regen = gr.Button("🔄 Regenerar (LTX)", size="sm", variant="secondary")
                    b_regen.click(fn=_mk_regen_clip5s(r["path"]),
                                  inputs=[pj_c, seed_c], outputs=[status_box]
                    ).then(fn=_bump, inputs=[tick], outputs=[tick])
                    b_up = gr.Button("📤 Usar mi clip", size="sm", variant="secondary")
                    b_up.click(fn=_mk_reemplazar_clip5s(r["path"]),
                               inputs=[up], outputs=[status_box]
                    ).then(fn=_bump, inputs=[tick], outputs=[tick])

# ── AUTOMATIZACIÓN por etapas ────────────────────────────────
_MODOS_DIALOGO = {
    "Completo (audio + lipsync LTX)": "completo",
    "Solo preview con imágenes (sin lipsync)": "preview",
}
_MODOS_GUION = {
    "Reutilizar guion si existe (si no, generar nuevo)": "auto",
    "Generar guion NUEVO": "nuevo",
    "Usar guion ACTUAL (editado)": "actual",
}

def ui_automatizar(tpl, personaje, historia, model_e2, model_e4, fast_e2, fast_e4,
                   inc_e2, inc_cancion, inc_e4, modo_dialogo_lbl, modo_guion_lbl,
                   reconstruir_tl, render_final):
    # Generador: va mostrando el progreso. Cada etapa es independiente: si una falla se anota
    # y se continúa con las demás. Orden: Etapa 2 → Canción → Etapa 4.
    if not (inc_e2 or inc_cancion or inc_e4):
        yield "❌ Marca al menos una etapa.", gr.update()
        return
    if not _GPU_LOCK.acquire(blocking=False):
        yield _BUSY_MSG, gr.update()
        return
    lines, last_video = [], None
    def _snap():
        return "\n".join(lines)
    try:
        vp = _get_vp(); tpl = tpl or _CUR_TPL
        modo_d = _MODOS_DIALOGO.get(modo_dialogo_lbl, "completo")
        modo_g = _MODOS_GUION.get(modo_guion_lbl, "auto")
        preview = (modo_d == "preview")
        necesita_ltx = bool(inc_cancion) or (bool(inc_e2 or inc_e4) and not preview)
        if necesita_ltx and not _LTX_PATHS:
            yield "❌ Descarga/inicializa LTX-2.5 primero (hace falta para lipsync y canción).", gr.update()
            return

        nombres = [n for n, a in (("Etapa 2", inc_e2), ("Canción", inc_cancion), ("Etapa 4", inc_e4)) if a]
        lines.append("📋 Plan: " + " → ".join(nombres) +
                     (f" | diálogos: {'PREVIEW con imágenes' if preview else 'completo con lipsync'}"
                      if (inc_e2 or inc_e4) else ""))
        yield _snap(), gr.update()

        def _run_dialog(etapa, hist, mid, fast):
            return ejecutar_etapa_dialogos(
                vp, tpl, personaje, etapa, _LTX_PATHS or {},
                historia_base=hist or "", model_id=mid, fast_mode_tts=bool(fast),
                regenerar_guion=(modo_g == "nuevo"), usar_guion_actual=(modo_g == "actual"),
                solo_preview=preview)

        def _run_cancion():
            song = vp.estado.get("etapa3", {}).get("song_wav", "")
            if not song or not os.path.exists(song):
                raise RuntimeError("Falta la canción: súbela y pulsa «💾 Guardar canción».")
            far5_dir = os.path.join(vp.video_dir, "etapa3_cancion", "planos_5s")
            tiene5 = os.path.isdir(far5_dir) and any(f.lower().endswith(".mp4") for f in os.listdir(far5_dir))
            if not tiene5:
                log("[AUTO] No hay clips 5s → generando (PASO 3.1)...")
                etapa3_paso1_generar_clips_5s(vp, tpl, personaje, _LTX_PATHS)
            if reconstruir_tl or not vp.timeline_items:
                log("[AUTO] Construyendo timeline de la canción (PASO 3.2)...")
                estilo_tl = cargar_plantilla(tpl).get("estilo", "got_talent")
                if estilo_tl == "la_voz":
                    build_song_timeline_lavoz(vp, tpl, personaje, song_audio_path=song,
                                              use_far_planes=["E", "F", "G"],
                                              intro_seconds=float(intro_seconds or 30))
                else:
                    build_song_timeline(vp, tpl, personaje, song_audio_path=song,
                                        use_far_planes=["E", "F", "G"])
            return etapa3_paso3_renderizar_secuencia(vp, tpl, personaje, _LTX_PATHS)

        pasos = []
        if inc_e2:
            pasos.append(("Etapa 2", lambda: _run_dialog("etapa2", historia, model_e2, fast_e2)))
        if inc_cancion:
            pasos.append(("Canción", _run_cancion))
        if inc_e4:
            pasos.append(("Etapa 4", lambda: _run_dialog("etapa4", "", model_e4, fast_e4)))

        for nombre, fn in pasos:
            lines.append(f"▶️ {nombre}...")
            yield _snap(), gr.update()
            try:
                res = fn()
                if res and os.path.exists(str(res)):
                    last_video = res
                lines.append(f"   ✅ {nombre} lista → {os.path.basename(str(res))}")
            except Exception as e:
                import traceback
                log(f"[AUTO {nombre}] ❌ {e}\n{traceback.format_exc()[-1500:]}")
                lines.append(f"   ❌ {nombre}: {e}")
            yield _snap(), (last_video if last_video else gr.update())

        if render_final:
            if preview and (inc_e2 or inc_e4):
                lines.append("ℹ️ VIDEO FINAL omitido: en modo preview los videos son solo de validación.")
            else:
                try:
                    vf = render_video_final(vp, {"etapa2": bool(inc_e2), "etapa3": bool(inc_cancion),
                                                 "etapa4": bool(inc_e4)})
                    last_video = vf
                    lines.append(f"🏁 VIDEO FINAL → {os.path.basename(vf)}")
                except Exception as e:
                    lines.append(f"❌ VIDEO FINAL: {e}")
        lines.append("✔️ Automatización terminada.")
        yield _snap(), (last_video if last_video else gr.update())
    except Exception as e:
        import traceback
        log(f"[AUTO] ❌ {e}\n{traceback.format_exc()[-1500:]}")
        lines.append(f"❌ Error: {e}")
        yield _snap(), gr.update()
    finally:
        _GPU_LOCK.release()

def ui_auto_e2_e4(tpl, personaje, historia, model_e2, model_e4, fast_e2, fast_e4,
                  modo_dialogo_lbl, modo_guion_lbl, render_final):
    yield from ui_automatizar(tpl, personaje, historia, model_e2, model_e4, fast_e2, fast_e4,
                              True, False, True, modo_dialogo_lbl, modo_guion_lbl, False, render_final)

def ui_auto_solo_cancion(tpl, personaje, reconstruir_tl):
    yield from ui_automatizar(tpl, personaje, "", "", "", False, False,
                              False, True, False, "", "", reconstruir_tl, False)

# ═══════════════════════════════════════════════════════════════
# v8 — PARCHE ADITIVO (solo UI). No modifica la lógica de las celdas anteriores:
#   1) Selección múltiple + «Regenerar seleccionados» en los timelines de E2, E3 y E4.
#   2) Contexto (prompt) para el guion de la Etapa 4 y casillas de contexto en Automatización.
# Cómo funciona: envuelve por nombre funciones ya existentes y SOLO actúa si hay un contexto
# escrito. Con las casillas vacías el comportamiento es idéntico al de antes.
# Las definiciones originales de _make_dialog_timeline / _make_song_timeline quedan arriba
# sin tocar; las de este bloque (mismo nombre) las sustituyen al construir la UI.
# ═══════════════════════════════════════════════════════════════
_CTX = {"e4_ui": "", "e2_auto": "", "e4_auto": "", "auto_run": False}

def _ctx_setter(clave):
    def _fn(txt):
        _CTX[clave] = (txt or "").strip()
    return _fn

def _ctx_cargar():
    # Al (re)cargar la página muestra lo que está realmente activo (evita contextos "fantasma").
    return _CTX["e4_ui"], _CTX["e2_auto"], _CTX["e4_auto"]

def _ctx_e4_activo():
    # Dentro de una automatización manda la casilla de la automatización; si está vacía,
    # se usa la de la sección Etapa 4.
    if _CTX["auto_run"] and _CTX["e4_auto"]:
        return _CTX["e4_auto"]
    return _CTX["e4_ui"]

# — Contexto para el guion de la ETAPA 4 (el prompt original no admitía historia) —
if not getattr(build_dialogo_final_system_prompt, "_ctx_wrapped", False):
    _build_final_orig = build_dialogo_final_system_prompt
    def _build_final_ctx(personaje_key, *a, **k):
        base = _build_final_orig(personaje_key, *a, **k)
        ctx = _ctx_e4_activo()
        if not ctx:
            return base
        marca = "REGLAS OBLIGATORIAS E INVIOLABLES:"
        bloque = f"Contexto a incorporar (historia / actuación del concursante): {ctx}\n\n"
        log(f"[CTX E4] Contexto aplicado al guion de la Etapa 4 ({len(ctx)} caracteres).")
        return base.replace(marca, bloque + marca, 1) if marca in base else base + "\n\n" + bloque
    _build_final_ctx._ctx_wrapped = True
    build_dialogo_final_system_prompt = _build_final_ctx

# — Contexto de la ETAPA 2 desde Automatización (solo si esa casilla tiene texto) —
if not getattr(build_dialogo_inicial_system_prompt, "_ctx_wrapped", False):
    _build_inicial_orig = build_dialogo_inicial_system_prompt
    def _build_inicial_ctx(personaje_key, historia_base="", *a, **k):
        if _CTX["auto_run"] and _CTX["e2_auto"]:
            historia_base = _CTX["e2_auto"]
            log(f"[CTX E2] Contexto de Automatización aplicado ({len(historia_base)} caracteres).")
        return _build_inicial_orig(personaje_key, historia_base, *a, **k)
    _build_inicial_ctx._ctx_wrapped = True
    build_dialogo_inicial_system_prompt = _build_inicial_ctx

# — Marca "estoy dentro de una automatización" (mismo nombre → el cableado existente lo recoge) —
import functools
if not getattr(ui_automatizar, "_ctx_wrapped", False):
    _ui_automatizar_base = ui_automatizar
    @functools.wraps(_ui_automatizar_base)
    def ui_automatizar(*args, **kwargs):
        _CTX["auto_run"] = True
        try:
            yield from _ui_automatizar_base(*args, **kwargs)
        finally:
            _CTX["auto_run"] = False
    ui_automatizar._ctx_wrapped = True

# — Regeneración por lote (reutiliza los handlers individuales ya existentes) —
def _regen_lote(sel, regen_fn, nombre_fn):
    if not sel:
        yield "ℹ️ No hay nada seleccionado. Marca uno o más con la casilla «Elegir»."
        return
    lineas = [f"🔄 Regenerando {len(sel)} seleccionado(s), uno tras otro (seed nueva en cada uno)..."]
    yield "\n".join(lineas)
    ok = 0
    for k, item in enumerate(sel, 1):
        lineas.append(f"▶️ ({k}/{len(sel)}) {nombre_fn(item)}...")
        yield "\n".join(lineas)
        res = str(regen_fn(item) or "").strip()
        if res == _BUSY_MSG:
            lineas.append(f"{_BUSY_MSG}\n⛔ Lote detenido.")
            yield "\n".join(lineas)
            return
        bien = res.startswith("✅")
        ok += 1 if bien else 0
        primera = (res.splitlines() or [""])[0][:200] or ("✅ listo" if bien else "❌ sin respuesta")
        lineas.append("   " + primera)
        yield "\n".join(lineas)
    lineas.append(f"✔️ Lote terminado: {ok}/{len(sel)} regenerados.")
    yield "\n".join(lineas)

def _mk_regen_lote_dialogo(etapa, idxs):
    def _fn(tpl, personaje, *marcas):
        sel = [i for i, m in zip(idxs, marcas) if m]
        yield from _regen_lote(sel, lambda i: ui_regen_clip_dialogo(etapa, i, tpl, personaje),
                               lambda i: f"fragmento #{int(i):02d}")
    return _fn

def _mk_regen_lote_cancion(clip_ids):
    def _fn(tpl, personaje, *marcas):
        sel = [c for c, m in zip(clip_ids, marcas) if m]
        yield from _regen_lote(sel, lambda c: ui_regen_clip_cancion(c, tpl, personaje),
                               lambda c: f"clip {c}")
    return _fn

def _mk_marcar(n, valor):
    def _fn():
        return tuple(valor for _ in range(n)) if n > 1 else valor
    return _fn

def _barra_lote(barra, chks, fn_lote, tick, status_box, tpl_c, pj_c):
    if not chks:
        return
    try:
        with barra:
            b_all  = gr.Button("☑️ Marcar todos", size="sm", variant="secondary")
            b_none = gr.Button("⬜ Desmarcar todos", size="sm", variant="secondary")
            b_go   = gr.Button("🔄 Regenerar seleccionados (LTX)", size="sm", variant="primary")
        b_all.click(fn=_mk_marcar(len(chks), True), inputs=[], outputs=chks)
        b_none.click(fn=_mk_marcar(len(chks), False), inputs=[], outputs=chks)
        b_go.click(fn=fn_lote, inputs=[tpl_c, pj_c] + chks, outputs=[status_box]
                   ).then(fn=_bump, inputs=[tick], outputs=[tick])
    except Exception as e:
        log(f"[UI v8] ⚠ No se pudo crear la barra de selección múltiple: {e}")

# — Timelines con casilla «Elegir» por fragmento (mismas firmas que los originales) —
def _make_dialog_timeline(etapa, tick, status_box, tpl_c, pj_c):
    @gr.render(inputs=[tick])
    def _render_dialog_tl(_t):
        data = ui_filas_timeline_dialogo(etapa)
        gr.Markdown(data["msg"])
        barra = gr.Row() if data["rows"] else None      # barra superior; se rellena al final
        chks = []
        for r in data["rows"]:
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=80):
                    chks.append(gr.Checkbox(label="Elegir", value=False))
                with gr.Column(scale=5, min_width=260):
                    gr.Markdown(r["md"])
                with gr.Column(scale=3, min_width=220):
                    if r["video"]:
                        gr.Video(value=r["video"], height=150, interactive=False, show_label=False)
                    elif r["audio"]:
                        gr.Audio(value=r["audio"], type="filepath", interactive=False, show_label=False)
                with gr.Column(scale=2, min_width=170):
                    btn = gr.Button("🔄 Regenerar lipsync (LTX)", size="sm", variant="secondary")
                    btn.click(fn=_mk_regen_dialogo(etapa, r["idx"]),
                              inputs=[tpl_c, pj_c], outputs=[status_box]
                    ).then(fn=_bump, inputs=[tick], outputs=[tick])
        _barra_lote(barra, chks, _mk_regen_lote_dialogo(etapa, [r["idx"] for r in data["rows"]]),
                    tick, status_box, tpl_c, pj_c)

def _make_song_timeline(tick, status_box, tpl_c, pj_c):
    @gr.render(inputs=[tick])
    def _render_song_tl(_t):
        data = ui_filas_timeline_cancion()
        gr.Markdown(data["msg"])
        barra = gr.Row() if any(r["lipsync"] for r in data["rows"]) else None
        chks, ids = [], []
        for r in data["rows"]:
            if not r["lipsync"]:
                gr.Markdown(r["md"])
                continue
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=80):
                    chks.append(gr.Checkbox(label="Elegir", value=False))
                    ids.append(r["clip_id"])
                with gr.Column(scale=5, min_width=260):
                    gr.Markdown(r["md"])
                with gr.Column(scale=3, min_width=220):
                    if r["video"]:
                        gr.Video(value=r["video"], height=150, interactive=False, show_label=False)
                with gr.Column(scale=2, min_width=170):
                    btn = gr.Button("🔄 Regenerar lipsync (LTX)", size="sm", variant="secondary")
                    btn.click(fn=_mk_regen_cancion(r["clip_id"]),
                              inputs=[tpl_c, pj_c], outputs=[status_box]
                    ).then(fn=_bump, inputs=[tick], outputs=[tick])
        _barra_lote(barra, chks, _mk_regen_lote_cancion(ids), tick, status_box, tpl_c, pj_c)


# ═══════════════════════════════════════════════════════════════
# v9 — PARCHE ANTI-REGENERACIÓN (solo UI, aditivo; no cambia ninguna otra celda)
#   Regla: un botón de etapa SOLO genera clips si esa etapa NO tiene ninguno.
#   Si ya tiene clips → únicamente ENSAMBLA/RENDERIZA lo que existe.
#   Para rehacer algo se usan los botones de regenerar (individual o «seleccionados»).
#   Cómo: envuelve por nombre 2 funciones de pipeline ya existentes. Las originales
#   quedan intactas y se llaman tal cual cuando la etapa está vacía.
# ═══════════════════════════════════════════════════════════════
def _v9_clip_ok(p):
    try:
        return bool(p) and os.path.exists(p) and os.path.getsize(p) > 10000
    except Exception:
        return False

# ── Etapa 3 (canción): render = ensamblar, nunca regenerar lo que ya existe ──────────
if not getattr(etapa3_paso3_renderizar_secuencia, "_v9", False):
    _v9_orig_render3 = etapa3_paso3_renderizar_secuencia
    _V9_MARK = "v9-reuse"

    def etapa3_paso3_renderizar_secuencia(vp, plantilla_slug, personaje_key, ltx_paths,
                                          fps=24, forzar_lipsync=False):
        _orig = lambda: _v9_orig_render3(vp, plantilla_slug, personaje_key, ltx_paths, fps, forzar_lipsync)
        song_wav = vp.estado.get("etapa3", {}).get("song_wav", "")
        if forzar_lipsync or not song_wav or not os.path.exists(song_wav) or not vp.timeline_items:
            return _orig()                       # forzado, o el original ya da su error claro
        clips_dir = os.path.join(vp.video_dir, "etapa3_cancion", "clips_renderizados")
        song_dur = _ffprobe_duration(song_wav) or 0.0
        existentes, faltan, t = [], [], 0.0
        for it in vp.timeline_items:
            dur = float(it.get("dur_s", 0))
            if it.get("tipo") == "lipsync_15s":
                if _v9_clip_ok(os.path.join(clips_dir, f"{it['clip_id']}.mp4")):
                    existentes.append(it)
                elif min(dur, max(0.1, song_dur - t)) >= 0.5:   # igual que el render: sin audio no cuenta
                    faltan.append(it["clip_id"])
            t += dur
        if not existentes:
            return _orig()                       # etapa vacía → genera normalmente
        if faltan:
            raise RuntimeError(
                f"La canción ya tiene {len(existentes)} lipsync, pero faltan {len(faltan)} "
                f"({', '.join(faltan[:8])}{'…' if len(faltan) > 8 else ''}). No se genera nada "
                f"automáticamente: márcalos en el timeline y pulsa «Regenerar seleccionados».")
        # Solo ensamblar: los lipsync existentes se marcan «vigentes» durante esta llamada
        guardado = [(it, it.get("estado"), it.get("sig")) for it in existentes]
        sig_prev = globals()["_sig_lipsync_cancion"]
        globals()["_sig_lipsync_cancion"] = lambda *a, **k: _V9_MARK
        for it in existentes:
            it["estado"], it["sig"] = "completado", _V9_MARK
        try:
            log("[PASO 3.3] Lipsync ya generados → solo se ensambla (no se regenera nada).")
            return _v9_orig_render3(vp, plantilla_slug, personaje_key, ltx_paths, fps, False)
        finally:
            globals()["_sig_lipsync_cancion"] = sig_prev
            for it, est, sg in guardado:         # deja el estado exactamente como estaba
                it["estado"] = est
                if sg is None:
                    it.pop("sig", None)
                else:
                    it["sig"] = sg
            try:
                vp.save()
            except Exception:
                pass
    etapa3_paso3_renderizar_secuencia._v9 = True

# ── Etapas 2 y 4 (diálogos): si ya hay clips lipsync, solo ensamblar ────────────────
#    «Ejecutar completa (guion NUEVO)» y el PREVIEW pasan directo al original (sin cambios).
if not getattr(ejecutar_etapa_dialogos, "_v9", False):
    _v9_orig_dialogos = ejecutar_etapa_dialogos

    def ejecutar_etapa_dialogos(vp, plantilla_slug, personaje_key, etapa, ltx_paths, *args, **kw):
        if kw.get("regenerar_guion") or kw.get("solo_preview"):
            return _v9_orig_dialogos(vp, plantilla_slug, personaje_key, etapa, ltx_paths, *args, **kw)
        guion = (vp.estado.get(etapa, {}) or {}).get("guion_json") or {}
        items = _items_validos_guion(guion)
        hay = [it for it in items
               if _v9_clip_ok(_frag_paths(vp, etapa, it["idx"], it["hablante"])["mp4"])]
        if not hay:
            return _v9_orig_dialogos(vp, plantilla_slug, personaje_key, etapa, ltx_paths, *args, **kw)
        if len(hay) < len(items):
            falt = [it["idx"] for it in items if it not in hay]
            raise RuntimeError(
                f"{etapa} ya tiene {len(hay)} clip(s) lipsync, pero faltan {len(falt)} "
                f"(fragmentos {', '.join('#%d' % i for i in falt[:8])}{'…' if len(falt) > 8 else ''}). "
                f"No se genera nada automáticamente: usa «🔄 Regenerar lipsync» o "
                f"«Regenerar seleccionados» en esos fragmentos.")
        log(f"[{etapa.upper()}] Ya tiene {len(hay)} clips lipsync → solo se ensambla (no se regenera nada).")
        _reensamblar_si_completo(vp, etapa, guion, kw.get("fps", 24))
        return (vp.estado.get(etapa, {}) or {}).get("video_final", "")
    ejecutar_etapa_dialogos._v9 = True


# ═══════════════════════════════════════════════════════════════
# CSS AIQUEST MODERNO
# ═══════════════════════════════════════════════════════════════
AIQUEST_CSS = '''
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
* { font-family: "Inter", sans-serif !important; }
.gradio-container { max-width: 1120px !important; margin: auto !important; }
.brand-header {
    text-align: center;
    background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
    padding: 28px; border-radius: 15px; margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(106,17,203,0.3);
}
.brand-title   { color: white; font-size: 2em; font-weight: 700; margin: 0 0 6px 0; }
.brand-subtitle{ color: rgba(255,255,255,0.88); font-size: 1em; margin-bottom: 12px; }
.voice-section {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 2px solid #6a11cb; border-radius: 12px; padding: 16px; margin: 12px 0;
}
button.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important; font-weight: 600 !important; border-radius: 12px !important;
}
button.secondary { border-radius: 10px !important; }
.voice-validated { color: #4caf50 !important; font-weight: 700 !important; }
.voice-pending   { color: #ff9800 !important; font-weight: 600 !important; }
'''


# ═══════════════════════════════════════════════════════════════
# UI GRADIO v6
# ═══════════════════════════════════════════════════════════════
with gr.Blocks(css=AIQUEST_CSS, theme=gr.themes.Soft(),
               title="🎤 Audiciones — Got Talent / La Voz") as demo:

    gr.HTML('''<div class="brand-header">
        <div class="brand-title">🎤 Sistema de Audiciones</div>
        <div class="brand-subtitle">
            Generación granular · Upload manual · Voz clonada consistente ·
            LTX-2.5 Wan2GP · Reanudación automática
        </div>
    </div>''')

    gr.Markdown("---")

    # ────────────────────────────────────────────────────────
    # PLANTILLA + NUEVO VIDEO
    # ────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=3):
            tpl_dd = gr.Dropdown(label="📁 Plantilla activa", choices=[], value=None,
                                 allow_custom_value=True)
            with gr.Row():
                new_tpl_box   = gr.Textbox(label="Nueva plantilla", value="Mi Plantilla")
                new_tpl_style = gr.Dropdown(label="Estilo de plantilla",
                    choices=[("Got Talent (actual)", "got_talent"), ("La Voz (giros de jueces)", "la_voz")],
                    value="got_talent")
                btn_crear_tpl = gr.Button("➕ Crear",  variant="primary")
                btn_refresh_tpl = gr.Button("🔄")
        with gr.Column(scale=2):
            personaje    = gr.Dropdown(label="🎭 Concursante",
                                       choices=list(PERSONAJE_OPCIONES.keys()), value="Mujer")
            p_reemplazar = gr.Dropdown(label="🔄 Reemplazar en plantilla",
                                       choices=list(PERSONAJE_OPCIONES.keys()), value="Mujer")
            btn_nuevo_video = gr.Button("🎬 Crear NUEVO video", variant="primary")
    with gr.Row():
        status_main   = gr.Textbox(label="Estado",      interactive=False, scale=5)
        video_dir_box = gr.Textbox(label="Directorio",  value=_CUR_VP_DIR,
                                   interactive=False, scale=3)
    _main_tabs = gr.Tabs()
    _main_tabs.__enter__()

    # ────────────────────────────────────────────────────────
    # REANUDAR
    # ────────────────────────────────────────────────────────
    with gr.Tab("🔄 REANUDAR VIDEO EXISTENTE"):
        with gr.Row():
            video_selector = gr.Dropdown(
                label="📂 Videos disponibles", choices=[], value=None,
                allow_custom_value=True, scale=4)
            btn_cargar  = gr.Button("▶️ Cargar",   variant="primary",   scale=1)
            btn_refprog = gr.Button("🔄 Refrescar", variant="secondary", scale=1)
            btn_listar  = gr.Button("📋 Listar",   variant="secondary", scale=1)
        progreso_box     = gr.Textbox(label="Estado del proyecto", value="Cargando...",
                                      lines=18, interactive=False)
        progreso_gallery = gr.Gallery(label="Imágenes Etapa 1", columns=4, height=200)
        with gr.Row():
            video_activo_label = gr.Textbox(
                label="Video activo",
                value=os.path.basename(_CUR_VP_DIR) if _CUR_VP_DIR else "Ninguno",
                interactive=False, scale=4)
            btn_abrir = gr.Button("📁 Ruta", scale=1)
        ruta_label = gr.Textbox(label="Ruta completa", value=_CUR_VP_DIR, interactive=False)

    # ────────────────────────────────────────────────────────
    # ASSETS JUECES INDEPENDIENTES
    # ────────────────────────────────────────────────────────
    with gr.Tab("📂 Assets de Plantilla (Jueces independientes)"):
        gr.Markdown("### 👨‍⚖️ Jueces — Imagen + Audio(s) por separado")
        gr.Markdown(
            "⚠️ **Al subir audios de jueces, la transcripción previa se borra** "
            "(se re-transcribirá automáticamente al ejecutar Etapa 2/4 "
            "o usando el botón de abajo)."
        )
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**Juez 1**")
                j1_img = gr.File(label="Imagen", file_types=[".png",".jpg",".jpeg"])
                j1_aud = gr.File(label="Audio(s)", file_count="multiple",
                                 file_types=[".wav",".mp3",".m4a"])
            with gr.Column(scale=1):
                gr.Markdown("**Juez 2**")
                j2_img = gr.File(label="Imagen", file_types=[".png",".jpg",".jpeg"])
                j2_aud = gr.File(label="Audio(s)", file_count="multiple",
                                 file_types=[".wav",".mp3",".m4a"])
            with gr.Column(scale=1):
                gr.Markdown("**Juez 3**")
                j3_img = gr.File(label="Imagen", file_types=[".png",".jpg",".jpeg"])
                j3_aud = gr.File(label="Audio(s)", file_count="multiple",
                                 file_types=[".wav",".mp3",".m4a"])
            with gr.Column(scale=1):
                gr.Markdown("**Juez 4**")
                j4_img = gr.File(label="Imagen", file_types=[".png",".jpg",".jpeg"])
                j4_aud = gr.File(label="Audio(s)", file_count="multiple",
                                 file_types=[".wav",".mp3",".m4a"])
        gr.Markdown("---")
        with gr.Row():
            esc_cercano  = gr.File(label="plano_cercano.png",
                                   file_types=[".png",".jpg"])
            escs_lejanos = gr.File(label="planos_lejanos 1..3",
                                   file_count="multiple", file_types=[".png",".jpg"])
        with gr.Row():
            reac1 = gr.File(label="Reacciones 1ª mitad", file_count="multiple",
                            file_types=[".mp4"])
            reac2 = gr.File(label="Reacciones 2ª mitad", file_count="multiple",
                            file_types=[".mp4"])
        with gr.Accordion("🎤 Reacciones La Voz por juez (solo plantillas La Voz)", open=False):
            gr.Markdown("Sube clips separados por juez y estado. Cada juez necesita al menos un clip «Girando».")
            lv_uploads = []
            for j in range(1, 5):
                with gr.Row():
                    gr.Markdown(f"**Juez {j}**")
                    lv_uploads.extend([
                        gr.File(label="Antes de girarse", file_count="multiple", file_types=[".mp4"], key=f"lv{j}a"),
                        gr.File(label="Girándose (obligatorio)", file_count="multiple", file_types=[".mp4"], key=f"lv{j}g"),
                        gr.File(label="Ya girado", file_count="multiple", file_types=[".mp4"], key=f"lv{j}d"),
                    ])
        with gr.Row():
            btn_assets = gr.Button("💾 Guardar TODOS los assets", variant="secondary")
            btn_transcribe_judges = gr.Button("🗣️ Transcribir audios de jueces ahora",
                                              variant="secondary")
        assets_status    = gr.Textbox(label="Estado assets/transcripción",
                                      interactive=False, lines=8)

    # ────────────────────────────────────────────────────────
    # ETAPA 1 — IMÁGENES (sin cambios)
    # ────────────────────────────────────────────────────────
    with gr.Tab("🖼️ ETAPA 1 — Imágenes A→G (Granular + Upload)"):
        gr.Markdown("### ⚙️ Parámetros")
        with gr.Row():
            edad     = gr.Number(label="Edad", value=28, precision=0)
            vestuario = gr.Textbox(label="Vestuario", value="vestido elegante de gala")
            planos   = gr.Radio(label="Planos lejanos", choices=[1,2,3], value=2)
        with gr.Row():
            dq = gr.Dropdown(label="Diffusion quant",
                             choices=list(QWEN_IMAGE_DIFFUSION_QUANTS.keys()),
                             value=list(QWEN_IMAGE_DIFFUSION_QUANTS.keys())[3],
                             allow_custom_value=True)
            eq = gr.Dropdown(label="Encoder quant",
                             choices=list(QWEN_VL_ENCODERS.keys()),
                             value=list(QWEN_VL_ENCODERS.keys())[0],
                             allow_custom_value=True)
            mq = gr.Dropdown(label="mmproj quant",
                             choices=list(QWEN_VL_MMPROJ.keys()),
                             value=list(QWEN_VL_MMPROJ.keys())[0],
                             allow_custom_value=True)
        with gr.Row():
            stA   = gr.Slider(5,40,value=18,step=1,label="Steps A")
            stE   = gr.Slider(5,40,value=18,step=1,label="Steps edit")
            cfg_v = gr.Slider(1.0,10.0,value=6.0,step=0.5,label="CFG")
            seed_v = gr.Number(label="Seed (-1=random)", value=-1, precision=0)
        gr.Markdown("---")
        gr.Markdown("### 🔑 PASO A — Imagen Base")
        with gr.Row():
            upload_A = gr.File(label="📤 Subir A", file_types=[".png",".jpg",".jpeg"])
            btn_A    = gr.Button("▶️ Generar/Subir A", variant="primary")
            btn_cont_A = gr.Button("⏩ Continuar desde A", variant="secondary")
        status_A  = gr.Textbox(label="Estado A", interactive=False)
        preview_A = gr.Image(label="Preview A", height=200)

        gr.Markdown("### 🔑 PASO B — Cuerpo Completo")
        with gr.Row():
            upload_B = gr.File(label="📤 Subir B", file_types=[".png",".jpg",".jpeg"])
            btn_B    = gr.Button("▶️ Generar/Subir B", variant="primary")
            btn_cont_B = gr.Button("⏩ Continuar desde B", variant="secondary")
        status_B  = gr.Textbox(label="Estado B", interactive=False)
        preview_B = gr.Image(label="Preview B", height=200)

        gr.Markdown("### PASO C — Plano Cercano en Escenario")
        with gr.Row():
            upload_C = gr.File(label="📤 Subir C", file_types=[".png",".jpg",".jpeg"])
            btn_C    = gr.Button("▶️ Generar/Subir C", variant="primary")
            btn_cont_C = gr.Button("⏩ Continuar desde C", variant="secondary")
        status_C  = gr.Textbox(label="Estado C", interactive=False)
        preview_C = gr.Image(label="Preview C", height=200)

        gr.Markdown("### PASO D — Close-up")
        with gr.Row():
            upload_D = gr.File(label="📤 Subir D", file_types=[".png",".jpg",".jpeg"])
            btn_D    = gr.Button("▶️ Generar/Subir D", variant="primary")
            btn_cont_D = gr.Button("⏩ Continuar desde D", variant="secondary")
        status_D  = gr.Textbox(label="Estado D", interactive=False)
        preview_D = gr.Image(label="Preview D", height=200)

        gr.Markdown("### PASOS E/F/G — Planos Lejanos")
        with gr.Row():
            upload_E = gr.File(label="📤 Subir E", file_types=[".png",".jpg",".jpeg"])
            btn_E    = gr.Button("▶️ E", variant="primary")
        with gr.Row():
            upload_F = gr.File(label="📤 Subir F", file_types=[".png",".jpg",".jpeg"])
            btn_F    = gr.Button("▶️ F", variant="primary")
        with gr.Row():
            upload_G = gr.File(label="📤 Subir G", file_types=[".png",".jpg",".jpeg"])
            btn_G    = gr.Button("▶️ G", variant="primary")
            btn_cont_E = gr.Button("⏩ Continuar desde E", variant="secondary")
        status_EFG = gr.Textbox(label="Estado E/F/G", interactive=False)

        gr.Markdown("---")
        gr.Markdown("### 🚀 MODO RÁPIDO: Ejecutar Etapa 1 Completa")
        btn_e1_todo = gr.Button("▶️ EJECUTAR ETAPA 1 COMPLETA", variant="primary", size="lg")
        e1_status   = gr.Textbox(label="Estado Etapa 1", interactive=False)
        e1_gallery  = gr.Gallery(label="Galería A..G", columns=4, height=280)

    # ────────────────────────────────────────────────────────
    # LTX-2.5
    # ────────────────────────────────────────────────────────
    with gr.Tab("⬇️ Modelos LTX (Wan2GP)"):
        ltx_preset_dd = gr.Dropdown(
            label="Preset", choices=list(LTX23_PRESETS.keys()),
            value=DEFAULT_LTX23_PRESET)
        for k, v in LTX23_PRESETS.items():
            gr.Markdown(f"- **{k}**: {v['desc']}")
        btn_ltx_dl = gr.Button("⬇️ Descargar e inicializar LTX", variant="secondary")
        ltx_status = gr.Textbox(label="Estado LTX", interactive=False, lines=7)
        with gr.Row():
            btn_ltx_check = gr.Button("🧪 Probar entorno LTX (diagnóstico lipsync)", variant="secondary")
        ltx_check_status = gr.Textbox(label="Diagnóstico entorno LTX", interactive=False, lines=4)

    # ════════════════════════════════════════════════════════
    # 🎙️ VOZ DEL CONCURSANTE — SECCIÓN CENTRAL (encima de E2 y E4)
    # Esta sección configura la voz que se usará en todo el guion.
    # ════════════════════════════════════════════════════════
    with gr.Tab("🎙️ VOZ DEL CONCURSANTE — Configuración (requerida para Etapas 2 y 4)"):
        gr.HTML('''<div style="background:linear-gradient(135deg,#1a1a2e,#16213e);
                   border:2px solid #6a11cb;border-radius:12px;padding:14px;margin-bottom:10px;">
            <p style="color:#b0b8ff;margin:0;font-size:0.95em;">
            ✨ <b>Pipeline de voz consistente:</b> Cualquier voz que configures aquí
            se usará mediante <b>clonación</b> en TODOS los fragmentos del guion
            (tanto Etapa 2 como Etapa 4), garantizando consistencia total.
            </p></div>''')

        voice_state_box = gr.Textbox(
            label="📊 Estado actual de la voz",
            value=ui_show_voice_state(),
            interactive=False, lines=5
        )
        btn_show_voice_state = gr.Button("🔄 Actualizar estado de voz", variant="secondary")

        # ── OPCIÓN A: Subir audio del concursante ────────────
        with gr.Tab("🎤 Opción A — Subir mi audio (clonación directa)"):
            gr.Markdown(
                "Sube un audio claro de **mínimo 3 segundos**. "
                "Se transcribirá automáticamente para mejorar la calidad de clonación."
            )
            with gr.Row():
                contestant_audio_upload = gr.File(
                    label="🎵 Audio del concursante",
                    file_types=[".wav",".mp3",".m4a",".flac",".ogg"],
                    scale=3
                )
                btn_process_contestant = gr.Button(
                    "⚙️ Procesar y transcribir", variant="primary", scale=1
                )
            contestant_upload_status = gr.Textbox(
                label="Estado del procesamiento",
                interactive=False, lines=5
            )
            with gr.Row():
                contestant_transcription_box = gr.Textbox(
                    label="📝 Transcripción detectada (editable)",
                    placeholder="Aquí aparecerá la transcripción automática...",
                    lines=3, interactive=True, scale=4
                )
                contestant_lang_box = gr.Textbox(
                    label="Idioma", interactive=False, scale=1
                )
            gr.Markdown("#### 🔊 Preview de clonación — valida antes de comprometerte")
            preview_text_clone = gr.Textbox(
                label="Texto para el preview",
                value=PREVIEW_TEXT_ES,
                lines=2, interactive=True
            )
            with gr.Row():
                btn_gen_clone_preview = gr.Button(
                    "▶️ Generar preview clonado", variant="primary"
                )
                btn_validate_clone = gr.Button(
                    "✅ Validar esta voz para el guion", variant="primary"
                )
            clone_preview_status = gr.Textbox(
                label="Estado del preview", interactive=False, lines=3
            )
            clone_preview_audio = gr.Audio(
                label="🔊 Preview clonado (escucha antes de validar)",
                type="filepath"
            )

        # ── OPCIÓN B: Diseñar voz con descripción ─────────────
        with gr.Tab("🎨 Opción B — Diseñar voz (VoiceDesign → Referencia)"):
            gr.Markdown(
                "Describe la voz que quieres. "
                "Se generará una **muestra de audio**. "
                "Si te gusta, la adoptamos como referencia y el guion completo "
                "usará **clonación** de esa voz para consistencia máxima."
            )
            gr.Markdown("**Ejemplos de descripción:**")
            gr.Markdown(
                "- `A young female, cheerful, singing naturally, Spanish speaking`\n"
                "- `A teenage girl, shy but expressive, soft and emotional, Spanish`\n"
                "- `A mature woman, warm and determined, clear pronunciation, Spanish`\n"
                "- `A young male, confident, energetic, moderate pitch, Spanish`"
            )
            design_instruct_box = gr.Textbox(
                label="✏️ Descripción de la voz (en inglés para mejores resultados)",
                placeholder="A young female, cheerful, speaking clearly with natural warmth, Spanish speaking",
                lines=3, interactive=True
            )
            preview_text_design = gr.Textbox(
                label="Texto para el preview",
                value=PREVIEW_TEXT_ES,
                lines=2, interactive=True
            )
            with gr.Row():
                btn_gen_design_preview = gr.Button(
                    "▶️ Generar preview VoiceDesign", variant="primary"
                )
                btn_adopt_design = gr.Button(
                    "✅ Usar esta voz como referencia y validar",
                    variant="primary"
                )
            design_preview_status = gr.Textbox(
                label="Estado del diseño", interactive=False, lines=4
            )
            design_preview_audio = gr.Audio(
                label="🔊 Preview de voz diseñada (escucha y compara)",
                type="filepath"
            )
            gr.Markdown(
                "ℹ️ Al pulsar **'Usar esta voz como referencia'**, "
                "el audio del preview se convierte en la referencia de clonación. "
                "A partir de ese momento, el guion completo clona esa voz."
            )

        # ── Estado final de validación ────────────────────────
        gr.Markdown("---")
        with gr.Row():
            btn_validate_final = gr.Button(
                "✅ CONFIRMAR voz como lista para el guion",
                variant="primary", size="lg"
            )
            btn_voice_state_refresh = gr.Button("🔄 Refrescar estado", variant="secondary")
        voice_validation_final_box = gr.Textbox(
            label="✅ Estado de validación final",
            interactive=False, lines=4
        )

    # ════════════════════════════════════════════════════════
    # ETAPA 3 — CANCIÓN
    # ════════════════════════════════════════════════════════
    with gr.Tab("🎵 ETAPA 3 — Canción con Lipsync"):
        with gr.Row():
            song_file  = gr.File(label="Audio canción",
                                 file_types=[".wav",".mp3",".m4a",".flac"])
            intro_seconds = gr.Number(label="Segundos iniciales de reacciones (La Voz)", value=30, minimum=0, maximum=120, precision=1)
            btn_song   = gr.Button("💾 Guardar canción")
            song_status = gr.Textbox(label="Estado canción", interactive=False)

        gr.Markdown("### 🎬 PASO 3.1 — Clips 5s (validación individual)")
        with gr.Row():
            far_res_dd    = gr.Dropdown(label="📐 Resolución clips lejanos", choices=LTX_RES_CHOICES, value=LTX_UI_CFG["far_res"])
            far_aspect_dd = gr.Dropdown(label="📏 Aspect ratio (lejanos)", choices=LTX_ASPECT_CHOICES, value=LTX_UI_CFG["far_aspect"],
                                        info="Auto sigue la imagen de referencia. Solo aplica si eliges 720p/540p/480p.")
            far_dur_dd    = gr.Dropdown(label="⏱️ Duración del fragmento", choices=["2s", "3s", "5s"], value=f"{int(LTX_UI_CFG['far_dur'])}s",
                                        info="También define el largo de los planos lejanos al construir el timeline.")
        with gr.Row():
            seed_5s       = gr.Number(label="Seed", value=-1, precision=0)
            btn_gen5s_E   = gr.Button("▶️ Clip E", variant="primary")
            btn_gen5s_F   = gr.Button("▶️ Clip F", variant="primary")
            btn_gen5s_G   = gr.Button("▶️ Clip G", variant="primary")
            btn_gen5s_all = gr.Button("🚀 Todos los clips 5s", variant="secondary")
        with gr.Row():
            btn_gen5s_Ex = gr.Button("➕ Clip E+ (otro de E)", variant="secondary")
            btn_gen5s_Fx = gr.Button("➕ Clip F+ (otro de F)", variant="secondary")
            btn_gen5s_Gx = gr.Button("➕ Clip G+ (otro de G)", variant="secondary")
        clips5s_status  = gr.Textbox(label="Estado clips 5s", interactive=False)
        clips5s_gallery = gr.Gallery(label="Clips 5s generados", columns=3, height=200)

        with gr.Accordion("🔄 Regenerar clip 5s específico", open=False):
            with gr.Row():
                clip_path_in  = gr.Textbox(label="Ruta del clip")
                seed_regen5s  = gr.Number(label="Seed", value=-1, precision=0)
                btn_regen5s   = gr.Button("🔄 Regenerar")
            regen5s_status = gr.Textbox(label="Estado", interactive=False)

        with gr.Accordion("🎞️ Clips 5s lejanos — ver, regenerar y subir los tuyos", open=False):
            gr.Markdown("Cada clip muestra su video, **🔄 Regenerar (LTX)** (con la seed de arriba; si LTX falla "
                        "se conserva el anterior) y **📤 Usar mi clip** para reemplazarlo por un video de tu PC "
                        "(se convierte a 1152x704, 24 fps, sin audio, 5 s). El resto del pipeline no cambia.")
            tick_5s = gr.State(0)
            btn_list5s = gr.Button("🔄 Actualizar lista de clips 5s", size="sm", variant="secondary")
            clips5s_st2 = gr.Textbox(label="Estado (clips 5s)", interactive=False, lines=2)
            _make_clips5s_panel(tick_5s, clips5s_st2, personaje, seed_5s)
            gr.Markdown("**➕ Añadir un clip nuevo desde mi PC** (como extra para la distribución):")
            with gr.Row():
                add5s_key  = gr.Radio(label="Imagen/plano", choices=["E", "F", "G"], value="E")
                add5s_file = gr.File(label="Video del PC",
                                     file_types=[".mp4", ".mov", ".webm", ".mkv", ".avi"],
                                     type="filepath")
                btn_add5s  = gr.Button("📤 Añadir clip", variant="secondary")

        gr.Markdown("### 📊 PASO 3.2 — Timeline")
        btn_tl     = gr.Button("📊 Construir timeline", variant="secondary")
        tl_status  = gr.Textbox(label="Estado timeline", interactive=False)
        tl_html    = gr.HTML(
            value="<div style='background:#1a1a1a;border:1px solid #444;"
                  "border-radius:8px;padding:8px;min-height:50px;"
                  "color:#aaa;font-family:monospace;'>Timeline aparecerá aquí.</div>"
        )

        gr.Markdown("### 🧩 Timeline por fragmentos (regenerar lipsync clip a clip)")
        gr.Markdown("Cada clip lipsync muestra su video y un botón **🔄 Regenerar lipsync (LTX)** que rehace "
                    "solo ese tramo con seed nueva (si LTX falla se conserva el anterior). "
                    "Al pulsar «Renderizar Etapa 3», los lipsync ya hechos se reutilizan.")
        tick_e3 = gr.State(0)
        btn_tl_frag = gr.Button("🔄 Actualizar fragmentos", size="sm", variant="secondary")
        regen_tl_st_e3 = gr.Textbox(label="Estado de regeneración (canción)", interactive=False, lines=2)
        _make_song_timeline(tick_e3, regen_tl_st_e3, tpl_dd, personaje)

        gr.Markdown("### 🎤 Resolución de los clips lipsync de la canción")
        with gr.Row():
            song_res_dd    = gr.Dropdown(label="📐 Resolución lipsync canción", choices=LTX_RES_CHOICES, value=LTX_UI_CFG["song_res"])
            song_aspect_dd = gr.Dropdown(label="📏 Aspect ratio (canción)", choices=LTX_ASPECT_CHOICES, value=LTX_UI_CFG["song_aspect"],
                                         info="Auto sigue la imagen de referencia. Solo aplica si eliges 720p/540p/480p.")
        gr.Markdown("### ▶️ PASO 3.3 — Renderizar")
        btn_render3  = gr.Button("▶️ Renderizar Etapa 3", variant="primary")
        e3_status    = gr.Textbox(label="Estado Etapa 3", interactive=False)
        e3_video     = gr.Video(label="Preview canción", height=240)

    # ════════════════════════════════════════════════════════
    # ETAPA 2 — DIÁLOGOS INICIALES
    # ════════════════════════════════════════════════════════
    with gr.Tab("🎙️ ETAPA 2 — Diálogos iniciales"):
        gr.HTML('''<div style="background:#0d2137;border-left:4px solid #2575fc;
                   padding:10px;border-radius:6px;margin-bottom:12px;">
            <p style="color:#7eb8f7;margin:0;font-size:0.9em;">
            🎙️ <b>Requiere voz del concursante validada</b> (sección de arriba).<br>
            Los jueces usarán sus audios de plantilla (auto-transcritos).<br>
            Si no hay audio de juez, se aplica VoiceDesign fallback automático.
            </p></div>''')
        with gr.Row():
            e2_dlg_res = gr.Dropdown(label="📐 Resolución lipsync diálogos", choices=LTX_RES_CHOICES, value=LTX_UI_CFG["dlg_res"])
            e2_dlg_asp = gr.Dropdown(label="📏 Aspect ratio (diálogos)", choices=LTX_ASPECT_CHOICES, value=LTX_UI_CFG["dlg_aspect"],
                                      info="Auto sigue la imagen de referencia. Solo aplica si eliges 720p/540p/480p.")

        historia_box = gr.Textbox(
            label="Historia personal del concursante",
            placeholder="Ej: 'Perdió su trabajo, cuida a sus hijos sola, "
                        "la música es su única salida...'",
            lines=4
        )
        with gr.Row():
            model_id_e2  = gr.Dropdown(
                label="Modelo xKiro para guion",
                choices=list(XKIRO_MODEL_OPTIONS.values()),
                value=DEFAULT_XKIRO_TEXT_MODEL_ID,
                allow_custom_value=True, scale=3
            )
            fast_mode_e2 = gr.Checkbox(
                label="Fast Mode TTS (sin transcripción ref)",
                value=False, scale=1
            )
        with gr.Row():
            btn_e2_regen_guion = gr.Button("📝 Generar solo guion (sin TTS)",
                                           variant="secondary")
            btn_e2_actual      = gr.Button("⏯️ Generar Etapa 2 con guion actual",
                                           variant="primary")
            btn_e2             = gr.Button("▶️ Ejecutar completa (guion NUEVO + TTS + video)",
                                           variant="secondary")
        with gr.Row():
            btn_e2_preview   = gr.Button("🖼️ Preview con imágenes (audio + imagen, sin lipsync)",
                                         variant="secondary")
            btn_e2_continuar = gr.Button("🎬 Continuar con lipsync (usa guion y audios del preview)",
                                         variant="primary")
        e2_status = gr.Textbox(label="Estado Etapa 2", interactive=False, lines=3)
        e2_preview_video = gr.Video(label="🖼️ Preview con imágenes (Etapa 2)", height=240)
        e2_video  = gr.Video(label="Preview Etapa 2", height=240)

        gr.Markdown("### 🧩 Timeline de fragmentos — Etapa 2")
        gr.Markdown("Cada fragmento muestra su estado y su clip. **🔄 Regenerar lipsync** rehace solo el clip LTX "
                    "de ese fragmento (reutiliza su audio) y reensambla el video de la etapa.")
        tick_e2 = gr.State(0)
        btn_tl_e2 = gr.Button("🔄 Actualizar timeline E2", size="sm", variant="secondary")
        regen_tl_st_e2 = gr.Textbox(label="Estado de regeneración (E2)", interactive=False, lines=2)
        _make_dialog_timeline("etapa2", tick_e2, regen_tl_st_e2, tpl_dd, personaje)

        with gr.Accordion("🔄 Regenerar fragmento individual (E2)", open=False):
            with gr.Row():
                etapa_sel_e2  = gr.Dropdown(label="Etapa", choices=["etapa2","etapa4"],
                                            value="etapa2")
                frag_idx_n    = gr.Number(label="Índice fragmento (0-based)",
                                          value=0, precision=0)
            nuevo_txt = gr.Textbox(label="Nuevo texto (vacío=usar original)", lines=3)
            btn_regen_frag  = gr.Button("🔄 Regenerar fragmento", variant="secondary")
            regen_frag_st   = gr.Textbox(label="Estado", interactive=False)

        with gr.Row():
            btn_ver_guion = gr.Button("📄 Ver guion JSON (E2 + E4)")
        guion_json_box = gr.Code(label="Guion JSON", language="json", lines=20)

        with gr.Accordion("✏️ EDITAR GUION MANUALMENTE", open=False):
            gr.Markdown(
                    "Edita el JSON directamente. Cambia textos, emociones o duraciones.\n"
                    "Al guardar, pulsa **'⏯️ Generar Etapa con guion actual'** para usar "
                    "**este guion editado** (no se llama a la IA).\n"
                    "⚠ 'Ejecutar completa' genera un guion NUEVO (el anterior queda respaldado).\n"
                    "Solo se rehacen audio/video de los fragmentos cuyo texto cambió."
            )
            with gr.Row():
                etapa_edit_sel = gr.Dropdown(
                    label="Etapa a editar",
                    choices=["etapa2", "etapa4"],
                    value="etapa2", scale=1
                )
                btn_cargar_guion_edit = gr.Button(
                    "📥 Cargar guion actual", variant="secondary", scale=1
                )
            guion_edit_box = gr.Code(
                label="Guion JSON editable",
                language="json",
                lines=25,
                interactive=True
            )
            with gr.Row():
                btn_guardar_guion_edit = gr.Button(
                    "💾 Guardar guion editado", variant="primary"
                )
                guion_edit_status = gr.Textbox(label="Estado", interactive=False)

            # Wiring
            btn_cargar_guion_edit.click(
                fn=lambda et: json.dumps(
                    (_get_vp().estado.get(et, {}).get("guion_json", {})),
                    ensure_ascii=False, indent=2
                ),
                inputs=[etapa_edit_sel],
                outputs=[guion_edit_box]
            )
            btn_guardar_guion_edit.click(
                fn=ui_guardar_guion_editado,
                inputs=[etapa_edit_sel, guion_edit_box],
                outputs=[guion_edit_status]
            )


    # ════════════════════════════════════════════════════════
    # ETAPA 4 — DIÁLOGOS FINALES
    # ════════════════════════════════════════════════════════
    with gr.Tab("🏆 ETAPA 4 — Diálogos finales de los jueces"):
        gr.HTML('''<div style="background:#0d2137;border-left:4px solid #6a11cb;
                   padding:10px;border-radius:6px;margin-bottom:12px;">
            <p style="color:#b0b8ff;margin:0;font-size:0.9em;">
            🎙️ <b>Requiere voz del concursante validada</b> (sección de arriba).<br>
            En esta etapa hablan principalmente los jueces (≥70% del tiempo).<br>
            La misma referencia del concursante se usa si éste responde brevemente.
            </p></div>''')
        with gr.Row():
            e4_dlg_res = gr.Dropdown(label="📐 Resolución lipsync diálogos", choices=LTX_RES_CHOICES, value=LTX_UI_CFG["dlg_res"])
            e4_dlg_asp = gr.Dropdown(label="📏 Aspect ratio (diálogos)", choices=LTX_ASPECT_CHOICES, value=LTX_UI_CFG["dlg_aspect"],
                                      info="Auto sigue la imagen de referencia. Solo aplica si eliges 720p/540p/480p.")

        ctx_e4_box = gr.Textbox(
            label="Contexto para el guion (opcional) — historia del concursante / cómo fue su actuación",
            placeholder="Ej: 'Perdió su trabajo, cuida a sus hijos sola, la música es su única salida... "
                        "cantó con la voz quebrada y el público se puso de pie...'",
            lines=4,
            info="Se usa al generar un guion NUEVO de la Etapa 4. Vacío = comportamiento de siempre."
        )

        with gr.Row():
            model_id_e4  = gr.Dropdown(
                label="Modelo xKiro para guion",
                choices=list(XKIRO_MODEL_OPTIONS.values()),
                value=DEFAULT_XKIRO_TEXT_MODEL_ID,
                allow_custom_value=True, scale=3
            )
            fast_mode_e4 = gr.Checkbox(
                label="Fast Mode TTS", value=False, scale=1
            )
        with gr.Row():
            btn_e4_regen_guion = gr.Button("📝 Generar solo guion (sin TTS)",
                                           variant="secondary")
            btn_e4_actual      = gr.Button("⏯️ Generar Etapa 4 con guion actual",
                                           variant="primary")
            btn_e4               = gr.Button("▶️ Ejecutar completa (guion NUEVO + TTS + video)",
                                           variant="secondary")
        with gr.Row():
            btn_e4_preview   = gr.Button("🖼️ Preview con imágenes (audio + imagen, sin lipsync)",
                                         variant="secondary")
            btn_e4_continuar = gr.Button("🎬 Continuar con lipsync (usa guion y audios del preview)",
                                         variant="primary")
        e4_status = gr.Textbox(label="Estado Etapa 4", interactive=False, lines=3)
        e4_preview_video = gr.Video(label="🖼️ Preview con imágenes (Etapa 4)", height=240)
        e4_video  = gr.Video(label="Preview Etapa 4", height=240)

        gr.Markdown("### 🧩 Timeline de fragmentos — Etapa 4")
        gr.Markdown("Cada fragmento muestra su estado y su clip. **🔄 Regenerar lipsync** rehace solo el clip LTX "
                    "de ese fragmento (reutiliza su audio) y reensambla el video de la etapa.")
        tick_e4 = gr.State(0)
        btn_tl_e4 = gr.Button("🔄 Actualizar timeline E4", size="sm", variant="secondary")
        regen_tl_st_e4 = gr.Textbox(label="Estado de regeneración (E4)", interactive=False, lines=2)
        _make_dialog_timeline("etapa4", tick_e4, regen_tl_st_e4, tpl_dd, personaje)

        with gr.Accordion("🔄 Regenerar fragmento individual (E4)", open=False):
            with gr.Row():
                etapa_sel_e4  = gr.Dropdown(label="Etapa", choices=["etapa2","etapa4"],
                                            value="etapa4")
                frag_idx_e4   = gr.Number(label="Índice fragmento (0-based)",
                                           value=0, precision=0)
            nuevo_txt_e4 = gr.Textbox(label="Nuevo texto (vacío=usar original)", lines=3)
            btn_regen_frag_e4 = gr.Button("🔄 Regenerar fragmento E4", variant="secondary")
            regen_frag_st_e4  = gr.Textbox(label="Estado", interactive=False)

    # ════════════════════════════════════════════════════════
    # RENDER FINAL
    # ════════════════════════════════════════════════════════
    # ════════════════════════════════════════════════════════
    # 🚀 AUTOMATIZACIÓN (v7)
    # ════════════════════════════════════════════════════════
    with gr.Tab("🚀 AUTOMATIZACIÓN — generar varias etapas de una vez"):
        gr.Markdown(
            "Marca qué etapas generar y pulsa **Generar lo marcado**. Si desmarcas una (o las dos de diálogos) "
            "esa etapa se omite; con solo «Canción» marcada se genera únicamente la canción. "
            "Cada etapa es independiente: si una falla se anota y se sigue con las demás. "
            "Orden: Etapa 2 → Canción → Etapa 4. Requiere voz validada (E2/E4) y canción guardada (Canción).\n\n"
            "💡 Para **validar primero y rehacer después**: modo «Solo preview con imágenes», revisa el timeline y "
            "luego «🎬 Continuar con lipsync» en cada etapa (reutiliza guion y audios)."
        )
        with gr.Row():
            au_dlg_res = gr.Dropdown(label="📐 Resolución lipsync diálogos", choices=LTX_RES_CHOICES, value=LTX_UI_CFG["dlg_res"])
            au_dlg_asp = gr.Dropdown(label="📏 Aspect ratio (diálogos)", choices=LTX_ASPECT_CHOICES, value=LTX_UI_CFG["dlg_aspect"],
                                      info="Auto sigue la imagen de referencia. Solo aplica si eliges 720p/540p/480p.")
        with gr.Row():
            au_far_res = gr.Dropdown(label="📐 Resolución clips lejanos", choices=LTX_RES_CHOICES, value=LTX_UI_CFG["far_res"])
            au_far_asp = gr.Dropdown(label="📏 Aspect ratio (lejanos)", choices=LTX_ASPECT_CHOICES, value=LTX_UI_CFG["far_aspect"])
            au_far_dur = gr.Dropdown(label="⏱️ Duración fragmento lejano", choices=["2s", "3s", "5s"], value=f"{int(LTX_UI_CFG['far_dur'])}s")
        with gr.Row():
            au_song_res = gr.Dropdown(label="📐 Resolución lipsync canción", choices=LTX_RES_CHOICES, value=LTX_UI_CFG["song_res"])
            au_song_asp = gr.Dropdown(label="📏 Aspect ratio (canción)", choices=LTX_ASPECT_CHOICES, value=LTX_UI_CFG["song_aspect"])
        with gr.Row():
            auto_e2 = gr.Checkbox(label="🎙️ Etapa 2 (diálogos iniciales)", value=True)
            auto_e3 = gr.Checkbox(label="🎵 Canción (Etapa 3)", value=True)
            auto_e4 = gr.Checkbox(label="🏆 Etapa 4 (jueces finales)", value=True)
        with gr.Row():
            auto_modo_dialogo = gr.Radio(label="Modo de diálogos (E2/E4)",
                                         choices=list(_MODOS_DIALOGO.keys()),
                                         value=list(_MODOS_DIALOGO.keys())[0])
            auto_modo_guion = gr.Radio(label="Guion de E2/E4",
                                       choices=list(_MODOS_GUION.keys()),
                                       value=list(_MODOS_GUION.keys())[0])
        with gr.Row():
            auto_ctx_e2 = gr.Textbox(
                label="📝 Contexto del guion — Etapa 2 (opcional)",
                lines=4,
                placeholder="Ej: historia personal del concursante...",
                info="Si lo rellenas, sustituye a «Historia personal» de la Etapa 2 solo dentro de la automatización."
            )
            auto_ctx_e4 = gr.Textbox(
                label="📝 Contexto del guion — Etapa 4 (opcional)",
                lines=4,
                placeholder="Ej: la misma historia + cómo fue la actuación...",
                info="Si está vacío se usa el contexto de la sección Etapa 4."
            )
        gr.Markdown("ℹ️ Los contextos solo se aplican cuando se **escribe un guion nuevo**: elige "
                    "«Generar guion NUEVO» (con «Reutilizar» y un guion ya existente no se vuelve a escribir).")
        with gr.Row():
            auto_reconstruir_tl = gr.Checkbox(
                label="Reconstruir el timeline de la canción (si no, se conserva el que ya validaste)",
                value=False)
            auto_render_final = gr.Checkbox(
                label="Renderizar VIDEO FINAL al terminar (solo con diálogos en modo Completo)", value=False)
        with gr.Row():
            btn_auto_todo = gr.Button("🚀 Generar lo marcado", variant="primary", size="lg")
            btn_auto_e2e4 = gr.Button("🎙️ Solo Etapas 2 y 4 (juntas)", variant="secondary")
            btn_auto_cancion = gr.Button("🎵 Solo canción", variant="secondary")
        auto_status = gr.Textbox(label="Progreso de la automatización", interactive=False, lines=10)
        auto_video  = gr.Video(label="Último video generado", height=300)

    with gr.Tab("🎞️ RENDER FINAL"):
        with gr.Row():
            etapa2_on = gr.Checkbox(label="Incluir Etapa 2 (diálogos iniciales)", value=True)
            gr.Markdown("**Etapa 3: siempre incluida**")
            etapa4_on = gr.Checkbox(label="Incluir Etapa 4 (jueces finales)", value=True)
        btn_final    = gr.Button("🏁 Renderizar VIDEO FINAL", variant="primary", size="lg")
        final_status = gr.Textbox(label="Estado", interactive=False)
        final_video  = gr.Video(label="VIDEO FINAL", height=360)

    # ════════════════════════════════════════════════════════
    # UTILIDADES
    # ════════════════════════════════════════════════════════
    with gr.Tab("🛠️ Utilidades"):
        with gr.Row():
            btn_estado  = gr.Button("📋 Ver estado completo")
            btn_dataset = gr.Button("☁️ Publicar → Kaggle Dataset")
        util_status = gr.Textbox(label="Info", interactive=False, lines=10)

    # ════════════════════════════════════════════════════════
    # LOGS
    # ════════════════════════════════════════════════════════
    with gr.Tab("📟 Consola (run.log)"):
        logs_box = gr.Textbox(label="Logs", value=tail_log(),
                              lines=28, max_lines=50, interactive=False)
    _main_tabs.__exit__(None, None, None)
    timer = gr.Timer(2.5)

    # ════════════════════════════════════════════════════════
    # WIRING — TODOS LOS EVENTOS
    # ════════════════════════════════════════════════════════

  # ── REEMPLAZAR POR (carga mínima inmediata):
    demo.load(
        fn=ui_initial_load,
        inputs=[],
        outputs=[progreso_box, progreso_gallery, video_selector, ruta_label, tpl_dd],
    )

    # Reanudación
    btn_cargar.click(
        fn=ui_cargar_video_seleccionado, inputs=[video_selector],
        outputs=[progreso_box, progreso_gallery, video_activo_label, ruta_label]
    ).then(fn=ui_refresh_tpl, inputs=[], outputs=[tpl_dd]).then(
        fn=_bump3, inputs=[tick_e2, tick_e3, tick_e4], outputs=[tick_e2, tick_e3, tick_e4])
    btn_refprog.click(fn=ui_refrescar_progreso, inputs=[],
                      outputs=[progreso_box, progreso_gallery])
    btn_listar.click(fn=ui_listar_videos, inputs=[], outputs=[video_selector])
    btn_abrir.click(fn=lambda: _CUR_VP_DIR or "", inputs=[], outputs=[ruta_label])

    # Plantillas
    btn_refresh_tpl.click(ui_refresh_tpl, [], [tpl_dd])
    btn_crear_tpl.click(ui_crear_plantilla, [new_tpl_box, new_tpl_style], [status_main, tpl_dd])
    btn_assets.click(
        ui_subir_assets_juez_independiente,
        [tpl_dd, j1_img, j1_aud, j2_img, j2_aud, j3_img, j3_aud, j4_img, j4_aud,
         esc_cercano, escs_lejanos, reac1, reac2] + lv_uploads,
        [assets_status]
    )
    btn_transcribe_judges.click(
        fn=ui_transcribe_judges, inputs=[tpl_dd], outputs=[assets_status]
    )

    # Nuevo video
    btn_nuevo_video.click(
        ui_nuevo_video, [tpl_dd, personaje],
        [status_main, video_dir_box, tpl_dd]
    ).then(ui_scan_ultimo_video, [],
           [progreso_box, progreso_gallery, video_selector, ruta_label]).then(
        _bump3, [tick_e2, tick_e3, tick_e4], [tick_e2, tick_e3, tick_e4])

    # LTX
    btn_ltx_dl.click(ui_descargar_ltx, [ltx_preset_dd], [ltx_status])

    # ── ETAPA 1 GRANULAR ──────────────────────────────────
    e1_inputs = [personaje, p_reemplazar, edad, vestuario, planos]

    def _make_paso_fn(key):
        def paso_wrapper(upload_file, personaje, p_reemplazar, edad, vestuario, planos):
            return ui_etapa1_paso(key, upload_file, personaje, p_reemplazar, edad, vestuario, planos)
        return paso_wrapper

    def _make_cont_fn(start_key):
        def cont_wrapper(personaje, p_reemplazar, edad, vestuario, planos):
            return ui_etapa1_continuar_desde(start_key, personaje, p_reemplazar, edad, vestuario, planos)
        return cont_wrapper

    btn_A.click(_make_paso_fn("A"), [upload_A]+e1_inputs, [status_A, preview_A]).then(
        ui_refrescar_progreso, [], [progreso_box, progreso_gallery])
    btn_B.click(_make_paso_fn("B"), [upload_B]+e1_inputs, [status_B, preview_B]).then(
        ui_refrescar_progreso, [], [progreso_box, progreso_gallery])
    btn_C.click(_make_paso_fn("C"), [upload_C]+e1_inputs, [status_C, preview_C]).then(
        ui_refrescar_progreso, [], [progreso_box, progreso_gallery])
    btn_D.click(_make_paso_fn("D"), [upload_D]+e1_inputs, [status_D, preview_D]).then(
        ui_refrescar_progreso, [], [progreso_box, progreso_gallery])
    btn_E.click(_make_paso_fn("E"), [upload_E]+e1_inputs, [status_EFG, preview_A]).then(
        ui_refrescar_progreso, [], [progreso_box, progreso_gallery])
    btn_F.click(_make_paso_fn("F"), [upload_F]+e1_inputs, [status_EFG, preview_A]).then(
        ui_refrescar_progreso, [], [progreso_box, progreso_gallery])
    btn_G.click(_make_paso_fn("G"), [upload_G]+e1_inputs, [status_EFG, preview_A]).then(
        ui_refrescar_progreso, [], [progreso_box, progreso_gallery])

    btn_cont_A.click(_make_cont_fn("A"), e1_inputs, [e1_status]).then(
        ui_refrescar_progreso, [], [progreso_box, progreso_gallery])
    btn_cont_B.click(_make_cont_fn("B"), e1_inputs, [e1_status]).then(
        ui_refrescar_progreso, [], [progreso_box, progreso_gallery])
    btn_cont_C.click(_make_cont_fn("C"), e1_inputs, [e1_status]).then(
        ui_refrescar_progreso, [], [progreso_box, progreso_gallery])
    btn_cont_D.click(_make_cont_fn("D"), e1_inputs, [e1_status]).then(
        ui_refrescar_progreso, [], [progreso_box, progreso_gallery])
    btn_cont_E.click(_make_cont_fn("E"), e1_inputs, [e1_status]).then(
        ui_refrescar_progreso, [], [progreso_box, progreso_gallery])

    btn_e1_todo.click(
        ui_etapa1_todo,
        e1_inputs + [dq, eq, mq, stA, stE, cfg_v, seed_v],
        [e1_status, e1_gallery]
    ).then(ui_refrescar_progreso, [], [progreso_box, progreso_gallery])

    # ── VOZ DEL CONCURSANTE ───────────────────────────────
    btn_show_voice_state.click(
        fn=ui_show_voice_state, inputs=[],
        outputs=[voice_state_box]
    )
    btn_voice_state_refresh.click(
        fn=ui_show_voice_state, inputs=[],
        outputs=[voice_state_box]
    )

    # Opción A: subir audio
    btn_process_contestant.click(
        fn=ui_upload_contestant_audio,
        inputs=[contestant_audio_upload],
        outputs=[contestant_upload_status,
                 contestant_transcription_box,
                 contestant_lang_box,
                 btn_gen_clone_preview]
    ).then(fn=ui_show_voice_state, inputs=[], outputs=[voice_state_box])

    btn_gen_clone_preview.click(
        fn=ui_generate_clone_preview,
        inputs=[contestant_transcription_box, preview_text_clone],
        outputs=[clone_preview_status, clone_preview_audio]
    )

    btn_validate_clone.click(
        fn=ui_validate_contestant_voice,
        inputs=[],
        outputs=[voice_validation_final_box]
    ).then(fn=ui_show_voice_state, inputs=[], outputs=[voice_state_box]).then(
        fn=ui_refrescar_progreso, inputs=[], outputs=[progreso_box, progreso_gallery])

    # Opción B: VoiceDesign
    btn_gen_design_preview.click(
        fn=ui_generate_design_preview,
        inputs=[design_instruct_box, preview_text_design],
        outputs=[design_preview_status, design_preview_audio]
    )

    btn_adopt_design.click(
        fn=ui_adopt_design_as_ref,
        inputs=[design_instruct_box],
        outputs=[design_preview_status, btn_validate_final]
    ).then(
        fn=ui_validate_contestant_voice,
        inputs=[],
        outputs=[voice_validation_final_box]
    ).then(fn=ui_show_voice_state, inputs=[], outputs=[voice_state_box]).then(
        fn=ui_refrescar_progreso, inputs=[], outputs=[progreso_box, progreso_gallery])

    # Validación final (botón grande)
    btn_validate_final.click(
        fn=ui_validate_contestant_voice,
        inputs=[],
        outputs=[voice_validation_final_box]
    ).then(fn=ui_show_voice_state, inputs=[], outputs=[voice_state_box]).then(
        fn=ui_refrescar_progreso, inputs=[], outputs=[progreso_box, progreso_gallery])

    # ── ETAPA 3 GRANULAR ─────────────────────────────────
    # Selectores LTX (resolución/duración): los de cada etapa y los de «Automatización» se mantienen sincronizados.
    def _ltx_mk(_k, _n):
        def _f(v):
            ltx_cfg_set(_k, v)
            if _n:
                return tuple(gr.update(value=v) for _ in range(_n))
        return _f
    _LTX_MIR = {"far_res": [far_res_dd, au_far_res], "far_aspect": [far_aspect_dd, au_far_asp],
                "far_dur": [far_dur_dd, au_far_dur], "song_res": [song_res_dd, au_song_res],
                "song_aspect": [song_aspect_dd, au_song_asp],
                "dlg_res": [e2_dlg_res, e4_dlg_res, au_dlg_res], "dlg_aspect": [e2_dlg_asp, e4_dlg_asp, au_dlg_asp]}
    for _k, _cs in _LTX_MIR.items():
        for _c in _cs:
            _o = [x for x in _cs if x is not _c]
            _ev = _c.input if hasattr(_c, "input") else _c.change
            _ev(_ltx_mk(_k, len(_o)), [_c], _o or None)
    btn_song.click(ui_subir_cancion, [song_file], [song_status])

    def _make_gen5s_fn(key):
        def gen5s_wrapper(personaje, seed_v):
            return ui_gen_5s_individual(key, personaje, seed_v)
        return gen5s_wrapper

    btn_gen5s_E.click(_make_gen5s_fn("E"), [personaje, seed_5s],
                      [clips5s_status, clips5s_gallery]).then(
        ui_refrescar_progreso, [], [progreso_box, progreso_gallery]).then(_bump, [tick_5s], [tick_5s])
    btn_gen5s_F.click(_make_gen5s_fn("F"), [personaje, seed_5s],
                      [clips5s_status, clips5s_gallery]).then(
        ui_refrescar_progreso, [], [progreso_box, progreso_gallery]).then(_bump, [tick_5s], [tick_5s])
    btn_gen5s_G.click(_make_gen5s_fn("G"), [personaje, seed_5s],
                      [clips5s_status, clips5s_gallery]).then(
        ui_refrescar_progreso, [], [progreso_box, progreso_gallery]).then(_bump, [tick_5s], [tick_5s])
    btn_gen5s_all.click(ui_gen_5s_todos, [personaje, seed_5s],
                        [clips5s_status, clips5s_gallery]).then(
        ui_refrescar_progreso, [], [progreso_box, progreso_gallery]).then(_bump, [tick_5s], [tick_5s])

    # v8: clips EXTRA (E+/F+/G+): un clip adicional por imagen, sin tocar los existentes
    for _btn, _k in ((btn_gen5s_Ex, "E"), (btn_gen5s_Fx, "F"), (btn_gen5s_Gx, "G")):
        _btn.click(_make_gen5s_extra_fn(_k), [personaje, seed_5s],
                   [clips5s_status, clips5s_gallery]).then(
            ui_refrescar_progreso, [], [progreso_box, progreso_gallery]).then(_bump, [tick_5s], [tick_5s])

    btn_regen5s.click(ui_regen_5s, [tpl_dd, personaje, clip_path_in, seed_regen5s],
                      [regen5s_status, clip_path_in]).then(_bump, [tick_5s], [tick_5s])

    # v8: lista de clips 5s + añadir clip propio
    btn_list5s.click(_bump, [tick_5s], [tick_5s])
    btn_add5s.click(ui_agregar_clip5s, [add5s_key, add5s_file], [clips5s_st2]).then(
        _bump, [tick_5s], [tick_5s])
    btn_tl.click(ui_build_tl, [tpl_dd, personaje, intro_seconds], [tl_status, tl_html]).then(
        ui_refrescar_progreso, [], [progreso_box, progreso_gallery]).then(_bump, [tick_e3], [tick_e3])
    btn_render3.click(ui_renderizar_cancion, [tpl_dd, personaje],
                      [e3_status, e3_video]).then(
        ui_refrescar_progreso, [], [progreso_box, progreso_gallery]).then(_bump, [tick_e3], [tick_e3])

    # ── ETAPA 2 ───────────────────────────────────────────
    btn_e2.click(
        fn=ui_etapa2,
        inputs=[tpl_dd, personaje, historia_box, model_id_e2, fast_mode_e2],
        outputs=[e2_status, e2_video]
    ).then(ui_refrescar_progreso, [], [progreso_box, progreso_gallery]).then(_bump, [tick_e2], [tick_e2])

    btn_e2_regen_guion.click(
        fn=lambda tpl, personaje, historia, mid: ui_generar_solo_guion(
            "etapa2", tpl, personaje, historia, mid),
        inputs=[tpl_dd, personaje, historia_box, model_id_e2],
        outputs=[e2_status, guion_edit_box, etapa_edit_sel]
    ).then(_bump, [tick_e2], [tick_e2])

    btn_e2_actual.click(
        fn=ui_etapa2_guion_actual,
        inputs=[tpl_dd, personaje, historia_box, model_id_e2, fast_mode_e2],
        outputs=[e2_status, e2_video]
    ).then(ui_refrescar_progreso, [], [progreso_box, progreso_gallery]).then(_bump, [tick_e2], [tick_e2])

    btn_regen_frag.click(
        fn=ui_regen_frag,
        inputs=[tpl_dd, personaje, etapa_sel_e2, frag_idx_n, nuevo_txt],
        outputs=[regen_frag_st]
    ).then(_bump, [tick_e2], [tick_e2]).then(_bump, [tick_e4], [tick_e4])

    # ── ETAPA 4 ───────────────────────────────────────────
    btn_e4.click(
        fn=ui_etapa4,
        inputs=[tpl_dd, personaje, model_id_e4, fast_mode_e4],
        outputs=[e4_status, e4_video]
    ).then(ui_refrescar_progreso, [], [progreso_box, progreso_gallery]).then(_bump, [tick_e4], [tick_e4])

    btn_e4_regen_guion.click(
        fn=lambda tpl, personaje, mid: ui_generar_solo_guion(
            "etapa4", tpl, personaje, "", mid),
        inputs=[tpl_dd, personaje, model_id_e4],
        outputs=[e4_status, guion_edit_box, etapa_edit_sel]
    ).then(_bump, [tick_e4], [tick_e4])

    btn_e4_actual.click(
        fn=ui_etapa4_guion_actual,
        inputs=[tpl_dd, personaje, model_id_e4, fast_mode_e4],
        outputs=[e4_status, e4_video]
    ).then(ui_refrescar_progreso, [], [progreso_box, progreso_gallery]).then(_bump, [tick_e4], [tick_e4])

    btn_regen_frag_e4.click(
        fn=ui_regen_frag,
        inputs=[tpl_dd, personaje, etapa_sel_e4, frag_idx_e4, nuevo_txt_e4],
        outputs=[regen_frag_st_e4]
    ).then(_bump, [tick_e2], [tick_e2]).then(_bump, [tick_e4], [tick_e4])

    btn_ver_guion.click(fn=ui_ver_guion_json, inputs=[], outputs=[guion_json_box])

    # ── RENDER FINAL ──────────────────────────────────────
    btn_final.click(
        fn=ui_render_final,
        inputs=[tpl_dd, etapa2_on, etapa4_on],
        outputs=[final_status, final_video]
    ).then(ui_refrescar_progreso, [], [progreso_box, progreso_gallery])

    # ── UTILIDADES ────────────────────────────────────────
    btn_estado.click(fn=ui_ver_estado, inputs=[], outputs=[util_status])
    btn_dataset.click(fn=ui_publicar_dataset, inputs=[], outputs=[util_status])

    # ══════════════════════════════════════════════════════
    # v7 — WIRING NUEVO
    # ══════════════════════════════════════════════════════
    # Diagnostico del entorno LTX
    btn_ltx_check.click(ui_probar_entorno_ltx, [], [ltx_check_status])

    # Preview con imagenes + Continuar con lipsync (E2)
    _e2_in = [tpl_dd, personaje, historia_box, model_id_e2, fast_mode_e2]
    btn_e2_preview.click(ui_etapa2_preview, _e2_in, [e2_status, e2_preview_video]).then(
        ui_refrescar_progreso, [], [progreso_box, progreso_gallery]).then(
        _bump, [tick_e2], [tick_e2])
    btn_e2_continuar.click(ui_etapa2_guion_actual, _e2_in, [e2_status, e2_video]).then(
        ui_refrescar_progreso, [], [progreso_box, progreso_gallery]).then(
        _bump, [tick_e2], [tick_e2])

    # Preview con imagenes + Continuar con lipsync (E4)
    _e4_in = [tpl_dd, personaje, model_id_e4, fast_mode_e4]
    btn_e4_preview.click(ui_etapa4_preview, _e4_in, [e4_status, e4_preview_video]).then(
        ui_refrescar_progreso, [], [progreso_box, progreso_gallery]).then(
        _bump, [tick_e4], [tick_e4])
    btn_e4_continuar.click(ui_etapa4_guion_actual, _e4_in, [e4_status, e4_video]).then(
        ui_refrescar_progreso, [], [progreso_box, progreso_gallery]).then(
        _bump, [tick_e4], [tick_e4])

    # Refresco manual de los timelines por fragmentos
    btn_tl_e2.click(_bump, [tick_e2], [tick_e2])
    btn_tl_e4.click(_bump, [tick_e4], [tick_e4])
    btn_tl_frag.click(_bump, [tick_e3], [tick_e3])

    # Automatizacion
    btn_auto_todo.click(
        ui_automatizar,
        [tpl_dd, personaje, historia_box, model_id_e2, model_id_e4, fast_mode_e2, fast_mode_e4,
         auto_e2, auto_e3, auto_e4, auto_modo_dialogo, auto_modo_guion,
         auto_reconstruir_tl, auto_render_final],
        [auto_status, auto_video]
    ).then(ui_refrescar_progreso, [], [progreso_box, progreso_gallery]).then(
        _bump3, [tick_e2, tick_e3, tick_e4], [tick_e2, tick_e3, tick_e4])
    btn_auto_e2e4.click(
        ui_auto_e2_e4,
        [tpl_dd, personaje, historia_box, model_id_e2, model_id_e4, fast_mode_e2, fast_mode_e4,
         auto_modo_dialogo, auto_modo_guion, auto_render_final],
        [auto_status, auto_video]
    ).then(ui_refrescar_progreso, [], [progreso_box, progreso_gallery]).then(
        _bump3, [tick_e2, tick_e3, tick_e4], [tick_e2, tick_e3, tick_e4])
    btn_auto_cancion.click(
        ui_auto_solo_cancion,
        [tpl_dd, personaje, auto_reconstruir_tl],
        [auto_status, auto_video]
    ).then(ui_refrescar_progreso, [], [progreso_box, progreso_gallery]).then(
        _bump3, [tick_e2, tick_e3, tick_e4], [tick_e2, tick_e3, tick_e4])

    # ── v8: contexto de guiones (E4 y Automatización) ──────
    ctx_e4_box.change(fn=_ctx_setter("e4_ui"), inputs=[ctx_e4_box], outputs=None,
                      queue=False, show_progress="hidden", api_name=False)
    auto_ctx_e2.change(fn=_ctx_setter("e2_auto"), inputs=[auto_ctx_e2], outputs=None,
                       queue=False, show_progress="hidden", api_name=False)
    auto_ctx_e4.change(fn=_ctx_setter("e4_auto"), inputs=[auto_ctx_e4], outputs=None,
                       queue=False, show_progress="hidden", api_name=False)
    demo.load(fn=_ctx_cargar, inputs=[], outputs=[ctx_e4_box, auto_ctx_e2, auto_ctx_e4],
              show_progress="hidden", api_name=False)
    # ── Logs (auto-refresh cada 2.5s) ─────────────────────
    timer.tick(fn=lambda: tail_log(12000), inputs=[], outputs=[logs_box])


# ════════════════════════════════════════════════════════════════
# LAUNCH — CSS y theme van aquí en Gradio 6
# ════════════════════════════════════════════════════════════════
demo.queue(max_size=10).launch(share=True, show_error=True)