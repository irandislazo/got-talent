# ═══════════════════════════════════════════════════════════════
# CELDA 5: ETAPA 1 — GENERACIÓN DE IMÁGENES (A→G) CON sd-cli (Qwen-Image-2.1)
# Implementa EXACTAMENTE la lógica de la especificación:
#  - A: text2img (9:16)
#  - B: edición desde A usando -r y prompt 2
#  - C/E/F/G: edición de plantilla usando SOLO B (como identidad) + plantilla como base (2 refs: -r B -r plantilla)
#  - D: edición desde C (close-up)
# Documentación sd.cpp Qwen-Image-2.1 (flags -r / --llm_vision / cfg 6 / euler / dims /32). <!--citation:1-->
# ═══════════════════════════════════════════════════════════════
import tempfile
import subprocess

def _ffprobe_duration(path: str) -> Optional[float]:
    try:
        cmd = ["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1", path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return float((r.stdout or "").strip())
    except Exception:
        return None

def sdcli_help_flags(sdcli_path: str) -> set:
    try:
        r = subprocess.run([sdcli_path, "--help"], capture_output=True, text=True, timeout=30)
        txt = (r.stdout or "") + (r.stderr or "")
        return set(re.findall(r'(--[a-zA-Z0-9][a-zA-Z0-9_-]*)', txt))
    except Exception:
        return set()

class SDQwenImage21:
    def __init__(self, sdcli_path: str, model_paths: Dict[str,str]):
        self.sdcli = sdcli_path
        self.mp = model_paths
        self.flags = sdcli_help_flags(sdcli_path)

    def _supports(self, flag: str) -> bool:
        return (not self.flags) or (flag in self.flags)

    def _run(self, args: List[str]):
        log("sd-cli CMD: " + " ".join(shlex_quote(a) for a in args))
        t0 = time.time()
        r = subprocess.run(args, capture_output=True, text=True)
        dt = time.time() - t0
        if r.returncode != 0:
            tail = ((r.stderr or "") + "\n" + (r.stdout or ""))[-4000:]
            raise RuntimeError(f"sd-cli falló (rc={r.returncode}) en {dt:.1f}s:\n{tail}")
        log(f"sd-cli OK en {dt:.1f}s")
        return r

    def generate_text2img(self, prompt: str, out_png: str,
                          w: int, h: int, steps: int, cfg: float, seed: int = -1,
                          offload_to_cpu: bool = True, use_fa: bool = True):
        os.makedirs(os.path.dirname(out_png), exist_ok=True)

        # Base según doc qwen_image_2.1.md: diffusion+vae+llm, cfg=6.0, euler, --offload-to-cpu, y --fa si existe. <!--citation:1-->
        cmd = [
            self.sdcli,
            "--diffusion-model", self.mp["diffusion"],
            "--vae", self.mp["vae"],
            "--llm", self.mp["encoder"],
            "-p", prompt,
            "--cfg-scale", str(cfg),
            "--sampling-method", "euler",
            "--steps", str(int(steps)),
            "--seed", str(int(seed)),
            "-o", out_png,
            "-W", str(int(w)),
            "-H", str(int(h)),
        ]

        if offload_to_cpu and self._supports("--offload-to-cpu"):
            cmd.append("--offload-to-cpu")

        # FA: según build puede ser --fa o --diffusion-fa; priorizar --fa por doc. <!--citation:1-->
        if use_fa:
            if self._supports("--fa"): cmd.append("--fa")
            elif self._supports("--diffusion-fa"): cmd.append("--diffusion-fa")

        self._run(cmd)
        if not os.path.exists(out_png) or os.path.getsize(out_png) < 5000:
            raise RuntimeError("No se generó PNG válido.")
        return out_png

    def edit_with_refs(self, prompt: str, refs: List[str], out_png: str,
                       w: int, h: int, steps: int, cfg: float, seed: int = -1,
                       offload_to_cpu: bool = True, use_fa: bool = True):
        os.makedirs(os.path.dirname(out_png), exist_ok=True)

        cmd = [
            self.sdcli,
            "--diffusion-model", self.mp["diffusion"],
            "--vae", self.mp["vae"],
            "--llm", self.mp["encoder"],
            "--llm_vision", self.mp["mmproj"],   # requerido para edición con encoder GGUF <!--citation:1-->
        ]
        for rimg in refs:
            cmd += ["-r", rimg]  # múltiples refs repitiendo -r <!--citation:1-->

        cmd += [
            "-p", prompt,
            "--cfg-scale", str(cfg),
            "--sampling-method", "euler",
            "--steps", str(int(steps)),
            "--seed", str(int(seed)),
            "-o", out_png,
            "-W", str(int(w)),
            "-H", str(int(h)),
        ]

        if offload_to_cpu and self._supports("--offload-to-cpu"):
            cmd.append("--offload-to-cpu")
        if use_fa:
            if self._supports("--fa"): cmd.append("--fa")
            elif self._supports("--diffusion-fa"): cmd.append("--diffusion-fa")

        self._run(cmd)
        if not os.path.exists(out_png) or os.path.getsize(out_png) < 5000:
            raise RuntimeError("No se generó PNG válido.")
        return out_png

def shlex_quote(s: str) -> str:
    import shlex
    return shlex.quote(str(s))

def etapa1_generar_imagenes(
    vp: VideoProject,
    plantilla_slug: str,
    personaje_key: str,
    personaje_plantilla_key: str,
    edad: int,
    vestuario: str,
    planos_lejanos_a_usar: int,
    diffusion_quant="Q8_0",
    encoder_quant="Q4_K_M",
    mmproj_quant="Q8_0",
    steps_A=30,
    steps_edit=30,
    cfg=2.0,
    seed: int = -1,
) -> Dict[str, str]:
    """
    Genera A,B,C,D y E/F/G opcionales.
    - A y B: 9:16 (múltiplos de 32)
    - C,D,E,F,G: 16:9 (múltiplos de 32)
    """
    tpl = cargar_plantilla(plantilla_slug)
    base = tpl["_base_dir"]

    # resoluciones (múltiplos de 32): 9:16 → 832x1472; 16:9 → 1376x768
    W_9_16, H_9_16 = 832, 1472
    W_16_9, H_16_9 = 1376, 768

    img_dir = os.path.join(vp.video_dir, "etapa1_imagenes")
    os.makedirs(img_dir, exist_ok=True)

    # Prompts (editables: guardamos en prompts.json)
    p1 = render_prompt_1(personaje_key, edad, vestuario)
    p2 = PROMPT_2_CUERPO_COMPLETO
    p3 = render_prompt_3(personaje_key, personaje_plantilla_key)
    p4 = PROMPT_4_CLOSEUP

    vp.prompts.setdefault("etapa1", {})
    vp.prompts["etapa1"].update({
        "prompt_1": p1,
        "prompt_2": p2,
        "prompt_3": p3,
        "prompt_4": p4,
    })

    vp.params.setdefault("etapa1", {})
    vp.params["etapa1"].update({
        "diffusion_quant": diffusion_quant,
        "encoder_quant": encoder_quant,
        "mmproj_quant": mmproj_quant,
        "steps_A": steps_A,
        "steps_edit": steps_edit,
        "cfg": cfg,
        "seed": seed,
        "res_A_B": [W_9_16, H_9_16],
        "res_rest": [W_16_9, H_16_9],
    })
    vp.save()

    # Modelos
    mp = ensure_qwen_image21_models(diffusion_quant, encoder_quant, mmproj_quant)
    sd = SDQwenImage21(SDCLI_PATH, mp)

    # Paths objetivo (A..G)
    paths = {
        "A": os.path.join(img_dir, "concursante_base.png"),
        "B": os.path.join(img_dir, "concursante_cuerpo_completo.png"),
        "C": os.path.join(img_dir, "concursante_plano_cercano.png"),
        "D": os.path.join(img_dir, "concursante_closeup.png"),
        "E": os.path.join(img_dir, "concursante_plano_lejano_1.png"),
        "F": os.path.join(img_dir, "concursante_plano_lejano_2.png"),
        "G": os.path.join(img_dir, "concursante_plano_lejano_3.png"),
    }

    # [PASO 1] A (text2img)
    log("[ETAPA 1] PASO 1/7: Generando A (text2img)...")
    sd.generate_text2img(p1, paths["A"], W_9_16, H_9_16, steps_A, cfg, seed=seed)

    # [PASO 2] B (edit desde A)
    log("[ETAPA 1] PASO 2/7: Generando B (edit desde A)...")
    sd.edit_with_refs(p2, refs=[paths["A"]], out_png=paths["B"], w=W_9_16, h=H_9_16, steps=steps_edit, cfg=cfg, seed=seed)

    # Plantilla plano cercano
    tpl_close = os.path.join(base, PLANTILLA_SUBDIRS["escenario_cercano"], "plantilla_plano_cercano.png")
    if not os.path.exists(tpl_close):
        raise RuntimeError(f"Falta plantilla_plano_cercano.png en {tpl_close}")

    # [PASO 3] C (edit: refs=[B, plantilla_cercano])
    log("[ETAPA 1] PASO 3/7: Generando C (edit plantilla plano cercano)...")
    sd.edit_with_refs(p3, refs=[paths["B"], tpl_close], out_png=paths["C"], w=W_16_9, h=H_16_9, steps=steps_edit, cfg=cfg, seed=seed)

    # [PASO 4] D (close-up desde C)
    log("[ETAPA 1] PASO 4/7: Generando D (close-up desde C)...")
    sd.edit_with_refs(p4, refs=[paths["C"]], out_png=paths["D"], w=W_16_9, h=H_16_9, steps=steps_edit, cfg=cfg, seed=seed)

    # [PASO 5-7] E/F/G opcionales (hasta planos_lejanos_a_usar)
    for i, key in enumerate(["E","F","G"], start=1):
        if i > int(planos_lejanos_a_usar):
            continue
        tpl_far = os.path.join(base, PLANTILLA_SUBDIRS["escenario_lejano"], f"plantilla_plano_lejano_{i}.png")
        if not os.path.exists(tpl_far):
            raise RuntimeError(f"Falta {os.path.basename(tpl_far)} en {tpl_far}")
        log(f"[ETAPA 1] PASO {4+i}/7: Generando {key} (edit plantilla plano lejano #{i})...")
        sd.edit_with_refs(p3, refs=[paths["B"], tpl_far], out_png=paths[key], w=W_16_9, h=H_16_9, steps=steps_edit, cfg=cfg, seed=seed)

    # Estado + persistencia
    for k,v in paths.items():
        vp.estado["aprobados"][f"etapa1_{k}"] = bool(os.path.exists(v))
    vp.save()

    log("[ETAPA 1] COMPLETADA.")
    return paths

log("CELDA 5 COMPLETADA.")