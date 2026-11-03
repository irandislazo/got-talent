# ═══════════════════════════════════════════════════════════════
# CELDA 3: UTILIDADES BASE + DISCO + PERSISTENCIA + DESCARGA MODELOS
# (Kaggle-friendly, sin reventar /kaggle/working)
# ═══════════════════════════════════════════════════════════════
import os, re, json, time, shutil, subprocess, math, glob
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from huggingface_hub import hf_hub_download

# ---------------- LOGS (a archivo + consola) ----------------
PROJECT_LOG_FILE = os.environ.get("PROJECT_LOG_FILE", f"{OUTPUT_DIR}/run.log")

def _append_log(line: str):
    try:
        with open(PROJECT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")
    except Exception:
        pass

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    _append_log(line)

def tail_log(max_chars: int = 12000) -> str:
    try:
        if not os.path.exists(PROJECT_LOG_FILE):
            return ""
        with open(PROJECT_LOG_FILE, "rb") as f:
            data = f.read()
        text = data.decode("utf-8", errors="ignore")
        return text[-max_chars:]
    except Exception:
        return ""

# ---------------- DISCO (working vs tmp) ----------------
def free_gb(path: str) -> float:
    st = os.statvfs(path)
    return (st.f_bavail * st.f_frsize) / 1024**3

log(f"Disco libre /kaggle/working: {free_gb('/kaggle/working'):.2f} GB")
log(f"Disco libre /kaggle/tmp:     {free_gb('/kaggle/tmp'):.2f} GB")

# ---------------- Kaggle INPUT find (dataset adjunto) ----------------
def find_in_kaggle_input(filename: str) -> str:
    if not os.path.isdir(KAGGLE_INPUT_DIR):
        return ""
    for root, _, files in os.walk(KAGGLE_INPUT_DIR):
        if filename in files:
            p = os.path.join(root, filename)
            if os.path.getsize(p) > 1024 * 1024:
                return p
    return ""

# ---------------- HF download (xet-safe) ----------------
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # acelera si hf_transfer está disponible

def hf_download_if_missing(repo_id: str, filename: str, dst_dir: str) -> str:
    """
    Descarga vía huggingface_hub (maneja xet / LFS). Si el archivo ya existe en dst_dir, lo reutiliza.
    """
    os.makedirs(dst_dir, exist_ok=True)
    out_path = os.path.join(dst_dir, filename)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1024 * 1024:
        return out_path

    # Si está en /kaggle/input, usarlo sin descargar:
    k_in = find_in_kaggle_input(filename)
    if k_in:
        return k_in

    log(f"[HF] Descargando {repo_id}/{filename} → {out_path}")
    tmp = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=dst_dir, local_dir_use_symlinks=False)
    # hf_hub_download devuelve path real; asegurar nombre esperado:
    if tmp != out_path and os.path.exists(tmp):
        shutil.copyfile(tmp, out_path)
    return out_path

# ---------------- PUBLICAR MODELOS COMO DATASET (opcional) ----------------
# (Reusa tu función original, pero generalizada. Kaggle dataset = persistencia gratuita fuera de /working.)
def publish_dir_as_kaggle_dataset(dataset_slug: str, src_dir: str, title: str = None, update_existing: bool = True) -> str:
    if PLATFORM != "kaggle":
        return "Solo Kaggle."

    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        os.environ["KAGGLE_USERNAME"] = secrets.get_secret("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = secrets.get_secret("KAGGLE_KEY")
    except Exception as e:
        return (f"No se pudieron leer credenciales desde Kaggle Secrets ({e}). "
                f"Add-ons → Secrets: KAGGLE_USERNAME y KAGGLE_KEY.")

    staging = f"{OUTPUT_DIR}/_dataset_staging/{dataset_slug}"
    os.makedirs(staging, exist_ok=True)

    # Copiar todo (solo archivos) para versionar
    for root, _, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        for fn in files:
            src = os.path.join(root, fn)
            dst = os.path.join(staging, rel, fn)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
                shutil.copyfile(src, dst)

    meta = {
        "title": title or dataset_slug,
        "id": f"{os.environ.get('KAGGLE_USERNAME', 'usuario')}/{dataset_slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(os.path.join(staging, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if update_existing:
        cmd = f"kaggle datasets version -p '{staging}' -m 'Actualización modelos/pipeline' -r zip"
    else:
        cmd = f"kaggle datasets create -p '{staging}' -r zip"

    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout + "\n" + r.stderr

# ---------------- MODELOS: Qwen-Image 2.1 (sd-cli) ----------------
QWEN_IMAGE_REPO = "leejet/Qwen-Image-2.1-GGUF"
QWEN_VL_REPO    = "Qwen/Qwen3-VL-8B-Instruct-GGUF"
QWEN_VAE_REPO   = "Comfy-Org/Qwen-Image-2.1"
QWEN_VAE_FILE   = "qwen_image_2.1_vae_bf16.safetensors"

QWEN_IMAGE_DIFFUSION_QUANTS = {
    "Q2_K":  "qwen_image_2.1-Q2_K.gguf",
    "Q3_K":  "qwen_image_2.1-Q3_K.gguf",
    "Q4_0":  "qwen_image_2.1-Q4_0.gguf",
    "Q4_K":  "qwen_image_2.1-Q4_K.gguf",
    "Q5_0":  "qwen_image_2.1-Q5_0.gguf",
    "Q6_K":  "qwen_image_2.1-Q6_K.gguf",
    "Q8_0":  "qwen_image_2.1-Q8_0.gguf",
}

QWEN_VL_ENCODERS = {
    "Q4_K_M": "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
    "Q8_0":   "Qwen3VL-8B-Instruct-Q8_0.gguf",
    "F16":    "Qwen3VL-8B-Instruct-F16.gguf",
}

QWEN_VL_MMPROJ = {
    "F16":  "mmproj-Qwen3VL-8B-Instruct-F16.gguf",
    "Q8_0": "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf",
}

def ensure_qwen_image21_models(diffusion_quant="Q8_0", encoder_quant="Q4_K_M", mmproj_quant="Q8_0") -> Dict[str, str]:
    """
    Descarga/ubica diffusion GGUF + encoder GGUF + mmproj + VAE.
    Para edición se requiere mmproj con --llm_vision (documentado). <!--citation:1-->
    """
    diff_file = QWEN_IMAGE_DIFFUSION_QUANTS[diffusion_quant]
    enc_file  = QWEN_VL_ENCODERS[encoder_quant]
    mm_file   = QWEN_VL_MMPROJ[mmproj_quant]

    diff_path = hf_download_if_missing(QWEN_IMAGE_REPO, diff_file, LOCAL_MODELS_DIR)
    enc_path  = hf_download_if_missing(QWEN_VL_REPO, enc_file, LOCAL_MODELS_DIR)
    mm_path   = hf_download_if_missing(QWEN_VL_REPO, mm_file, LOCAL_MODELS_DIR)
    vae_path  = hf_download_if_missing(QWEN_VAE_REPO, f"vae/{QWEN_VAE_FILE}", LOCAL_MODELS_DIR)

    # hf_hub_download con subcarpeta 'vae/' queda en LOCAL_MODELS_DIR/vae/...
    if "/vae/" in vae_path or vae_path.endswith(QWEN_VAE_FILE):
        pass
    else:
        # fallback
        pass

    return {"diffusion": diff_path, "encoder": enc_path, "mmproj": mm_path, "vae": vae_path}

# ---------------- MODELOS: Qwen3-TTS (GGUF) ----------------
QWEN_TTS_REPO = "Serveurperso/Qwen3-TTS-GGUF"  # listado de talkers + tokenizers en HF <!--citation:7-->

QWEN_TTS_PRESETS = {
    "BASE_Q8": {
        "talker": "qwen-talker-1.7b-base-Q8_0.gguf",
        "codec":  "qwen-tokenizer-12hz-Q8_0.gguf",
        "mode": "base"
    },
    "BASE_Q4": {
        "talker": "qwen-talker-1.7b-base-Q4_K_M.gguf",
        "codec":  "qwen-tokenizer-12hz-Q4_K_M.gguf",
        "mode": "base"
    },
    "VOICEDESIGN_Q8": {
        "talker": "qwen-talker-1.7b-voicedesign-Q8_0.gguf",
        "codec":  "qwen-tokenizer-12hz-Q8_0.gguf",
        "mode": "voicedesign"
    },
    "CUSTOMVOICE_Q8": {
        "talker": "qwen-talker-1.7b-customvoice-Q8_0.gguf",
        "codec":  "qwen-tokenizer-12hz-Q8_0.gguf",
        "mode": "customvoice"
    },
}

def ensure_qwen_tts_models(preset_key="VOICEDESIGN_Q8") -> Dict[str, str]:
    p = QWEN_TTS_PRESETS[preset_key]
    talker = hf_download_if_missing(QWEN_TTS_REPO, p["talker"], LOCAL_MODELS_DIR)
    codec  = hf_download_if_missing(QWEN_TTS_REPO, p["codec"],  LOCAL_MODELS_DIR)
    return {"talker": talker, "codec": codec, "mode": p["mode"]}

# ---------------- Workaround PR #483 (ComfyUI-GGUF) ----------------
def patch_comfyui_gguf_for_qwen_image21(comfy_custom_nodes_dir: str):
    """
    Workaround documentado en PR #483: añade detección de arquitectura qwen_image cuando
    sd.cpp exporta GGUF sin general.architecture. <!--citation:6-->
    Solo se aplica si tienes ComfyUI-GGUF instalado (opcional).
    """
    gguf_dir = os.path.join(comfy_custom_nodes_dir, "ComfyUI-GGUF")
    convert_py = os.path.join(gguf_dir, "tools", "convert.py")
    if not os.path.exists(convert_py):
        return "ComfyUI-GGUF no encontrado; patch omitido."

    txt = open(convert_py, "r", encoding="utf-8", errors="ignore").read()
    if "class ModelQwenImage" in txt or "ModelQwenImage" in txt:
        return "Patch PR #483 ya presente (ModelQwenImage)."

    # Inserción mínima: agregar clase cerca de otros ModelTemplate detectors
    # (No cambia nada más; solo habilita fingerprint por keys).
    inject = """
class ModelQwenImage(ModelTemplate):
    arch = "qwen_image"
    keys_detect = [
        ("img_in.weight", "modulation.1.weight"),
        ("img_in.weight", "img_mlp.gate_layer.weight"),
    ]
"""
    # Heurística: insertar antes de la lista/registro final si existe, sino al final.
    marker = "class Model"
    idx = txt.rfind("class Model")
    if idx == -1:
        txt2 = txt + "\n" + inject + "\n"
    else:
        # insertar antes del último bloque de modelos para minimizar riesgos
        txt2 = txt[:idx] + inject + "\n" + txt[idx:]

    open(convert_py, "w", encoding="utf-8").write(txt2)
    return "Patch PR #483 aplicado en tools/convert.py (ModelQwenImage)."

log("CELDA 3 COMPLETADA.")