# ═══════════════════════════════════════════════════════════════
# CELDA 12 (v3 — DISK-SAFE): LTX-2.5 MODELOS
# CAMBIO CRÍTICO vs v2:
#   - TODO va a /tmp/models (NO toca /kaggle/working)
#   - Wan2GP/models/ y Wan2GP/ckpts/ son SOLO symlinks → /tmp/models/
#   - Companions (~7GB) + Transformer (19.5GB) + Gemma4 → /tmp/
#   - /kaggle/working permanece libre para el pipeline
# ═══════════════════════════════════════════════════════════════
import os, sys, time, json, shutil

for mod in list(sys.modules.keys()):
    if mod.startswith('huggingface_hub'):
        del sys.modules[mod]

from huggingface_hub import hf_hub_download

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# ── Rutas: TODO en /tmp (scratch ilimitado, NO /kaggle/working) ──
TMP_MODELS    = '/tmp/models'          # único directorio real de modelos
WAN2GP_DIR    = '/kaggle/working/Wan2GP'
MODEL_DIR     = os.path.join(WAN2GP_DIR, 'models')   # SOLO symlinks
CKPTS_DIR     = os.path.join(WAN2GP_DIR, 'ckpts')    # SOLO symlinks
OUTPUTS_DIR   = os.path.join(WAN2GP_DIR, 'outputs')

for d in [TMP_MODELS, MODEL_DIR, CKPTS_DIR, OUTPUTS_DIR,
          '/kaggle/working/outputs', '/tmp/ltx25_outputs']:
    os.makedirs(d, exist_ok=True)

LTX25_REPO    = 'DeepBeepMeep/LTX-2'
GEMMA4_SUB    = 'gemma4-12b-ltx-v1'
GEMMA4_TMP    = os.path.join(TMP_MODELS, GEMMA4_SUB)
os.makedirs(GEMMA4_TMP, exist_ok=True)

def safe_symlink(src, dest):
    """Crea symlink solo si el destino no existe."""
    abs_src = os.path.abspath(src)
    if not os.path.exists(abs_src):
        return
    if os.path.lexists(dest):
        return
    try:
        os.symlink(abs_src, dest)
    except Exception as e:
        log(f"  [symlink] {os.path.basename(dest)}: {e}")

def hf_to_tmp(filename, subfolder=None):
    """
    Descarga a /tmp/models (siempre).
    Si subfolder, descarga a /tmp/models/<subfolder>/<filename>.
    Retorna la ruta real del archivo descargado.
    """
    if subfolder:
        dst_dir = os.path.join(TMP_MODELS, subfolder)
        dst_path = os.path.join(dst_dir, filename)
    else:
        dst_dir = TMP_MODELS
        dst_path = os.path.join(dst_dir, filename)

    os.makedirs(dst_dir, exist_ok=True)

    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 1024 * 1024:
        return dst_path
    # v8: archivos pequeños (config/tokenizer): reutilizar solo si son válidos (no vacíos / JSON correcto)
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        try:
            if filename.endswith('.json'):
                json.load(open(dst_path, 'r', encoding='utf-8'))
            return dst_path
        except Exception:
            log(f"  [fix] {filename} inválido → se vuelve a descargar")
            os.remove(dst_path)
    elif os.path.lexists(dst_path):
        os.remove(dst_path)

    # Buscar en /kaggle/input primero
    for root, _, files in os.walk('/kaggle/input'):
        if filename in files:
            p = os.path.join(root, filename)
            if os.path.getsize(p) > 1024 * 1024:
                log(f"  [cache] {filename} ← /kaggle/input")
                return p

    log(f"  [HF↓] {filename} → /tmp/models/{subfolder or ''}")
    sys.stdout.flush()
    t0 = time.time()

    if subfolder:
        got = hf_hub_download(
            repo_id=LTX25_REPO, subfolder=subfolder,
            filename=filename, local_dir=dst_dir
        )
        # hf_hub_download con subfolder puede crear un nivel extra
        nested = os.path.join(dst_dir, subfolder, filename)
        if os.path.exists(nested) and not os.path.exists(dst_path):
            shutil.move(nested, dst_path)
        # v8: usar la ruta que DEVUELVE hf_hub_download (cambia entre versiones de huggingface_hub)
        elif (not os.path.exists(dst_path) and got and os.path.exists(got)
              and os.path.abspath(got) != os.path.abspath(dst_path)):
            shutil.copyfile(got, dst_path)
    else:
        hf_hub_download(repo_id=LTX25_REPO, filename=filename, local_dir=dst_dir)

    # Limpiar cache HF para no desperdiciar espacio en /tmp
    for cache_dir in [os.path.join(dst_dir, '.cache'), os.path.join(TMP_MODELS, '.cache')]:
        shutil.rmtree(cache_dir, ignore_errors=True)

    elapsed = time.time() - t0
    size_gb = os.path.getsize(dst_path) / 1024**3 if os.path.exists(dst_path) else 0
    log(f"    ✓ {filename} ({size_gb:.2f} GB, {elapsed:.0f}s)")
    return dst_path

def link_to_wan2gp(real_path, filename, subfolder=None):
    """
    Crea symlinks en Wan2GP/models/ y Wan2GP/ckpts/ apuntando al archivo real en /tmp/.
    NO copia archivos, solo symlinks.
    """
    if subfolder:
        m_dir = os.path.join(MODEL_DIR, subfolder)
        c_dir = os.path.join(CKPTS_DIR, subfolder)
        os.makedirs(m_dir, exist_ok=True)
        os.makedirs(c_dir, exist_ok=True)
        safe_symlink(real_path, os.path.join(m_dir, filename))
        safe_symlink(real_path, os.path.join(c_dir, filename))
    else:
        safe_symlink(real_path, os.path.join(MODEL_DIR, filename))
        safe_symlink(real_path, os.path.join(CKPTS_DIR, filename))

_SKIP25 = globals().get('LTX_VERSION', '2.3') == '2.3'   # LTX-2.3 activo: no se descargan los ~30 GB de LTX-2.5
# ═══ 1. TRANSFORMER (19.5 GB) → /tmp/models/ ═══
TRANSFORMER_FILE = 'ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors'
if not _SKIP25:
    log('Verificando Transformer LTX-2.5...')
    tf_real = hf_to_tmp(TRANSFORMER_FILE)
    link_to_wan2gp(tf_real, TRANSFORMER_FILE)

# ═══ 2. COMPANION FILES (~7 GB) → /tmp/models/ ═══
COMPANION_FILES = [
    'ltx-2.5-spatial-upscaler-x2-1.0_bf16.safetensors',
    'ltx-2.5-temporal-upscaler-x2-1.0_bf16.safetensors',
    'ltx-2.5-22b_video_embeddings_connector_int8_convrot.safetensors',
    'ltx-2.5-22b_audio_embeddings_connector_int8_convrot.safetensors',
    'ltx-2.5-22b_text_embedding_projection_bf16.safetensors',
    'ltx-2.5-22b_video_vae_bf16.safetensors',
    'ltx-2.5-22b_audio_vae_bf16.safetensors',
    'ltx-2.5-22b_vocoder_bf16.safetensors',
]
if not _SKIP25:
    log('Verificando companion files...')
    for f in COMPANION_FILES:
        real = hf_to_tmp(f)
        link_to_wan2gp(real, f)

# ═══ 3. GEMMA4 TEXT ENCODER → /tmp/models/gemma4-12b-ltx-v1/ ═══
GEMMA4_FILES = [
    'gemma4-12b-ltx-v1_int8_convrot.safetensors',
    'config.json', 'chat_template.jinja',
    'tokenizer.json', 'tokenizer_config.json',
]
if not _SKIP25:
    log('Verificando Gemma4 text encoder...')
    for gf in GEMMA4_FILES:
        real = hf_to_tmp(gf, subfolder=GEMMA4_SUB)
        link_to_wan2gp(real, gf, subfolder=GEMMA4_SUB)

# ═══ 4. CONFIGURAR DEFAULTS WAN2GP ═══
DEFAULTS_DIR = os.path.join(WAN2GP_DIR, 'defaults')
os.makedirs(DEFAULTS_DIR, exist_ok=True)
cfg = {
    "model": {"name": "LTX-2 2.5 Distilled 22B", "architecture": "ltx2_25_22B",
              "ltx2_pipeline": "distilled"},
    "guidance_phases": 1, "num_inference_steps": 8,
    "video_length": 73, "resolution": "640x384"
}
for cn in ['ltx2_25_22B.json', 'ltx2_25_22B_distilled.json', 'ltx2_25_22B_distilled_nvfp4.json']:
    try:
        json.dump(cfg, open(os.path.join(DEFAULTS_DIR, cn), 'w'), indent=4)
    except Exception:
        pass

# ═══ 5. VERIFICACIÓN FINAL ═══
log('\n=== Verificación Celda 12 ===')
critical = [
    (TMP_MODELS,  TRANSFORMER_FILE),
    (MODEL_DIR,   TRANSFORMER_FILE),       # symlink
    (CKPTS_DIR,   TRANSFORMER_FILE),       # symlink
    (TMP_MODELS,  'ltx-2.5-22b_video_vae_bf16.safetensors'),
    (GEMMA4_TMP,  'gemma4-12b-ltx-v1_int8_convrot.safetensors'),
    (GEMMA4_TMP,  'config.json'),
    (GEMMA4_TMP,  'tokenizer.json'),
]
if _SKIP25: critical = []
all_ok = True
for dir_, fname in critical:
    path = os.path.join(dir_, fname)
    exists = os.path.exists(path)   # os.path.exists sigue symlinks
    size   = os.path.getsize(path) if exists else 0
    ok     = exists and size > 1000
    if not ok: all_ok = False
    status = '✅' if ok else '❌'
    rel    = os.path.relpath(path, '/kaggle/working')
    log(f'  {status} {rel} ({size/1024**2:.0f} MB)' if exists else f'  {status} MISSING: {rel}')

st_w = os.statvfs('/kaggle/working')
st_t = os.statvfs('/tmp')
log(f'\n📦 /kaggle/working libre: {(st_w.f_bavail*st_w.f_frsize)/1024**3:.1f} GB  ← NO debe bajar')
log(f'📦 /tmp libre:           {(st_t.f_bavail*st_t.f_frsize)/1024**3:.1f} GB')

# ── Alias para Celda 18 (compatibilidad) ──
LTX23_PRESETS = {
    "⚡ RÁPIDO — LTX-2.5 Distilled INT8 (~25 GB /tmp, 8 steps)": {
        "transformer_key": "INT8_CONVROT", "encoder_key": "GEMMA4_INT8",
        "desc": "LTX-2.5 22B Distilled. 8 pasos. Todo en /tmp, cero impacto en /working.",
    },
}
DEFAULT_LTX23_PRESET = list(LTX23_PRESETS.keys())[0]

def ensure_ltx23_models(preset_label=None):
    """Retorna dict de rutas. Todos los archivos en /tmp/models/."""
    paths = {
        'transformer':           os.path.join(TMP_MODELS, TRANSFORMER_FILE),
        'encoder':               os.path.join(GEMMA4_TMP, 'gemma4-12b-ltx-v1_int8_convrot.safetensors'),
        'video_vae':             os.path.join(TMP_MODELS, 'ltx-2.5-22b_video_vae_bf16.safetensors'),
        'audio_vae':             os.path.join(TMP_MODELS, 'ltx-2.5-22b_audio_vae_bf16.safetensors'),
        'embeddings_connector':  os.path.join(TMP_MODELS, 'ltx-2.5-22b_video_embeddings_connector_int8_convrot.safetensors'),
        'gemma4_dir':            GEMMA4_TMP,
        '_preset':               preset_label or DEFAULT_LTX23_PRESET,
        '_steps': 8, '_cfg': 1.0, '_vram_estimado': 6, '_tipo': 'distilled',
    }
    for k, v in paths.items():
        if k.startswith('_'): continue
        if not os.path.exists(v):
            log(f'[LTX25] ❌ Faltante: {k} → {v}')
            return None
    return paths

ensure_ltx25_models = ensure_ltx23_models

if all_ok:
    log('\n✅ CELDA 12 (v3 DISK-SAFE) COMPLETADA')
    log('   ► Todos los modelos en /tmp/models/ (cero impacto en /kaggle/working)')
    log('   ► Wan2GP/models/ y Wan2GP/ckpts/ = solo symlinks')
else:
    log('\n⚠ Algunos archivos faltan. Revisa la descarga.')