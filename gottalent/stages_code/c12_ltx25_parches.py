# ═══════════════════════════════════════════════════════════════
# CELDA 12b (UNIFICADA DEFINITIVA): PARCHES WAN2GP + run_ltx25_generate.py
#
# CONSOLIDA EN UNA SOLA CELDA:
#   - Cell 12b original (parches disco + script v4)
#   - REESCRITURA COMPLETA (VERSIÓN DEFINITIVA)
#   - CELDA FIX I2V RAÍZ (video_encoder fp32, resize exacto, strength=1.0)
#
# REGLA: NO tocar nada de lo que funciona. Escritura única del script final.
# ═══════════════════════════════════════════════════════════════
import subprocess as _sp, sys, os
from pathlib import Path

WAN2GP_DIR   = '/kaggle/working/Wan2GP'
LTX25_SCRIPT = os.path.join(WAN2GP_DIR, 'run_ltx25_generate.py')
TMP_MODELS   = '/tmp/models'
GEMMA4_TMP   = os.path.join(TMP_MODELS, 'gemma4-12b-ltx-v1')

# ══════════════════════════════════════════════════════════════
# BLOQUE 1: PARCHES EN DISCO A ARCHIVOS WAN2GP
# Idénticos al proyecto de referencia AIQUEST + fixes adicionales validados.
# Todos son idempotentes (verifican antes de aplicar).
# ══════════════════════════════════════════════════════════════
print("Aplicando parches a Wan2GP en disco...")
patches_applied = 0

def patch_file(fpath, old, new, label):
    global patches_applied
    if not os.path.isfile(fpath): return
    code = open(fpath, encoding='utf-8').read()
    if new in code or old not in code: return
    open(fpath, 'w', encoding='utf-8').write(code.replace(old, new))
    patches_applied += 1
    print(f"  ✅ {label}")

for base in [WAN2GP_DIR, '.']:
    # P1: FP16 en lugar de BFloat16 (T4 Tensor Cores ~24s/paso)
    patch_file(f"{base}/models/ltx2/ltx_pipelines/distilled.py",
        "self.dtype = torch.bfloat16",
        "self.dtype = getattr(models, 'dtype', torch.float16)",
        'distilled.py → FP16 (a)')
    patch_file(f"{base}/models/ltx2/ltx_pipelines/distilled.py",
        "dtype = torch.bfloat16",
        "dtype = getattr(self, 'dtype', torch.float16)",
        'distilled.py → FP16 (b)')

    # P2: PixelNorm en FP32 (evita overflow FP16 → fondo gris/negro)
    patch_file(f"{base}/models/ltx2/ltx_core/model/common/normalization.py",
        "        mean_sq = torch.mean(x**2, dim=self.dim, keepdim=True)\n"
        "        # Normalize by the root-mean-square (RMS).\n"
        "        rms = torch.sqrt(mean_sq + self.eps)\n"
        "        return x / rms",
        "        orig_dtype = x.dtype\n"
        "        x_float = x.float()\n"
        "        mean_sq = torch.mean(x_float**2, dim=self.dim, keepdim=True)\n"
        "        rms = torch.sqrt(mean_sq + self.eps)\n"
        "        return (x_float / rms).to(orig_dtype)",
        'normalization.py → PixelNorm FP32')

    # P3: feature_extractor RMS en FP32
    patch_file(f"{base}/models/ltx2/ltx_core/text_encoders/gemma/feature_extractor.py",
        "def _norm_and_concat_per_token_rms(encoded_text: torch.Tensor, "
        "attention_mask: torch.Tensor) -> torch.Tensor:\n"
        "    b, t, d, l = encoded_text.shape\n"
        "    variance = torch.mean(encoded_text**2, dim=2, keepdim=True)\n"
        "    normed = encoded_text * torch.rsqrt(variance + 1e-6)\n"
        "    normed = normed.reshape(b, t, d * l)\n"
        "    mask_3d = attention_mask.bool().unsqueeze(-1)\n"
        "    return torch.where(mask_3d, normed, torch.zeros_like(normed))",
        "def _norm_and_concat_per_token_rms(encoded_text: torch.Tensor, "
        "attention_mask: torch.Tensor) -> torch.Tensor:\n"
        "    orig_dtype = encoded_text.dtype\n"
        "    enc_f = encoded_text.float()\n"
        "    b, t, d, l = enc_f.shape\n"
        "    variance = torch.mean(enc_f**2, dim=2, keepdim=True)\n"
        "    normed = enc_f * torch.rsqrt(variance + 1e-6)\n"
        "    normed = normed.reshape(b, t, d * l).to(orig_dtype)\n"
        "    mask_3d = attention_mask.bool().unsqueeze(-1)\n"
        "    return torch.where(mask_3d, normed, torch.zeros_like(normed))",
        'feature_extractor.py → RMS FP32')

    # P4: CausalConv3d auto-cast (dtype mismatch VAE)
    patch_file(f"{base}/models/ltx2/ltx_core/model/video_vae/convolution.py",
        "        x = self.conv(x)",
        "        if hasattr(self, 'conv') and hasattr(self.conv, 'weight') "
        "and x.dtype != self.conv.weight.dtype:\n"
        "            x = x.to(dtype=self.conv.weight.dtype)\n"
        "        x = self.conv(x)",
        'convolution.py → CausalConv3d auto-cast')

    # P5: audio_vae latent auto-cast
    patch_file(f"{base}/models/ltx2/ltx_core/model/audio_vae/audio_vae.py",
        "    decoded_audio = audio_decoder(latent)",
        "    if hasattr(audio_decoder, 'conv_in') and hasattr(audio_decoder.conv_in, 'weight'):\n"
        "        latent = latent.to(dtype=audio_decoder.conv_in.weight.dtype)\n"
        "    decoded_audio = audio_decoder(latent)\n"
        "    if hasattr(vocoder, 'vocoder') and hasattr(vocoder.vocoder, 'conv_pre'):\n"
        "        decoded_audio = decoded_audio.to(dtype=vocoder.vocoder.conv_pre.weight.dtype)",
        'audio_vae.py → latent auto-cast')

    # P6: vocoder buffers auto-cast (LowPassFilter1d)
    patch_file(f"{base}/models/ltx2/ltx_core/model/audio_vae/vocoder.py",
        "return F.conv1d(x, self.filter.expand(n_channels, -1, -1), "
        "stride=self.stride, groups=n_channels)",
        "filt = self.filter.to(dtype=x.dtype, device=x.device).expand(n_channels, -1, -1)\n"
        "        return F.conv1d(x, filt, stride=self.stride, groups=n_channels)",
        'vocoder.py → LowPassFilter buffer')

    # P7: vocoder STFT buffer
    patch_file(f"{base}/models/ltx2/ltx_core/model/audio_vae/vocoder.py",
        "spec = F.conv1d(y, self.forward_basis, stride=self.hop_length, padding=0)",
        "basis = self.forward_basis.to(dtype=y.dtype, device=y.device)\n"
        "        spec = F.conv1d(y, basis, stride=self.hop_length, padding=0)",
        'vocoder.py → STFT buffer')

    # P8: ltx2.py requires_grad_(False) para carga limpia de pesos INT8
    patch_file(f"{base}/models/ltx2/ltx2.py",
        'def _load_component(model, path, sd_ops=None, postprocess=None, '
        'ignore_unused_weights=False):\n            if postprocess is None and sd_ops is not None:',
        'def _load_component(model, path, sd_ops=None, postprocess=None, '
        'ignore_unused_weights=False):\n            model.requires_grad_(False)\n'
        '            if postprocess is None and sd_ops is not None:',
        'ltx2.py → requires_grad_(False)')

print(f"  → {patches_applied} parches en disco aplicados.")

# ══════════════════════════════════════════════════════════════
# BLOQUE 2: ESCRIBIR run_ltx25_generate.py — VERSIÓN DEFINITIVA FINAL
#
# Esta es la versión que fue VALIDADA funcionando:
#   ✅ video_encoder → float32 (fix frames negros I2V)
#   ✅ PNG frames → ffmpeg (guardado robusto, reemplaza save_video)
#   ✅ _patch_video_encoder_float32 hook (seguridad adicional)
#   ✅ image_start resize EXACTO a width×height
#   ✅ input_video_strength = 1.0 SIEMPRE (sin ruido sobre latente)
#   ✅ Progreso correcto (sin 5000%, anti-loop)
#   ✅ nan_to_num antes de guardar
#   ✅ convertWeightsFloatTo=torch.bfloat16 (validado en perfil 4)
#   ✅ SDPA: flash=False, mem_efficient=True (T4/Turing)
# ══════════════════════════════════════════════════════════════
print("\nEscribiendo run_ltx25_generate.py (VERSIÓN DEFINITIVA FINAL)...")

GENERATE_SCRIPT = r'''#!/usr/bin/env python3
"""
run_ltx25_generate.py — VERSIÓN DEFINITIVA FINAL
Consolidación de todos los fixes validados:
  - video_encoder float32 (fix I2V frames negros)
  - PNG frames → ffmpeg pipe (guardado robusto)
  - _patch_video_encoder_float32 hook (seguridad extra)
  - image_start resize EXACTO a width×height
  - input_video_strength = 1.0 SIEMPRE
  - Progreso correcto (sin 5000%)
  - nan_to_num antes de guardar
"""
import sys, os, json, gc, time, random, traceback, tempfile, shutil
import torch
import numpy as np
import io as _io

# stdout sin buffer (crítico para que Popen lea línea a línea en tiempo real)
sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)

def _p(msg):
    print(msg, flush=True)

# ── Bootstrap ──────────────────────────────────────────────────
WAN2GP_DIR = os.environ.get("WAN2GP_DIR", "/kaggle/working/Wan2GP")
TMP_MODELS = "/tmp/models"
GEMMA4_TMP = os.path.join(TMP_MODELS, "gemma4-12b-ltx-v1")
sys.path.insert(0, WAN2GP_DIR)
os.chdir(WAN2GP_DIR)

# ── Parches en disco ANTES de importar (seguridad en caso de reinicio) ─
def _patch(fpath, old, new):
    if not os.path.isfile(fpath): return False
    c = open(fpath, encoding="utf-8").read()
    if new in c or old not in c: return False
    open(fpath, "w", encoding="utf-8").write(c.replace(old, new))
    return True

_r = _patch(
    f"{WAN2GP_DIR}/models/ltx2/ltx_core/model/common/normalization.py",
    "        mean_sq = torch.mean(x**2, dim=self.dim, keepdim=True)\n"
    "        # Normalize by the root-mean-square (RMS).\n"
    "        rms = torch.sqrt(mean_sq + self.eps)\n"
    "        return x / rms",
    "        orig_dtype = x.dtype\n"
    "        x_float = x.float()\n"
    "        mean_sq = torch.mean(x_float**2, dim=self.dim, keepdim=True)\n"
    "        rms = torch.sqrt(mean_sq + self.eps)\n"
    "        return (x_float / rms).to(orig_dtype)"
)
_p(f"[PATCH] PixelNorm FP32: {'aplicado' if _r else 'ya activo'}")

_patch(f"{WAN2GP_DIR}/models/ltx2/ltx_pipelines/distilled.py",
       "self.dtype = torch.bfloat16",
       "self.dtype = getattr(models, 'dtype', torch.float16)")

_patch(f"{WAN2GP_DIR}/models/ltx2/ltx_core/model/video_vae/convolution.py",
       "        x = self.conv(x)",
       "        if hasattr(self,'conv') and hasattr(self.conv,'weight') "
       "and x.dtype != self.conv.weight.dtype:\n"
       "            x = x.to(dtype=self.conv.weight.dtype)\n"
       "        x = self.conv(x)")

_patch(f"{WAN2GP_DIR}/models/ltx2/ltx_core/text_encoders/gemma/feature_extractor.py",
       "    b, t, d, l = encoded_text.shape\n"
       "    variance = torch.mean(encoded_text**2, dim=2, keepdim=True)\n"
       "    normed = encoded_text * torch.rsqrt(variance + 1e-6)\n"
       "    normed = normed.reshape(b, t, d * l)\n"
       "    mask_3d = attention_mask.bool().unsqueeze(-1)\n"
       "    return torch.where(mask_3d, normed, torch.zeros_like(normed))",
       "    orig_dtype = encoded_text.dtype\n"
       "    enc_f = encoded_text.float()\n"
       "    b, t, d, l = enc_f.shape\n"
       "    variance = torch.mean(enc_f**2, dim=2, keepdim=True)\n"
       "    normed = enc_f * torch.rsqrt(variance + 1e-6)\n"
       "    normed = normed.reshape(b, t, d * l).to(orig_dtype)\n"
       "    mask_3d = attention_mask.bool().unsqueeze(-1)\n"
       "    return torch.where(mask_3d, normed, torch.zeros_like(normed))")

_patch(f"{WAN2GP_DIR}/models/ltx2/ltx_core/model/audio_vae/audio_vae.py",
       "    decoded_audio = audio_decoder(latent)",
       "    if hasattr(audio_decoder,'conv_in') and hasattr(audio_decoder.conv_in,'weight'):\n"
       "        latent = latent.to(dtype=audio_decoder.conv_in.weight.dtype)\n"
       "    decoded_audio = audio_decoder(latent)\n"
       "    if hasattr(vocoder,'vocoder') and hasattr(vocoder.vocoder,'conv_pre'):\n"
       "        decoded_audio = decoded_audio.to(dtype=vocoder.vocoder.conv_pre.weight.dtype)")

_patch(f"{WAN2GP_DIR}/models/ltx2/ltx_core/model/audio_vae/vocoder.py",
       "return F.conv1d(x, self.filter.expand(n_channels, -1, -1), stride=self.stride, groups=n_channels)",
       "filt = self.filter.to(dtype=x.dtype, device=x.device).expand(n_channels, -1, -1)\n"
       "        return F.conv1d(x, filt, stride=self.stride, groups=n_channels)")
_patch(f"{WAN2GP_DIR}/models/ltx2/ltx_core/model/audio_vae/vocoder.py",
       "spec = F.conv1d(y, self.forward_basis, stride=self.hop_length, padding=0)",
       "basis = self.forward_basis.to(dtype=y.dtype, device=y.device)\n"
       "        spec = F.conv1d(y, basis, stride=self.hop_length, padding=0)")

# ── HfFolder backward-compat ───────────────────────────────────
try:
    import huggingface_hub.utils as _hfu
    if not hasattr(_hfu, "HfFolder"):
        class HfFolder:
            @staticmethod
            def get_token():
                try:
                    from huggingface_hub import get_token
                    return get_token() or os.environ.get("HF_TOKEN")
                except: return os.environ.get("HF_TOKEN")
            @staticmethod
            def save_token(t): pass
        setattr(_hfu, "HfFolder", HfFolder)
        if "huggingface_hub.utils" in sys.modules:
            setattr(sys.modules["huggingface_hub.utils"], "HfFolder", HfFolder)
except: pass

# ── PyTorch optimizaciones ─────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(False)       # T4/Turing: usar mem_efficient
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# ── Imports Wan2GP ─────────────────────────────────────────────
from mmgp import offload, quant_router
from models.ltx2.ltx2_handler import family_handler

# Registrar todos los handlers de cuantización
for h in [
    "shared.qtypes.scaled_fp8", "shared.qtypes.nvfp4",
    "shared.qtypes.bnb_nf4",   "shared.qtypes.nunchaku_int4",
    "shared.qtypes.nunchaku_fp4", "shared.qtypes.asym_w4a8_int8",
    "shared.qtypes.int8_convrot", "shared.qtypes.gguf",
]:
    try: quant_router.register_handler(h)
    except: pass
quant_router.unregister_handler(".fp8_quanto_bridge")
try:
    from shared.qtypes import gguf as _gguf
    quant_router.register_file_extension("gguf", _gguf)
except: pass

from PIL import Image
import subprocess as _sp

# Limpiar lock files de ejecuciones anteriores
for lf in ["startup.lock", "wgp.lock"]:
    lp = os.path.join(WAN2GP_DIR, lf)
    if os.path.exists(lp):
        try: os.remove(lp)
        except: pass

# ── Resolver rutas de modelos ──────────────────────────────────
TF = "ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors"
GF = "gemma4-12b-ltx-v1_int8_convrot.safetensors"

def _find(*candidates):
    for c in candidates:
        if os.path.exists(c) and os.path.getsize(c) > 1024*1024: return c
    return ""

transformer_path = _find(
    os.path.join(TMP_MODELS, TF),
    os.path.join(WAN2GP_DIR, "models", TF),
    os.path.join(WAN2GP_DIR, "ckpts",  TF))
text_encoder_file = _find(
    os.path.join(GEMMA4_TMP, GF),
    os.path.join(WAN2GP_DIR, "models", "gemma4-12b-ltx-v1", GF),
    os.path.join(WAN2GP_DIR, "ckpts",  "gemma4-12b-ltx-v1", GF))

if not transformer_path:  _p("ERROR: Transformer no encontrado."); sys.exit(1)
if not text_encoder_file: _p("ERROR: Text encoder no encontrado."); sys.exit(1)

_p(f"[SUBPROCESS] Transformer : {os.path.basename(transformer_path)}")
_p(f"[SUBPROCESS] Text Encoder: {os.path.basename(text_encoder_file)}")
_p("[SUBPROCESS] Cargando modelo LTX-2.5 22B Distilled...")

# ── Cargar modelo ──────────────────────────────────────────────
base_model_type = "ltx2_25_22B"
model_def = {
    "ltx2_pipeline": "distilled",
    "name":          "LTX-2 2.5 Distilled 22B",
    "architecture":  "ltx2_25_22B",
}
try:
    model_def.update(family_handler.query_model_def(base_model_type, model_def))
except: pass

with torch.inference_mode():
    with torch.set_grad_enabled(False):
        wan_model, pipe = family_handler.load_model(
            model_filename        = transformer_path,
            model_type            = "ltx2_25_22B_distilled",
            base_model_type       = base_model_type,
            model_def             = model_def,
            dtype                 = torch.float16,
            VAE_dtype             = torch.float16,
            text_encoder_filename = text_encoder_file,
        )

kwargs = {}
if isinstance(pipe, dict) and "pipe" in pipe:
    kwargs = pipe; pipe = kwargs.pop("pipe")
loras = kwargs.pop("loras", [])
if "transformer" in pipe and "transformer" not in loras:
    loras.append("transformer")

# ── FIX CRÍTICO: video_encoder → float32 ANTES del offload ────
# Sin este fix: overflow FP16 → NaN → frames completamente negros en I2V
def _patch_video_encoder_float32(pipe_obj):
    """Fuerza video_encoder a float32 para evitar overflow en I2V."""
    # Caso 1: pipe es dict
    if isinstance(pipe_obj, dict):
        ve = pipe_obj.get("video_encoder")
        if ve is not None:
            try:
                ve.to(torch.float32)
                _p("[SUBPROCESS] ✅ video_encoder (dict) → float32 (fix I2V negro)")
                # Parchear el método encode si existe
                if hasattr(ve, 'encode'):
                    _orig = ve.encode
                    def _enc_fp32(*a, **kw):
                        with torch.autocast('cuda', dtype=torch.float32):
                            return _orig(*a, **kw)
                    ve.encode = _enc_fp32
                    _p("[SUBPROCESS] ✅ video_encoder.encode → autocast float32")
                return
            except Exception as e:
                _p(f"[SUBPROCESS] ⚠ video_encoder dict float32: {e}")
    # Caso 2: pipe es objeto, buscar atributos
    for attr in ["video_encoder", "vae", "encoder"]:
        ve = getattr(pipe_obj, attr, None) if not isinstance(pipe_obj, dict) else pipe_obj.get(attr)
        if ve is not None:
            try:
                ve.to(torch.float32)
                _p(f"[SUBPROCESS] ✅ {attr} → float32")
                return
            except: pass
    _p("[SUBPROCESS] ⚠ video_encoder no encontrado en pipe (se continuará sin fix)")

_patch_video_encoder_float32(pipe)

_p("[SUBPROCESS] Aplicando offload profile 4 (Partial Pinning, Async DMA)...")
offload.profile(
    pipe,
    profile_no            = 4,
    pinnedMemory          = "transformer",
    partialPinning        = True,
    perc_reserved_mem_max = 0.35,
    asyncTransfers        = True,
    quantizeTransformer   = False,
    convertWeightsFloatTo = torch.bfloat16,   # validado: bfloat16 para perfil 4
    loras                 = loras,
    budgets = {
        "transformer":       6000,
        "text_encoder":      1500,
        "video_encoder":     2000,
        "video_decoder":     3000,
        "audio_encoder":     1000,
        "audio_decoder":     1000,
        "vocoder":           500,
        "spatial_upsampler": 1500,
        "vae":               1000,
        "":                  1000,
    },
    **kwargs
)
offload.shared_state["_attention"] = "sdpa"
offload.shared_state["_radial"]    = False
_p("[SUBPROCESS] ✅ Modelo listo para generar.")

# ══════════════════════════════════════════════════════════════
# GUARDADO ROBUSTO: PNG frames → ffmpeg
# Reemplaza save_video() que fallaba silenciosamente (6 KB).
# Método validado: T frames PNG a disco → ffmpeg libx264.
# ══════════════════════════════════════════════════════════════
def save_robust(tensor, out_path, fps=24):
    """Guarda tensor de video de forma robusta vía PNG frames + ffmpeg."""
    # ── FIX RAM: antes se hacía tensor.cpu().float() COMPLETO y luego nan_to_num,
    #    (t+1)/2*255, clamp, astype... = 5-6 copias float32 del video (>12 GB en 10 s
    #    a 1152x704) → OOM → el kernel se reiniciaba. Ahora se procesa por BLOQUES de
    #    frames (GPU→CPU de pocos frames) y se acumula directo en uint8 (1 byte/valor).
    #    La lógica de conversión es IDÉNTICA a la anterior.
    t = tensor.detach()          # se queda donde esté (GPU): NO se copia entero a CPU

    # Quitar batch dim si existe
    if t.ndim == 5: t = t[0]

    # Normalizar a (T, H, W, C)  (son vistas, no copian memoria)
    if t.ndim == 4:
        s = t.shape
        # (C, T, H, W) → (T, H, W, C) si C<=4 y T>4
        if s[0] <= 4 and s[1] > 4 and s[2] > 4 and s[3] > 4:
            t = t.permute(1, 2, 3, 0)
    if t.ndim == 3: t = t.unsqueeze(0)

    T, H, W, C = t.shape
    if C not in (1, 3, 4):
        raise RuntimeError(f"Canales inesperados: C={C}, shape={tuple(t.shape)}")
    _p(f"[SUBPROCESS] Tensor recibido: shape={list(t.shape)}, dtype={tensor.dtype} "
       f"(procesado por bloques para ahorrar RAM)")

    _C0  = C   # canales de ENTRADA (C se fija a 3 más abajo)
    _BLK = 8   # frames por bloque (8 frames a 1152x704 ≈ 80 MB en float32)

    def _prep(b):
        """Bloque → float32 CPU, sin NaN/Inf, con 3 canales."""
        b = b.cpu().float()
        if not torch.isfinite(b).all():
            b = torch.nan_to_num(b, nan=0.0, posinf=255.0, neginf=0.0)
        if _C0 == 1:   b = b.repeat(1, 1, 1, 3)
        elif _C0 == 4: b = b[..., :3]
        return b

    # Pasada 1: rango global (min/max) y detección de NaN/Inf
    vmin, vmax, had_bad = float("inf"), float("-inf"), False
    for i in range(0, T, _BLK):
        raw_b = t[i:i + _BLK].cpu().float()
        if not torch.isfinite(raw_b).all():
            had_bad = True
        del raw_b
        b = _prep(t[i:i + _BLK])
        vmin = min(vmin, b.min().item())
        vmax = max(vmax, b.max().item())
        del b
    if had_bad:
        _p("[SUBPROCESS] ⚠ NaN/Inf detectado — aplicado nan_to_num por bloques")
    C = 3

    # Pasada 2: convertir a uint8 [0, 255] bloque a bloque
    frames = np.empty((T, H, W, 3), dtype=np.uint8)
    for i in range(0, T, _BLK):
        b = _prep(t[i:i + _BLK])
        if vmax <= 1.0 + 1e-4:
            # Rango [0,1] o [-1,1]
            b = ((b + 1) / 2 * 255 if vmin < -0.5 else b * 255).clamp(0, 255)
        else:
            b = b.clamp(0, 255)
        frames[i:i + b.shape[0]] = b.numpy().astype(np.uint8)
        del b

    _p(f"[SUBPROCESS] Frames numpy: T={T} H={H} W={W} C={C} | "
       f"uint8 min={frames.min()} max={frames.max()}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # ── Guardar PNGs a disco → ffmpeg ─────────────────────────
    frames_dir = tempfile.mkdtemp(prefix="ltx_fr_")
    try:
        _p(f"[SUBPROCESS] Guardando {T} frames PNG...")
        from PIL import Image as _PIL
        for i, fr in enumerate(frames):
            _PIL.fromarray(fr).save(os.path.join(frames_dir, f"f{i:06d}.png"))

        _p(f"[SUBPROCESS] Codificando con ffmpeg → {out_path}")
        r = _sp.run([
            "ffmpeg", "-y",
            "-framerate", str(int(fps)),
            "-i", os.path.join(frames_dir, "f%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-preset", "fast",
            "-movflags", "+faststart",
            out_path
        ], capture_output=True, timeout=300)

        if r.returncode == 0 and os.path.exists(out_path):
            sz = os.path.getsize(out_path)
            _p(f"[SUBPROCESS] ✅ ffmpeg PNG→MP4 OK: {sz // 1024} KB")
            return out_path, sz
        _p(f"[SUBPROCESS] ⚠ ffmpeg rc={r.returncode}: "
           f"{r.stderr[-300:].decode(errors='ignore')}")
    except Exception as e:
        _p(f"[SUBPROCESS] ⚠ PNG frames falló: {e}")
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)

    # ── Fallback: ffmpeg rawvideo pipe ─────────────────────────
    try:
        _p(f"[SUBPROCESS] Fallback: ffmpeg rawvideo pipe...")
        proc = _sp.Popen([
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{W}x{H}", "-pix_fmt", "rgb24",
            "-r", str(int(fps)), "-i", "pipe:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", "-preset", "fast",
            out_path
        ], stdin=_sp.PIPE, stdout=_sp.PIPE, stderr=_sp.PIPE)
        for i in range(T):
            proc.stdin.write(frames[i].tobytes())
        proc.stdin.close()
        _, stderr = proc.communicate(timeout=120)
        if proc.returncode == 0 and os.path.exists(out_path):
            sz = os.path.getsize(out_path)
            if sz > 10000:
                _p(f"[SUBPROCESS] ✅ ffmpeg pipe OK: {sz // 1024} KB")
                return out_path, sz
        _p(f"[SUBPROCESS] ⚠ pipe rc={proc.returncode}: "
           f"{stderr[-200:].decode(errors='ignore')}")
    except Exception as e:
        _p(f"[SUBPROCESS] ⚠ ffmpeg pipe falló: {e}")

    # ── Último recurso: devolver lo que haya si es > 1 KB ─────
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        sz = os.path.getsize(out_path)
        _p(f"[SUBPROCESS] ⚠ Usando archivo parcial existente: {sz} bytes")
        return out_path, sz

    raise RuntimeError(f"Todos los métodos de guardado fallaron para {out_path}")

# ── Callback de progreso ───────────────────────────────────────
_pstate = {
    "phase": "", "total": 8, "done": 0,
    "t0": time.monotonic(), "lt": time.monotonic()
}

def progress_cb(step, latent=None, is_start=False,
                override_num_inference_steps=None,
                pass_no=None, **kw):
    idx   = int(step)
    phase = str(kw.get("progress_title") or "Denoising")
    if pass_no is not None: phase += f"/p{pass_no}"
    phase = phase.replace(":", "")

    supplied = None
    if override_num_inference_steps is not None and not isinstance(override_num_inference_steps, bool):
        supplied = max(1, int(override_num_inference_steps))

    now = time.monotonic()
    if is_start or _pstate["phase"] != phase:
        _pstate.update({
            "phase": phase, "total": supplied or 8,
            "done": 0, "t0": now, "lt": now
        })
        print(f"PROGRESS:stage:{phase}", flush=True)
        print(f"PROGRESS:start:{_pstate['total']}", flush=True)
        return

    if idx < 0: return   # consulta de interrupción interna de mmgp

    if supplied: _pstate["total"] = supplied
    total   = _pstate["total"]
    current = min(total, max(1, idx + 1))
    if current <= _pstate["done"]: return

    dt  = max(0.0, now - _pstate["lt"]) / max(1, current - _pstate["done"])
    avg = max(0.0, now - _pstate["t0"]) / current
    rem = max(0, total - current) * avg
    free = torch.cuda.mem_get_info()[0] / 1024**3 if torch.cuda.is_available() else 0
    pct  = min(100, int(current * 100 / total))

    _pstate["done"] = current
    _pstate["lt"]   = now
    print(f"PROGRESS:step:{current}:{total}:{dt:.1f}:{rem:.0f}:{free:.1f}:{phase}:{pct}",
          flush=True)

def set_stage(s):
    print(f"PROGRESS:stage:{s}", flush=True)
    if "Decod" in s or "Upsamp" in s:
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

def _audio_inputs_for_ltx(params, frame_num, fps):
    import wave

    audio_path = params.get("audio_path", "")
    if not audio_path:
        # Los clips de 5s siguen usando exactamente la llamada anterior.
        return {}

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio no encontrado: {audio_path}")

    with wave.open(audio_path, "rb") as wav:
        if wav.getsampwidth() != 2:
            raise RuntimeError(
                "La entrada A2V debe ser WAV PCM de 16 bits. "
                "Normalízala con ffmpeg_to_wav_mono_24k()."
            )

        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        raw_audio = wav.readframes(wav.getnframes())

    waveform = (
        np.frombuffer(raw_audio, dtype="<i2")
        .reshape(-1, channels)
        .astype(np.float32)
        / 32768.0
    )

    if waveform.shape[0] == 0:
        raise RuntimeError("El WAV está vacío.")

    # Cubrir los frames extra de la cuadrícula temporal de LTX.
    # No modifica el archivo original que después se adjunta al vídeo.
    needed_samples = int(
        np.ceil(float(frame_num) / float(fps) * sample_rate)
    )

    if waveform.shape[0] < needed_samples:
        waveform = np.pad(
            waveform,
            ((0, needed_samples - waveform.shape[0]), (0, 0)),
            mode="constant",
        )

    waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    peak = float(np.abs(waveform).max())

    _p(
        f"[SUBPROCESS] A2V: audio={os.path.basename(audio_path)} | "
        f"sr={sample_rate} | shape={waveform.shape} | "
        f"peak={peak:.4f} | audio_scale=1.0"
    )

    if peak < 1e-5:
        _p(
            "[SUBPROCESS] AVISO: audio prácticamente silencioso; "
            "no sirve para validar sincronización de habla."
        )

    return {
        "input_waveform": waveform,
        "input_waveform_sample_rate": int(sample_rate),
        "audio_prompt_type": "A",
        "audio_scale": 1.0,
    }



# ── Función principal de generación ───────────────────────────
def generate(params):
    prompt  = params.get("prompt", "")
    imgpath = params.get("image_path", "")
    outpath = params.get("output_path", "")
    W       = int(params.get("width",     768))
    H       = int(params.get("height",    432))
    NF      = int(params.get("num_frames", 121))
    fps     = float(params.get("fps",      24))
    seed    = int(params.get("seed",       -1))
    steps   = int(params.get("steps",       8))
    cfg     = float(params.get("cfg",      1.0))
    acfg    = float(params.get("audio_cfg", 1.0))

    if seed < 0: seed = random.randint(0, 2**32 - 1)
    if not outpath:
        os.makedirs("/tmp/ltx25_outputs", exist_ok=True)
        outpath = f"/tmp/ltx25_outputs/ltx25_{int(time.time())}_{seed}.mp4"
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)

    # Resetear estado de progreso
    _pstate.update({
        "phase": "", "done": 0,
        "t0": time.monotonic(), "lt": time.monotonic()
    })

    # ── Pre-procesar imagen I2V ────────────────────────────────
    image_start = None
    if imgpath and os.path.exists(imgpath):
        img = Image.open(imgpath).convert("RGB")
        # Resize EXACTO a W×H (un pixel de diferencia causa mismatch en VAE → frames negros)
        image_start = img.resize((W, H), Image.LANCZOS)
        _p(f"[SUBPROCESS] image_start: {image_start.size} (exacto {W}×{H})")

    mode = "I2V" if image_start else "T2V"
    _p(f"[SUBPROCESS] {mode} | {W}x{H} | {NF}f@{fps}fps | steps={steps} | cfg={cfg} | seed={seed}")

    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    t0 = time.time()
    result = wan_model.generate(
        input_prompt         = prompt,
        image_start          = image_start,
        height               = H,
        width                = W,
        frame_num            = NF,
        fps                  = fps,
        seed                 = seed,
        callback             = progress_cb,
        set_progress_status  = set_stage,
        VAE_tile_size        = 0,          # sin seams de tiling
        input_video_strength = 1.0,        # SIEMPRE 1.0: sin ruido sobre latente de imagen
        denoising_strength   = 1.0,
        guide_scale          = cfg,
        audio_cfg_scale      = acfg,
        sampling_steps       = steps,
        guide_phases         = 1,
        sample_solver        = "distilled_8_steps",
        self_refiner_setting = 0,
        n_prompt             = "",
        **_audio_inputs_for_ltx(params, NF, fps),  # NUEVO
    )
    elapsed = time.time() - t0
    _p(f"[SUBPROCESS] Pipeline completado en {elapsed:.1f}s")

    # ── Extraer tensor de video ────────────────────────────────
    vt = None
    if isinstance(result, dict):
        # Elegir tensor con mayor rango dinámico (el más "real")
        best = -1.0
        for k, v in result.items():
            if not torch.is_tensor(v): continue
            rng = (v.float().max() - v.float().min()).item()
            _p(f"[SUBPROCESS] result['{k}']: shape={list(v.shape)} "
               f"dtype={v.dtype} range={rng:.3f}")
            if rng > best: best = rng; vt = v
    elif isinstance(result, (list, tuple)):
        vt = next((x for x in result if torch.is_tensor(x)), None)
    elif torch.is_tensor(result):
        vt = result

    if vt is None:
        raise RuntimeError(f"Sin tensor de video. Tipo resultado: {type(result)}")

    _p("[SUBPROCESS] Guardando video...")
    op, sz = save_robust(vt, outpath, fps=fps)

    # Copia a almacenamiento persistente
    try:
        bn = os.path.basename(op)
        for d in ["/kaggle/working/outputs",
                  os.path.join(WAN2GP_DIR, "outputs")]:
            os.makedirs(d, exist_ok=True)
            shutil.copy2(op, os.path.join(d, bn))
    except Exception as e:
        _p(f"[SUBPROCESS] copia persistente: {e}")

    del vt
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    _p(f"[SUBPROCESS] ✅ Generado en {elapsed:.1f}s → {op} ({sz // 1024} KB)")
    return op


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        _p("Uso: python run_ltx25_generate.py '<json_params>'")
        sys.exit(1)
    try:
        op = generate(json.loads(sys.argv[1]))
        print(f"OUTPUT_PATH:{op}", flush=True)
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
'''

SCRIPT = Path(LTX25_SCRIPT)
SCRIPT.write_text(GENERATE_SCRIPT, encoding="utf-8")
SCRIPT.chmod(0o755)

# ══════════════════════════════════════════════════════════════
# BLOQUE 3: VERIFICACIÓN FINAL
# ══════════════════════════════════════════════════════════════
content = SCRIPT.read_text(encoding="utf-8")

checks = {
    "PNG frames (guardado robusto)":        "f%06d.png"                          in content,
    "video_encoder float32 (hook)":         "_patch_video_encoder_float32"       in content,
    "video_encoder.to(float32)":            "ve.to(torch.float32)"               in content,
    "image_start resize exacto W×H":        "img.resize((W, H)"                  in content,
    "input_video_strength = 1.0 SIEMPRE":   "input_video_strength = 1.0"         in content,
    "PixelNorm FP32 (patch runtime)":       "x_float = x.float()"                in content,
    "Progreso anti-loop (idx<0)":           "if idx < 0: return"                 in content,
    "nan_to_num antes de guardar":          "nan_to_num"                          in content,
    "convertWeightsFloatTo=bfloat16":       "convertWeightsFloatTo = torch.bfloat16" in content,
    "SDPA mem_efficient (T4)":              "enable_mem_efficient_sdp(True)"      in content,
    "flash_sdp=False (T4/Turing)":          "enable_flash_sdp(False)"             in content,
    "stdout line_buffering":                "line_buffering=True"                 in content,
    "OUTPUT_PATH: en stdout":               "OUTPUT_PATH:"                        in content,
    "PROGRESS:step: en stdout":             "PROGRESS:step:"                      in content,
}

print(f"\n✅ Script escrito: {SCRIPT.stat().st_size:,} bytes\n")
print("── Verificación de checks críticos ──")
all_ok = True
for nombre, ok in checks.items():
    symbol = "✅" if ok else "❌"
    print(f"  {symbol} {nombre}")
    if not ok: all_ok = False

print(f"\n{'✅ CELDA 12b UNIFICADA LISTA — Sin cambios pendientes.' if all_ok else '❌ Revisar los checks fallidos.'}")
print("\n── Resumen ──────────────────────────────────────────────")
print(f"  Parches en disco Wan2GP: {patches_applied}")
print(f"  Script:  {LTX25_SCRIPT}")
print(f"  Tamaño:  {SCRIPT.stat().st_size // 1024} KB")
print("\nFlujo de ejecución correcto:")
print("  Celda 12  → descarga modelos a /tmp/models")
print("  Celda 12b → parches Wan2GP + escribe script definitivo  ← ESTA CELDA")
print("  Celda 13  → SubprocessLTXBackend (Popen + progreso)")
print("  Celda 14  → wrappers generar_clip_5s / lipsync")
print("  Celda 15  → Etapa 3 completa")
print("  Celda 16  → Etapas 2 y 4 (diálogos)")
print("  Celda 17  → Render final")
print("  Celda 18  → UI Gradio")