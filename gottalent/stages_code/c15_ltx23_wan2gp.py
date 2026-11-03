# ═══════════════════════════════════════════════════════════════
# CELDA 14b (NUEVA): LTX-2.3 (Wan2GP aislado, 2×T4) + selectores de resolución/duración
#   - LTX-2.3 corre en su PROPIO Wan2GP (commit fijado) y su PROPIO venv → no toca el entorno de LTX-2.5/Qwen/UI.
#   - Misma interfaz que el motor anterior (get_ltx_backend().generate) → Celdas 15, 16 y la UI no cambian.
#   - Para volver a LTX-2.5: define LTX_VERSION = "2.5" ANTES de la Celda 12 y re-ejecuta esta celda.
#   - Va DESPUÉS de la Celda 14 y ANTES de la UI (Celda 18).
# ═══════════════════════════════════════════════════════════════
import os, sys, re, json, time, glob, shutil, random, subprocess, threading

LTX_VERSION = globals().get("LTX_VERSION", "2.3")

# ── Ajustes elegidos en la UI (los valores por defecto = comportamiento anterior) ──
LTX_RES_DEFAULT   = "Por defecto (1152x704)"
LTX_AUTO_ASPECT   = "Auto (match reference image)"
LTX_RES_CHOICES   = [LTX_RES_DEFAULT, "720p", "540p", "480p"]
LTX_ASPECT_CHOICES = [LTX_AUTO_ASPECT, "16:9 Landscape", "4:3 Standard", "1:1 Square", "3:4 Portrait", "9:16 Portrait"]
LTX_UI_CFG = globals().get("LTX_UI_CFG") or {
    "far_res": LTX_RES_DEFAULT, "far_aspect": LTX_AUTO_ASPECT, "far_dur": 5.0,
    "song_res": LTX_RES_DEFAULT, "song_aspect": LTX_AUTO_ASPECT,
}
LTX_UI_CFG.setdefault("dlg_res", LTX_RES_DEFAULT)           # Etapas 2 y 4 (diálogos)
LTX_UI_CFG.setdefault("dlg_aspect", LTX_AUTO_ASPECT)
FAR_CLIP_DUR_S = float(LTX_UI_CFG["far_dur"])      # la lee el timeline (Celda 9) para el largo de los planos lejanos

def ltx_cfg_set(key, value):
    global FAR_CLIP_DUR_S
    if key == "far_dur":
        value = float(re.sub(r"[^0-9.]", "", str(value)) or 5.0)
        FAR_CLIP_DUR_S = value
    LTX_UI_CFG[key] = value
    log(f"[LTX-UI] {key} = {value}")

def ltx_resolve(kind, image_path, default=None):
    """Misma lógica que get_resolution() del notebook LTX-2.3. Devuelve (w, h) o `default` si se dejó 'Por defecto'."""
    res, asp = LTX_UI_CFG.get(kind + "_res"), LTX_UI_CFG.get(kind + "_aspect")
    if not res or res == LTX_RES_DEFAULT:
        return default
    base = {"1080p": 1088, "720p": 704, "540p": 544, "480p": 480}.get(res, 704)
    ratios = {"16:9 Landscape": 16/9, "4:3 Standard": 4/3, "1:1 Square": 1.0, "3:4 Portrait": 3/4, "9:16 Portrait": 9/16}
    ratio = None
    if asp == LTX_AUTO_ASPECT and image_path and os.path.exists(image_path):
        try:
            from PIL import Image
            with Image.open(image_path) as im:
                ratio = im.width / im.height
        except Exception:
            ratio = None
    if ratio is None:
        ratio = ratios.get(asp, 16/9)
    if ratio >= 1.0:
        h, w = base, int(base * ratio)
    else:
        w, h = base, int(base / ratio)
    return (w // 32) * 32, (h // 32) * 32

# ── Wrappers: clips lejanos (resolución + duración) y lipsync de la canción (resolución) ──
if not getattr(generar_clip_5s, "_ltx23", False):
    _LTX_ORIG_5S = generar_clip_5s
    def generar_clip_5s(vp, image_path, prompt, out_mp4, ltx_paths=None,
                        seed=-1, fps=24, width=1152, height=704):
        w, h = ltx_resolve("far", image_path, (width, height))
        dur = float(LTX_UI_CFG.get("far_dur", 5.0))
        if dur == 5.0:
            return _LTX_ORIG_5S(vp, image_path, prompt, out_mp4, ltx_paths, seed=seed, fps=fps, width=w, height=h)
        os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)
        engine = get_ltx_backend()
        num_frames = calc_num_frames(dur, fps)
        steps = int(ltx_paths.get("_steps", 8)) if ltx_paths else 8
        cfg = float(ltx_paths.get("_cfg", 1.0)) if ltx_paths else 1.0
        log(f"[CLIP {dur:.0f}s] I2V | {w}x{h} | {num_frames}f@{fps}fps | steps={steps}")
        raw = engine.generate(prompt=prompt, image_path=image_path, num_frames=num_frames, width=w, height=h,
                              seed=seed, steps=steps, cfg=cfg, fps=fps,
                              input_video_strength=0.8, denoise_strength=1.0)
        _to_mp4_exact(raw, out_mp4, target_s=dur, fps=fps, mute=True)
        log(f"[CLIP {dur:.0f}s] ✅ → {out_mp4}")
        return out_mp4
    generar_clip_5s._ltx23 = True

if not getattr(generar_clip_lipsync_15s, "_ltx23", False):
    _LTX_ORIG_LIP15 = generar_clip_lipsync_15s
    def generar_clip_lipsync_15s(vp, image_path, audio_path, prompt, out_mp4,
                                 ltx_paths=None, seed=-1, fps=24, **kwargs):
        if "width" not in kwargs and "height" not in kwargs:
            r = ltx_resolve("song", image_path, None)
            if r:
                kwargs["width"], kwargs["height"] = r
        return _LTX_ORIG_LIP15(vp, image_path, audio_path, prompt, out_mp4,
                               ltx_paths=ltx_paths, seed=seed, fps=fps, **kwargs)
    generar_clip_lipsync_15s._ltx23 = True

if not getattr(generar_clip_lipsync_dialogo, "_ltx23", False):
    import inspect as _insp
    _LTX_ORIG_DLG = generar_clip_lipsync_dialogo
    def generar_clip_lipsync_dialogo(*args, **kwargs):
        try:
            _ps = _insp.signature(_LTX_ORIG_DLG).parameters
            _ok = "width" in _ps or any(p.kind == p.VAR_KEYWORD for p in _ps.values())
        except Exception:
            _ok = False
        if _ok and "width" not in kwargs and "height" not in kwargs:
            _img = args[1] if len(args) > 1 else (kwargs.get("image_path") or kwargs.get("img_ref"))
            _r = ltx_resolve("dlg", _img, None)
            if _r:
                kwargs["width"], kwargs["height"] = _r
        return _LTX_ORIG_DLG(*args, **kwargs)
    generar_clip_lipsync_dialogo._ltx23 = True

# ═══════════════════════════════════════════════════════════════
# LTX-2.3 — preparación aislada (Wan2GP fijado + venv + modelos + script)
# ═══════════════════════════════════════════════════════════════
LTX23_BASE    = "/tmp/ltx23"
LTX23_REPO    = f"{LTX23_BASE}/Wan2GP"
LTX23_MODELS  = f"{LTX23_BASE}/models"
LTX23_VENV    = f"{LTX23_BASE}/venv"
LTX23_PY      = f"{LTX23_VENV}/bin/python"
LTX23_SCRIPT  = f"{LTX23_BASE}/run_ltx23_generate.py"
LTX23_COMMIT  = "b8880bd5d292edfbe2384f5c951d95757ad59ea0"
LTX23_HF_REPO = "DeepBeepMeep/LTX-2"
_LTX23_READY  = {"ok": False}

def _ltx23_apply_patches(REPO_DIR):
    import os
    # 5. Apply Wan2GP Core Runtime Patches
    print('🔧 Applying Wan2GP runtime patches...')

    # Patch 1: ltx2.py runtime fixes
    ltx2_py = os.path.join(REPO_DIR, 'models/ltx2/ltx2.py')
    if os.path.exists(ltx2_py):
        with open(ltx2_py, 'r') as f:
            content = f.read()
        old_line1 = 'input_video_strength = max(0.0, min(1.0, input_video_strength))'
        new_line1 = 'input_video_strength = max(0.0, min(1.0, input_video_strength)) if input_video_strength is not None else 1.0'
        if old_line1 in content:
            content = content.replace(old_line1, new_line1)
        old_line2 = 'source = source.to(device=self.device, dtype=torch.bfloat16)'
        new_line2 = 'source = source.to(device=self.device, dtype=self.dtype)'
        if old_line2 in content:
            content = content.replace(old_line2, new_line2)
        with open(ltx2_py, 'w') as f:
            f.write(content)
        print('  ✓ Patched Wan2GP ltx2.py routines')

    # Patch 2: Spatial Upsampler Dtype Auto-Alignment
    upsampler_file = os.path.join(REPO_DIR, 'models/ltx2/ltx_core/model/upsampler/model.py')
    if os.path.exists(upsampler_file):
        with open(upsampler_file, 'r') as f:
            ucontent = f.read()
        old_up = 'latent = upsampler(latent)'
        new_up = ('upsampler_dtype = next(upsampler.parameters()).dtype if any(True for _ in upsampler.parameters()) else torch.float16\n'
                  '    if latent.dtype != upsampler_dtype:\n'
                  '        latent = latent.to(upsampler_dtype)\n'
                  '    latent = upsampler(latent)')
        if old_up in ucontent:
            ucontent = ucontent.replace(old_up, new_up)
        with open(upsampler_file, 'w') as f:
            f.write(ucontent)
        print('  ✓ Patched Wan2GP spatial upsampler auto-alignment')

    # Patch 3: Video VAE Dtype Auto-Alignment
    vae_file = os.path.join(REPO_DIR, 'models/ltx2/ltx_core/model/video_vae/video_vae.py')
    if os.path.exists(vae_file):
        with open(vae_file, 'r') as f:
            vcontent = f.read()
        old_v = 'sample = self.conv_in(sample, causal=self.causal)'
        new_v = ('conv_param = next(self.conv_in.parameters(), None)\n'
                 '        if conv_param is not None and sample.dtype != conv_param.dtype:\n'
                 '            sample = sample.to(conv_param.dtype)\n'
                 '        sample = self.conv_in(sample, causal=self.causal)')
        if old_v in vcontent:
            vcontent = vcontent.replace(old_v, new_v)
        old_enc = 'sample = self.conv_in(sample)'
        new_enc = ('enc_param = next(self.conv_in.parameters(), None)\n'
                   '        if enc_param is not None and sample.dtype != enc_param.dtype:\n'
                   '            sample = sample.to(enc_param.dtype)\n'
                   '        sample = self.conv_in(sample)')
        if old_enc in vcontent:
            vcontent = vcontent.replace(old_enc, new_enc)
        with open(vae_file, 'w') as f:
            f.write(vcontent)
        print('  ✓ Patched Wan2GP video VAE auto-alignment')

    # Patch 4: Audio VAE Dtype Auto-Alignment
    audio_vae_file = os.path.join(REPO_DIR, 'models/ltx2/ltx_core/model/audio_vae/audio_vae.py')
    if os.path.exists(audio_vae_file):
        with open(audio_vae_file, 'r') as f:
            acontent = f.read()
        old_a = 'decoded_audio = audio_decoder(latent)'
        new_a = ('decoder_param = next(audio_decoder.parameters(), None)\n'
                 '    if decoder_param is not None and latent.dtype != decoder_param.dtype:\n'
                 '        latent = latent.to(decoder_param.dtype)\n'
                 '    decoded_audio = audio_decoder(latent)\n'
                 '    vocoder_param = next(vocoder.parameters(), None)\n'
                 '    if vocoder_param is not None and decoded_audio.dtype != vocoder_param.dtype:\n'
                 '        decoded_audio = decoded_audio.to(vocoder_param.dtype)')
        if old_a in acontent:
            acontent = acontent.replace(old_a, new_a)
        with open(audio_vae_file, 'w') as f:
            f.write(acontent)
        print('  ✓ Patched Wan2GP audio VAE auto-alignment')

    # Patch 5: Vocoder STFT & Filter Dtype Auto-Alignment
    vocoder_file = os.path.join(REPO_DIR, 'models/ltx2/ltx_core/model/audio_vae/vocoder.py')
    if os.path.exists(vocoder_file):
        with open(vocoder_file, 'r') as f:
            voc_content = f.read()
        old_stft = 'spec = F.conv1d(y, self.forward_basis, stride=self.hop_length, padding=0)'
        new_stft = ('forward_basis = self.forward_basis.to(dtype=y.dtype, device=y.device) if self.forward_basis.dtype != y.dtype else self.forward_basis\n'
                    '        spec = F.conv1d(y, forward_basis, stride=self.hop_length, padding=0)')
        if old_stft in voc_content:
            voc_content = voc_content.replace(old_stft, new_stft)
        old_low = 'return F.conv1d(x, self.filter.expand(n_channels, -1, -1), stride=self.stride, groups=n_channels)'
        new_low = ('filt = self.filter.to(dtype=x.dtype, device=x.device) if self.filter.dtype != x.dtype else self.filter\n'
                   '        return F.conv1d(x, filt.expand(n_channels, -1, -1), stride=self.stride, groups=n_channels)')
        if old_low in voc_content:
            voc_content = voc_content.replace(old_low, new_low)
        with open(vocoder_file, 'w') as f:
            f.write(voc_content)
        print('  ✓ Patched Wan2GP vocoder auto-alignment')

    # Patch 6: Wan2GP Base Text Encoder Device Alignment (Fixes Multi-GPU device mismatch)
    # Gemma's hidden states stay in BF16: the feature extractor squares them for RMS normalization, and in
    # FP16 any value above 256 overflows (inf -> NaN text features -> black video).
    base_enc_file = os.path.join(REPO_DIR, 'models/ltx2/ltx_core/text_encoders/gemma/encoders/base_encoder.py')
    if os.path.exists(base_enc_file):
        with open(base_enc_file, 'r') as f:
            bcontent = f.read()
        old_b = 'attention_mask = item.attention_mask.to("cuda")\n        encoded_video_input, encoded_audio_input = _apply_feature_extractor(\n            item.hidden_states,'
        new_b = ('target_dev = next(feature_extractor_linear.parameters()).device if any(True for _ in feature_extractor_linear.parameters()) else torch.device("cuda:0")\n'
                 '        attention_mask = item.attention_mask.to(target_dev)\n'
                 '        hidden_states = tuple(h.to(device=target_dev) for h in item.hidden_states)\n'
                 '        encoded_video_input, encoded_audio_input = _apply_feature_extractor(\n'
                 '            hidden_states,')
        if old_b in bcontent:
            bcontent = bcontent.replace(old_b, new_b)
            with open(base_enc_file, 'w') as f:
                f.write(bcontent)
            print('  ✓ Patched Wan2GP base_encoder device & dtype auto-alignment')

    # Patch 7: Wan2GP Gemma Feature Extractor & Connector Dtype Auto-alignment
    fe_file = os.path.join(REPO_DIR, 'models/ltx2/ltx_core/text_encoders/gemma/feature_extractor.py')
    if os.path.exists(fe_file):
        with open(fe_file, 'r') as f:
            fe_content = f.read()
        old_fe = 'normed = _norm_and_concat_per_token_rms(encoded, attention_mask).to(encoded.dtype)\n            v_dim = self.video_aggregate_embed.out_features'
        new_fe = ('v_target_dtype = next(self.video_aggregate_embed.parameters()).dtype if any(True for _ in self.video_aggregate_embed.parameters()) else encoded.dtype\n'
                  '            v_target_dev = next(self.video_aggregate_embed.parameters()).device if any(True for _ in self.video_aggregate_embed.parameters()) else encoded.device\n'
                  '            normed = _norm_and_concat_per_token_rms(encoded.to(device=v_target_dev), attention_mask.to(v_target_dev)).to(dtype=v_target_dtype)\n'
                  '            v_dim = self.video_aggregate_embed.out_features')
        if old_fe in fe_content:
            fe_content = fe_content.replace(old_fe, new_fe)
        old_fe_aud = 'audio = self.audio_aggregate_embed(_rescale_norm(normed, a_dim, self.embedding_dim))'
        new_fe_aud = ('a_target_dtype = next(self.audio_aggregate_embed.parameters()).dtype if any(True for _ in self.audio_aggregate_embed.parameters()) else v_target_dtype\n'
                      '                normed_a = normed.to(dtype=a_target_dtype) if a_target_dtype != v_target_dtype else normed\n'
                      '                audio = self.audio_aggregate_embed(_rescale_norm(normed_a, a_dim, self.embedding_dim))')
        if old_fe_aud in fe_content:
            fe_content = fe_content.replace(old_fe_aud, new_fe_aud)
        with open(fe_file, 'w') as f:
            f.write(fe_content)
        print('  ✓ Patched Wan2GP feature_extractor dtype auto-alignment')

    conn_file = os.path.join(REPO_DIR, 'models/ltx2/ltx_core/text_encoders/gemma/embeddings_connector.py')
    if os.path.exists(conn_file):
        with open(conn_file, 'r') as f:
            conn_content = f.read()
        old_reg = 'learnable_registers = torch.tile(self.learnable_registers, (num_registers_duplications, 1))'
        new_reg = 'learnable_registers = torch.tile(self.learnable_registers.to(device=hidden_states.device, dtype=hidden_states.dtype), (num_registers_duplications, 1))'
        if old_reg in conn_content:
            conn_content = conn_content.replace(old_reg, new_reg)
            with open(conn_file, 'w') as f:
                f.write(conn_content)
            print('  ✓ Patched Wan2GP embeddings_connector register auto-alignment')

LTX23_SCRIPT_SRC = r'''import gc
import os
import sys
import json
import random
import tempfile
import glob
import time
import traceback
import numpy as np
import subprocess
import psutil
import soundfile as sf
from PIL import Image

# 1. Environment & Performance Settings
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["MALLOC_TRIM_THRESHOLD_"] = "0"

# ---- bootstrap Wan2GP ----
WAN2GP_DIR = os.environ.get("WAN2GP_DIR23", "/tmp/ltx23/Wan2GP")
# Videos are saved in the working folder (/kaggle/working/outputs), visible in Kaggle's Output panel
OUTPUT_DIR = "/tmp/ltx23/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", os.path.join(OUTPUT_DIR, ".gradio_cache"))
if WAN2GP_DIR not in sys.path:
    sys.path.insert(0, WAN2GP_DIR)
if os.path.exists(WAN2GP_DIR):
    os.chdir(WAN2GP_DIR)

import torch
try:
    from shared.utils.audio_video import save_video
except Exception:
    save_video = None

# Detect Hardware
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
_GPU_SM = torch.cuda.get_device_capability(0) if NUM_GPUS > 0 else (0, 0)
_IS_SM60 = (_GPU_SM[0] == 6)

print("=" * 65)
print("🎤 LTX-2.3 22B Distilled - Audio-to-Video Engine")
print("📺 Created by: AIQUEST Academy")
print(f"🎮 Accelerator: {'Dual GPU (T4 x2)' if NUM_GPUS > 1 else 'Single GPU'} | Compute: sm_{_GPU_SM[0]}{_GPU_SM[1]}")
print("=" * 65)

# CUDA Attention & Convolution Optimizations for Tesla T4 (Turing sm_75)
torch.backends.cuda.enable_flash_sdp(False)          # sm_75 does not support FlashAttention v2
torch.backends.cuda.enable_mem_efficient_sdp(True)   # Fast memory-efficient attention (Cutlass kernel)
torch.backends.cuda.enable_math_sdp(True)            # Safe fallback
torch.backends.cudnn.benchmark = False               # Autotuning FP32 conv3d on a T4 stalls the VAE for 10+ minutes

# ==== 1. GEMMA 3 SDPA & ROPE DTYPE PATCH ====
def _patch_gemma3_attention():
    """
    Fixes RuntimeError: Expected query, key, and value to have the same dtype,
    got query.dtype: float key.dtype: float and value.dtype: c10::BFloat16.
    """
    try:
        import transformers.models.gemma3.modeling_gemma3 as g3_mod
        _orig_rope = g3_mod.apply_rotary_pos_emb

        def _safe_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
            if cos.dtype != q.dtype:
                cos = cos.to(dtype=q.dtype)
            if sin.dtype != q.dtype:
                sin = sin.to(dtype=q.dtype)
            return _orig_rope(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

        g3_mod.apply_rotary_pos_emb = _safe_apply_rotary_pos_emb
        print("  [Gemma3 Patch] ✅ apply_rotary_pos_emb patched (RoPE dtype aligned with query)")
    except Exception as e:
        print(f"  [Gemma3 Patch] ⚠️ Could not patch apply_rotary_pos_emb: {e}")

    try:
        import transformers.integrations.sdpa_attention as sdpa_mod
        _orig_sdpa = sdpa_mod.sdpa_attention_forward

        def _safe_sdpa_attention_forward(module, query, key, value, *args, **kwargs):
            if key.dtype != query.dtype:
                key = key.to(dtype=query.dtype)
            if value.dtype != query.dtype:
                value = value.to(dtype=query.dtype)
            return _orig_sdpa(module, query, key, value, *args, **kwargs)

        sdpa_mod.sdpa_attention_forward = _safe_sdpa_attention_forward
        print("  [SDPA Patch] ✅ sdpa_attention_forward patched (query/key/value dtype auto-aligned)")
    except Exception as e:
        print(f"  [SDPA Patch] ⚠️ Could not patch sdpa_attention_forward: {e}")

_patch_gemma3_attention()

# ==== 2. GGUF EXTENSION HANDLER & CONFIG PATCH ====
def _register_gguf_handler():
    """Register the GGUF handler with mmgp's quant_router."""
    try:
        import shared.qtypes.gguf
        print("  [GGUF] ✅ Extension handler registered with mmgp (Wan2GP native)")
    except Exception as e:
        print(f"  [GGUF] ⚠️ Could not register gguf handler: {e}")

def _patch_ltx2_config_loading():
    """Patch _load_config_from_checkpoint to handle GGUF metadata gracefully."""
    import models.ltx2.ltx2 as ltx2_mod
    _original = ltx2_mod._load_config_from_checkpoint

    def _patched(path, fallback_config_path=None):
        from mmgp import quant_router
        if isinstance(path, (list, tuple)):
            path = path[0] if path else ""
        if not path:
            return {}
        try:
            _, metadata = quant_router.load_metadata_state_dict(path)
            if metadata:
                config_raw = metadata.get("config")
                if config_raw:
                    config = ltx2_mod._normalize_config(config_raw)
                    if config:
                        return config
        except Exception:
            pass
        if fallback_config_path and os.path.isfile(fallback_config_path):
            try:
                with open(fallback_config_path, "r", encoding="utf-8") as f:
                    config = ltx2_mod._normalize_config(json.load(f))
                    if config:
                        print(f"  [GGUF Patch] ✅ Config loaded from {os.path.basename(fallback_config_path)}")
                        return config
            except Exception:
                pass
        return {}

    ltx2_mod._load_config_from_checkpoint = _patched
    print("  [GGUF Patch] ✅ Config loading patched for GGUF")

_register_gguf_handler()
_patch_ltx2_config_loading()

# ==== 3. GPU SPECIFICATIONS & MEMORY STATS ====
if torch.cuda.is_available():
    print(f"Primary GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    if NUM_GPUS > 1:
        print(f"Secondary GPU: {torch.cuda.get_device_name(1)} (VRAM: {torch.cuda.get_device_properties(1).total_memory / 1024**3:.1f} GB)")
ram = psutil.virtual_memory()
print(f"System RAM: {ram.total / 1024**3:.1f} GB total, {ram.available / 1024**3:.1f} GB available")
sys.stdout.flush()

# ==== 4. LOAD LTX-2.3 MODEL COMPONENTS ====
print("\nLoading LTX-2.3 22B Distilled (GGUF Q4_K_M)...")
sys.stdout.flush()

from mmgp import offload
from shared.utils import files_locator as fl
from models.ltx2.ltx2_handler import family_handler

fl.set_checkpoints_paths(["models", "ckpts", "."])

base_model_type = "ltx2_22B"
model_def = {"ltx2_pipeline": "distilled"}
extra = family_handler.query_model_def(base_model_type, model_def)
model_def.update(extra)

gemma_folder = "models/gemma-3-12b-it-qat-q4_0-unquantized"
gemma_files = sorted(glob.glob(os.path.join(gemma_folder, "*.safetensors")))
quanto_files = [f for f in gemma_files if "quanto" in f]
text_encoder_file = quanto_files[0] if quanto_files else (gemma_files[0] if gemma_files else None)
if not text_encoder_file:
    raise FileNotFoundError(f"No .safetensors in {gemma_folder}. Check download cell.")
print(f"  Text encoder: {os.path.basename(text_encoder_file)}")

transformer_path = os.path.join("models", "ltx-2.3-22b-distilled-Q4_K_M_light.gguf")
if not os.path.isfile(transformer_path):
    raise FileNotFoundError(f"{transformer_path} missing. Check download cell.")
print(f"  Transformer : {os.path.basename(transformer_path)}")
sys.stdout.flush()

# FP16 transformer: the T4's tensor cores make it ~3x faster than BF16 (the original May setup).
# Gemma's hidden states are kept in BF16 (see the dual-GPU section); a NaN guard after Stage 1 stops
# early if anything overflows. The video VAE + upsampler run in FP32 (converted below).
MODEL_DTYPE = torch.float16
VAE_DTYPE   = torch.float32

ltx2_model, pipe = family_handler.load_model(
    model_filename=transformer_path,
    model_type="ltx2_22B_distilled",
    base_model_type=base_model_type,
    model_def=model_def,
    dtype=MODEL_DTYPE,
    VAE_dtype=VAE_DTYPE,
    text_encoder_filename=text_encoder_file,
)

# Align Gemma embed_tokens to bfloat16 to avoid float32 inputs_embeds
try:
    _g_embed = ltx2_model.text_encoder.model.model.embed_tokens
    _g_embed.to(torch.bfloat16)
except Exception:
    try:
        ltx2_model.text_encoder.model.embed_tokens.to(torch.bfloat16)
    except Exception:
        pass

# ==== 4b. FP32 VIDEO VAE, DTYPE-SAFE FORWARDS & STAGE PROBES ====
# The LTX video VAE + latent upsampler must run in FP32 on a T4: in FP16/BF16 the decoded video
# collapses to flat grey with neon edges. The pipeline feeds them BF16 latents (the cause of
# "Input type (BFloat16) and bias type (Half)"), so their forwards cast the input to the weight
# dtype/device and the encoder hands its result back in the caller's dtype and device.
for _key in ("video_decoder", "video_encoder", "spatial_upsampler"):
    _mod = getattr(ltx2_model, _key, None)
    if _mod is None:
        continue
    _mod.to(torch.float32)
    _mod._model_dtype = torch.float32
    for _m in _mod.modules():
        _m._lock_dtype = torch.float32
print("  [VAE] ✅ Video VAE + spatial upsampler converted to FP32")

import models.ltx2.ltx_core.model.video_vae.video_vae as _video_vae_mod
_orig_video_decoder_forward = _video_vae_mod.VideoDecoder.forward
_orig_video_encoder_forward = _video_vae_mod.VideoEncoder.forward

def _fp32_video_decoder_forward(self, sample, *args, **kwargs):
    param = next(self.parameters(), None)
    if param is not None and sample.dtype != param.dtype:
        sample = sample.to(param.dtype)
    return _orig_video_decoder_forward(self, sample, *args, **kwargs)

def _fp32_video_encoder_forward(self, sample, *args, **kwargs):
    in_dtype, in_device = sample.dtype, sample.device
    param = next(self.parameters(), None)
    if param is not None and param.device.type == "cuda" and sample.device != param.device:
        sample = sample.to(param.device)
    if param is not None and sample.dtype != param.dtype:
        sample = sample.to(param.dtype)
    with torch.cuda.device(sample.device if sample.is_cuda else torch.cuda.current_device()):
        out = _orig_video_encoder_forward(self, sample, *args, **kwargs)
    if torch.is_tensor(out) and out.is_floating_point():
        out = out.to(device=in_device, dtype=in_dtype if in_dtype.is_floating_point else out.dtype)
    return out

_video_vae_mod.VideoDecoder.forward = _fp32_video_decoder_forward
_video_vae_mod.VideoEncoder.forward = _fp32_video_encoder_forward

# distilled.py binds these helpers by name at import, so wrap them inside that module
import models.ltx2.ltx_pipelines.distilled as _distilled_mod
from models.ltx2.ltx_core.model.video_vae.tiling import TilingConfig, SpatialTilingConfig, TemporalTilingConfig
_orig_pipeline_upsample = _distilled_mod.upsample_video
_orig_pipeline_decode = _distilled_mod.vae_decode_video_to_tensor
UPS_ON_GPU1 = False  # set below once the upsampler + encoder are moved to GPU 1

def _latent_stats(tag, t):
    try:
        t = t[0] if isinstance(t, (list, tuple)) else t
        f = t.detach().float()
        nonfinite = int((~torch.isfinite(f)).sum().item())
        f = torch.nan_to_num(f)
        print(f"  🔬 {tag}: shape={tuple(t.shape)} mean={f.mean().item():+.3f} std={f.std().item():.3f} nonfinite={nonfinite}")
        sys.stdout.flush()
        return nonfinite
    except Exception as _e:
        print(f"  🔬 {tag}: stats unavailable ({_e})")
        return 0

def _probed_upsample_video(latent, video_encoder, upsampler):
    if _latent_stats("Stage 1 latent", latent):
        # Fail fast: NaN here always ends as a black video, no point spending minutes on Stage 2 + VAE
        raise RuntimeError("Stage 1 produced NaN latents (FP16 overflow). Check the '🔬 Gemma hidden states' line: "
                           "they must stay BF16. As a last resort set MODEL_DTYPE = torch.bfloat16 (slower).")
    in_dtype, in_device = latent.dtype, latent.device
    if UPS_ON_GPU1:
        with torch.cuda.device(1):
            out = _orig_pipeline_upsample(latent=latent.to("cuda:1"), video_encoder=video_encoder, upsampler=upsampler)
    else:
        out = _orig_pipeline_upsample(latent=latent, video_encoder=video_encoder, upsampler=upsampler)
    out = out.to(device=in_device, dtype=in_dtype)
    _latent_stats("Upscaled latent", out)
    return out

def _fast_vae_tiling(height, width):
    # Whole frames when they fit, official LTX overlaps (24 frames / 64 px), longest chunks the
    # T4 allows while the transformer is offloaded. ~217 B per pixel-frame measured for this decoder.
    budget = torch.cuda.get_device_properties(0).total_memory - 1.6e9 - 3.0e9
    bytes_per_pixel_frame = 230
    spatial = None
    tile_area = height * width
    if budget / (bytes_per_pixel_frame * tile_area) < 96:
        spatial = SpatialTilingConfig(tile_size_in_pixels=512, tile_overlap_in_pixels=64)
        tile_area = min(height, 512) * min(width, 512)
    frames = max(32, min(160, int(budget / (bytes_per_pixel_frame * tile_area)) // 8 * 8))
    return TilingConfig(spatial_config=spatial,
                        temporal_config=TemporalTilingConfig(tile_size_in_frames=frames, tile_overlap_in_frames=24))

def _probed_decode(latent, video_decoder, tiling_config=None, *args, **kwargs):
    _latent_stats("Final latent -> VAE", latent)
    latent_tensor = latent[0] if isinstance(latent, list) else latent  # decode clears the list; keep it for a retry
    fast_config = _fast_vae_tiling(kwargs.get("expected_height") or 480, kwargs.get("expected_width") or 832)
    print(f"  🎞️ VAE tiling: {'whole frame' if fast_config.spatial_config is None else '512 px tiles'}, "
          f"{fast_config.temporal_config.tile_size_in_frames}-frame chunks (24 overlap)")
    t_dec = time.time()
    torch.cuda.reset_peak_memory_stats(0)
    try:
        out = _orig_pipeline_decode([latent_tensor] if isinstance(latent, list) else latent_tensor,
                                    video_decoder, fast_config, *args, **kwargs)
    except torch.OutOfMemoryError:
        print("  ⚠️ VAE out of memory with large chunks, retrying with Wan2GP's default tiling...")
        gc.collect()
        torch.cuda.empty_cache()
        out = _orig_pipeline_decode([latent_tensor] if isinstance(latent, list) else latent_tensor,
                                    video_decoder, tiling_config, *args, **kwargs)
    print(f"  🎞️ VAE decode: {time.time() - t_dec:.1f}s (peak VRAM {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB)")
    return out

_distilled_mod.upsample_video = _probed_upsample_video
_distilled_mod.vae_decode_video_to_tensor = _probed_decode

# ==== 5. DUAL T4 ACCELERATION (GPU T4 x2) ====
AUX_ON_GPU1 = False

if NUM_GPUS > 1:
    print("\n🚀 Dual GPU Active (GPU T4 x2) - Offloading Gemma 3 Text Encoder to GPU 1...")
    try:
        # Move Gemma 3 text encoder to GPU 1
        ltx2_model.text_encoder.to("cuda:1")
        _gemma = ltx2_model.text_encoder.model
        _gemma_lm = _gemma.model if hasattr(_gemma, "model") else _gemma
        if getattr(_gemma, "lm_head", None) is not None:
            _gemma.lm_head = torch.nn.Identity()

        # Token-embedding lookup on CPU (few ms) gives GPU 1 maximum free VRAM
        _embed = _gemma_lm.embed_tokens
        _embed.to("cpu")
        _orig_embed_fwd = _embed.forward

        def _cpu_embed_fwd(input_ids, *args, **kwargs):
            return _orig_embed_fwd(input_ids.to("cpu"), *args, **kwargs).to("cuda:1")

        _embed.forward = _cpu_embed_fwd
        _gemma.__class__ = type(_gemma.__class__.__name__, (_gemma.__class__,),
                                {"device": property(lambda self: torch.device("cuda", 1))})
        pipe.pop("text_encoder", None)

        # Wrap encode_raw to send output hidden_states & attention_mask back to cuda:0 for downstream pipeline
        _orig_encode_raw = ltx2_model.text_encoder.encode_raw

        def _dual_gpu_encode_raw(text, padding_side="left"):
            raw_emb = _orig_encode_raw(text, padding_side=padding_side)
            # Device move only: Gemma's hidden states must stay BF16 (values far above FP16's range
            # once squared by the feature extractor's RMS norm; casting them to FP16 gave all-NaN latents)
            hs_cuda0 = tuple(h.to(device="cuda:0") for h in raw_emb.hidden_states)
            _absmax = max(float(h.abs().max()) for h in hs_cuda0)
            print(f"  🔬 Gemma hidden states: {len(hs_cuda0)} layers, dtype={hs_cuda0[0].dtype}, absmax={_absmax:.0f}")
            mask_cuda0 = raw_emb.attention_mask.to("cuda:0")
            return raw_emb.__class__(hs_cuda0, mask_cuda0, raw_emb.padding_side)

        ltx2_model.text_encoder.encode_raw = _dual_gpu_encode_raw

        # Fail-safe device and dtype alignment in postprocess_text_embeddings
        try:
            import models.ltx2.ltx_core.text_encoders.gemma.encoders.base_encoder as _base_enc_mod
            _orig_postprocess = _base_enc_mod.postprocess_text_embeddings

            def _safe_postprocess(embeddings, feat_ext, emb_conn, aud_conn, return_attention_masks=False):
                target_dev = next(feat_ext.parameters()).device if any(True for _ in feat_ext.parameters()) else torch.device("cuda:0")
                safe_embs = []
                for item in embeddings:
                    hs = tuple(h.to(device=target_dev) for h in item.hidden_states)  # keep BF16 (see above)
                    mask = item.attention_mask.to(target_dev)
                    safe_embs.append(item.__class__(hs, mask, item.padding_side))
                return _orig_postprocess(safe_embs, feat_ext, emb_conn, aud_conn, return_attention_masks=return_attention_masks)

            _base_enc_mod.postprocess_text_embeddings = _safe_postprocess
            print("  [GPU T4 x2] ✅ postprocess_text_embeddings device & dtype alignment active.")
        except Exception as _pe:
            print(f"  [GPU T4 x2] ⚠️ Postprocess patch note: {_pe}")

        AUX_ON_GPU1 = True
        _free1 = torch.cuda.mem_get_info(1)[0] / 1024**3
        print(f"  [GPU T4 x2] ✅ Gemma 3 text encoder resident on GPU 1 ({_free1:.1f} GB free on GPU 1).")
        print("  [GPU T4 x2] ✅ GPU 0 is dedicated to the Transformer and VAE decoder.")
    except Exception as e:
        print(f"  [GPU T4 x2] ⚠️ Dual GPU setup notice: {e}, falling back to single GPU mode")
        AUX_ON_GPU1 = False

# The upsampler (between the stages) and the video encoder (reference image, once per stage) would
# each evict the transformer from GPU 0 under mmgp. Both fit next to Gemma on GPU 1 in FP32.
if AUX_ON_GPU1 and pipe.get("spatial_upsampler") is not None:
    with torch.cuda.device(1):
        torch.cuda.empty_cache()
    try:
        for _key in ("spatial_upsampler", "video_encoder"):
            pipe[_key].to("cuda:1")
        for _key in ("spatial_upsampler", "video_encoder"):
            pipe.pop(_key)
        UPS_ON_GPU1 = True
        print(f"  [GPU T4 x2] ✅ Spatial upsampler + video encoder resident on GPU 1 ({torch.cuda.mem_get_info(1)[0] / 1024**3:.1f} GB free).")
    except Exception as e:
        print(f"  [GPU T4 x2] ⚠️ Upsampler / encoder stay on GPU 0 ({e})")
        for _key in ("spatial_upsampler", "video_encoder"):
            getattr(ltx2_model, _key).to("cpu")
        with torch.cuda.device(1):
            torch.cuda.empty_cache()

sys.stdout.flush()

# ==== 6. APPLY MMGP OFFLOADING ON GPU 0 ====
# When Gemma 3 is on GPU 1, transformer budget on GPU 0 can be increased to 11000 MB!
# This keeps the ~11.8 GB Q4_K_M model resident in VRAM during diffusion (no PCIe streaming).
transformer_budget = 11000 if AUX_ON_GPU1 else 6000
print(f"\nApplying mmgp Profile 4 (Transformer VRAM budget: {transformer_budget} MB)...")
sys.stdout.flush()

budgets = {
    "transformer":       transformer_budget,
    "vae":               3500,   # FP32 video decoder (1.5 GB) + working memory
    "spatial_upsampler": 2000,
    "video_encoder":     2000,
    "audio_encoder":     1000,
    "audio_decoder":     1000,
    "vocoder":           500,
    "*":                 1000,
}
if not AUX_ON_GPU1:
    budgets["text_encoder"] = 1500
if UPS_ON_GPU1:
    budgets.pop("spatial_upsampler")
    budgets.pop("video_encoder")

offload.profile(
    pipe,
    profile_no=4,
    quantizeTransformer=False,
    convertWeightsFloatTo=torch.float16,
    budgets=budgets,
)
offload.shared_state["_attention"] = "sdpa"

print("\n✅ Setup complete! Distilled Audio-to-Video pipeline active.")
sys.stdout.flush()

# ==== 7. ENTRADA PARA LA AUTOMATIZACIÓN (sustituye la UI Gradio; mismo pipeline del notebook) ====
def snap_frames(n, max_frames=481):
    k = max(0, round((int(n) - 1) / 8))
    return int(max(49, min(8 * k + 1, max_frames)))

_st = {"total": 8, "cur": 0, "pass": 1, "t0": time.time(), "lt": time.time()}

def _cb(step, latent=None, is_start=False, override_num_inference_steps=None, pass_no=None, **kwargs):
    if kwargs.get("progress_unit") is not None:
        return
    if is_start:
        if override_num_inference_steps is not None:
            _st["total"] = int(override_num_inference_steps)
        if pass_no is not None:
            _st["pass"] = pass_no
        _st["cur"] = 0
        _st["t0"] = _st["lt"] = time.time()
        print(f"PROGRESS:start:{_st['total']}", flush=True)
        return
    _st["cur"] += 1
    now = time.time()
    cur, tot = min(_st["cur"], _st["total"]), max(_st["total"], 1)
    dt = now - _st["lt"]
    rem = max(0, tot - cur) * (now - _st["t0"]) / max(cur, 1)
    _st["lt"] = now
    free = torch.cuda.mem_get_info(0)[0] / 1024**3
    print(f"PROGRESS:step:{cur}:{tot}:{dt:.1f}:{rem:.0f}:{free:.1f}:Stage{_st['pass']}:{int(cur * 100 / tot)}", flush=True)

def _stage(status):
    print(f"PROGRESS:stage:{status}", flush=True)

def _save_fallback(t, out_path, fps):
    t = t.detach().cpu()
    if t.ndim == 5:
        t = t[0]
    if t.ndim == 4 and t.shape[0] <= 4 and t.shape[-1] > 4:
        t = t.permute(1, 2, 3, 0)
    t = t.float()
    t = ((t + 1) / 2 * 255 if float(t.min()) < -0.5 else (t * 255 if float(t.max()) <= 1.0 else t))
    arr = t.clamp(0, 255).to(torch.uint8).numpy()
    T, H, W, _ = arr.shape
    p = subprocess.Popen(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(int(fps)),
                          "-i", "pipe:0", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "fast", out_path],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.stdin.write(arr.tobytes()); p.stdin.close(); p.communicate(timeout=600)

@torch.inference_mode()
def generate(params):
    prompt = params.get("prompt", "")
    imgp, audp = params.get("image_path") or "", params.get("audio_path") or ""
    out_path = params["output_path"]
    width, height = int(params["width"]), int(params["height"])
    frame_rate = float(params.get("fps", 24))
    num_frames = snap_frames(params.get("num_frames", 121))
    seed = int(params.get("seed", -1))
    if seed < 0:
        seed = random.randint(0, 2**32 - 1)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    gc.collect(); torch.cuda.empty_cache()

    input_waveform, sr = None, None
    if audp and os.path.isfile(audp):
        wav, sr = sf.read(audp)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)
        need = int(np.ceil(num_frames / frame_rate * sr))
        if len(wav) < need:
            wav = np.pad(wav, (0, need - len(wav)))      # cola en silencio hasta cubrir todos los frames
        input_waveform, sr = wav, int(sr)
        print(f"  [Audio] sr={sr}, {len(wav) / sr:.1f}s")
    image_start = Image.open(imgp).convert("RGB") if imgp and os.path.exists(imgp) else None

    print(f"🎬 Generating: {width}x{height}, {num_frames} frames, seed={seed}, audio={'YES' if input_waveform is not None else 'NO'}", flush=True)
    gen_kwargs = dict(
        input_prompt=prompt, image_start=image_start, height=height, width=width,
        frame_num=num_frames, fps=frame_rate, seed=seed, callback=_cb, VAE_tile_size=512,
        input_video_strength=1.0, denoising_strength=1.0, guide_scale=1.0, audio_cfg_scale=1.0,
        alt_guide_scale=1.0, sampling_steps=8, guide_phases=2, n_prompt="",
        video_prompt_type="", audio_prompt_type="", set_progress_status=_stage,
    )
    if input_waveform is not None:
        gen_kwargs.update(input_waveform=input_waveform, input_waveform_sample_rate=sr, audio_scale=1.0)

    t_start = time.time()
    result = ltx2_model.generate(**gen_kwargs)
    print(f"  [Pipeline Completed in {time.time() - t_start:.1f}s]", flush=True)
    if result is None:
        raise RuntimeError("Generation failed or was interrupted.")
    video_tensor = result.get("x") if isinstance(result, dict) else (result[0] if isinstance(result, tuple) else result)
    if video_tensor is None or not torch.is_tensor(video_tensor):
        raise RuntimeError(f"No video tensor produced: {type(video_tensor)}")
    video_tensor = video_tensor.cpu()
    gc.collect(); torch.cuda.empty_cache()

    _stage("Saving")
    try:
        if save_video is None:
            raise RuntimeError("save_video no disponible")
        if video_tensor.dtype == torch.uint8:
            save_video(tensor=video_tensor.unsqueeze(0), save_file=out_path, fps=frame_rate, nrow=1)
        else:
            save_video(tensor=video_tensor.unsqueeze(0).float() / 127.5 - 1.0, save_file=out_path, fps=frame_rate,
                       nrow=1, normalize=True, value_range=(-1, 1))
        if not os.path.exists(out_path) or os.path.getsize(out_path) < 50_000:
            raise RuntimeError("save_video produjo un archivo vacío")
    except Exception as e:
        print(f"  ⚠ save_video falló ({e}); usando guardado ffmpeg", flush=True)
        _save_fallback(video_tensor, out_path, frame_rate)
    del video_tensor
    gc.collect(); torch.cuda.empty_cache()
    return out_path

if __name__ == "__main__":
    try:
        op = generate(json.loads(sys.argv[1]))
        print(f"OUTPUT_PATH:{op}", flush=True)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
'''

def _ltx23_run(cmd, what, timeout=3600, env=None):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"[LTX-2.3] {what} falló:\n{((r.stderr or '') + (r.stdout or ''))[-1500:]}")
    return r

def _ltx23_import_check():
    r = subprocess.run([LTX23_PY, "-c",
        "import torch, torchaudio, transformers; from mmgp import offload, quant_router; "
        "import transformers.models.gemma3.modeling_gemma3"],
        capture_output=True, text=True, timeout=900)
    return r.returncode == 0, ((r.stderr or "") + (r.stdout or ""))[-2000:]

def _ltx23_setup_repo():
    os.makedirs(LTX23_REPO, exist_ok=True)
    g = lambda *a: subprocess.run(["git", "-C", LTX23_REPO, *a], capture_output=True, text=True)
    if not os.path.isdir(os.path.join(LTX23_REPO, ".git")):
        g("init", "-q"); g("remote", "add", "origin", "https://github.com/DeepBeepMeep/Wan2GP.git")
    head = g("rev-parse", "HEAD").stdout.strip()
    if head != LTX23_COMMIT:
        log(f"[LTX-2.3] Descargando Wan2GP @ {LTX23_COMMIT[:10]} (carpeta aislada {LTX23_REPO})...")
        r = g("fetch", "-q", "--depth", "1", "origin", LTX23_COMMIT)
        if r.returncode != 0:
            raise RuntimeError("[LTX-2.3] git fetch falló: " + (r.stderr or "")[-500:])
        g("checkout", "-q", "-f", "FETCH_HEAD")
    else:
        g("reset", "-q", "--hard")
    _ltx23_apply_patches(LTX23_REPO)

def _ltx23_pip(pc, args, what, timeout=3600):
    """pip con salida en vivo (para ver si avanza o está atascado)."""
    p = subprocess.Popen(pc + ["install", "--timeout", "120", "--retries", "5", "--no-warn-conflicts"] + args,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    t0, last, tail = time.time(), 0.0, []
    for l in p.stdout:
        l = l.rstrip(); tail = (tail + [l])[-40:]
        if l.startswith(("ERROR", "Successfully", "Installing collected")) or (l.startswith(("Collecting", "Downloading")) and time.time() - last > 8):
            log(f"[LTX-2.3] pip: {l[:140]}"); last = time.time()
        if time.time() - t0 > timeout:
            p.kill(); raise RuntimeError(f"[LTX-2.3] {what}: timeout\n" + "\n".join(tail))
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"[LTX-2.3] {what} falló:\n" + "\n".join(tail))

def _ltx23_setup_venv():
    """venv 100% aislado (SIN system-site-packages): su propio torch/torchaudio cu128 coherentes."""
    marker = os.path.join(LTX23_BASE, ".deps_ok3")
    if os.path.exists(marker) and os.path.exists(LTX23_PY):
        return
    shutil.rmtree(LTX23_VENV, ignore_errors=True)
    log("[LTX-2.3] Creando entorno aislado (venv propio)...")
    _ltx23_run([sys.executable, "-m", "venv", "--without-pip", LTX23_VENV], "crear venv")
    pc = [sys.executable, "-m", "pip", "--python", LTX23_PY]     # pip del sistema apuntando al venv (ensurepip no funciona aquí)
    if subprocess.run(pc + ["--version"], capture_output=True).returncode != 0:
        raise RuntimeError("[LTX-2.3] pip no soporta --python (actualiza pip) o no está disponible.")
    log("[LTX-2.3] Instalando torch/torchvision/torchaudio cu128 en el venv (~4 GB, solo la primera vez)...")
    _ltx23_pip(pc, ["torch==2.11.0+cu128", "torchvision==0.26.0+cu128", "torchaudio==2.11.0+cu128", "--index-url", "https://download.pytorch.org/whl/cu128",
                    "--extra-index-url", "https://pypi.org/simple"], "pip torch", timeout=3600)
    r = subprocess.run([LTX23_PY, "-c", "import torch,torchvision,torchaudio as a;"
                        "print(torch.__version__,torchvision.__version__,a.__version__)"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("[LTX-2.3] torch del venv no importa:\n" + (r.stderr or "")[-800:])
    tv, vv, av = r.stdout.split()
    if "cu128" not in tv:
        raise RuntimeError(f"[LTX-2.3] torch del venv no es cu128: {tv}")
    cfile = os.path.join(LTX23_BASE, "constraints.txt")
    open(cfile, "w").write(f"torch=={tv.split('+')[0]}\ntorchvision=={vv.split('+')[0]}\ntorchaudio=={av.split('+')[0]}\n")   # sin sufijo local: coincide con +cu128
    log(f"[LTX-2.3] torch del venv: {tv} | {vv} | {av}")
    with open(os.path.join(LTX23_REPO, "requirements.txt")) as f:
        reqs = [("onnxruntime-gpu>=1.22.0" if l.startswith("onnxruntime-gpu==") and "dev" in l else l.rstrip("\n")) for l in f]
    req_file = os.path.join(LTX23_BASE, "wan2gp_requirements.txt")
    open(req_file, "w").write("\n".join(reqs) + "\n")
    log("[LTX-2.3] Instalando dependencias de Wan2GP en el venv...")
    _ltx23_pip(pc, ["-c", cfile, "-r", req_file], "pip requirements", timeout=3600)
    _ltx23_pip(pc, ["-c", cfile, "gradio>=5.0.0", "gguf", "soundfile", "einops", "sentencepiece", "accelerate",
                    "psutil", "pillow", "numpy"], "pip extras", timeout=1800)
    ok, err = _ltx23_import_check()
    if not ok and ("huggingface_hub" in err.lower() or "is_offline_mode" in err):
        log("[LTX-2.3] Ajustando huggingface_hub dentro del venv...")
        _ltx23_pip(pc, ["-c", cfile, "--upgrade", "huggingface_hub>=1.5.0,<2.0.0"], "pip huggingface_hub")
        ok, err = _ltx23_import_check()
    if not ok:
        raise RuntimeError("[LTX-2.3] El venv no puede importar torch/transformers/mmgp:\n" + err)
    open(marker, "w").write("ok")

def _ltx23_download_models():
    from huggingface_hub import hf_hub_download
    os.makedirs(LTX23_MODELS, exist_ok=True)
    mdir = os.path.join(LTX23_REPO, "models")
    os.makedirs(mdir, exist_ok=True)
    files = ["ltx-2.3-22b-distilled-Q4_K_M_light.gguf", "ltx-2.3-22b_embeddings_connector.safetensors",
             "ltx-2.3-22b_text_embedding_projection.safetensors", "ltx-2.3-22b_vae.safetensors",
             "ltx-2.3-22b_audio_vae.safetensors", "ltx-2.3-22b_vocoder.safetensors",
             "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"]
    def link(real, dest):
        if os.path.lexists(dest) and not os.path.exists(dest):
            os.remove(dest)
        if not os.path.lexists(dest):
            os.symlink(real, dest)
    for f in files:
        real = os.path.join(LTX23_MODELS, f)
        if not (os.path.exists(real) and os.path.getsize(real) > 1024 * 1024):
            log(f"[LTX-2.3] ↓ {f}")
            hf_hub_download(repo_id=LTX23_HF_REPO, filename=f, local_dir=LTX23_MODELS)
        link(real, os.path.join(mdir, f))
    gf = "gemma-3-12b-it-qat-q4_0-unquantized"
    gfiles = ["gemma-3-12b-it-qat-q4_0-unquantized_quanto_bf16_int8.safetensors", "added_tokens.json",
              "chat_template.json", "config_light.json", "generation_config.json", "preprocessor_config.json",
              "processor_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer.model",
              "tokenizer_config.json"]
    gdir = os.path.join(LTX23_MODELS, gf)
    os.makedirs(gdir, exist_ok=True)
    for g in gfiles:
        p = os.path.join(gdir, g)
        if not (os.path.exists(p) and os.path.getsize(p) > 0):
            log(f"[LTX-2.3] ↓ gemma/{g}")
            hf_hub_download(repo_id=LTX23_HF_REPO, filename=f"{gf}/{g}", local_dir=LTX23_MODELS)
    link(gdir, os.path.join(mdir, gf))
    for d in (os.path.join(LTX23_MODELS, ".cache"),):
        shutil.rmtree(d, ignore_errors=True)

def ltx23_prepare(force=False):
    """Idempotente: deja listo Wan2GP-2.3 + venv + modelos + script. No toca nada de LTX-2.5."""
    if _LTX23_READY["ok"] and not force and os.path.exists(LTX23_SCRIPT):
        return
    os.makedirs(LTX23_BASE, exist_ok=True)
    _ltx23_setup_repo()
    _ltx23_setup_venv()
    _ltx23_download_models()
    open(LTX23_SCRIPT, "w", encoding="utf-8").write(LTX23_SCRIPT_SRC)
    _LTX23_READY["ok"] = True
    log("[LTX-2.3] ✅ Entorno aislado listo.")

def ltx23_diag():
    try:
        ltx23_prepare()
        ok, err = _ltx23_import_check()
        return "✅ Entorno LTX-2.3 OK (venv aislado: transformers + mmgp)." if ok else f"❌ {err}"
    except Exception as e:
        return f"❌ {e}"

class _LTX23Backend:
    """Motor LTX-2.3 (subproceso, 2×T4). Misma interfaz/progreso que SubprocessLTXBackend."""
    def __init__(self, base):
        self._base = base
    def is_available(self):
        return True            # se prepara solo al primer uso (o con «Descargar e inicializar LTX»)
    def generate(self, prompt="", image_path=None, audio_path=None, num_frames=121, width=1152, height=704,
                 seed=-1, steps=8, cfg=1.0, fps=24, audio_cfg=1.0, input_video_strength=0.8,
                 denoise_strength=1.0, output_path=None):
        ltx23_prepare()
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
        out_path = self._base._resolve_output_path(output_path)
        params = {"prompt": prompt, "image_path": image_path or "", "audio_path": audio_path or "",
                  "output_path": out_path, "width": int(width), "height": int(height),
                  "num_frames": int(num_frames), "fps": fps, "seed": seed}
        mode = "A2V" if audio_path else ("I2V" if (image_path and os.path.exists(image_path)) else "T2V")
        log(f"[LTX23] {mode} | {width}x{height} | {num_frames}f@{fps}fps | seed={seed}")
        _write_progress({"status": "⏳ Iniciando subproceso LTX-2.3...", "active": True, "step": 0, "total": 8, "pct": 0})
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(p for p in env.get("PYTHONPATH", "").split(os.pathsep)
                                            if p and "ltx_hf_compat" not in p)
        env.pop("CUDA_VISIBLE_DEVICES", None)           # LTX-2.3 usa las dos T4 (como su notebook)
        env.update(WAN2GP_DIR23=LTX23_REPO, PYTHONUNBUFFERED="1", HF_HUB_ENABLE_HF_TRANSFER="0")
        t0, final_path, errs = time.time(), out_path, []
        proc = subprocess.Popen([LTX23_PY, "-u", LTX23_SCRIPT, json.dumps(params)], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
        def _rd():
            for l in proc.stderr:
                l = l.rstrip()
                if l: errs.append(l)
        th = threading.Thread(target=_rd, daemon=True); th.start()
        for raw in proc.stdout:
            line = raw.rstrip()
            if not line: continue
            if line.startswith("OUTPUT_PATH:"):
                cand = line.split("OUTPUT_PATH:", 1)[1].strip()
                if os.path.exists(cand): final_path = cand
                continue
            if line.startswith("PROGRESS:"):
                msg = self._base._parse_progress_line(line)
                if msg: log(f"[LTX23] {msg}")
                continue
            log(f"  {line}")
        proc.wait(); th.join(timeout=5)
        el = time.time() - t0
        if proc.returncode != 0:
            tail = "\n".join(errs[-30:])
            _write_progress({"status": f"❌ Error (rc={proc.returncode})", "active": False})
            raise RuntimeError(f"LTX-2.3 subprocess falló (rc={proc.returncode}):\n{tail[-2000:]}")
        if not os.path.exists(final_path) or os.path.getsize(final_path) < 50_000:
            raise RuntimeError(f"Video LTX-2.3 no válido: {final_path}")
        _write_progress({"status": f"✅ Completado en {el:.1f}s", "active": False, "pct": 100, "path": final_path})
        log(f"[LTX23] ✅ {el:.1f}s → {final_path} ({os.path.getsize(final_path)//1024} KB)")
        return final_path

class _LTXRouter:
    """Reparte según LTX_VERSION; con '2.5' delega en el motor original sin cambios."""
    _is_router = True
    def __init__(self, base):
        self._base = base
        self._v23 = _LTX23Backend(base)
    def _cur(self):
        return self._v23 if LTX_VERSION == "2.3" else self._base
    def is_available(self):
        return self._cur().is_available()
    def generate(self, *a, **k):
        return self._cur().generate(*a, **k)
    def __getattr__(self, name):
        return getattr(self._base, name)

_b0 = get_ltx_backend()
if not getattr(_b0, "_is_router", False):
    _b0 = _LTXRouter(_b0)
_LTX_BACKEND = _b0

# ── Compatibilidad con la UI: preset + ensure_ltx23_models (descarga/prepara LTX-2.3 en vez de 2.5) ──
if LTX_VERSION == "2.3":
    _LBL = "⚡ RÁPIDO — LTX-2.3 Distilled Q4_K_M (GGUF, 2×T4, 8 steps)"
    LTX23_PRESETS = globals().get("LTX23_PRESETS") or {}
    LTX23_PRESETS.clear()
    LTX23_PRESETS[_LBL] = {"desc": "LTX-2.3 22B Distilled GGUF Q4. Wan2GP y venv aislados en /tmp/ltx23 (no afecta a LTX-2.5)."}
    DEFAULT_LTX23_PRESET = _LBL
    def ensure_ltx23_models(preset_label=None):
        ltx23_prepare()
        m = LTX23_MODELS
        return {"transformer": f"{m}/ltx-2.3-22b-distilled-Q4_K_M_light.gguf",
                "encoder": f"{m}/gemma-3-12b-it-qat-q4_0-unquantized/gemma-3-12b-it-qat-q4_0-unquantized_quanto_bf16_int8.safetensors",
                "video_vae": f"{m}/ltx-2.3-22b_vae.safetensors", "audio_vae": f"{m}/ltx-2.3-22b_audio_vae.safetensors",
                "embeddings_connector": f"{m}/ltx-2.3-22b_embeddings_connector.safetensors",
                "_preset": preset_label or _LBL, "_steps": 8, "_cfg": 1.0, "_vram_estimado": 12, "_tipo": "distilled"}
    ensure_ltx25_models = ensure_ltx23_models

log(f"CELDA 14b COMPLETADA — motor LTX activo: {LTX_VERSION}")
log(f"  ► Clips lejanos: resolución/duración desde la UI | Canción: resolución desde la UI")
