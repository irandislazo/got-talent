# ═══════════════════════════════════════════════════════════════
# CELDA 6 (v7): QWEN3-TTS AISLADO EN SUBENTORNO
#
# OBJETIVO CRÍTICO:
#   Qwen3-TTS NO instala ni modifica dependencias del entorno principal.
#
# El proyecto utiliza varios backends con requisitos incompatibles
# (Wan2GP/LTX, Whisper, Qwen, Gradio, etc.). Por eso Qwen se ejecuta
# en un proceso Python independiente con un virtualenv propio.
#
# STACK INTERNO QWEN — basado en el notebook AIQUEST:
#   qwen-tts==0.0.5
#   transformers==4.57.3
#   huggingface-hub==1.3.3
#   accelerate==1.12.0
#
# La interfaz pública de qwen3_tts se conserva para no cambiar la
# lógica del pipeline existente.
# ═══════════════════════════════════════════════════════════════
import os
import gc
import sys
import time
import shutil
import subprocess
import tempfile
import json
import threading
from typing import Optional, Dict, Tuple

import torch
import soundfile as sf


# ---------------------------------------------------------------
# CACHE EXCLUSIVA DEL WORKER QWEN
# ---------------------------------------------------------------
TTS_CACHE_DIR = "/tmp/hf_cache_tts"
os.makedirs(TTS_CACHE_DIR, exist_ok=True)

st_tmp = os.statvfs("/tmp")
_tmp_free_gb = (st_tmp.f_bavail * st_tmp.f_frsize) / 1024**3
log(f"[TTS] Cache Qwen aislada → {TTS_CACHE_DIR}")
log(f"[TTS] /tmp libre: {_tmp_free_gb:.1f} GB")


# ---------------------------------------------------------------
# PYAV / SOUNDFILE — SOLO para el entorno principal de Whisper.
# No cambiamos versiones globales de Hugging Face.
# ---------------------------------------------------------------
def _patch_audio_readers() -> None:
    try:
        import soundfile as _sf
        if not getattr(_sf.read, "_patched_metadata_errors", False):
            _orig_sf_read = _sf.read

            def _patched_sf_read(file, *args, **kwargs):
                kwargs.pop("metadata_errors", None)
                kwargs.pop("metadata_encoding", None)
                return _orig_sf_read(file, *args, **kwargs)

            _patched_sf_read._patched_metadata_errors = True
            _sf.read = _patched_sf_read
            log("[TTS] ✅ soundfile.read protegido contra metadata_errors.")
    except Exception as e:
        log(f"[TTS] ⚠ Patch soundfile omitido: {e}")

    try:
        import av as _av
        if not getattr(_av.open, "_patched_metadata_errors", False):
            _orig_av_open = _av.open

            def _patched_av_open(file, mode="r", **kwargs):
                kwargs.pop("metadata_errors", None)
                kwargs.pop("metadata_encoding", None)
                return _orig_av_open(file, mode=mode, **kwargs)

            _patched_av_open._patched_metadata_errors = True
            _av.open = _patched_av_open
            log(f"[TTS] ✅ av.open protegido | PyAV={getattr(_av, '__version__', '?')}")
    except Exception as e:
        log(f"[TTS] ⚠ Patch PyAV omitido: {e}")


_patch_audio_readers()


# ---------------------------------------------------------------
# WORKER AISLADO DE QWEN3-TTS
# ---------------------------------------------------------------
# Qwen ejecuta su backend Python en un virtualenv separado para no tocar
# las dependencias del resto del notebook.
#
# Stack interno basado en el notebook AIQUEST:
#   qwen-tts==0.0.5
#   transformers==4.57.3
#   huggingface-hub==0.36.0
#   accelerate==1.12.0
#
# Transformers 4.57.3 requiere huggingface-hub <1.0, por lo que 0.36.0
# es la versión correcta para este entorno aislado.
# ---------------------------------------------------------------

QWEN_TTS_VENV = "/tmp/qwen3_tts_env"
QWEN_TTS_WORKER = "/tmp/qwen3_tts_worker.py"
QWEN_TTS_PYTHON = os.path.join(QWEN_TTS_VENV, "bin", "python")
QWEN_TTS_READY_FILE = os.path.join(QWEN_TTS_VENV, ".qwen_ready.json")
QWEN_TTS_LOG_FILE = os.path.join(TTS_CACHE_DIR, "qwen_worker.log")
CONTESTANT_VOICE_STATE_FILE = os.path.join(OUTPUT_DIR, "contestant_voice_state.json")

QWEN_TTS_WORKER_SOURCE = r'''# -*- coding: utf-8 -*-
import gc
import json
import os
import sys
import time
import traceback

# --- Aislar stdout: SOLO el protocolo JSON puede salir por el stdout real.
# Cualquier print() de librerias (qwen_tts, sox, transformers...) se desvia a stderr.
_REAL_STDOUT = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr

import torch
import soundfile as sf

TTS_CACHE_DIR = "/tmp/hf_cache_tts"
os.makedirs(TTS_CACHE_DIR, exist_ok=True)

MODEL_IDS = {
    "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "custom": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}

current_model = None
current_model_type = None


def out_json(data):
    _REAL_STDOUT.write(json.dumps(data, ensure_ascii=False) + "\n")
    _REAL_STDOUT.flush()


def err_log(message):
    sys.stderr.write(str(message).rstrip() + "\n")
    sys.stderr.flush()


def unload_model():
    global current_model, current_model_type
    if current_model is not None:
        try:
            del current_model
        except Exception:
            pass
    current_model = None
    current_model_type = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_model(model_type):
    global current_model, current_model_type

    if model_type not in MODEL_IDS:
        raise ValueError(f"Tipo de modelo desconocido: {model_type}")

    if current_model is not None and current_model_type == model_type:
        return current_model

    unload_model()

    from huggingface_hub import snapshot_download
    from qwen_tts import Qwen3TTSModel

    model_name = MODEL_IDS[model_type]
    err_log(f"[QWEN-WORKER] Cargando '{model_type}': {model_name}")

    local_path = snapshot_download(
        repo_id=model_name,
        cache_dir=TTS_CACHE_DIR,
        local_files_only=False,
    )

    speech_tokenizer_dir = os.path.join(local_path, "speech_tokenizer")
    os.makedirs(speech_tokenizer_dir, exist_ok=True)

    for cfg_name in ("preprocessor_config.json", "config.json"):
        src = os.path.join(local_path, cfg_name)
        dst = os.path.join(speech_tokenizer_dir, cfg_name)
        if os.path.exists(src) and not os.path.lexists(dst):
            try:
                os.symlink(src, dst)
            except Exception:
                pass

    if torch.cuda.is_available():
        device = "cuda:0"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    current_model = Qwen3TTSModel.from_pretrained(
        local_path,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation="sdpa",
    )
    current_model_type = model_type

    vram = 0.0
    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated(0) / 1024**3
    err_log(f"[QWEN-WORKER] Modelo '{model_type}' listo | VRAM={vram:.2f} GB")
    return current_model


def save_wav(wavs, sr, out_path):
    if not wavs:
        raise RuntimeError("Qwen no devolvió audio.")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, wavs[0], sr)
    if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
        raise RuntimeError(f"No se pudo crear WAV válido: {out_path}")
    return out_path, float(len(wavs[0]) / sr), int(sr)


def handle(req):
    command = req.get("command", "")

    if command == "ping":
        import importlib.metadata as md
        return {
            "ok": True,
            "command": "ping",
            "qwen_tts": md.version("qwen-tts"),
            "transformers": md.version("transformers"),
            "huggingface_hub": md.version("huggingface-hub"),
            "accelerate": md.version("accelerate"),
            "torch": torch.__version__,
            "cuda": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        }

    if command == "unload":
        unload_model()
        return {"ok": True, "command": "unload"}

    if command == "clone":
        text = str(req.get("text", "")).strip()
        ref_wav = str(req.get("ref_wav", ""))
        ref_text = str(req.get("ref_text", "")).strip()
        out_wav = str(req.get("out_wav", ""))
        fast_mode = bool(req.get("fast_mode", False))

        if not text:
            raise ValueError("Texto vacío.")
        if not ref_wav or not os.path.exists(ref_wav):
            raise FileNotFoundError(f"Referencia inexistente: {ref_wav}")
        if not out_wav:
            raise ValueError("Falta out_wav.")

        model = ensure_model("base")
        t0 = time.time()

        if fast_mode or not ref_text:
            prompt_items = model.create_voice_clone_prompt(
                ref_audio=ref_wav,
                x_vector_only_mode=True,
            )
        else:
            prompt_items = model.create_voice_clone_prompt(
                ref_audio=ref_wav,
                ref_text=ref_text,
                x_vector_only_mode=False,
            )

        with torch.inference_mode():
            wavs, sr = model.generate_voice_clone(
                text=text,
                voice_clone_prompt=prompt_items,
            )

        out_path, duration, sample_rate = save_wav(wavs, sr, out_wav)
        elapsed = time.time() - t0
        return {
            "ok": True,
            "command": "clone",
            "out_wav": out_path,
            "duration": duration,
            "sample_rate": sample_rate,
            "elapsed": elapsed,
        }

    if command == "design":
        text = str(req.get("text", "")).strip()
        instruct = str(req.get("instruct", "")).strip()
        out_wav = str(req.get("out_wav", ""))

        if not text:
            raise ValueError("Texto vacío.")
        if not instruct:
            raise ValueError("Instrucción de VoiceDesign vacía.")
        if not out_wav:
            raise ValueError("Falta out_wav.")

        model = ensure_model("design")
        t0 = time.time()
        with torch.inference_mode():
            wavs, sr = model.generate_voice_design(
                text=text,
                instruct=instruct,
            )

        out_path, duration, sample_rate = save_wav(wavs, sr, out_wav)
        elapsed = time.time() - t0
        return {
            "ok": True,
            "command": "design",
            "out_wav": out_path,
            "duration": duration,
            "sample_rate": sample_rate,
            "elapsed": elapsed,
        }

    if command == "custom":
        text = str(req.get("text", "")).strip()
        speaker = str(req.get("speaker", "serena")).strip().lower()
        instruct = str(req.get("instruct", "")).strip()
        out_wav = str(req.get("out_wav", ""))

        if not text:
            raise ValueError("Texto vacío.")
        if not out_wav:
            raise ValueError("Falta out_wav.")

        model = ensure_model("custom")
        t0 = time.time()
        kwargs = {"text": text, "speaker": speaker}
        if instruct:
            kwargs["instruct"] = instruct

        with torch.inference_mode():
            wavs, sr = model.generate_custom_voice(**kwargs)

        out_path, duration, sample_rate = save_wav(wavs, sr, out_wav)
        elapsed = time.time() - t0
        return {
            "ok": True,
            "command": "custom",
            "out_wav": out_path,
            "duration": duration,
            "sample_rate": sample_rate,
            "elapsed": elapsed,
        }

    raise ValueError(f"Comando desconocido: {command}")


def main():
    err_log(
        "[QWEN-WORKER] Proceso iniciado | "
        f"GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            result = handle(req)
            out_json(result)
        except Exception as exc:
            err_log(traceback.format_exc())
            out_json({"ok": False, "error": str(exc), "traceback": traceback.format_exc()})


if __name__ == "__main__":
    main()
'''

# ---------------------------------------------------------------
# CREACIÓN DEL ENTORNO AISLADO
# ---------------------------------------------------------------
def _run_qwen_venv_command(args, check=True, extra_env=None):
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONPATH", None)
    if extra_env:
        env.update(extra_env)
    r = subprocess.run(args, capture_output=True, text=True, env=env)
    if check and r.returncode != 0:
        tail = ((r.stderr or "") + "\n" + (r.stdout or "")).strip()[-1500:]
        raise RuntimeError(f"rc={r.returncode} en: {' '.join(str(a) for a in args)}\n{tail}")
    return r


def _create_qwen_venv() -> str:
    """Crea el venv aislado probando varias estrategias (Kaggle suele no traer
    python3-venv/ensurepip). NO modifica el entorno principal."""
    py = sys.executable
    errors = []

    def _clean():
        shutil.rmtree(QWEN_TTS_VENV, ignore_errors=True)

    def _venv_has_pip() -> bool:
        if not os.path.exists(QWEN_TTS_PYTHON):
            return False
        return _run_qwen_venv_command([QWEN_TTS_PYTHON, "-m", "pip", "--version"], check=False).returncode == 0

    # 1) venv estándar
    # 2) venv sin ensurepip (usa el pip del sistema visible via system-site-packages)
    for label, cmd in [
        ("venv", [py, "-m", "venv", "--system-site-packages", QWEN_TTS_VENV]),
        ("venv --without-pip", [py, "-m", "venv", "--system-site-packages", "--without-pip", QWEN_TTS_VENV]),
    ]:
        _clean()
        try:
            _run_qwen_venv_command(cmd)
            if _venv_has_pip():
                return label
            errors.append(f"[{label}] creado pero sin pip utilizable")
        except Exception as e:
            errors.append(f"[{label}] {e}")

    # 3) uv (si existe en el sistema)
    uv_bin = shutil.which("uv")
    if uv_bin:
        _clean()
        try:
            _run_qwen_venv_command([uv_bin, "venv", "--system-site-packages", "--seed", "--python", py, QWEN_TTS_VENV])
            if _venv_has_pip():
                return "uv venv"
            errors.append("[uv] creado pero sin pip utilizable")
        except Exception as e:
            errors.append(f"[uv] {e}")

    # 4) virtualenv instalado en carpeta temporal (--target) → NO toca site-packages principal
    boot = "/tmp/qwen_bootstrap_virtualenv"
    _clean()
    try:
        _run_qwen_venv_command([py, "-m", "pip", "install", "-q", "--target", boot, "virtualenv"])
        _run_qwen_venv_command(
            [py, "-m", "virtualenv", "--system-site-packages", QWEN_TTS_VENV],
            extra_env={"PYTHONPATH": boot},
        )
        if _venv_has_pip():
            return "virtualenv"
        errors.append("[virtualenv] creado pero sin pip utilizable")
    except Exception as e:
        errors.append(f"[virtualenv] {e}")

    raise RuntimeError("No se pudo crear el venv de Qwen con ningún método:\n" + "\n".join(errors))


def _qwen_versions_in_venv():
    if not os.path.exists(QWEN_TTS_PYTHON):
        return None
    code = ("import importlib.metadata as m, json; "
            "print(json.dumps({"
            "'qwen_tts':m.version('qwen-tts'),"
            "'transformers':m.version('transformers'),"
            "'huggingface_hub':m.version('huggingface-hub'),"
            "'accelerate':m.version('accelerate')}))")
    try:
        r = _run_qwen_venv_command([QWEN_TTS_PYTHON, "-c", code], check=False)
        if r.returncode != 0:
            return None
        return json.loads((r.stdout or "").strip())
    except Exception:
        return None


def _ensure_qwen_isolated_environment() -> bool:
    expected = {
        "qwen_tts": "0.0.5",
        "transformers": "4.57.3",
        "huggingface_hub": "0.36.0",
        "accelerate": "1.12.0",
    }
    current = _qwen_versions_in_venv()
    if current != expected:
        log("[TTS] Preparando entorno Python AISLADO de Qwen3-TTS...")
        try:
            if os.path.isdir(QWEN_TTS_VENV):
                shutil.rmtree(QWEN_TTS_VENV, ignore_errors=True)
            _how = _create_qwen_venv()
            log(f"[TTS] Venv aislado creado con: {_how}")
            _run_qwen_venv_command([
                QWEN_TTS_PYTHON, "-m", "pip", "install", "-q", "--upgrade", "pip"
            ])
            _run_qwen_venv_command([
                QWEN_TTS_PYTHON, "-m", "pip", "install", "-q", "--no-cache-dir",
                "transformers==4.57.3", "huggingface-hub==0.36.0", "accelerate==1.12.0",
                "gradio==6.4.0", "librosa", "soundfile", "sox", "onnxruntime", "einops"
            ])
            _run_qwen_venv_command([
                QWEN_TTS_PYTHON, "-m", "pip", "install", "-q", "--no-cache-dir", "--no-deps", "qwen-tts==0.0.5"
            ])
            with open(QWEN_TTS_WORKER, "w", encoding="utf-8") as f:
                f.write(QWEN_TTS_WORKER_SOURCE)
            verified = _qwen_versions_in_venv()
            if verified != expected:
                raise RuntimeError(f"Versiones inesperadas: {verified}")
            with open(QWEN_TTS_READY_FILE, "w", encoding="utf-8") as f:
                json.dump(expected, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log(f"[TTS] ❌ No se pudo preparar el entorno aislado: {e}")
            return False
    try:
        with open(QWEN_TTS_WORKER, "w", encoding="utf-8") as f:
            f.write(QWEN_TTS_WORKER_SOURCE)
    except Exception as e:
        log(f"[TTS] ❌ No se pudo escribir worker Qwen: {e}")
        return False
    return True


class _QwenWorkerClient:
    def __init__(self):
        self.process = None
        self.lock = threading.Lock()
        self.stderr_thread = None

    def _drain_stderr(self):
        if self.process is None or self.process.stderr is None:
            return
        try:
            for line in self.process.stderr:
                line = line.rstrip()
                if line:
                    log(line)
        except Exception:
            pass

    def start(self):
        if self.process is not None and self.process.poll() is None:
            return
        if not _ensure_qwen_isolated_environment():
            raise RuntimeError("No se pudo preparar el entorno aislado de Qwen3-TTS.")
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env.pop("PYTHONPATH", None)
        env["HF_HOME"] = TTS_CACHE_DIR
        env["HUGGINGFACE_HUB_CACHE"] = TTS_CACHE_DIR
        env["HF_DATASETS_CACHE"] = TTS_CACHE_DIR
        env["PYTHONUNBUFFERED"] = "1"
        log("[TTS] Iniciando worker Qwen aislado...")
        self.process = subprocess.Popen(
            [QWEN_TTS_PYTHON, "-u", QWEN_TTS_WORKER],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=env
        )
        self.stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True, name="qwen-tts-worker-stderr")
        self.stderr_thread.start()

    def request(self, payload: dict) -> dict:
        with self.lock:
            self.start()
            if self.process is None or self.process.stdin is None or self.process.stdout is None:
                raise RuntimeError("Worker Qwen no está operativo.")
            if self.process.poll() is not None:
                code = self.process.returncode
                self.process = None
                raise RuntimeError(f"Worker Qwen terminó antes de responder (rc={code}).")
            self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            data = None
            while data is None:
                line = self.process.stdout.readline()
                if not line:
                    code = self.process.poll()
                    self.process = None
                    raise RuntimeError(f"Worker Qwen cerró stdout sin respuesta (rc={code}).")
                line = line.strip()
                if not line:
                    continue                      # línea en blanco → ruido, ignorar
                if not line.startswith("{"):
                    log(f"[QWEN-WORKER stdout] {line[:300]}")
                    continue                      # print() ajeno → ignorar
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    log(f"[QWEN-WORKER stdout no-JSON] {line[:300]}")
            if not data.get("ok", False):
                raise RuntimeError(data.get("error", "Error desconocido del worker Qwen."))
            return data

    def close(self):
        with self.lock:
            proc = self.process
            self.process = None
            if proc is None:
                return
            try:
                if proc.stdin:
                    proc.stdin.write(json.dumps({"command": "unload"}) + "\n")
                    proc.stdin.flush()
            except Exception:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


class Qwen3TTSEngine:
    MODEL_IDS = {
        "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "custom": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    }

    def __init__(self):
        self._worker = _QwenWorkerClient()
        self._available = None

    @property
    def available(self):
        if self._available is not None:
            return self._available
        try:
            r = self._worker.request({"command": "ping"})
            log(
                "[TTS] ✅ Worker Qwen disponible | "
                f"qwen-tts={r['qwen_tts']} | transformers={r['transformers']} | "
                f"huggingface-hub={r['huggingface_hub']} | accelerate={r['accelerate']} | GPU={r['gpu']}"
            )
            self._available = True
        except Exception as e:
            log(f"[TTS] ❌ Worker Qwen no disponible: {e}")
            self._available = False
        return self._available

    def _request(self, payload):
        if not self.available:
            raise RuntimeError("qwen_tts no disponible.")
        return self._worker.request(payload)

    def synth_clone(self, text, out_wav, ref_wav, ref_text="", fast_mode=False, seed=-1):
        r = self._request({"command":"clone","text":text,"out_wav":out_wav,"ref_wav":ref_wav,"ref_text":ref_text,"fast_mode":bool(fast_mode),"seed":int(seed)})
        log(f"[TTS Clone] {r['duration']:.1f}s audio | {r['elapsed']:.1f}s generación | RTF={r['elapsed']/max(r['duration'],0.1):.2f}x")
        return out_wav

    def synth_voice_design(self, text, out_wav, instruct, seed=-1):
        r = self._request({"command":"design","text":text,"out_wav":out_wav,"instruct":instruct,"seed":int(seed)})
        log(f"[TTS Design] {r['duration']:.1f}s audio | {r['elapsed']:.1f}s generación")
        return out_wav

    def synth_custom_voice(self, text, out_wav, speaker="Serena", instruction="", seed=-1):
        r = self._request({"command":"custom","text":text,"out_wav":out_wav,"speaker":speaker,"instruct":instruction,"seed":int(seed)})
        log(f"[TTS Custom] speaker={speaker} | {r['duration']:.1f}s audio | {r['elapsed']:.1f}s generación")
        return out_wav

    def unload(self):
        try:
            if self._available:
                self._worker.request({"command":"unload"})
        except Exception:
            pass
        self._worker.close()
        self._available = None

    def report_cache_usage(self):
        total=0
        if os.path.isdir(TTS_CACHE_DIR):
            for root,_,files in os.walk(TTS_CACHE_DIR):
                for fname in files:
                    try: total += os.path.getsize(os.path.join(root,fname))
                    except Exception: pass
        log(f"[TTS Cache] {TTS_CACHE_DIR}: {total/1024**3:.2f} GB usados en /tmp")


def load_contestant_voice_state():
    if os.path.exists(CONTESTANT_VOICE_STATE_FILE):
        try:
            with open(CONTESTANT_VOICE_STATE_FILE,"r",encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"mode":None,"ref_wav_path":"","ref_text":"","design_instruct":"","preview_wav":"","validated":False}


def save_contestant_voice_state(state):
    os.makedirs(os.path.dirname(CONTESTANT_VOICE_STATE_FILE),exist_ok=True)
    with open(CONTESTANT_VOICE_STATE_FILE,"w",encoding="utf-8") as f:
        json.dump(state,f,ensure_ascii=False,indent=2)


_patch_audio_readers()
qwen3_tts = Qwen3TTSEngine()
_QWEN_TTS_PY_AVAILABLE = bool(qwen3_tts.available)
log(f"[TTS] Engine disponible: {_QWEN_TTS_PY_AVAILABLE}")
if _QWEN_TTS_PY_AVAILABLE:
    qwen3_tts.report_cache_usage()

CONTESTANT_VOICE_STATE = load_contestant_voice_state()


# ═══════════════════════════════════════════════════════════════
# SECCIÓN 6.4 — DIRECTORIOS DE REFERENCIA
# AUDIO de referencia → /kaggle/working/voice_refs (pequeños WAVs, persistentes)
# MODELOS HF         → /tmp/hf_cache_tts           (grandes, no persistentes)
# ═══════════════════════════════════════════════════════════════
VOICE_REFS_DIR         = os.path.join(OUTPUT_DIR, "voice_refs")
JUECES_REF_DIR         = os.path.join(VOICE_REFS_DIR, "jueces")
CONTESTANT_REF_WAV     = os.path.join(VOICE_REFS_DIR, "contestant_ref.wav")
CONTESTANT_REF_TXT     = os.path.join(VOICE_REFS_DIR, "contestant_ref.txt")
CONTESTANT_PREVIEW_WAV = os.path.join(VOICE_REFS_DIR, "contestant_preview.wav")
DESIGN_REF_WAV         = os.path.join(VOICE_REFS_DIR, "design_as_ref.wav")
DESIGN_REF_TXT         = os.path.join(VOICE_REFS_DIR, "design_as_ref.txt")

for _d in [VOICE_REFS_DIR, JUECES_REF_DIR]:
    os.makedirs(_d, exist_ok=True)
log(f"[TTS] WAVs de referencia → {VOICE_REFS_DIR} (/working, persistentes, <50MB)")


def get_juez_ref_wav(plantilla_slug: str, juez_id: int) -> str:
    if not plantilla_slug:
        return ""
    try:
        tpl = cargar_plantilla(plantilla_slug)
        voz_dir = os.path.join(tpl["_base_dir"], PLANTILLA_SUBDIRS["voces_jueces"])
        wav_path = os.path.join(voz_dir, f"juez_{juez_id}.wav")
        return wav_path if os.path.exists(wav_path) else ""
    except Exception:
        return ""

def get_juez_ref_text(plantilla_slug: str, juez_id: int) -> str:
    txt_path = os.path.join(JUECES_REF_DIR, f"juez_{juez_id}_ref.txt")
    if os.path.exists(txt_path):
        try: return open(txt_path, "r", encoding="utf-8").read().strip()
        except Exception: pass
    return ""

# ---------------------------------------------------------------
# TRANSCRIPTOR (faster-whisper, entorno principal)
# Contrato usado por el resto del pipeline:
#   transcribe_audio_file(wav, out_txt_path=None, language=None) -> (texto, idioma)
# Carga el modelo bajo demanda y lo libera al terminar para no robar VRAM
# a LTX / Qwen. Nunca lanza excepción: si falla devuelve ("", "").
# ---------------------------------------------------------------
WHISPER_MODEL_SIZE = "small"

def transcribe_audio_file(wav_path: str, out_txt_path: Optional[str] = None,
                          language: Optional[str] = None) -> Tuple[str, str]:
    if not wav_path or not os.path.exists(wav_path):
        log(f"[Transcriber] ❌ Audio inexistente: {wav_path}")
        return "", ""

    tmp16 = None
    model = None
    try:
        from faster_whisper import WhisperModel

        # Decodificar a 16 kHz mono con ffmpeg → evita depender de PyAV.
        tmp16 = tempfile.mktemp(suffix="_whisper16k.wav", dir="/tmp")
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-ac", "1", "-ar", "16000",
             "-acodec", "pcm_s16le", tmp16, "-loglevel", "error"],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg: {(r.stderr or '')[-500:]}")
        audio, _sr = sf.read(tmp16, dtype="float32")

        try:
            if torch.cuda.is_available():
                model = WhisperModel(WHISPER_MODEL_SIZE, device="cuda", compute_type="float16")
            else:
                raise RuntimeError("sin CUDA")
        except Exception as e_gpu:
            log(f"[Transcriber] Whisper en CPU ({e_gpu})")
            model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

        segments, info = model.transcribe(
            audio, language=language, beam_size=5, vad_filter=True)
        texto = " ".join(seg.text.strip() for seg in segments).strip()
        idioma = (getattr(info, "language", None) or language or "").strip()

        if out_txt_path and texto:
            os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)
            with open(out_txt_path, "w", encoding="utf-8") as f:
                f.write(texto)
        log(f"[Transcriber] ✅ idioma={idioma or '?'} | {len(texto)} chars")
        return texto, idioma
    except Exception as e:
        log(f"[Transcriber] ❌ Falló la transcripción: {e}")
        return "", ""
    finally:
        try:
            del model
        except Exception:
            pass
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        if tmp16 and os.path.exists(tmp16):
            try: os.remove(tmp16)
            except Exception: pass


def auto_transcribe_juez(plantilla_slug: str, juez_id: int) -> str:
    wav = get_juez_ref_wav(plantilla_slug, juez_id)
    if not wav:
        return ""
    txt_path = os.path.join(JUECES_REF_DIR, f"juez_{juez_id}_ref.txt")
    existing = get_juez_ref_text(plantilla_slug, juez_id)
    if existing:
        log(f"[Transcriber] Juez {juez_id}: transcripción existente reutilizada.")
        return existing
    texto, _ = transcribe_audio_file(wav, out_txt_path=txt_path)
    log(f"[Transcriber] Juez {juez_id}: '{texto[:80]}...'")
    return texto

def auto_transcribe_all_judges(plantilla_slug: str) -> Dict[str, str]:
    result = {}
    for i in range(1, 5):
        wav = get_juez_ref_wav(plantilla_slug, i)
        if wav:
            txt = auto_transcribe_juez(plantilla_slug, i)
            result[f"juez_{i}"] = txt
    return result


# ═══════════════════════════════════════════════════════════════
# SECCIÓN 6.5 — VALIDACIÓN DE VOZ DEL CONCURSANTE
# ═══════════════════════════════════════════════════════════════
PREVIEW_TEXT_ES = (
    "Hola, muchas gracias por estar aquí. "
    "Llevo años soñando con este momento y hoy estoy lista para dar todo."
)

def process_contestant_upload(audio_path: str) -> Tuple[str, str, str]:
    global CONTESTANT_VOICE_STATE
    if not audio_path or not os.path.exists(audio_path):
        return "", "", "❌ No se recibió audio válido."
    try:
        ffmpeg_to_wav_mono_24k(audio_path, CONTESTANT_REF_WAV)
        dur = _ffprobe_duration(CONTESTANT_REF_WAV) or 0
        if dur < 2.0:
            return "", "", f"⚠ Audio muy corto ({dur:.1f}s). Mínimo 3 segundos."

        log("[Contestant] Auto-transcribiendo audio subido...")
        texto, idioma = transcribe_audio_file(
            CONTESTANT_REF_WAV, out_txt_path=CONTESTANT_REF_TXT)

        # FIX: Si la transcripción falla, usar placeholder para NO bloquear el preview
        if not texto.strip():
            texto = PREVIEW_TEXT_ES
            idioma = idioma or "es"
            log("[Contestant] ⚠ Transcripción vacía, usando texto placeholder para preview.")

        CONTESTANT_VOICE_STATE.update({
            "mode": "clone_upload",
            "ref_wav_path": CONTESTANT_REF_WAV,
            "ref_text": texto,
            "validated": False,
            "preview_wav": "",
        })
        save_contestant_voice_state(CONTESTANT_VOICE_STATE)

        msg = (f"✅ Audio cargado ({dur:.1f}s) | Idioma: {idioma}\n"
               f"📝 «{texto[:200]}{'...' if len(texto)>200 else ''}»\n"
               f"→ Genera un preview para validar.")
        return texto, idioma, msg
    except Exception as e:
        log(f"[Contestant] Error: {e}")
        return "", "", f"❌ Error: {e}"

def generate_contestant_clone_preview(
    ref_wav: str = "", ref_text: str = "",
    preview_text: str = PREVIEW_TEXT_ES) -> Tuple[str, str]:
    global CONTESTANT_VOICE_STATE
    wav = ref_wav or CONTESTANT_REF_WAV
    txt = ref_text or CONTESTANT_VOICE_STATE.get("ref_text", "")
    if not wav or not os.path.exists(wav):
        return "", "❌ Primero sube el audio del concursante."
    if not preview_text or not preview_text.strip():
        preview_text = PREVIEW_TEXT_ES
    try:
        log("[Contestant Preview] Generando preview por clonación...")
        qwen3_tts.synth_clone(
            text=preview_text.strip(), out_wav=CONTESTANT_PREVIEW_WAV,
            ref_wav=wav, ref_text=txt,
            fast_mode=(not bool(txt.strip())))
        dur = _ffprobe_duration(CONTESTANT_PREVIEW_WAV) or 0
        CONTESTANT_VOICE_STATE["preview_wav"] = CONTESTANT_PREVIEW_WAV
        save_contestant_voice_state(CONTESTANT_VOICE_STATE)
        return CONTESTANT_PREVIEW_WAV, f"✅ Preview generado ({dur:.1f}s). Escucha y valida."
    except Exception as e:
        return "", f"❌ Error generando preview: {e}"

def generate_voice_design_preview(
    design_instruct: str,
    preview_text: str = PREVIEW_TEXT_ES) -> Tuple[str, str]:
    global CONTESTANT_VOICE_STATE
    if not design_instruct or not design_instruct.strip():
        return "", "❌ Escribe una descripción de voz."
    if not preview_text or not preview_text.strip():
        preview_text = PREVIEW_TEXT_ES
    preview_path = os.path.join(VOICE_REFS_DIR, "design_preview.wav")
    try:
        log("[VoiceDesign Preview] Generando preview...")
        qwen3_tts.synth_voice_design(
            text=preview_text.strip(), out_wav=preview_path,
            instruct=design_instruct.strip())
        dur = _ffprobe_duration(preview_path) or 0
        CONTESTANT_VOICE_STATE["preview_wav"] = preview_path
        CONTESTANT_VOICE_STATE["design_instruct"] = design_instruct.strip()
        save_contestant_voice_state(CONTESTANT_VOICE_STATE)
        return preview_path, (
            f"✅ Preview generado ({dur:.1f}s).\n"
            f"Descripción: «{design_instruct[:100]}»\n"
            f"Si te gusta, pulsa '✅ Usar esta voz como referencia'.")
    except Exception as e:
        return "", f"❌ Error: {e}"

def adopt_design_as_reference(design_instruct: str = "") -> Tuple[str, str]:
    global CONTESTANT_VOICE_STATE
    preview_src = CONTESTANT_VOICE_STATE.get("preview_wav", "")
    if not preview_src or not os.path.exists(preview_src):
        return "", "❌ Primero genera un preview de diseño de voz."
    try:
        shutil.copyfile(preview_src, DESIGN_REF_WAV)
        with open(DESIGN_REF_TXT, "w", encoding="utf-8") as f:
            f.write(PREVIEW_TEXT_ES)
        CONTESTANT_VOICE_STATE.update({
            "mode": "clone_design",
            "ref_wav_path": DESIGN_REF_WAV,
            "ref_text": PREVIEW_TEXT_ES,
            "design_instruct": design_instruct or CONTESTANT_VOICE_STATE.get("design_instruct", ""),
            "validated": True,
            "preview_wav": preview_src,
        })
        save_contestant_voice_state(CONTESTANT_VOICE_STATE)
        dur = _ffprobe_duration(DESIGN_REF_WAV) or 0
        msg = (f"✅ Voz de diseño adoptada ({dur:.1f}s).\n"
               f"Todo el guion usará clonación de esta voz.\n"
               f"Descripción: «{CONTESTANT_VOICE_STATE['design_instruct'][:100]}»")
        return DESIGN_REF_WAV, msg
    except Exception as e:
        return "", f"❌ Error: {e}"

def validate_contestant_voice() -> Tuple[bool, str]:
    global CONTESTANT_VOICE_STATE
    ref = CONTESTANT_VOICE_STATE.get("ref_wav_path", "")
    if not ref or not os.path.exists(ref):
        return False, "❌ No hay referencia de voz. Sube audio o diseña una voz primero."
    CONTESTANT_VOICE_STATE["validated"] = True
    save_contestant_voice_state(CONTESTANT_VOICE_STATE)
    txt = CONTESTANT_VOICE_STATE.get("ref_text", "")
    dur = _ffprobe_duration(ref) or 0
    msg = (f"✅ Voz VALIDADA | Modo: {CONTESTANT_VOICE_STATE.get('mode','?')}\n"
           f"Ref: {os.path.basename(ref)} ({dur:.1f}s)\n"
           f"«{txt[:120]}{'...' if len(txt)>120 else ''}»\n"
           f"→ Ya puedes ejecutar Etapas 2 y 4.")
    return True, msg

def get_contestant_ref_for_tts() -> Dict[str, str]:
    state = CONTESTANT_VOICE_STATE
    if not state.get("validated"):
        raise RuntimeError(
            "Voz del concursante no validada. "
            "Ve a '🎙️ VOZ DEL CONCURSANTE' y valídala primero.")
    ref_wav = state.get("ref_wav_path", "")
    ref_text = state.get("ref_text", "")
    if not ref_wav or not os.path.exists(ref_wav):
        raise RuntimeError("El WAV de referencia no existe.")
    return {"ref_wav": ref_wav, "ref_text": ref_text}


# ═══════════════════════════════════════════════════════════════
# SECCIÓN 6.6 — SÍNTESIS UNIFICADA DE FRAGMENTO
# ═══════════════════════════════════════════════════════════════
def synth_fragment_for_dialog(texto: str, hablante: str, out_wav: str,
                               plantilla_slug: str, fast_mode: bool = False) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_wav)), exist_ok=True)

    if qwen3_tts.available:
        try:
            if hablante == "concursante":
                ref_info = get_contestant_ref_for_tts()
                return qwen3_tts.synth_clone(
                    text=texto, out_wav=out_wav,
                    ref_wav=ref_info["ref_wav"], ref_text=ref_info["ref_text"],
                    fast_mode=fast_mode or not bool(ref_info["ref_text"].strip()))
            elif hablante.startswith("juez_"):
                juez_id  = int(hablante.split("_")[1])
                ref_wav  = get_juez_ref_wav(plantilla_slug, juez_id)
                ref_text = get_juez_ref_text(plantilla_slug, juez_id)
                if not ref_wav or not os.path.exists(ref_wav):
                    log(f"[TTS] ⚠ Sin audio ref para {hablante}. VoiceDesign fallback.")
                    instruct_fb = (
                        "adult male, warm, calm voice, Spanish speaking"
                        if juez_id % 2 == 1 else
                        "adult female, warm, professional voice, Spanish speaking")
                    return qwen3_tts.synth_voice_design(
                        text=texto, out_wav=out_wav, instruct=instruct_fb)
                return qwen3_tts.synth_clone(
                    text=texto, out_wav=out_wav, ref_wav=ref_wav,
                    ref_text=ref_text,
                    fast_mode=fast_mode or not bool(ref_text.strip()))
            else:
                raise ValueError(f"Hablante desconocido: {hablante}")
        except Exception as e:
            log(f"[TTS] Error qwen3_tts '{hablante}': {e}. Fallback edge-tts.")

    # ── Fallback: edge-tts ────────────────────────────────────
    import asyncio
    import edge_tts as _edge_tts

    voice_map = {
        "concursante": "es-ES-AbrilNeural",
        "juez_1": "es-ES-AlvaroNeural",
        "juez_2": "es-ES-ElviraNeural",
        "juez_3": "es-MX-JorgeNeural",
        "juez_4": "es-MX-DaliaNeural",
    }
    voice = voice_map.get(hablante, "es-ES-AbrilNeural")

    async def _edge_synth():
        communicate = _edge_tts.Communicate(texto, voice)
        await communicate.save(out_wav)

    asyncio.run(_edge_synth())
    log(f"[TTS Fallback edge] {hablante} → {os.path.basename(out_wav)}")
    return out_wav


# ═══════════════════════════════════════════════════════════════
# SECCIÓN 6.7 — FFMPEG HELPERS
# ═══════════════════════════════════════════════════════════════
def _ffprobe_duration(path: str) -> Optional[float]:
    if not path or not os.path.exists(path):
        return None
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", path],
            capture_output=True, text=True)
        data = json.loads(r.stdout or "{}")
        d = float(data.get("format", {}).get("duration", 0))
        return d if d > 0 else None
    except Exception:
        return None

def ffmpeg_to_wav_mono_24k(src: str, dst: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    cmd = (f"ffmpeg -y -i '{src}' -ac 1 -ar 24000 "
           f"-acodec pcm_s16le '{dst}' -loglevel error")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg falló:\n{(r.stderr or '')[-2000:]}")
    return dst

def fit_audio_exact_duration(in_wav: str, out_wav: str, target_s: float) -> str:
    d = _ffprobe_duration(in_wav) or 0.0
    os.makedirs(os.path.dirname(os.path.abspath(out_wav)), exist_ok=True)
    if d <= 0:
        raise RuntimeError(f"Duración WAV inválida: {in_wav}")
    if abs(d - target_s) < 0.02:
        cmd = (f"ffmpeg -y -i '{in_wav}' "
               f"-af 'atrim=0:{target_s:.6f},asetpts=N/SR/TB' "
               f"-acodec pcm_s16le -ar 24000 '{out_wav}' -loglevel error")
        subprocess.run(cmd, shell=True, capture_output=True)
        return out_wav
    if d > target_s:
        ratio  = d / target_s
        parts  = []
        r      = ratio
        while r > 2.0:
            parts.append("atempo=2.0")
            r /= 2.0
        parts.append(f"atempo={r:.6f}")
        chain = ",".join(parts)
        cmd = (f"ffmpeg -y -i '{in_wav}' "
               f"-af '{chain},atrim=0:{target_s:.6f},asetpts=N/SR/TB' "
               f"-acodec pcm_s16le -ar 24000 '{out_wav}' -loglevel error")
    else:
        cmd = (f"ffmpeg -y -i '{in_wav}' "
               f"-af 'apad=pad_dur=10,atrim=0:{target_s:.6f},asetpts=N/SR/TB' "
               f"-acodec pcm_s16le -ar 24000 '{out_wav}' -loglevel error")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg falló:\n{(r.stderr or '')[-2000:]}")
    return out_wav


# Alias legacy
qwen_tts = qwen3_tts

"""
# NOTA HISTÓRICA: este bloque quedó deshabilitado deliberadamente.
# NO volver a forzar Transformers 4.x aquí: rompería el backend
# Qwen3-TTS aislado utilizado por la Celda 6.
"""



log(f"\n✅ CELDA 6 (v7) COMPLETADA")
log(f"   Package: {'Qwen3-TTS aislado | qwen-tts 0.0.5 + transformers 4.57.3 + huggingface-hub 0.36.0' if _QWEN_TTS_PY_AVAILABLE else 'NO disponible → edge-tts fallback'}")
log(f"   TTS Engine disponible: {qwen3_tts.available}")
log(f"   Modelos cache: {TTS_CACHE_DIR} (en /tmp, ~15GB max, no toca /working)")
log(f"   WAVs referencia: {VOICE_REFS_DIR} (en /working, persistentes, <50MB)")
log(f"   Transcriptor: faster-whisper")
log(f"   Voz concursante validada: {CONTESTANT_VOICE_STATE.get('validated', False)}")