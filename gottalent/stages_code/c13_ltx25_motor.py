# ═══════════════════════════════════════════════════════════════
# CELDA 13 (v4): MOTOR LTX-2.5 CON PROGRESO EN TIEMPO REAL
# CAMBIOS vs v3:
#   - Usa subprocess.Popen en vez de run() → lee stdout línea a línea
#   - Parsea líneas PROGRESS: del script → actualiza archivo de progreso
#   - Expone get_ltx_progress() para que la UI la lea vía timer
#   - Mantiene 100% de compatibilidad con Celdas 14-18
# ═══════════════════════════════════════════════════════════════
import subprocess, sys, os, json, random, time, threading

WAN2GP_DIR   = "/kaggle/working/Wan2GP"
LTX25_SCRIPT = os.path.join(WAN2GP_DIR, "run_ltx25_generate.py")
TMP_MODELS   = "/tmp/models"
GEMMA4_TMP   = os.path.join(TMP_MODELS, "gemma4-12b-ltx-v1")

# ── Archivo de progreso (compartido entre subprocess y UI timer) ──
LTX_PROGRESS_FILE = "/tmp/ltx25_progress.json"

def _write_progress(data: dict):
    try:
        import json as _j
        with open(LTX_PROGRESS_FILE, "w") as f:
            _j.dump(data, f)
    except: pass

def get_ltx_progress() -> dict:
    """Llamada por el timer de Gradio para actualizar la UI."""
    try:
        import json as _j
        if os.path.exists(LTX_PROGRESS_FILE):
            return _j.load(open(LTX_PROGRESS_FILE))
    except: pass
    return {}

def calc_num_frames(target_s, fps=24):
    raw = max(9, int(round(target_s * fps)))
    n   = max(1, (raw - 1) // 8)
    return max(9, min([8*n+1, 8*(n+1)+1], key=lambda x: abs(x - raw)))



# ── FIX LIPSYNC (v7): compatibilidad de huggingface_hub SOLO para el subproceso LTX ──
# Síntoma en el log:  ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'
# Causa: la Celda 17 deja huggingface_hub<1.0 (lo necesita gradio) pero el transformers
#        instalado exige huggingface_hub>=1.0 → el subproceso LTX moría en 7 s y el
#        pipeline caía EN SILENCIO al clip estático.
# Solución: NO se toca el kernel. Se instala huggingface_hub>=1.0 en una carpeta aparte
#           que solo ve el subproceso LTX (vía PYTHONPATH).
LTX_HF_TARGET = "/kaggle/tmp/ltx_hf_compat"
_LTX_ENV = {"ok": False, "pythonpath": ""}

def _ltx_import_check(pythonpath: str = ""):
    env = os.environ.copy()
    if pythonpath:
        env["PYTHONPATH"] = pythonpath + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run(
        [sys.executable, "-c", "import transformers; from mmgp import offload, quant_router"],
        capture_output=True, text=True, env=env, timeout=900)
    return r.returncode == 0, ((r.stderr or "") + (r.stdout or ""))[-2500:]

# v8: rango de huggingface_hub que exige el transformers instalado (5.x → >=1.5,<2.0).
#     Si el kernel tiene hub 0.x (is_offline_mode) o 2.x (version check) el subproceso LTX muere;
#     la solución es darle SOLO al subproceso un hub 1.x compatible, sin tocar kernel ni Qwen.
LTX_HF_SPEC = "huggingface_hub>=1.5.0,<2.0.0"

def _ltx_err_is_hf(err: str) -> bool:
    e = (err or "").lower().replace("-", "_")
    return ("huggingface_hub" in e) or ("is_offline_mode" in e)

def ensure_ltx_env(force: bool = False) -> str:
    """Verifica que el subproceso LTX pueda importar transformers+mmgp.
    Devuelve el PYTHONPATH extra a usar ('' si no hace falta). Lanza RuntimeError si no hay arreglo."""
    import shutil as _sh
    if _LTX_ENV["ok"] and not force:
        return _LTX_ENV["pythonpath"]
    # Si ya existe un target válido de una ejecución previa, probarlo primero (evita reinstalar).
    if os.path.isdir(LTX_HF_TARGET) and os.listdir(LTX_HF_TARGET):
        ok, err = _ltx_import_check(LTX_HF_TARGET)
        if ok:
            _LTX_ENV.update(ok=True, pythonpath=LTX_HF_TARGET)
            return LTX_HF_TARGET
    ok, err = _ltx_import_check("")
    if ok:
        _LTX_ENV.update(ok=True, pythonpath="")
        return ""
    if _ltx_err_is_hf(err):
        log(f"[LTX25] ⚠ Conflicto huggingface_hub/transformers → instalando '{LTX_HF_SPEC}' "
            f"aislado en {LTX_HF_TARGET} (solo para el subproceso LTX)...")
        # Intento 1: solo huggingface_hub (--no-deps) → no pisa ninguna otra librería del subproceso.
        # Intento 2: con dependencias, por si falta algún módulo auxiliar de hub 1.x.
        for extra in (["--no-deps"], []):
            _sh.rmtree(LTX_HF_TARGET, ignore_errors=True)
            os.makedirs(LTX_HF_TARGET, exist_ok=True)
            r = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--no-cache-dir",
                 "--target", LTX_HF_TARGET] + extra + [LTX_HF_SPEC],
                capture_output=True, text=True)
            if r.returncode != 0:
                log(f"[LTX25] pip falló ({' '.join(extra) or 'con deps'}): "
                    f"{((r.stderr or '') + (r.stdout or ''))[-600:]}")
                continue
            ok, err = _ltx_import_check(LTX_HF_TARGET)
            if ok:
                _LTX_ENV.update(ok=True, pythonpath=LTX_HF_TARGET)
                log(f"[LTX25] ✅ Entorno LTX corregido (huggingface_hub aislado, {' '.join(extra) or 'con deps'}).")
                return LTX_HF_TARGET
    raise RuntimeError("El entorno de LTX-2.5 no puede importar transformers/mmgp:\n" + err[-1500:])


# ── FIX LIPSYNC (v8): archivos pequeños del Gemma4 (config/tokenizer) corruptos o vacíos ──
# Síntoma:  json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
#           en build_gemma_text_encoder → json.load(config.json)  (el subproceso moría a los ~30 s)
# Causa:    uno de los json de /tmp/models/gemma4-12b-ltx-v1 (o una copia en Wan2GP/ckpts|models) está
#           vacío/no es JSON (descarga hecha con otra versión de huggingface_hub, o interrumpida), y
#           ensure_ltx23_models() solo comprueba que el archivo EXISTA.
# Solución: validar esos archivos y, si están mal, volver a bajarlos por URL directa (independiente de
#           la versión de huggingface_hub) y re-enlazarlos. No toca el .safetensors (19+ GB).
GEMMA_SUB    = "gemma4-12b-ltx-v1"
GEMMA_SMALL  = ["config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja"]
GEMMA_REPO   = "DeepBeepMeep/LTX-2"
_GEMMA_OK    = {"ok": False}

def _gemma_valid(path: str, name: str) -> bool:
    try:
        if not os.path.exists(path) or os.path.getsize(path) <= 0:
            return False
        if name.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                json.load(f)
        return True
    except Exception:
        return False

def _gemma_diag(path: str) -> str:
    try:
        if not os.path.lexists(path):
            return "no existe"
        real = os.path.realpath(path)
        if not os.path.exists(real):
            return f"symlink roto → {real}"
        with open(real, "rb") as f:
            head = f.read(80)
        return f"{os.path.getsize(real)} bytes, inicio={head!r}"
    except Exception as e:
        return f"error leyendo: {e}"

def _gemma_download(name: str, dst: str):
    import urllib.request
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    url = f"https://huggingface.co/{GEMMA_REPO}/resolve/main/{GEMMA_SUB}/{name}"
    err = None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        if _gemma_valid(tmp, name):
            os.replace(tmp, dst)
            return
        err = f"descarga directa inválida ({_gemma_diag(tmp)})"
    except Exception as e:
        err = f"descarga directa falló: {e}"
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except Exception: pass
    # Respaldo: huggingface_hub a una carpeta temporal y copia del archivo devuelto
    log(f"[LTX25]   ⚠ {name}: {err} → probando huggingface_hub...")
    from huggingface_hub import hf_hub_download
    td = "/tmp/_gemma_dl"
    shutil.rmtree(td, ignore_errors=True)
    got = hf_hub_download(repo_id=GEMMA_REPO, subfolder=GEMMA_SUB, filename=name, local_dir=td)
    shutil.copyfile(got, dst + ".part")
    shutil.rmtree(td, ignore_errors=True)
    if not _gemma_valid(dst + ".part", name):
        bad = _gemma_diag(dst + ".part")
        try: os.remove(dst + ".part")
        except Exception: pass
        raise RuntimeError(f"No se pudo obtener un {name} válido ({bad})")
    os.replace(dst + ".part", dst)

def ensure_gemma_files(force: bool = False):
    """Garantiza que config/tokenizer del Gemma4 sean válidos en /tmp/models y en Wan2GP/ckpts|models."""
    if _GEMMA_OK["ok"] and not force:
        return
    for name in GEMMA_SMALL:
        real = os.path.join(GEMMA4_TMP, name)
        if not _gemma_valid(real, name):
            log(f"[LTX25] ⚠ Gemma4 {name} inválido ({_gemma_diag(real)}) → re-descargando...")
            try:
                if os.path.lexists(real): os.remove(real)
            except Exception: pass
            _gemma_download(name, real)
            log(f"[LTX25] ✅ Gemma4 {name} restaurado ({os.path.getsize(real)} bytes)")
        # Copias/symlinks que Wan2GP puede leer en lugar del de /tmp
        for base in (os.path.join(WAN2GP_DIR, "ckpts"), os.path.join(WAN2GP_DIR, "models")):
            dst = os.path.join(base, GEMMA_SUB, name)
            try:
                if os.path.lexists(dst) and not _gemma_valid(dst, name):
                    log(f"[LTX25] ⚠ {os.path.relpath(dst, WAN2GP_DIR)} inválido ({_gemma_diag(dst)}) → re-enlazado.")
                    os.remove(dst)
                if not os.path.lexists(dst):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    os.symlink(os.path.abspath(real), dst)
            except Exception as e:
                log(f"[LTX25] ⚠ no se pudo re-enlazar {dst}: {e}")
    _GEMMA_OK["ok"] = True


class SubprocessLTXBackend:
    """Motor LTX-2.5 con progreso en tiempo real vía Popen."""

    def is_available(self):
        return (os.path.isfile(LTX25_SCRIPT) and
                os.path.isdir(WAN2GP_DIR)    and
                os.path.isdir(TMP_MODELS))

    def _resolve_output_path(self, suggested=None):
        os.makedirs("/tmp/ltx25_outputs", exist_ok=True)
        if suggested:
            return os.path.join("/tmp/ltx25_outputs", os.path.basename(suggested))
        return f"/tmp/ltx25_outputs/ltx25_{int(time.time())}_{random.randint(0,99999)}.mp4"

    def _parse_progress_line(self, line: str) -> str:
        """
        Parsea líneas PROGRESS: del script y actualiza el archivo de progreso.
        Retorna un string legible para mostrar en UI.
        """
        if not line.startswith("PROGRESS:"): return ""
        parts = line.split(":")
        kind  = parts[1] if len(parts) > 1 else ""

        if kind == "start":
            total = parts[2] if len(parts) > 2 else "?"
            msg   = f"⚡ Iniciando denoising ({total} pasos)..."
            _write_progress({"status": msg, "step": 0,
                             "total": int(total) if total.isdigit() else 8,
                             "pct": 0, "active": True})
            return msg

        elif kind == "step":
            # PROGRESS:step:current:total:dt:rem:free_vram
            try:
                cur   = int(parts[2])
                total = int(parts[3])
                dt    = float(parts[4])
                rem   = float(parts[5])
                free  = float(parts[6])
                pct   = int(cur / total * 100) if total > 0 else 0
                bar   = "█" * (pct // 10) + "░" * (10 - pct // 10)
                msg   = (f"⚡ [{bar}] {pct}% | Paso {cur}/{total} | "
                         f"{dt:.1f}s/paso | ~{rem:.0f}s restantes | "
                         f"VRAM libre: {free:.1f}GB")
                _write_progress({"status": msg, "step": cur, "total": total,
                                 "pct": pct, "dt": dt, "rem": rem,
                                 "free_vram": free, "active": True})
                return msg
            except: return line

        elif kind == "stage":
            stage = ":".join(parts[2:])
            stage_icons = {
                "VAE Encoding": "🎞️ Codificando frames entrada...",
                "VAE Decoding": "🎬 Decodificando latents → frames...",
                "Upsampling":   "🔭 Upsampling espacial...",
            }
            msg = stage_icons.get(stage, f"⏳ {stage}...")
            _write_progress({"status": msg, "active": True})
            return msg

        return ""

    def generate(self,
                 prompt="", image_path=None, audio_path=None,
                 num_frames=121, width=1152, height=704,       #768×432   848×480 
                 seed=-1, steps=8, cfg=1.0, fps=24,
                 audio_cfg=1.0,
                 input_video_strength=0.8, denoise_strength=1.0,
                 output_path=None):
        """
        Genera video con Popen (lectura línea a línea de stdout).
        Actualiza LTX_PROGRESS_FILE en tiempo real.
        Retorna ruta del MP4 generado.
        """
        if not self.is_available():
            raise RuntimeError(
                f"Backend no disponible.\n"
                f"  Script: {os.path.isfile(LTX25_SCRIPT)}\n"
                f"  /tmp/models: {os.path.isdir(TMP_MODELS)}\n"
                "Ejecuta Celdas 12 y 12b."
            )

        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
        out_path = self._resolve_output_path(output_path)

        params = {
            "prompt": prompt,
            "image_path": image_path or "",
            "audio_path": audio_path or "",  # NUEVO
            "output_path":          out_path,
            "width":                width,
            "height":               height,
            "num_frames":           num_frames,
            "fps":                  fps,
            "seed":                 seed,
            "steps":                steps,
            "cfg":                  cfg,
            "audio_cfg":            audio_cfg,
            "input_video_strength": input_video_strength,
        }

        mode = "I2V" if (image_path and os.path.exists(image_path)) else "T2V"
        log(f"[LTX25] {mode} | {width}x{height} | {num_frames}f@{fps}fps | "
            f"steps={steps} | cfg={cfg} | seed={seed}")

        _write_progress({"status": "⏳ Iniciando subproceso LTX-2.5...",
                         "active": True, "step": 0, "total": steps, "pct": 0})

        ensure_gemma_files()                  # v8: config/tokenizer de Gemma4 válidos antes de lanzar el subproceso
        _extra_pp = ensure_ltx_env()          # v7: arregla el import de huggingface_hub (ver arriba)
        env = os.environ.copy()
        if _extra_pp:
            env["PYTHONPATH"] = _extra_pp + os.pathsep + env.get("PYTHONPATH", "")
        env["WAN2GP_DIR"]       = WAN2GP_DIR
        env["PYTHONUNBUFFERED"] = "1"

        t0           = time.time()
        final_path   = out_path
        stderr_lines = []

        # ── Popen: leer stdout línea a línea en tiempo real ────
        proc = subprocess.Popen(
            [sys.executable, "-u", LTX25_SCRIPT, json.dumps(params)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,          # line-buffered
            env=env,
        )

        # Hilo separado para capturar stderr sin bloquear
        def _read_stderr():
            for line in proc.stderr:
                line = line.rstrip()
                if line:
                    stderr_lines.append(line)
        stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
        stderr_thread.start()

        # Leer stdout línea a línea
        for raw_line in proc.stdout:
            line = raw_line.rstrip()
            if not line: continue

            if line.startswith("OUTPUT_PATH:"):
                candidate = line.split("OUTPUT_PATH:", 1)[1].strip()
                if os.path.exists(candidate):
                    final_path = candidate
                log(f"[LTX25] OUTPUT_PATH detectado: {final_path}")
                continue

            if line.startswith("PROGRESS:"):
                ui_msg = self._parse_progress_line(line)
                if ui_msg:
                    log(f"[LTX25] {ui_msg}")
                continue

            # Líneas normales de log del subprocess
            if line.startswith("[SUBPROCESS]"):
                log(f"  {line}")
            else:
                # mmgp y otros mensajes informativos
                log(f"  {line}")

        proc.wait()
        stderr_thread.join(timeout=5)
        elapsed = time.time() - t0

        # ── Resultado ──────────────────────────────────────────
        if proc.returncode != 0:
            err_tail = "\n".join(stderr_lines[-30:])
            _write_progress({"status": f"❌ Error (rc={proc.returncode})",
                             "active": False})
            log(f"[LTX25] ❌ Subprocess falló (rc={proc.returncode}, {elapsed:.1f}s)")
            log(f"  STDERR:\n{err_tail[-2000:]}")
            raise RuntimeError(
                f"LTX-2.5 subprocess falló (rc={proc.returncode}):\n{err_tail[-2000:]}"
            )

        # Verificar archivo generado
        if not os.path.exists(final_path):
            raise RuntimeError(f"Video no encontrado: {final_path}")
        sz = os.path.getsize(final_path)
        if sz < 50_000:
            raise RuntimeError(
                f"Video demasiado pequeño ({sz} bytes): {final_path}\n"
                f"El guardado falló. Revisa stderr en los logs."
            )

        _write_progress({
            "status": f"✅ Completado en {elapsed:.1f}s | {sz/1024/1024:.1f} MB",
            "active": False, "pct": 100, "path": final_path
        })
        log(f"[LTX25] ✅ {elapsed:.1f}s → {final_path} ({sz/1024:.0f} KB)")
        return final_path


# ── Singleton ──────────────────────────────────────────────────
_LTX_BACKEND = None

def get_ltx_backend():
    global _LTX_BACKEND
    if _LTX_BACKEND is None:
        _LTX_BACKEND = SubprocessLTXBackend()
    return _LTX_BACKEND

def get_ltx_engine():
    return get_ltx_backend()

_b = get_ltx_backend()
log("CELDA 13 (v4 TIEMPO REAL) COMPLETADA.")
log(f"  ► Backend: {_b.is_available()}")
log(f"  ► Progreso en tiempo real: {LTX_PROGRESS_FILE}")
log(f"  ► /tmp libre: {free_gb('/tmp'):.1f} GB | /working libre: {free_gb('/kaggle/working'):.1f} GB")