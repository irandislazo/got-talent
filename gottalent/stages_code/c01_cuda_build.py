# ═══════════════════════════════════════════════════════════════
# CELDA 1 (MODIFICADA): COMPILACIÓN CUDA — sd-cli + qwen-tts
# CAMBIOS vs original:
#   - Eliminado constraint de NumPy (Wan2GP requiere numpy==2.1.2)
#   - Eliminado transformers==4.49.0 (Wan2GP requiere 4.54.0)
#   - Ref: https://github.com/deepbeepmeep/Wan2GP/blob/main/requirements.txt
#   - Ref: https://www.kaggle.com/code/jigs0907/ltx-2-5-wan2gp-kaggle
# ═══════════════════════════════════════════════════════════════
import os, sys, subprocess, shutil, json, time

FORCE_REBUILD_SDCPP   = False
FORCE_REBUILD_QWENTTS = False

def sh(cmd, fatal=True, cwd=None):
    print(f"→ {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    out = (r.stderr or "") + (r.stdout or "")
    if r.returncode != 0:
        print(out[-6000:])
        if fatal:
            raise RuntimeError(f"Fallo: {cmd}")
    return r

IS_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or os.path.isdir("/kaggle/working")
if not IS_KAGGLE:
    raise RuntimeError("Esta celda está pensada para Kaggle.")

SDCPP_DIR    = "/kaggle/working/stable-diffusion.cpp"
QWENTTS_DIR  = "/kaggle/working/qwentts.cpp"
KAGGLE_TMP   = "/kaggle/tmp"
os.makedirs(KAGGLE_TMP, exist_ok=True)

# ── SIN CONSTRAINT DE NUMPY ──
# Wan2GP/requirements.txt especifica numpy==2.1.2 y transformers==4.54.0
# sd-cli y qwen-tts son binarios C++ independientes de NumPy/transformers Python
# Ref: https://github.com/deepbeepmeep/Wan2GP/blob/main/requirements.txt
# --- SIN CONSTRAINT DE NUMPY (Wan2GP requiere numpy==2.1.2) ---
# Ref: https://github.com/deepbeepmeep/Wan2GP/blob/main/requirements.txt
C = ""

sh("apt-get update -y > /dev/null 2>&1")
sh("apt-get install -y git cmake ninja-build ffmpeg libsndfile1 sox > /dev/null 2>&1")
sh(f"pip install -q {C} openai edge-tts gradio psutil kaggle")
sh(f"pip install -q {C} faster-whisper ctranslate2>=4.5.0 \'av==18.1.0\'")
sh("pip install -q 'pyannote.audio>=4.0.1' torchcodec")
sh(f"pip install -q {C} pandas>=2.2.3 nltk omegaconf>=2.3.0")
sh("pip install -q --no-deps git+https://github.com/m-bain/whisperx.git")
# NOTA: NO instalar transformers/diffusers/huggingface_hub aquí.
# El backend TTS usa su propio pin estable de Transformers 5.9.0.
# No se instala aquí ninguna versión 4.x de Transformers.
# Qwen3-TTS Python NO se instala en el entorno principal.
# Su backend Python se ejecuta en un virtualenv aislado creado en Celda 6.
# Esto evita que Qwen cambie transformers / huggingface_hub del resto del proyecto.
sh("pip install -q 'numba>=0.61.0' 'librosa' 'soundfile' 'sox'")
sh("pip install -q mmgp")
sh("pip install -q 'numba>=0.61.0'")
sh(f"pip install -q {C} langdetect")
# comfy-cli YA NO ES NECESARIO (Wan2GP reemplaza ComfyUI para video)

# --- GPU / CUDA ---
r = sh("nvidia-smi -L", fatal=False)
if r.returncode != 0 or not (r.stdout or "").strip():
    raise RuntimeError("No hay GPU (nvidia-smi). Celda CUDA-only.")

r = sh("nvidia-smi --query-gpu=compute_cap --format=csv,noheader", fatal=False)
CUDA_ARCH = ((r.stdout or "").strip().splitlines() or ["7.5"])[0].replace(".", "") or "75"
print(f"CUDA_ARCH: {CUDA_ARCH}")

CUDA_HOME = "/usr/local/cuda"
NVCC = shutil.which("nvcc") or f"{CUDA_HOME}/bin/nvcc"
if not os.path.exists(NVCC):
    raise RuntimeError(f"nvcc no encontrado: {NVCC}")
sh(f"'{NVCC}' --version")

def find_libcuda_so1() -> str:
    r = subprocess.run("ldconfig -p 2>/dev/null | grep 'libcuda.so.1' || true",
                       shell=True, capture_output=True, text=True)
    for line in (r.stdout or "").splitlines():
        if "=>" in line:
            p = line.split("=>", 1)[1].strip()
            if os.path.exists(p): return p
    for p in ["/usr/local/nvidia/lib64/libcuda.so.1",
              "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
              "/lib/x86_64-linux-gnu/libcuda.so.1"]:
        if os.path.exists(p): return p
    r = subprocess.run(
        "find /usr/local/nvidia /usr/lib /lib -name 'libcuda.so.1' 2>/dev/null | head -n 1",
        shell=True, capture_output=True, text=True)
    p = (r.stdout or "").strip()
    return p if p and os.path.exists(p) else ""

LIBCUDA_SO1 = find_libcuda_so1()
print(f"libcuda.so.1: {LIBCUDA_SO1}")
if not LIBCUDA_SO1:
    raise RuntimeError("No se encontró libcuda.so.1")

CUDA_LINK_DIR = "/kaggle/working/cuda-link"
os.makedirs(CUDA_LINK_DIR, exist_ok=True)
LIBCUDA_SO = os.path.join(CUDA_LINK_DIR, "libcuda.so")
if os.path.lexists(LIBCUDA_SO): os.remove(LIBCUDA_SO)
os.symlink(LIBCUDA_SO1, LIBCUDA_SO)

os.environ["LIBRARY_PATH"] = f"{CUDA_LINK_DIR}:/usr/local/nvidia/lib64:" + os.environ.get("LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = f"{CUDA_LINK_DIR}:/usr/local/nvidia/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["CPATH"] = f"{CUDA_HOME}/include:" + os.environ.get("CPATH", "")
os.environ["CUDA_PATH"] = CUDA_HOME
os.environ["CUDAToolkit_ROOT"] = CUDA_HOME

def git_head(repo_dir):
    r = subprocess.run(f"git -C '{repo_dir}' rev-parse HEAD", shell=True, capture_output=True, text=True)
    return (r.stdout or "").strip()

# ── sd-cli ──
sdcli_bin = f"{SDCPP_DIR}/build/bin/sd-cli"
fp_path_sd = f"{SDCPP_DIR}/build/BUILD_FINGERPRINT.json"

if FORCE_REBUILD_SDCPP and os.path.exists(SDCPP_DIR):
    sh(f"rm -rf '{SDCPP_DIR}'")

if not os.path.exists(SDCPP_DIR):
    sh(f"git clone --recursive https://github.com/leejet/stable-diffusion.cpp '{SDCPP_DIR}'")
else:
    sh(f"git -C '{SDCPP_DIR}' reset --hard HEAD")
    sh(f"git -C '{SDCPP_DIR}' submodule foreach --recursive 'git reset --hard || true'", fatal=False)
    sh(f"git -C '{SDCPP_DIR}' submodule foreach --recursive 'git clean -fdx || true'", fatal=False)
    _need = FORCE_REBUILD_SDCPP or (not os.path.exists(sdcli_bin))
    if _need:
        sh(f"git -C '{SDCPP_DIR}' pull --recurse-submodules || true", fatal=False)
        sh(f"git -C '{SDCPP_DIR}' submodule update --init --recursive --force")
    else:
        sh(f"git -C '{SDCPP_DIR}' submodule update --init --recursive || true", fatal=False)
        print("✓ Fuentes sd.cpp sin pull (binario existente).")

def sd_up_to_date():
    if not (os.path.exists(sdcli_bin) and os.path.exists(fp_path_sd)): return False
    try:
        fp = json.load(open(fp_path_sd))
        return fp.get("commit") == git_head(SDCPP_DIR) and fp.get("backend") == "cuda" and fp.get("cuda_arch") == CUDA_ARCH
    except: return False

if sd_up_to_date():
    print("✓ sd-cli CUDA ya compilado.")
else:
    sh(f"rm -rf '{SDCPP_DIR}/build'")
    cmake_cmd = (
        f"cd '{SDCPP_DIR}' && cmake -S . -B build -G Ninja "
        f"-DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON -DGGML_CUDA=ON "
        f"-DCMAKE_CUDA_COMPILER='{NVCC}' -DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCH} "
        f"-DCUDAToolkit_ROOT='{CUDA_HOME}' -DCUDA_cuda_LIBRARY='{LIBCUDA_SO}' "
        f"-DCMAKE_LIBRARY_PATH='{CUDA_LINK_DIR};/usr/local/nvidia/lib64' "
        f"-DCMAKE_EXE_LINKER_FLAGS='-L{CUDA_LINK_DIR} -L/usr/local/nvidia/lib64'"
    )
    sh(cmake_cmd)
    sh(f"cd '{SDCPP_DIR}' && cmake --build build --config Release --target sd-cli")
    if not os.path.exists(sdcli_bin):
        raise RuntimeError("No se generó build/bin/sd-cli")
    json.dump({"commit": git_head(SDCPP_DIR), "backend": "cuda", "cuda_arch": CUDA_ARCH, "libcuda": LIBCUDA_SO1},
              open(fp_path_sd, "w"), indent=2)
    print("¡Build sd-cli CUDA exitoso!")

open(f"{SDCPP_DIR}/build/SDCLI_BACKEND.txt", "w").write("cuda")

# ── qwen-tts ──
qwentts_bin = f"{QWENTTS_DIR}/build/bin/qwen-tts"
fp_path_tts = f"{QWENTTS_DIR}/build/BUILD_FINGERPRINT.json"

if FORCE_REBUILD_QWENTTS and os.path.exists(QWENTTS_DIR):
    sh(f"rm -rf '{QWENTTS_DIR}'")

if not os.path.exists(QWENTTS_DIR):
    sh(f"git clone --recurse-submodules https://github.com/ServeurpersoCom/qwentts.cpp.git '{QWENTTS_DIR}'")

def tts_up_to_date():
    if not (os.path.exists(qwentts_bin) and os.path.exists(fp_path_tts)): return False
    try:
        fp = json.load(open(fp_path_tts))
        return fp.get("commit") == git_head(QWENTTS_DIR) and fp.get("backend") == "cuda" and fp.get("cuda_arch") == CUDA_ARCH
    except: return False

if tts_up_to_date():
    print("✓ qwen-tts CUDA ya compilado.")
else:
    print("→ Compilando qwen-tts...")
    sh(f"rm -rf '{QWENTTS_DIR}/build'")
    cmake_tts = (
        f"cd '{QWENTTS_DIR}' && cmake -S . -B build -G Ninja "
        f"-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON "
        f"-DCMAKE_CUDA_COMPILER='{NVCC}' -DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCH} "
        f"-DCUDAToolkit_ROOT='{CUDA_HOME}' "
        f"-DCMAKE_LIBRARY_PATH='{CUDA_LINK_DIR};/usr/local/nvidia/lib64' "
        f"-DCUDA_cuda_LIBRARY='{LIBCUDA_SO}' "
        f"-DCMAKE_EXE_LINKER_FLAGS='-L{CUDA_LINK_DIR} -L/usr/local/nvidia/lib64 -lcuda' "
        f"-DCMAKE_SHARED_LINKER_FLAGS='-L{CUDA_LINK_DIR} -L/usr/local/nvidia/lib64 -lcuda'"
    )
    r_cmake = sh(cmake_tts, fatal=False)
    if r_cmake.returncode != 0:
        print("⚠ CMake falló. Fallback CPU...")
        sh(f"cd '{QWENTTS_DIR}' && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF", fatal=False)
    sh(f"cd '{QWENTTS_DIR}' && cmake --build build --config Release", fatal=False)
    if not os.path.exists(qwentts_bin):
        r_find = subprocess.run(f"find '{QWENTTS_DIR}' -maxdepth 5 -type f -name 'qwen-tts' 2>/dev/null",
                                shell=True, capture_output=True, text=True)
        found = (r_find.stdout or "").strip().splitlines()
        if found:
            os.makedirs(os.path.dirname(qwentts_bin), exist_ok=True)
            shutil.copy2(found[0], qwentts_bin)
    if os.path.exists(qwentts_bin):
        os.chmod(qwentts_bin, 0o755)
        json.dump({"commit": git_head(QWENTTS_DIR), "backend": "cuda", "cuda_arch": CUDA_ARCH},
                  open(fp_path_tts, "w"), indent=2)
        print("¡Build qwen-tts exitoso!")
    else:
        print("⚠ qwen-tts no compiló. Edge-tts como fallback.")
"""
# ── ComfyUI (se mantiene para compatibilidad, pero NO se usa para LTX-2.5) ──
COMFYUI_DIR = "/kaggle/working/ComfyUI"
if not os.path.isdir(COMFYUI_DIR):
    print("Instalando ComfyUI...")
    sh(f"yes | comfy --workspace='{COMFYUI_DIR}' install --nvidia", fatal=False)
else:
    print("✓ ComfyUI ya instalado.")

COMFYUI_GGUF_DIR = f"{COMFYUI_DIR}/custom_nodes/ComfyUI-GGUF"
if not os.path.isdir(COMFYUI_GGUF_DIR):
    sh(f"git clone https://github.com/city96/ComfyUI-GGUF '{COMFYUI_GGUF_DIR}'", fatal=False)
    sh(f"cd '{COMFYUI_GGUF_DIR}' && pip install -q -r requirements.txt", fatal=False)
"""

# ── Wan2GP (código fuente para LTX-2.5) ──────────────────────────────
# Sin esto, /kaggle/working/Wan2GP/models/ltx2/ no existe en entornos nuevos.
WAN2GP_DIR    = "/kaggle/working/Wan2GP"
WAN2GP_REPO   = "https://github.com/deepbeepmeep/Wan2GP"
WAN2GP_COMMIT = "b8b18f8114e432eea8f3d7e853a51dd91fa99571"   # ← pega aquí el hash de tu entorno que funciona (ver abajo). Vacío = main
_WAN2GP_MARK  = os.path.join(WAN2GP_DIR, "models", "ltx2", "ltx2_handler.py")

if os.path.isfile(_WAN2GP_MARK):
    print(f"✓ Wan2GP ya presente ({git_head(WAN2GP_DIR)[:10] or 'sin .git'}).")
else:
    print("Clonando Wan2GP...")
    _src = "/kaggle/tmp/Wan2GP_src"
    sh(f"rm -rf '{_src}'")
    sh(f"git clone {WAN2GP_REPO} '{_src}'")
    if WAN2GP_COMMIT:
        sh(f"git -C '{_src}' checkout {WAN2GP_COMMIT}")
    os.makedirs(WAN2GP_DIR, exist_ok=True)
    # Fusiona sin pisar lo que ya haya (symlinks de models/ckpts, script, defaults)
    sh(f"cp -a '{_src}/.' '{WAN2GP_DIR}/'")
    sh(f"rm -rf '{_src}'")
    if not os.path.isfile(_WAN2GP_MARK):
        raise RuntimeError("Wan2GP clonado pero falta models/ltx2/ltx2_handler.py "
                           "(¿commit incorrecto o cambió la estructura del repo?)")
    print(f"✅ Wan2GP listo ({git_head(WAN2GP_DIR)[:10]}).")

# Permisos
for _bin in [sdcli_bin, qwentts_bin]:
    if os.path.exists(_bin): os.chmod(_bin, 0o755)
for _bin_dir in [f"{SDCPP_DIR}/build/bin", f"{QWENTTS_DIR}/build/bin"]:
    if os.path.isdir(_bin_dir):
        for fname in os.listdir(_bin_dir):
            fpath = os.path.join(_bin_dir, fname)
            if os.path.isfile(fpath): os.chmod(fpath, 0o755)

print("\nVerificando sd-cli...")
r = subprocess.run([sdcli_bin, "--help"], capture_output=True, text=True)
help_txt = (r.stdout or "") + (r.stderr or "")
print(f"OK: {sdcli_bin} | --llm_vision: {'SÍ' if '--llm_vision' in help_txt else 'NO'}")

if os.path.exists(qwentts_bin):
    print(f"OK: {qwentts_bin}")
else:
    print("qwen-tts no disponible.")

# ── FIX CRÍTICO: Restaurar Gradio tras qwen-tts-compat ──
sh("pip install -q --force-reinstall 'gradio>=6.0.0,<7.0.0'")
print("✅ Gradio reinstalado correctamente")

print("\n✅ CELDA 1 COMPLETADA (stack TTS fijado y sin mezclar Transformers 4.x/5.x).")
print("   Qwen3-TTS Python quedará aislado en /tmp/qwen3_tts_env")
print("   Reinicia el kernel después de ejecutar todas las celdas.")
time.sleep(2)
os.kill(os.getpid(), 9)
