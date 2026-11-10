# Etapa del proyecto (no existía en el notebook): repara en el venv de Qwen el desajuste torch/torchvision
# ("operator torchvision::nms does not exist"). Idempotente: si el venv ya importa bien, no hace nada.
import os, re, subprocess

def _fix_torchvision_qwen():
    py = "/tmp/qwen3_tts_env/bin/python"
    sh = lambda c: (lambda r: (r.stdout + r.stderr).strip())(subprocess.run(c, shell=True, capture_output=True, text=True))
    test = f'{py} -c "import torchvision; from transformers import AutoProcessor; print(\'OK\', torchvision.__version__)"'
    if not os.path.exists(py):
        print("⚠ fix torchvision Qwen: el venv de Qwen aún no existe, se omite"); return
    t = sh(test)
    if "OK" in t.split("\n")[-1]:
        print("✅ venv Qwen: torch/torchvision compatibles —", t.splitlines()[-1]); return
    if "torchvision" not in t and "nms" not in t:
        print("⚠ venv Qwen falla por otra causa (no torchvision):\n", t[-400:]); return
    v = sh(f'{py} -c "import torch;print(torch.__version__)"').splitlines()[-1]
    tag = v.partition("+")[2]
    idx = f"--index-url https://download.pytorch.org/whl/{tag}" if tag else ""
    print(f"🔧 venv Qwen: torch {v} con torchvision incompatible → buscando versión compatible...")
    out = sh(f"{py} -m pip index versions torchvision {idx}")
    cands = [x.strip().split("+")[0] for x in out.split("Available versions:")[-1].split("\n")[0].split(",")
             if re.match(r"\s*0\.\d+\.\d+", x) and (not tag or x.strip().endswith("+" + tag))]
    for ver in cands[:6]:
        sh(f"{py} -m pip install -q --no-deps --force-reinstall torchvision=={ver}{'+' + tag if tag else ''} {idx}")
        t = sh(test)
        ok = "OK" in t.split("\n")[-1]
        print(f"   torchvision {ver} ->", "✅ OK" if ok else "falla")
        if ok: return
    print("❌ No se encontró un torchvision compatible para el venv de Qwen.")

_fix_torchvision_qwen()
