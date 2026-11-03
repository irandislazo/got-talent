import subprocess, os

QWEN_PY = "/tmp/qwen3_tts_env/bin/python"

if os.path.exists(QWEN_PY):
    r = subprocess.run([QWEN_PY, "-m", "pip", "install", "-q", "hf_transfer"],
                       capture_output=True, text=True)
    print("venv Qwen:", "✅ OK" if r.returncode == 0 else f"❌ {r.stderr[-500:]}")
else:
    print("⚠ El venv de Qwen aún no existe (ejecuta primero la celda de Qwen)")

r = subprocess.run(["pip", "install", "-q", "hf_transfer"], capture_output=True, text=True)
print("entorno principal:", "✅ OK" if r.returncode == 0 else f"❌ {r.stderr[-500:]}")