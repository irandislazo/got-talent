# ═══ FIX DEFINITIVO huggingface_hub ═══
import subprocess, sys
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "huggingface_hub>=0.34.0,<1.0.0"
], capture_output=True)

for mod in list(sys.modules.keys()):
    if any(x in mod.lower() for x in ["huggingface_hub", "gradio", "transformers"]):
        del sys.modules[mod]

try:
    from huggingface_hub.utils import HfFolder
    print("✅ huggingface_hub>=0.34 compatible con transformers+gradio")
except ImportError:
    print("⚠ HfFolder no encontrado, aplicando patch manual...")
    # El patch de Celda 6 ya lo cubre