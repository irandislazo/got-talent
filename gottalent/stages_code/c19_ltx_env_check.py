# ═══ CELDA 17 (v8): COMPATIBILIDAD huggingface_hub — SIN tocar el kernel ═══
# ANTES: pip install "huggingface_hub<1.0" + borrar transformers/gradio de sys.modules.
#   Eso dejaba un huggingface_hub demasiado viejo para el transformers instalado y el
#   subproceso de LTX moría a los ~7 s con:
#       ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'
#   (la versión anterior estable NO tenía esta celda). Los parches HfFolder para gradio ya
#   existen en la Celda 12b (script LTX) y en la Celda 18 (UI), así que no hace falta bajar la versión.
# AHORA: solo se comprueba que el subproceso LTX pueda importar transformers + mmgp.
#   Si no puede, ensure_ltx_env() (Celda 13) instala huggingface_hub>=1.0 en una carpeta
#   aislada que SOLO ve el subproceso LTX. Qwen-TTS (venv propio) y el kernel no se modifican.
import importlib.metadata as _md

def _ver(pkg):
    try:
        return _md.version(pkg)
    except Exception:
        return "no instalado"

print(f"ℹ️ kernel: huggingface_hub={_ver('huggingface_hub')} | transformers={_ver('transformers')} | gradio={_ver('gradio')}")
try:
    _pp = ensure_ltx_env(force=True)
    print("✅ Entorno LTX-2.5 OK (transformers + mmgp importables en el subproceso)"
          + (f" — huggingface_hub aislado en {_pp}" if _pp else ""))
except Exception as _e:
    print(f"❌ Entorno LTX-2.5 con problemas:\n{_e}")
