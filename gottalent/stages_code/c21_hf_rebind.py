import importlib.metadata as _md, huggingface_hub
print("en memoria:", huggingface_hub.__version__, "| en disco:", _md.version("huggingface_hub"))
from huggingface_hub import hf_hub_download   # reemplaza la función vieja de las celdas 3 y 12
print("✅ hf_hub_download re-enlazado")