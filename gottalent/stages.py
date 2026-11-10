"""Orden de ejecución. Editar aquí para activar/desactivar etapas (nº = celda original del notebook)."""

# Se ejecutan en este orden, en el mismo espacio de nombres (igual que celdas de un kernel).
SETUP = [
    2,   # CELDA 2  configuración general
    3,   # CELDA 3  utilidades base
    4,   # CELDA 4  estructuras de proyecto
    5,   # CELDA 5  etapa 1 imágenes
    6,   # CELDA 6  Qwen3-TTS aislado
    7,   # CELDA 8  etapas 2 y 4 diálogos
    8,   # (sin etiqueta) FAR_DUR ...
    9,   # CELDA 10 funciones auxiliares UI
    10,  # (sin etiqueta) HF_HUB_ENABLE_HF_TRANSFER=0
    11,  # CELDA 12 LTX-2.5 modelos
    12,  # CELDA 12b parches Wan2GP
    13,  # CELDA 13 motor LTX-2.5
    14,  # CELDA 14 clips LTX-2.5
    15,  # CELDA 14b LTX-2.3
    16,  # CELDA 15 etapa 3 canción
    17,  # CELDA 16 pipeline diálogos
    18,  # CELDA 17 render final
    21,  # (sin etiqueta) re-enlazar hf_hub_download
    22,  # (sin etiqueta) hf_transfer en venvs
    99,  # (proyecto) repara torch/torchvision del venv de Qwen si hace falta
    23,  # (sin etiqueta) cerrar Gradio/túneles
]  # luego UI = 27 (CELDA 18 v7). El número = posición real en el notebook.

UI = 27  # UI principal (Blocks 'demo' con pestañas)

# Mini-apps que se montan como pestañas extra dentro de la UI: (celda, variable Blocks, título)
EXTRA_TABS = [
    (0,  "demo_visor",      "🎬 Visor de videos"),
    (32, "demo_yt",         "📺 YouTube"),
    (33, "demo_plantillas", "📦 Plantillas"),
    (34, "demo_exportar",   "📤 Exportar video"),
    (35, "demo_importar",   "📥 Importar video"),
    (39, "demo_clean",      "🧹 Limpiador Qwen"),
    (40, "demo_limpieza",   "🧹 Limpieza"),
]

# Celdas sueltas que se lanzan a demanda con gottalent.tool("nombre")
TOOLS = {
    "fix_torchvision_qwen": 99,
    "cuda": 1,   # CELDA 1: compila CUDA y REINICIA el kernel (ejecutar sola, primero)
    "visor": 0, "youtube": 32, "plantillas": 33, "exportar_video": 34, "importar_video": 35,
    "limpiador_qwen": 39, "limpieza": 40,
    "ltx_env_check": 19, "hf_fix_antiguo": 20, "ltx_transformers_pin": 24, "wan2gp_hash_memwatch": 25,
    "selector_proyecto_fijo": 26, "auditoria_timeline": 28, "inspeccion_selector": 29,
    "listar_proyectos": 30, "borrar_duplicado": 31, "detener_ltx": 36, "probar_upload": 37,
    "diagnostico": 38, "borrar_escobas": 41,
}
