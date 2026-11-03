import os

PROYECTO_CORRECTO = (
    "/kaggle/working/VideoAutomationProject/videos/"
    "20261022_205741_la_voz_1_mujer"
)

# 1. Verificar que el proyecto y sus clips existen
CARPETA_CLIPS = os.path.join(
    PROYECTO_CORRECTO, "etapa3_cancion", "planos_5s"
)

assert os.path.isdir(PROYECTO_CORRECTO), (
    f"No existe el proyecto: {PROYECTO_CORRECTO}"
)
assert os.path.isdir(CARPETA_CLIPS), (
    f"No existe la carpeta de clips: {CARPETA_CLIPS}"
)

clips = [
    f for f in os.listdir(CARPETA_CLIPS)
    if f.lower().endswith(".mp4")
]

assert clips, "No hay archivos MP4 en la carpeta de clips."

# 2. Sincronizar el selector antiguo con el nuevo
_CUR_VP_DIR = PROYECTO_CORRECTO
CURRENT_VIDEO_DIR = PROYECTO_CORRECTO

if globals().get("_CUR_TPL"):
    CURRENT_TEMPLATE = _CUR_TPL

# Sincronizar el estado guardado de la interfaz
for nombre_estado in ("UI_STATE", "_UI_STATE"):
    estado = globals().get(nombre_estado)
    if isinstance(estado, dict):
        estado["current_video_dir"] = PROYECTO_CORRECTO
        if globals().get("_CUR_TPL"):
            estado["current_template_slug"] = _CUR_TPL

        if callable(globals().get("save_ui_state")):
            save_ui_state(estado)

# 3. Hacer que las funciones antiguas usen el selector correcto
def get_current_vp() -> VideoProject:
    return _get_vp()

# 4. Comprobación final
vp = get_current_vp()

print("✅ Selector corregido.")
print("📂 Proyecto seleccionado:", vp.video_dir)
print("🎬 Clips 5s encontrados:", len(clips))
for nombre in sorted(clips):
    print("  -", nombre)
print("\nYa puedes probar de nuevo la acción que construye la secuencia.")