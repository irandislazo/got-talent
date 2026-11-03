from pathlib import Path
import inspect

ACTIVO = Path(
    "/kaggle/working/VideoAutomationProject/videos/"
    "20261022_205741_la_voz_1_mujer"
).resolve()

print("=== VARIABLES ===")
for nombre in ["_CUR_VP_DIR", "CURRENT_VIDEO_DIR"]:
    valor = globals().get(nombre, "NO EXISTE")
    print(f"{nombre} = {valor}")

print("\n=== SELECTORES ===")
for nombre in ["_get_vp", "get_current_vp"]:
    funcion = globals().get(nombre)
    if callable(funcion):
        try:
            print(f"{nombre}() = {funcion()}")
            print(f"Código de {nombre}:")
            print(inspect.getsource(funcion))
        except Exception as e:
            print(f"No se pudo inspeccionar {nombre}: {e}")
    else:
        print(f"{nombre}: no está definida")

print("\n=== CLIPS DEL PROYECTO ACTIVO ===")
carpeta = ACTIVO / "etapa3_cancion" / "planos_5s"
print("Carpeta:", carpeta)
print("Existe:", carpeta.is_dir())

if carpeta.is_dir():
    print("Clips MP4:", [p.name for p in carpeta.glob("*.mp4")])