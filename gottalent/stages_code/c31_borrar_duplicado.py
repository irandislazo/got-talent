from pathlib import Path
import shutil

BASE_DIR = Path("/kaggle/working/VideoAutomationProject/videos").resolve()

ACTIVO = (BASE_DIR / "20261022_205741_la_voz_1_mujer").resolve()
DUPLICADO = (BASE_DIR / "20261022_205644_mi_plantilla_1_mujer").resolve()

# Comprobaciones de seguridad
assert ACTIVO.is_dir(), "No se encuentra el proyecto activo."
assert DUPLICADO.is_dir(), "No se encuentra el directorio candidato."
assert ACTIVO != DUPLICADO, "Error: las rutas coinciden."
assert DUPLICADO.parent == BASE_DIR, "Ruta fuera de la carpeta de proyectos."

print("✅ Se conservará:", ACTIVO)
print("⚠️ Candidato a eliminar:", DUPLICADO)
print("\nContenido del candidato:")
for item in sorted(DUPLICADO.iterdir()):
    print(" -", item.name)

confirmacion = input(
    "\nPara eliminar SOLO este candidato, escribe exactamente:\n"
    "20261022_205644_mi_plantilla_1_mujer\n> "
)

if confirmacion == DUPLICADO.name:
    shutil.rmtree(DUPLICADO)
    print("✅ Directorio candidato eliminado.")
else:
    print("❎ Cancelado. No se eliminó nada.")