from pathlib import Path
from datetime import datetime

BASE_DIR = Path("/kaggle/working/VideoAutomationProject/videos")
PROYECTO_ACTIVO = BASE_DIR / "20261022_205741_la_voz_1_mujer"

print("✅ PROYECTO QUE SE CONSERVARÁ:")
print(PROYECTO_ACTIVO)
print("\n📂 PROYECTOS ENCONTRADOS:\n")

if not BASE_DIR.exists():
    print("❌ No existe la carpeta base:", BASE_DIR)
else:
    proyectos = [
        p for p in BASE_DIR.iterdir()
        if p.is_dir() and p.resolve() != PROYECTO_ACTIVO.resolve()
    ]

    proyectos.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    if not proyectos:
        print("No se encontraron otros directorios de proyectos.")
    else:
        for p in proyectos:
            fecha = datetime.fromtimestamp(p.stat().st_mtime).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            print(f"📁 {p.name} | Modificado: {fecha}")
            print(f"   Ruta: {p}")