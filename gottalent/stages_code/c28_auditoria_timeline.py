
# ============================================================
# AUDITORÍA DEL TIMELINE — SOLO LECTURA
# Comprueba orden, duraciones, giros y distribución de reacciones.
# No modifica el proyecto ni genera clips.
# ============================================================

import os, json
import pandas as pd
from IPython.display import display

# 1. Obtener el proyecto activo usando el selector actualizado
if "_get_vp" in globals() and callable(_get_vp):
    vp = _get_vp()
elif "get_current_vp" in globals() and callable(get_current_vp):
    vp = get_current_vp()
else:
    raise RuntimeError("No encuentro el selector del proyecto activo.")

print("Proyecto:", getattr(vp, "video_dir", "(ruta no disponible)"))

# 2. Leer el timeline guardado en memoria
items = getattr(vp, "timeline_items", None)

if isinstance(items, str):
    try:
        items = json.loads(items)
    except Exception:
        items = None

# Si no está cargado en memoria, intentar leer el JSON del proyecto
if not items:
    project_dir = getattr(vp, "video_dir", "")
    candidates = [
        os.path.join(project_dir, "timeline.json"),
        os.path.join(project_dir, "timeline_items.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    items = data.get("timeline_items", data.get("clips", []))
                if items:
                    print("Timeline leído desde:", path)
                    break
            except Exception:
                pass

if not isinstance(items, list) or not items:
    raise RuntimeError(
        "No pude leer el timeline guardado. No se modificó nada. "
        "Hay que revisar cómo lo carga esta versión de VideoProject."
    )

# 3. Normalizar los registros para la auditoría
rows = []
for i, item in enumerate(items, 1):
    if not isinstance(item, dict):
        try:
            item = vars(item)
        except Exception:
            item = {"valor": str(item)}

    meta = item.get("meta") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {"meta_texto": meta}
    if not isinstance(meta, dict):
        meta = {}

    dur = item.get("dur_s", item.get("duracion", 0))
    try:
        dur = float(dur or 0)
    except (TypeError, ValueError):
        dur = 0.0

    start = item.get("start")
    end = item.get("end")
    try:
        start = float(start) if start is not None else None
    except (TypeError, ValueError):
        start = None
    try:
        end = float(end) if end is not None else None
    except (TypeError, ValueError):
        end = None

    rows.append({
        "n": i,
        "inicio_s": start,
        "fin_s": end,
        "dur_s": dur,
        "tipo": item.get("tipo", ""),
        "hablante": item.get("hablante", ""),
        "estado": item.get("estado", ""),
        "archivo": os.path.basename(str(item.get("path_video", ""))),
        "meta": meta,
    })

df = pd.DataFrame(rows)

# Calcular tiempos acumulados si el JSON no los incluye
if df["inicio_s"].isna().all():
    df["inicio_s"] = df["dur_s"].shift(fill_value=0).cumsum()
if df["fin_s"].isna().all():
    df["fin_s"] = df["inicio_s"] + df["dur_s"]

df["tipo"] = df["tipo"].fillna("").astype(str)
df["hablante"] = df["hablante"].fillna("").astype(str)

# 4. Resumen
print("\n" + "=" * 65)
print("RESUMEN DEL TIMELINE")
print("=" * 65)
print("Número de clips:", len(df))
print(f"Duración acumulada: {df['dur_s'].sum():.2f} s")
print("\nClips por tipo:")
display(df["tipo"].value_counts(dropna=False).rename_axis("tipo").to_frame("cantidad"))

print("\nDistribución por juez/hablante:")
display(df["hablante"].value_counts(dropna=False).rename_axis("hablante").to_frame("cantidad"))

# 5. Validaciones automáticas básicas
issues = []

for _, r in df.iterrows():
    if r["dur_s"] <= 0:
        issues.append(f"Clip {r['n']}: duración no positiva ({r['dur_s']}).")

# Comprobar continuidad entre los tiempos guardados
if df["inicio_s"].notna().all() and df["fin_s"].notna().all():
    for i in range(1, len(df)):
        prev_end = df.iloc[i - 1]["fin_s"]
        cur_start = df.iloc[i]["inicio_s"]
        delta = cur_start - prev_end
        if delta > 0.10:
            issues.append(f"Hueco de {delta:.2f} s antes del clip {i+1}.")
        elif delta < -0.10:
            issues.append(f"Solapamiento de {-delta:.2f} s en el clip {i+1}.")

# Comprobar reacciones consecutivas
reaction_mask = df["tipo"].str.contains("reaccion", case=False, na=False)
max_streak = streak = 0
for is_reaction in reaction_mask.tolist():
    streak = streak + 1 if is_reaction else 0
    max_streak = max(max_streak, streak)

print("\nMáximo de clips de reacción consecutivos:", max_streak)

# Buscar clips de giro por el tipo que aparece en este notebook
turn_mask = df["tipo"].str.contains("girando|giro", case=False, na=False)
turns = df[turn_mask]
print("Clips identificados como giros:", len(turns))
if not turns.empty:
    display(turns[["n", "inicio_s", "fin_s", "tipo", "hablante", "archivo"]])

# 6. Tabla cronológica completa
print("\n" + "=" * 65)
print("TIMELINE COMPLETO — ORDEN CRONOLÓGICO")
print("=" * 65)
display(df[[
    "n", "inicio_s", "fin_s", "dur_s",
    "tipo", "hablante", "estado", "archivo", "meta"
]])

# 7. Diagnóstico
print("\n" + "=" * 65)
print("DIAGNÓSTICO")
print("=" * 65)

if issues:
    print(f"Se encontraron {len(issues)} posibles problemas estructurales:")
    for msg in issues[:100]:
        print("⚠️", msg)
    if len(issues) > 100:
        print(f"... y {len(issues) - 100} avisos más.")
else:
    print("✅ No se detectaron duraciones no positivas, huecos ni solapamientos "
          "en los datos disponibles.")

print(
    "\nIMPORTANTE: esta auditoría comprueba estructura y distribución básica. "
    "Para confirmar que cada juez tiene la reacción correcta antes/después de girar, "
    "hay que revisar los metadatos de estado del juez y las imágenes de los clips."
)
