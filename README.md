# 🎤 Got Talent — Automatización (proyecto Python)

El código de cada celda del notebook original está **intacto** en `gottalent/stages_code/` (`cNN_*.py`, NN = nº de celda original).
El runner las ejecuta en orden en el mismo espacio global que un kernel (mismos venv aislados, mismas dependencias; el proyecto **no instala nada**).
Única diferencia: en `c27_ui_principal.py` los acordeones principales son **pestañas**, y las mini-apps (YouTube, plantillas, exportar/importar, limpieza, visor) se montan como pestañas extra.

## Uso en Kaggle (2 celdas)
```python
!rm -rf /kaggle/working/got-talent && git clone -q --depth 1 https://github.com/TU_USUARIO/got-talent.git /kaggle/working/got-talent
```
```python
import sys; sys.path.insert(0, "/kaggle/working/got-talent"); import gottalent; gottalent.run()
```
Al terminar imprime el enlace público (`share`) de la UI; ábrelo en otra pestaña (no se muestra bajo la celda).

## Opciones
- `gottalent.run(skip=[1])` salta celdas · `run(only=[27])`… · `run(desde=13)` · `run(extras=False)`
- Orden/activación: `gottalent/stages.py` (`SETUP`, `EXTRA_TABS`, `TOOLS`).
- Utilidades sueltas: `gottalent.tool("detener_ltx")`, `"diagnostico"`, `"probar_upload"`…
- Solo reabrir la UI (tras `run` previo en la misma sesión): `gottalent.launch_ui()`

## Solo a demanda
El arranque ejecuta las celdas 1-18 + 21, 22, 23 (LTX 2.3) + la UI. Las demás celdas posteriores a la 18 (fixes, diagnóstico, detener LTX, borrados…) NO se ejecutan solas: lánzalas cuando quieras con `gottalent.tool("nombre")`. Nombres en `gottalent/stages.py` (`TOOLS`).

## Mini-celdas (una por herramienta; requieren haber ejecutado antes `gottalent.run()` en la sesión)
```python
import gottalent as g; g.tool("visor")            # 🎬 visor de videos terminados (celda 0)
g.tool("youtube")      # 📺 subida a YouTube
g.tool("plantillas")   # 📦 exportar/importar plantillas
g.tool("exportar_video"); g.tool("importar_video")
g.tool("limpiador_qwen"); g.tool("limpieza")
g.tool("detener_ltx"); g.tool("diagnostico"); g.tool("probar_upload"); g.tool("auditoria_timeline")
```
