# ═══════════════════════════════════════════════════════════════
# CELDA 2: CONFIGURACIÓN GENERAL — Sistema de Audiciones (Got Talent / La Voz)
# Plantillas · Selectores globales · Prompts oficiales · Voces · Persistencia
# ═══════════════════════════════════════════════════════════════

import os, gc, json, subprocess, shutil, time, random
from IPython.display import display, Javascript

IS_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or os.path.isdir("/kaggle/working")
KAGGLE_INPUT_DIR = "/kaggle/input"
KAGGLE_TMP_DIR   = "/kaggle/tmp"
os.makedirs(KAGGLE_TMP_DIR, exist_ok=True)

if not IS_KAGGLE:
    raise RuntimeError("Notebook diseñado para Kaggle.")

PLATFORM   = "kaggle"
OUTPUT_DIR = "/kaggle/working/VideoAutomationProject"
SDCPP_DIR  = "/kaggle/working/stable-diffusion.cpp"
QWENTTS_DIR = "/kaggle/working/qwentts.cpp"
COMFYUI_DIR = "/kaggle/working/ComfyUI"

os.makedirs(OUTPUT_DIR, exist_ok=True)
PLANTILLAS_DIR = f"{OUTPUT_DIR}/plantillas"
VIDEOS_DIR     = f"{OUTPUT_DIR}/videos"
LOCAL_MODELS_DIR = f"{OUTPUT_DIR}/local_models"          # caché persistente 20GB
LOCAL_MODELS_TMP_DIR = f"{KAGGLE_TMP_DIR}/local_models_tmp"  # scratch ~60GB no persistente (LTX pesado)
for d in [PLANTILLAS_DIR, VIDEOS_DIR, LOCAL_MODELS_DIR, LOCAL_MODELS_TMP_DIR]:
    os.makedirs(d, exist_ok=True)

LOG_FILE = f"{OUTPUT_DIR}/run.log"
os.environ["PROJECT_LOG_FILE"] = LOG_FILE
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("=== SISTEMA DE AUDICIONES — LOG ===\n")

def _append_log(line: str):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")
    except Exception:
        pass

def _log_cell2(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    _append_log(line)

_log_cell2(f"OUTPUT_DIR: {OUTPUT_DIR}")
_log_cell2(f"PLANTILLAS_DIR: {PLANTILLAS_DIR}")
_log_cell2(f"VIDEOS_DIR: {VIDEOS_DIR}")
_log_cell2(f"LOCAL_MODELS_DIR (persistente, 20GB compartidos): {LOCAL_MODELS_DIR}")
_log_cell2(f"LOCAL_MODELS_TMP_DIR (scratch /kaggle/tmp, ~60GB, NO persistente): {LOCAL_MODELS_TMP_DIR}")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
CPU_THREADS = max(1, (os.cpu_count() or 2) // 2)
os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
SDCLI_THREADS = max(1, os.cpu_count() or 2)

SDCLI_GPU_AVAILABLE = shutil.which("nvidia-smi") is not None
if SDCLI_GPU_AVAILABLE:
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        SDCLI_GPU_AVAILABLE = r.returncode == 0 and bool((r.stdout or "").strip())
    except Exception:
        SDCLI_GPU_AVAILABLE = False

# ---------------------------------------------------------------
# API KEY xKIRO
# ---------------------------------------------------------------
XKIRO_API_KEY = "sk-xt-4a5373fc29fc7292c082ad27e10b5333952cacf3fbf30c1c"
XKIRO_BASE_URL = "https://api.xkiro.com/v1"
os.environ["XKIRO_API_KEY"] = XKIRO_API_KEY

XKIRO_MODEL_OPTIONS = {
    "🇫🇷 Mistral Large 3 (MoE, el más capaz)":            "mistralai/mistral-large-2512",
    "🇫🇷 Mistral Medium 3.5 (agentic, texto+imagen)":     "mistralai/mistral-medium-3.5",
    "🇫🇷 Mistral Small 4 (instruct+razonamiento+código)": "mistralai/mistral-small-2603",
    "🇫🇷 Codestral (código)":                             "mistralai/codestral-2508",
    "🇫🇷 Devstral 2 Medium (agentic, código)":            "mistralai/devstral-medium",
    "🇫🇷 Ministral 3 14B (texto+visión)":                 "mistralai/ministral-14b",
    "🇫🇷 Ministral 3 8B (eficiente)":                     "mistralai/ministral-8b",
    "🇫🇷 Ministral 3 3B (ultraligero)":                   "mistralai/ministral-3b",
    "🌀 MiniMax M3 (1M contexto, frontier)":               "minimax/minimax-m3:free",
    "🌀 MiniMax M2.7":                                      "minimax/minimax-m2.7:free",
    "🌀 MiniMax M2.7 Highspeed":                            "minimax/minimax-m2.7-highspeed:free",
    "🌀 MiniMax M2.5 (código)":                             "minimax/minimax-m2.5:free",
    "🌀 MiniMax M2.5 Highspeed":                            "minimax/minimax-m2.5-highspeed:free",
    "🌀 MiniMax M2.1 (razonamiento)":                       "minimax/minimax-m2.1:free",
    "🌀 MiniMax M2.1 Highspeed":                            "minimax/minimax-m2.1-highspeed:free",
    "🌀 MiniMax M2 (agentic, function calling)":            "minimax/minimax-m2:free",
    "🟣 Qwen3.8 Omni Flash (multimodal, 1M)":               "qwen/qwen3.8-omni-flash:free",
    "🟣 Qwen3.8 Max (flagship, 1M) [RECOMENDADO]":          "qwen/qwen3.8-max:free",
    "🟣 Qwen3.7 Max":                                       "qwen/qwen3.7-max:free",
    "🟣 Qwen3.7 Plus (económico)":                          "qwen/qwen3.7-plus:free",
    "🟣 Qwen3.7 Flash (vision-language)":                   "qwen/qwen3.7-flash:free",
    "🟣 Qwen3.6 Plus (MoE híbrido)":                        "qwen/qwen3.6-plus:free",
    "🟣 Qwen3.6 Max Preview (~1T params)":                  "qwen/qwen3.6-max-preview:free",
    "🟣 Qwen3.6 27B (denso)":                               "qwen/qwen3.6-27b:free",
    "🟣 Qwen3.5 Plus":                                      "qwen/qwen3.5-plus:free",
    "🟣 Qwen3.5 Omni Plus (multimodal)":                    "qwen/qwen3.5-omni-plus:free",
    "🟣 Qwen3.6 35B A3B (multimodal)":                      "qwen/qwen3.6-35b-a3b:free",
    "🟣 Qwen3.5 Flash":                                     "qwen/qwen3.5-flash:free",
    "🟣 Qwen3.5 397B A17B (vision-language)":               "qwen/qwen3.5-397b-a17b:free",
    "🟣 Qwen3.5 Omni Flash":                                "qwen/qwen3.5-omni-flash:free",
    "🟣 Qwen3 Max":                                         "qwen/qwen3-max:free",
    "🟣 Qwen Plus 0728 (razonamiento híbrido)":             "qwen/qwen-plus-2025-07-28:free",
    "🟣 Qwen3 Coder Plus (código)":                         "qwen/qwen3-coder-plus:free",
    "🟣 Qwen3 VL Plus (visión, OCR)":                       "qwen/qwen3-vl-plus:free",
    "🟣 Qwen3 Omni Flash":                                  "qwen/qwen3-omni-flash:free",
    "🔵 SenseNova 6.8 Flash-Lite (multimodal)":             "sensenova/sensenova-6.8-flash-lite",
    "🔵 SenseNova 6.7 Flash-Lite (multimodal)":             "sensenova/sensenova-6.7-flash-lite",
}
DEFAULT_XKIRO_TEXT_MODEL_ID = XKIRO_MODEL_OPTIONS["🟣 Qwen3.5 Flash"]
XKIRO_FALLBACK_POOL = [
    "qwen/qwen3.8-max:free",
    "minimax/minimax-m2:free",
    "mistralai/mistral-large-2512",
]

SDCLI_PATH   = f"{SDCPP_DIR}/build/bin/sd-cli"
QWENTTS_PATH = f"{QWENTTS_DIR}/build/bin/qwen-tts"
try:
    with open(f"{SDCPP_DIR}/build/SDCLI_BACKEND.txt") as _f:
        SDCLI_BACKEND = _f.read().strip() or "cpu"
except Exception:
    SDCLI_BACKEND = "cpu"

QWEN_TTS_AVAILABLE = os.path.exists(QWENTTS_PATH)

# ═══════════════════════════════════════════════════════════════
# 4.1 / 4.2 — SELECTORES GLOBALES DE PERSONAJE (afectan TODOS los prompts)
# ═══════════════════════════════════════════════════════════════
PERSONAJE_OPCIONES = {
    "Niña":   {"sujeto": "niña",   "el_ella": "ella", "articulo": "la",  "concord_a": "a", "genero": "f", "edad_tipo": "infantil"},
    "Niño":   {"sujeto": "niño",   "el_ella": "él",   "articulo": "el",  "concord_a": "o", "genero": "m", "edad_tipo": "infantil"},
    "Chica":  {"sujeto": "chica",  "el_ella": "ella", "articulo": "la",  "concord_a": "a", "genero": "f", "edad_tipo": "adolescente"},
    "Chico":  {"sujeto": "chico",  "el_ella": "él",   "articulo": "el",  "concord_a": "o", "genero": "m", "edad_tipo": "adolescente"},
    "Mujer":  {"sujeto": "mujer",  "el_ella": "ella", "articulo": "la",  "concord_a": "a", "genero": "f", "edad_tipo": "adulto"},
    "Hombre": {"sujeto": "hombre", "el_ella": "él",   "articulo": "el",  "concord_a": "o", "genero": "m", "edad_tipo": "adulto"},
}

def concordar(base: str, sufijo_f: str, sufijo_m: str, genero: str) -> str:
    """Aplica concordancia de género a una palabra base (p.ej. 'atractiv' -> 'atractiva'/'atractivo')."""
    return f"{base}{sufijo_f if genero == 'f' else sufijo_m}"

def frase_el_ella(genero: str, mayus_inicial: bool = True) -> str:
    palabra = "Ella" if genero == "f" else "Él"
    return palabra if mayus_inicial else palabra.lower()

def concursante_articulado(personaje_key: str) -> str:
    """'La niña' / 'El niño' / 'La mujer' ... para insertar directamente en prompts."""
    p = PERSONAJE_OPCIONES[personaje_key]
    return f"{p['articulo']} {p['sujeto']}"

# ═══════════════════════════════════════════════════════════════
# 3. SISTEMA DE PLANTILLAS — estructura persistente en disco
# ═══════════════════════════════════════════════════════════════
PLANTILLA_SUBDIRS = {
    "jueces_imagenes":        "imagenes/jueces",            # J1..J4 (nunca se editan)
    "escenario_lejano":       "imagenes/escenario_lejano",  # plantilla_plano_lejano_1..3
    "escenario_cercano":      "imagenes/escenario_cercano",  # plantilla_plano_cercano
    "voces_jueces":           "voces/jueces",               # clips fuente para clonar (3-10s)
    "reacciones_primera_mitad": "video/reacciones_primera_mitad",
    "reacciones_segunda_mitad": "video/reacciones_segunda_mitad",
    "reacciones_lavoz": "video/reacciones_lavoz",
}

def crear_plantilla(nombre: str) -> str:
    """Crea la estructura de carpetas de una plantilla nueva. Infinitas plantillas, siempre editables."""
    slug = "".join(c if c.isalnum() or c in "_-" else "_" for c in nombre.strip().lower())
    base = f"{PLANTILLAS_DIR}/{slug}"
    for sub in PLANTILLA_SUBDIRS.values():
        os.makedirs(f"{base}/{sub}", exist_ok=True)
    meta_path = f"{base}/plantilla.json"
    if not os.path.exists(meta_path):
        meta = {
            "nombre": nombre,
            "slug": slug,
            "creada_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "jueces": [{"id": i, "nombre": f"Juez {i}", "voz_clonada": False} for i in range(1, 5)],
            "planos_lejanos_disponibles": 0,
            "reacciones_primera_mitad": [],
            "reacciones_segunda_mitad": [],
            "estilo": "got_talent",
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    _log_cell2(f"[PLANTILLA] Creada/verificada: {slug} → {base}")
    return base

def listar_plantillas() -> list:
    if not os.path.isdir(PLANTILLAS_DIR):
        return []
    out = []
    for slug in sorted(os.listdir(PLANTILLAS_DIR)):
        meta_path = f"{PLANTILLAS_DIR}/{slug}/plantilla.json"
        if os.path.exists(meta_path):
            try:
                meta = json.load(open(meta_path, encoding="utf-8"))
                out.append((meta.get("nombre", slug), slug))
            except Exception:
                out.append((slug, slug))
    return out

def cargar_plantilla(slug: str) -> dict:
    meta_path = f"{PLANTILLAS_DIR}/{slug}/plantilla.json"
    if not os.path.exists(meta_path):
        raise RuntimeError(f"Plantilla '{slug}' no encontrada")
    meta = json.load(open(meta_path, encoding="utf-8"))
    meta["_base_dir"] = f"{PLANTILLAS_DIR}/{slug}"
    return meta

def guardar_plantilla(meta: dict) -> None:
    base = meta.pop("_base_dir", None) or f"{PLANTILLAS_DIR}/{meta['slug']}"
    meta_path = f"{base}/plantilla.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    meta["_base_dir"] = base

# ═══════════════════════════════════════════════════════════════
# 5.2 — PROMPTS OFICIALES DE ETAPA 1 (editables, con placeholders)
# ═══════════════════════════════════════════════════════════════
PROMPT_1_CREACION_BASE = (
    "Créame una imagen de {ART_UNA} {SUJETO} {ATRACTIV} de unos {EDAD} años vestid{CONCORD} "
    "para un concurso de talentos de canto {VESTUARIO}, plano cercano de su torso y rostro, "
    "que esté de frente a la cámara, micrófono en su mano derecha y cerca de su boca, exprecion del rostro tranquilo y relajado, "
    "azul plano, no puede tener microfonos en sus orejas ni audifonos,no puede tener ningun audicular en sus orejas o cualquier elemento tecnologico en sus orejas, máximo realismo, resolución 9:16."
)

PROMPT_2_CUERPO_COMPLETO = (
    "Plano donde se pueda ver todo su cuerpo y zapatos, resolución 9:16."
)

PROMPT_3_INTERCAMBIO_PLANTILLA = (
    "Coloca a {NUEVO_PERSONAJE} de la imagen 1 en el escenario de la imagen 2, elimina a "
    "{PERSONAJE_PLANTILLA} de la imagen 2 y sustitúyela por {NUEVO_PERSONAJE}, no cambies nada "
    "más en la imagen 2, la escala de {NUEVO_PERSONAJE} tiene que ser la misma de el de la de la imagen 2 "
    "para que se vea coherente y realista con la lejanía del plano, mantén el vestuario de "
    "{NUEVO_PERSONAJE} original y el micrófono en la mano derecha. Que se vea muy realista y alta "
    "resolución de detalles."
)

PROMPT_4_CLOSEUP = "Plano más cercano, máximo realismo y detalles."

def render_prompt_1(personaje_key: str, edad: int, vestuario: str) -> str:
    p = PERSONAJE_OPCIONES[personaje_key]
    art_una = "una" if p["genero"] == "f" else "un"
    atractiv = concordar("atractiv", "a", "o", p["genero"])
    concord = concordar("", "a", "o", p["genero"])
    return PROMPT_1_CREACION_BASE.format(
        ART_UNA=art_una, SUJETO=p["sujeto"], ATRACTIV=atractiv,
        EDAD=int(edad), CONCORD=concord, VESTUARIO=vestuario,
    )

def render_prompt_3(personaje_key: str, personaje_plantilla_key: str) -> str:
    nuevo = concursante_articulado(personaje_key)
    plantilla_p = concursante_articulado(personaje_plantilla_key)
    return PROMPT_3_INTERCAMBIO_PLANTILLA.format(
        NUEVO_PERSONAJE=nuevo, PERSONAJE_PLANTILLA=plantilla_p,
    )

# ═══════════════════════════════════════════════════════════════
# 7.5 — PROMPT LIPSYNC DE DIÁLOGOS (Etapas 2 y 4)
# ═══════════════════════════════════════════════════════════════
PROMPT_LIPSYNC_DIALOGO = (
    "{EL_ELLA} hablando, sincronización labial natural y muy realista, no modificar los rasgos "
    "y la apariencia de su rostro. Nunca mira a la cámara y mantiene la mirada al frente siempre, "
    "nunca girar la cabeza ni cambia la mirada, la camara se mantiene estática sin desplasamientos."
)

def render_prompt_lipsync_dialogo(personaje_key: str) -> str:
    p = PERSONAJE_OPCIONES[personaje_key]
    return PROMPT_LIPSYNC_DIALOGO.format(EL_ELLA=frase_el_ella(p["genero"]))

# ═══════════════════════════════════════════════════════════════
# 8.2 — PROMPT DE 5s (plano lejano, doblemente dinámico)
# ═══════════════════════════════════════════════════════════════
PROMPT_5S_PLANO_LEJANO = (
    "{CONCURSANTE_ART} del escenario cantando muy relajada, sus movimientos son muy sutiles y leves, micrófono en su mano derecha cerca de su "
    "boca todo el tiempo, todos en silencio sin aplaudir, prohibido añadir más personas que no estén "
    "en la imagen o personas con escalas grandes que no respeten el plano lejano, la cámara "
    "manteniendo el plano lejano todo el tiempo y se desplaza a {DIRECCION} muy lentamente, no "
    "realices cambios en los letreros de la imagen, que se vea muy real ."
   
)

def render_prompt_5s(personaje_key: str, direccion: str = None) -> str:
    direccion = direccion or random.choice(["izquierda", "derecha"])
    return PROMPT_5S_PLANO_LEJANO.format(
        CONCURSANTE_ART=concursante_articulado(personaje_key).capitalize(),
        DIRECCION=direccion,
    ), direccion

# ═══════════════════════════════════════════════════════════════
# 8.4 — PROMPT LIPSYNC DE CANCIÓN (15s)
# ═══════════════════════════════════════════════════════════════
PROMPT_LIPSYNC_CANCION = (
    "Sincronización labial natural, que el fondo esté animado, máximo realismo, emociones "
    "naturales al cantar."
)

# ═══════════════════════════════════════════════════════════════
# 7.2 / 9 — PROMPTS DE CONTEXTO PARA GUIONES DE DIÁLOGO (xKiro)
#   Entregables creados desde cero según especificación (secciones 7.2 y 9)
# ═══════════════════════════════════════════════════════════════

def build_dialogo_inicial_system_prompt(personaje_key: str, historia_base: str = "") -> str:
    p = PERSONAJE_OPCIONES[personaje_key]
    el_ella = frase_el_ella(p["genero"], mayus_inicial=False)
    return f"""Eres un guionista experto en programas de talento tipo Got Talent / La Voz, especializado en
generar EMPATÍA PROFUNDA y EMOCIÓN para un público mayor de 40 años.

CONTEXTO: Estás escribiendo los DIÁLOGOS INICIALES de una audición, antes de que el concursante
cante. El concursante es {concursante_articulado(personaje_key)}.
{f"Historia personal a incorporar: {historia_base}" if historia_base else ""}

REGLAS OBLIGATORIAS E INVIOLABLES:
1. Idioma: ESPAÑOL, natural y emotivo.
2. DISTRIBUCIÓN DE VOZ: el 65-70% del tiempo total de habla debe ser del CONCURSANTE. Los jueces
   solo hacen preguntas breves (máximo 1-2 frases cada intervención).
3. El concursante se presenta, cuenta su historia personal (dificultades superadas, por qué canta,
   qué significa este momento para {el_ella}), y genera conexión emocional real con el espectador.
4. Los jueces intervienen con preguntas cortas y naturales: "¿Cómo te llamas?", "¿Por qué has venido
   hoy?", "¿Qué vas a cantar?", reacciones breves de sorpresa o ternura.
5. FRAGMENTOS: cada fragmento de texto (hablado por una sola persona) debe poder decirse en MÁXIMO
   15 SEGUNDOS (LTX-Video no puede sincronizar labios en clips más largos). Estima ~2.5 palabras
   por segundo en español hablado pausado y emotivo.
6. DURACIÓN TOTAL: la suma de duraciones estimadas de todos los fragmentos debe rondar los 3.5
   MINUTOS (210 segundos), nunca menos de 120 segundos.
7. TONO: cálido, humano, que genere empatía y conmueva. Nada de humor forzado ni cinismo.
8. Concordancia de género: usa "{el_ella}" y las concordancias correctas para {concursante_articulado(personaje_key)}
   en TODO el guion.
9. FORMATO DE SALIDA: SOLO un JSON válido, sin texto adicional, con esta estructura EXACTA:
{{
  "fragmentos": [
    {{
      "hablante": "concursante" | "juez_1" | "juez_2" | "juez_3" | "juez_4",
      "texto": "texto exacto que dirá, en español",
      "duracion_estimada_s": 12.5,
      "emocion": "nervioso" | "emocionado" | "triste" | "esperanzado" | "agradecido" | "tímido" | "orgulloso"
    }}
  ],
  "duracion_total_estimada_min": 3.5
}}
10. Ningún fragmento puede superar 15.0 segundos estimados. Si una idea es larga, divídela en
    varios fragmentos consecutivos del mismo hablante.
"""

def build_dialogo_final_system_prompt(personaje_key: str) -> str:
    p = PERSONAJE_OPCIONES[personaje_key]
    el_ella = frase_el_ella(p["genero"], mayus_inicial=False)
    return f"""Eres un guionista experto en programas de talento tipo Got Talent / La Voz, especializado en
generar CIERRES EMOTIVOS que conmuevan a un público mayor de 40 años.

CONTEXTO: Estás escribiendo los DIÁLOGOS FINALES de una audición, justo después de que el
concursante ({concursante_articulado(personaje_key)}) ha terminado de cantar. Los jueces van a dar
su veredicto y opinión.

REGLAS OBLIGATORIAS E INVIOLABLES:
1. Idioma: ESPAÑOL, natural y emotivo.
2. DISTRIBUCIÓN DE VOZ: el 70% O MÁS del tiempo total de habla debe ser de LOS JUECES (repartido
   entre juez_1, juez_2, juez_3, juez_4). El concursante solo responde brevemente si es necesario
   (agradecimientos cortos, reacciones de emoción).
3. Los jueces dan su opinión sincera y detallada: valoran la técnica vocal, la emoción transmitida,
   comentan la historia personal del concursante, y generan un CIERRE EMOTIVO que conecte con el
   público (ej. reflexiones sobre la superación, el talento, la segunda oportunidad).
4. Al menos un juez debe mencionar explícitamente la historia personal del concursante para reforzar
   la empatía.
5. FRAGMENTOS: cada fragmento debe poder decirse en MÁXIMO 15 SEGUNDOS. Estima ~2.5 palabras por
   segundo en español hablado pausado y emotivo.
6. DURACIÓN TOTAL: la suma debe rondar los 3.5 MINUTOS (210 segundos), nunca menos de 120 segundos.
7. TONO: cálido, sincero, emotivo, con un cierre que resuene y conmueva, apto para mayores de 40 años.
8. Concordancia de género: usa "{el_ella}" y las concordancias correctas para
   {concursante_articulado(personaje_key)} en TODO el guion.
9. FORMATO DE SALIDA: SOLO un JSON válido, sin texto adicional, MISMA estructura que el guion inicial:
{{
  "fragmentos": [
    {{
      "hablante": "concursante" | "juez_1" | "juez_2" | "juez_3" | "juez_4",
      "texto": "texto exacto que dirá, en español",
      "duracion_estimada_s": 12.5,
      "emocion": "emocionado" | "orgulloso" | "conmovido" | "sorprendido" | "agradecido"
    }}
  ],
  "duracion_total_estimada_min": 3.5
}}
10. Ningún fragmento puede superar 15.0 segundos estimados.
"""

# ═══════════════════════════════════════════════════════════════
# 7.1 — LISTA EXTENSA DE TIPOS DE VOZ PARA QWEN3-TTS (VoiceDesign)
#   Cada entrada genera el string exacto para "--instruct" de qwen-tts
#   (documentado: --instruct "male, young adult, moderate pitch" --lang Spanish)
# ═══════════════════════════════════════════════════════════════
VOICE_DESIGN_CATALOG = {
    # ---------------- NIÑA ----------------
    "Niña — Dulce y tierna":         {"personaje": "Niña", "instruct": "female child, sweet and tender voice, soft pitch, gentle warmth"},
    "Niña — Tímida e inocente":      {"personaje": "Niña", "instruct": "female child, shy and innocent voice, soft and hesitant tone"},
    "Niña — Energética y traviesa":  {"personaje": "Niña", "instruct": "female child, energetic and mischievous voice, playful high pitch"},
    "Niña — Soñadora y poética":     {"personaje": "Niña", "instruct": "female child, dreamy and poetic voice, slow gentle cadence"},
    "Niña — Valiente y decidida":    {"personaje": "Niña", "instruct": "female child, brave and determined voice, firm clear tone"},
    "Niña — Alegre y juguetona":     {"personaje": "Niña", "instruct": "female child, cheerful and playful voice, bright happy tone"},

    # ---------------- NIÑO ----------------
    "Niño — Valiente y aventurero":  {"personaje": "Niño", "instruct": "male child, brave and adventurous voice, confident bright tone"},
    "Niño — Curioso y preguntón":    {"personaje": "Niño", "instruct": "male child, curious and inquisitive voice, quick excited pitch"},
    "Niño — Juguetón y bromista":    {"personaje": "Niño", "instruct": "male child, playful and joking voice, mischievous tone"},
    "Niño — Serio y maduro":         {"personaje": "Niño", "instruct": "male child, serious and mature-sounding voice, calm steady tone"},
    "Niño — Tímido pero expresivo":  {"personaje": "Niño", "instruct": "male child, shy but expressive voice, soft with emotional bursts"},
    "Niño — Enérgico y competitivo": {"personaje": "Niño", "instruct": "male child, energetic and competitive voice, fast excited pitch"},

    # ---------------- CHICA ADOLESCENTE ----------------
    "Chica — Rebelde y segura":      {"personaje": "Chica", "instruct": "female teenager, rebellious and self-assured voice, confident edge"},
    "Chica — Dulce e insegura":      {"personaje": "Chica", "instruct": "female teenager, sweet and insecure voice, soft hesitant tone"},
    "Chica — Apasionada y soñadora": {"personaje": "Chica", "instruct": "female teenager, passionate and dreamy voice, warm emotional tone"},
    "Chica — Sarcástica pero cálida":{"personaje": "Chica", "instruct": "female teenager, sarcastic but warm voice, dry humor undertone"},
    "Chica — Introvertida y profunda":{"personaje": "Chica", "instruct": "female teenager, introverted and thoughtful voice, low soft pitch"},

    # ---------------- CHICO ADOLESCENTE ----------------
    "Chico — Confiado y carismático":{"personaje": "Chico", "instruct": "male teenager, confident and charismatic voice, warm bright tone"},
    "Chico — Torpe pero encantador": {"personaje": "Chico", "instruct": "male teenager, clumsy but charming voice, awkward warm tone"},
    "Chico — Apasionado y romántico":{"personaje": "Chico", "instruct": "male teenager, passionate and romantic voice, soft emotional tone"},
    "Chico — Rebelde y enérgico":    {"personaje": "Chico", "instruct": "male teenager, rebellious and energetic voice, edgy fast pitch"},
    "Chico — Sensible y reflexivo":  {"personaje": "Chico", "instruct": "male teenager, sensitive and reflective voice, slow thoughtful tone"},

    # ---------------- MUJER ----------------
    "Mujer — Cálida y maternal":     {"personaje": "Mujer", "instruct": "adult female, warm and maternal voice, soft nurturing tone, moderate pitch"},
    "Mujer — Profesional y elegante":{"personaje": "Mujer", "instruct": "adult female, professional and elegant voice, clear articulate tone"},
    "Mujer — Sensual y suave":       {"personaje": "Mujer", "instruct": "adult female, sensual and soft voice, low breathy tone"},
    "Mujer — Fuerte y decidida":     {"personaje": "Mujer", "instruct": "adult female, strong and determined voice, firm confident tone"},
    "Mujer — Joven y vibrante":      {"personaje": "Mujer", "instruct": "young adult female, vibrant and lively voice, bright energetic pitch"},
    "Mujer — Madura y sabia":        {"personaje": "Mujer", "instruct": "mature adult female, wise and calm voice, low steady pitch"},
    "Mujer — Nerviosa y emotiva":    {"personaje": "Mujer", "instruct": "adult female, nervous and emotional voice, trembling soft tone"},

    # ---------------- HOMBRE ----------------
    "Hombre — Grave y profundo":     {"personaje": "Hombre", "instruct": "adult male, deep and profound voice, very low pitch, resonant tone"},
    "Hombre — Juvenil y ligero":     {"personaje": "Hombre", "instruct": "young adult male, light and youthful voice, moderate bright pitch"},
    "Hombre — Ronco y curtido":      {"personaje": "Hombre", "instruct": "adult male, rough and weathered voice, gravelly low tone"},
    "Hombre — Elegante y pausado":   {"personaje": "Hombre", "instruct": "adult male, elegant and measured voice, calm articulate tone"},
    "Hombre — Enérgico y firme":     {"personaje": "Hombre", "instruct": "adult male, energetic and firm voice, confident moderate pitch"},
    "Hombre — Emotivo y tembloroso": {"personaje": "Hombre", "instruct": "adult male, emotional and trembling voice, soft unsteady tone"},
    "Hombre — Maduro y sabio":       {"personaje": "Hombre", "instruct": "mature adult male, wise and calm voice, low steady resonant tone"},
}

# Añadir idioma español a todas las entradas (parámetro --lang de qwen-tts)
for _k in VOICE_DESIGN_CATALOG:
    VOICE_DESIGN_CATALOG[_k]["lang"] = "Spanish"

_log_cell2(f"Catálogo de voces (VoiceDesign): {len(VOICE_DESIGN_CATALOG)} variantes en 6 categorías.")

# ═══════════════════════════════════════════════════════════════
# ESTILO VISUAL DEL PROGRAMA (para prompts de escenario/plantilla si se regeneran)
# ═══════════════════════════════════════════════════════════════
AUDICION_STYLE_GUIDE = {
    "resolucion_9_16": ["A", "B"],
    "resolucion_16_9": ["C", "D", "E", "F", "G", "J1", "J2", "J3", "J4"],
    "dim_multiplo": 32,   # OBLIGATORIO por documentación sd.cpp Qwen-Image-2.1
    "cfg_scale_recomendado": 6.0,  # documentado en qwen_image_2.1.md
    "sampling_method": "euler",
}

_log_cell2("\nCELDA 2 COMPLETADA — Sistema de Audiciones configurado.")
_log_cell2(f"sd-cli: {SDCLI_PATH} (existe={os.path.exists(SDCLI_PATH)}) | backend: {SDCLI_BACKEND}")
_log_cell2(f"qwen-tts: {QWENTTS_PATH} (disponible={QWEN_TTS_AVAILABLE})")
_log_cell2(f"ComfyUI: {COMFYUI_DIR} (existe={os.path.isdir(COMFYUI_DIR)})")
_log_cell2(f"Plantillas existentes: {listar_plantillas()}")