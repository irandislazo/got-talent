# ═══════════════════════════════════════════════════════════════
# CELDA 4: ESTRUCTURAS DE PROYECTO (VIDEO) + TIMELINE JSON + RENDER HTML
# ═══════════════════════════════════════════════════════════════
import datetime
import random

def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9áéíóúüñ_-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:80] or "video"

def now_slug() -> str:
    return __import__("datetime").datetime.utcnow().strftime("%Y%m%d_%H%M%S")   # robusto: otras celdas rebindean `datetime` a la clase

@dataclass
class TimelineClip:
    clip_id: str
    tipo: str                 # "dialogo" | "plano_lejano_5s" | "lipsync_15s" | "reaccion"
    hablante: str             # concursante/juez_1..4/""
    dur_s: float
    path_video: str
    path_audio_ref: str = ""  # si aplica (p.ej. lipsync con audio de canción recortado)
    estado: str = "pendiente" # pendiente/generando/completado/error
    meta: dict = None

def timeline_to_json(clips: List[TimelineClip]) -> list:
    out = []
    t = 0.0
    for c in clips:
        d = float(c.dur_s)
        out.append({
            "clip_id": c.clip_id,
            "tipo": c.tipo,
            "hablante": c.hablante,
            "start": t,
            "end": t + d,
            "dur_s": d,
            "path_video": c.path_video,
            "path_audio_ref": c.path_audio_ref,
            "estado": c.estado,
            "meta": c.meta or {},
        })
        t += d
    return out

def render_timeline_html(timeline_items: list, width_px: int = 1100) -> str:
    """
    Timeline estilo editor: una sola pista (simplificado, pero muy claro),
    coloreado por tipo.
    """
    if not timeline_items:
        return "<div style='font-family:monospace'>Timeline vacío.</div>"

    total = float(timeline_items[-1]["end"]) if timeline_items else 1.0
    if total <= 0:
        total = 1.0

    colors = {
        "dialogo": "#4e79a7",
        "plano_lejano_5s": "#59a14f",
        "lipsync_15s": "#f28e2b",
        "reaccion": "#af7aa1",
    }

    bars = []
    for it in timeline_items:
        start = float(it["start"])
        end   = float(it["end"])
        left  = int((start / total) * width_px)
        w     = max(2, int(((end - start) / total) * width_px))
        tipo  = it["tipo"]
        c     = colors.get(tipo, "#999")
        label = f"{tipo} | {it.get('hablante','')} | {it['dur_s']:.2f}s | {it.get('estado','')}"
        bars.append(
            f"<div title='{label}' "
            f"style='position:absolute;left:{left}px;top:0px;width:{w}px;height:34px;"
            f"background:{c};border:1px solid #111;border-radius:6px;overflow:hidden;"
            f"color:#111;font-family:monospace;font-size:11px;line-height:14px;padding:3px;'>"
            f"{tipo}<br>{it.get('hablante','')}"
            f"</div>"
        )
    html = (
        "<div style='font-family:monospace;margin:4px 0;'>"
        f"<div style='position:relative;width:{width_px}px;height:38px;background:#222;border-radius:8px;border:1px solid #444;'>"
        + "".join(bars) +
        "</div>"
        f"<div style='color:#333;font-size:12px;margin-top:4px;'>Duración total: {total:.2f}s</div>"
        "</div>"
    )
    return html

# ---------------- VideoProject (persistente) ----------------
class VideoProject:
    def __init__(self, video_dir: str):
        self.video_dir = video_dir
        os.makedirs(video_dir, exist_ok=True)

        self.paths = {
            "estado":  os.path.join(video_dir, "estado.json"),
            "prompts": os.path.join(video_dir, "prompts.json"),
            "params":  os.path.join(video_dir, "params.json"),
            "timeline": os.path.join(video_dir, "timeline.json"),
        }
        self.estado = {}
        self.prompts = {}
        self.params = {}
        self.timeline_items = []

        self._load()

    def _load(self):
        for k, p in self.paths.items():
            if os.path.exists(p):
                try:
                    data = json.load(open(p, "r", encoding="utf-8"))
                    if k == "estado": self.estado = data
                    if k == "prompts": self.prompts = data
                    if k == "params": self.params = data
                    if k == "timeline": self.timeline_items = data
                except Exception:
                    pass

    def save(self):
        json.dump(self.estado, open(self.paths["estado"], "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump(self.prompts, open(self.paths["prompts"], "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump(self.params, open(self.paths["params"], "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump(self.timeline_items, open(self.paths["timeline"], "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def create_new_video_project(template_slug: str, personaje_key: str) -> VideoProject:
    vid_slug = f"{now_slug()}_{_slug(template_slug)}_{_slug(personaje_key)}"
    vd = os.path.join(VIDEOS_DIR, vid_slug)
    for sub in ["etapa1_imagenes","etapa2_dialogos","etapa3_cancion","etapa4_final","tmp","comfy_input","comfy_output"]:
        os.makedirs(os.path.join(vd, sub), exist_ok=True)
    vp = VideoProject(vd)
    vp.estado.setdefault("aprobados", {})
    vp.save()
    return vp

log("CELDA 4 COMPLETADA.")