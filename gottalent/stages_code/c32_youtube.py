# ═══════════════════════════════════════════════════════════════
# CELDA FINAL: 📺 SUBIDA A YOUTUBE (multi-canal) + xKiro
#  - Varios canales (cada uno con su propia API/OAuth) guardados en JSON
#  - xKiro genera títulos (eliges 1), descripción, tags y hashtags
#  - Todo editable antes de subir; ajustes por canal que se autorrellenan
#  - Subida resumible, miniatura, lista de reproducción, programación
# No instala nada ni modifica otras celdas.
# ═══════════════════════════════════════════════════════════════
import os, sys, json, re, time, mimetypes, subprocess, threading, tempfile
from datetime import datetime, timezone
import requests
import gradio as gr

CONFIG_DIR  = "/kaggle/working/yt_uploader"          # ← contiene secretos/tokens
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")
os.makedirs(CONFIG_DIR, exist_ok=True)
_CFG_LOCK = threading.Lock()

OAUTH_SCOPE = "https://www.googleapis.com/auth/youtube"
YT_API      = "https://www.googleapis.com/youtube/v3"
CHUNK       = 8 * 1024 * 1024                         # múltiplo de 256 KiB

# ───────────────────────── Config persistente ─────────────────────────
def _load_cfg():
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    cfg.setdefault("channels", {})
    cfg.setdefault("history", [])
    cfg.setdefault("last_channel", "")
    cfg.setdefault("last_client", {})
    return cfg

def _save_cfg(cfg):
    with _CFG_LOCK:
        tmp = CONFIG_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        os.replace(tmp, CONFIG_PATH)
        try: os.chmod(CONFIG_PATH, 0o600)
        except Exception: pass

DEFAULT_PROFILE = {
    "privacy": "private", "category": "22", "language": "es",
    "tags": "", "footer": "",
    "kids": False, "synthetic": True, "embeddable": True, "stats": True,
    "license": "youtube", "notify": False, "playlist": "",
    "ai_context": "", "timezone": "UTC", "ai_model": "", "n_titles": 6,
}
PROF_KEYS = ["privacy", "category", "language", "tags", "footer", "kids", "synthetic",
             "embeddable", "stats", "license", "notify", "playlist", "ai_context",
             "timezone", "ai_model", "n_titles"]

def _profile(alias):
    p = dict(DEFAULT_PROFILE)
    p.update(_load_cfg()["channels"].get(alias, {}).get("defaults", { }))
    return p

def _ch_choices():
    cfg = _load_cfg()
    return [(f"{a} — {c.get('channel_title') or 'sin verificar'}", a) for a, c in cfg["channels"].items()]

# ───────────────────────── Utilidades ─────────────────────────
def _clean_text(s):
    return (s or "").replace("<", "").replace(">", "").strip()

def _parse_tags(s, limit=450):
    raw = re.split(r"[,\n]", s or "")
    out, seen, total = [], set(), 0
    for t in raw:
        t = t.strip().lstrip("#").strip()
        t = _clean_text(t)
        if not t or t.lower() in seen:
            continue
        cost = len(t) + (2 if " " in t else 0) + 1
        if total + cost > limit:
            break
        seen.add(t.lower()); out.append(t); total += cost
    return out

def _yt_err(r):
    try:
        e = r.json().get("error", {})
        reasons = ",".join(x.get("reason", "") for x in e.get("errors", []) if isinstance(x, dict))
        return f"HTTP {r.status_code}: {e.get('message', r.text[:300])} [{reasons}]"
    except Exception:
        return f"HTTP {r.status_code}: {r.text[:300]}"

_TOK = {}
def _token(alias):
    ch = _load_cfg()["channels"].get(alias)
    if not ch:
        raise RuntimeError(f"Canal '{alias}' no encontrado.")
    c = _TOK.get(alias)
    if c and c["exp"] > time.time() + 90:
        return c["tok"]
    r = requests.post("https://oauth2.googleapis.com/token", data={
        "client_id": ch["client_id"], "client_secret": ch["client_secret"],
        "refresh_token": ch["refresh_token"], "grant_type": "refresh_token"}, timeout=30)
    j = r.json()
    if "access_token" not in j:
        raise RuntimeError(f"No se pudo renovar el token de '{alias}': {j.get('error')} "
                           f"{j.get('error_description', '')}. Reconecta el canal en la pestaña 🔑.")
    _TOK[alias] = {"tok": j["access_token"], "exp": time.time() + int(j.get("expires_in", 3500))}
    return j["access_token"]

def _auth(alias):
    return {"Authorization": f"Bearer {_token(alias)}"}

def _channel_info(alias):
    r = requests.get(f"{YT_API}/channels", params={"part": "snippet", "mine": "true"},
                     headers=_auth(alias), timeout=30)
    if r.status_code != 200:
        raise RuntimeError(_yt_err(r))
    items = r.json().get("items", [])
    if not items:
        return None
    return {"id": items[0]["id"], "title": items[0]["snippet"]["title"]}

# ───────────────────────── xKiro ─────────────────────────
def _xk_models():
    m = globals().get("XKIRO_MODEL_OPTIONS")
    if isinstance(m, dict) and m:
        return dict(m)
    d = globals().get("DEFAULT_XKIRO_TEXT_MODEL_ID")
    return {str(d): d} if d else {}

def _xk_json(system_prompt, user_prompt, model_id):
    fn = globals().get("call_xkiro_json")
    if callable(fn):
        return fn(system_prompt, user_prompt, model_id=model_id, temperature=0.8, max_tokens=3500)
    from openai import OpenAI
    key = os.environ.get("XKIRO_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Falta XKIRO_API_KEY.")
    cli = OpenAI(api_key=key, base_url=globals().get("XKIRO_BASE_URL", "https://api.xkiro.com/v1"))
    resp = cli.chat.completions.create(
        model=model_id, temperature=0.8, max_tokens=3500,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
    txt = (resp.choices[0].message.content or "").strip()
    txt = re.sub(r"^```(?:json)?", "", txt).rstrip("`").strip()
    return json.loads(txt)

def ai_generate(prompt, ctx, model_name, n, tags_cur, footer, lang):
    if not (prompt or "").strip():
        raise gr.Error("Escribe el prompt / idea del video.")
    models = _xk_models()
    model_id = models.get(model_name) or globals().get("DEFAULT_XKIRO_TEXT_MODEL_ID")
    if not model_id:
        raise gr.Error("No hay modelo xKiro disponible (ejecuta la celda de xKiro primero).")
    n = int(n or 6)
    system = ("Eres un experto en SEO y estrategia de YouTube. "
              "Respondes ÚNICAMENTE con JSON válido, sin markdown ni texto extra.")
    user = f"""IDEA / CONTEXTO DEL VIDEO:
{prompt.strip()}

CONTEXTO DEL CANAL (tono, público, estilo):
{(ctx or '').strip() or 'No especificado'}

IDIOMA DE SALIDA (código): {lang or 'es'}

Genera:
- "titles": {n} títulos distintos entre sí (curiosidad, directo, emocional, SEO...). Ideal ≤ 70 caracteres, NUNCA más de 100. Sin clickbait falso, sin los símbolos < >.
- "description": descripción lista para publicar. Las 2 primeras líneas son un gancho con palabras clave. 150–300 palabras, párrafos cortos. NO inventes enlaces, datos ni marcas de tiempo. Sin hashtags dentro del texto.
- "tags": 12 a 20 etiquetas relevantes (sin #).
- "hashtags": 3 hashtags (con #).

Formato exacto:
{{"titles": ["..."], "description": "...", "tags": ["..."], "hashtags": ["#..."]}}"""
    try:
        data = _xk_json(system, user, model_id)
    except Exception as e:
        raise gr.Error(f"xKiro falló: {e}")
    titles, seen = [], set()
    for t in data.get("titles", []):
        t = _clean_text(str(t)).strip('"“”')[:100]
        if t and t.lower() not in seen:
            seen.add(t.lower()); titles.append(t)
    desc = _clean_text(str(data.get("description", "")))
    if footer and footer.strip():
        desc += "\n\n" + _clean_text(footer)
    tags_ai = data.get("tags", [])
    if isinstance(tags_ai, str):
        tags_ai = tags_ai.split(",")
    merged = _parse_tags(", ".join([*(_parse_tags(tags_cur)), *map(str, tags_ai)]))
    hs = []
    for h in data.get("hashtags", [])[:3]:
        h = "#" + re.sub(r"\s+", "", str(h)).lstrip("#")
        if len(h) > 1: hs.append(h)
    if hs:
        desc += "\n\n" + " ".join(hs)
    return (gr.Radio(choices=titles, value=None), desc, ", ".join(merged),
            f"✅ {len(titles)} títulos generados con {model_name or model_id}. Elige uno abajo.")

def _skip():
    return gr.skip() if hasattr(gr, "skip") else gr.update()

def pick_title(t):
    return t if t else _skip()

# ───────────────────────── Vídeos ─────────────────────────
_SKIP_DIRS = {"Wan2GP", "stable-diffusion.cpp", "qwentts.cpp", "ComfyUI", "env_snapshot",
              "__pycache__", "build", "models", "ckpts", "node_modules", "yt_uploader"}
_VID_EXT = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")

def _scan_videos(limit=80):
    roots = []
    for g in ("VIDEOS_DIR", "OUTPUT_DIR"):
        v = globals().get(g)
        if v and os.path.isdir(v): roots.append(v)
    roots.append("/kaggle/working/outputs")
    found = {}
    for root in roots:
        if not os.path.isdir(root): continue
        for dp, dn, fn in os.walk(root):
            dn[:] = [d for d in dn if d not in _SKIP_DIRS and not d.startswith(".")]
            for f in fn:
                if f.lower().endswith(_VID_EXT):
                    full = os.path.join(dp, f)
                    try: found[full] = (os.path.getmtime(full), os.path.getsize(full))
                    except OSError: pass
    items = sorted(found.items(), key=lambda kv: kv[1][0], reverse=True)[:limit]
    return [(f"{os.path.relpath(p, '/kaggle/working')}  ·  {sz/1024**2:.0f} MB  ·  "
             f"{time.strftime('%d/%m %H:%M', time.localtime(mt))}", p) for p, (mt, sz) in items]

def _resolve_video(sel, man, upl):
    upl = getattr(upl, "name", upl)
    for p in (upl, (man or "").strip().strip("\"'"), sel):
        if p and isinstance(p, str) and os.path.isfile(p):
            return p
    return None

def video_info(sel, man, upl):
    p = _resolve_video(sel, man, upl)
    if not p: return "—"
    size = os.path.getsize(p) / 1024**2
    dur = ""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=noprint_wrappers=1:nokey=1", p],
                           capture_output=True, text=True, timeout=30)
        d = float(r.stdout.strip()); dur = f" · {int(d//60)}:{int(d%60):02d} min"
    except Exception: pass
    return f"📁 {p}\n{size:.1f} MB{dur}"

def extract_frame(sel, man, upl, sec):
    p = _resolve_video(sel, man, upl)
    if not p: raise gr.Error("Elige un video primero.")
    out = os.path.join(tempfile.gettempdir(), f"yt_thumb_{int(time.time())}.jpg")
    r = subprocess.run(["ffmpeg", "-y", "-ss", str(float(sec or 0)), "-i", p, "-frames:v", "1",
                        "-vf", "scale=1280:-2", "-q:v", "3", out], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(out):
        raise gr.Error("No se pudo extraer el frame.")
    return out

def _prep_thumb(path):
    path = getattr(path, "name", path)
    if not path or not os.path.isfile(path): return None
    ok_ext = path.lower().endswith((".jpg", ".jpeg", ".png"))
    if ok_ext and os.path.getsize(path) <= 2 * 1024 * 1024:
        return path
    out = os.path.join(tempfile.gettempdir(), f"yt_thumb_conv_{int(time.time())}.jpg")
    subprocess.run(["ffmpeg", "-y", "-i", path, "-vf", "scale=1280:-2", "-q:v", "4", out],
                   capture_output=True, text=True)
    return out if os.path.isfile(out) and os.path.getsize(out) <= 2 * 1024 * 1024 else None

# ───────────────────────── Subida a YouTube ─────────────────────────
def _resumable_upload(alias, path, body, notify):
    size = os.path.getsize(path)
    if size == 0: raise RuntimeError("El archivo de video está vacío.")
    mime = mimetypes.guess_type(path)[0] or "video/*"
    params = {"uploadType": "resumable", "part": "snippet,status",
              "notifySubscribers": "true" if notify else "false"}
    def _init():
        h = dict(_auth(alias))
        h.update({"Content-Type": "application/json; charset=UTF-8",
                  "X-Upload-Content-Length": str(size), "X-Upload-Content-Type": mime})
        return requests.post("https://www.googleapis.com/upload/youtube/v3/videos",
                             params=params, headers=h, data=json.dumps(body), timeout=60)
    r = _init()
    if r.status_code == 400 and "containsSyntheticMedia" in r.text:
        body["status"].pop("containsSyntheticMedia", None)
        r = _init()
    if r.status_code != 200:
        raise RuntimeError(_yt_err(r))
    loc = r.headers["Location"]
    offset, fails = 0, 0
    with open(path, "rb") as f:
        while offset < size:
            f.seek(offset)
            chunk = f.read(CHUNK)
            end = offset + len(chunk) - 1
            h = dict(_auth(alias))
            h.update({"Content-Length": str(len(chunk)),
                      "Content-Range": f"bytes {offset}-{end}/{size}", "Content-Type": mime})
            rr = None
            try:
                rr = requests.put(loc, headers=h, data=chunk, timeout=300)
            except requests.RequestException:
                pass
            if rr is not None and rr.status_code in (200, 201):
                yield 100, rr.json(); return
            if rr is not None and rr.status_code == 308:
                rng = rr.headers.get("Range")
                offset = int(rng.split("-")[-1]) + 1 if rng else 0
                fails = 0
                yield int(offset * 100 / size), None
                continue
            if rr is not None and rr.status_code == 401:
                _TOK.pop(alias, None)
            fails += 1
            if fails > 6 or (rr is not None and rr.status_code in (400, 403, 404)):
                raise RuntimeError(_yt_err(rr) if rr is not None else "Error de red persistente")
            time.sleep(min(30, 2 ** fails))
            try:   # preguntar hasta dónde llegó el servidor
                h2 = dict(_auth(alias))
                h2.update({"Content-Length": "0", "Content-Range": f"bytes */{size}"})
                q = requests.put(loc, headers=h2, timeout=60)
                if q.status_code in (200, 201):
                    yield 100, q.json(); return
                if q.status_code == 308:
                    rng = q.headers.get("Range")
                    offset = int(rng.split("-")[-1]) + 1 if rng else 0
            except requests.RequestException:
                pass
    raise RuntimeError("La subida terminó sin respuesta final de YouTube.")

def _parse_sched(s, tz):
    s = (s or "").strip()
    if not s: return None
    from zoneinfo import ZoneInfo
    try: z = ZoneInfo((tz or "UTC").strip() or "UTC")
    except Exception: raise ValueError(f"Zona horaria inválida: {tz}")
    dt = None
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M"):
        try: dt = datetime.strptime(s, fmt); break
        except ValueError: pass
    if not dt: raise ValueError("Fecha programada inválida. Formato: 2026-10-15 18:30")
    dt = dt.replace(tzinfo=z).astimezone(timezone.utc)
    if dt <= datetime.now(timezone.utc): raise ValueError("La fecha programada debe estar en el futuro.")
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

def _history_md():
    h = _load_cfg()["history"][-8:][::-1]
    if not h: return "_Sin subidas registradas._"
    return "\n".join(f"- {x['t']} · **{x['channel']}** · [{x['title']}](https://youtu.be/{x['id']}) · {x['privacy']}" for x in h)

def do_upload(alias, vsel, vman, vupl, title, desc, tags, privacy, cat, lang, kids, synth,
              emb, stats, lic, notify, playlist, sched, tz, thumb):
    log = []
    def out(): return "\n".join(log), _history_md()
    try:
        if not alias or alias not in _load_cfg()["channels"]:
            raise ValueError("Elige un canal (o conecta uno en la pestaña 🔑).")
        path = _resolve_video(vsel, vman, vupl)
        if not path: raise ValueError("Elige un video (lista, ruta manual o subida).")
        title = _clean_text(title)
        if not title: raise ValueError("Falta el título.")
        if len(title) > 100: raise ValueError(f"El título tiene {len(title)} caracteres (máx. 100).")
        desc = _clean_text(desc)
        if len(desc.encode("utf-8")) > 5000: raise ValueError("La descripción supera 5000 bytes.")
        tag_list = _parse_tags(tags)
        publish_at = _parse_sched(sched, tz)
        if publish_at: privacy = "private"
        body = {
            "snippet": {"title": title, "description": desc, "tags": tag_list,
                        "categoryId": str(cat), "defaultLanguage": lang, "defaultAudioLanguage": lang},
            "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": bool(kids),
                       "embeddable": bool(emb), "publicStatsViewable": bool(stats),
                       "license": lic, "containsSyntheticMedia": bool(synth)},
        }
        if publish_at: body["status"]["publishAt"] = publish_at
        log.append(f"🚀 Subiendo a «{alias}»: {os.path.basename(path)}"
                   + (f"\n🕒 Programado (UTC): {publish_at}" if publish_at else ""))
        pi = len(log); log.append("⬆️ 0%"); yield out()
        last, resp = -1, None
        for pct, r in _resumable_upload(alias, path, body, notify):
            if pct != last:
                log[pi] = f"⬆️ {pct}%"; last = pct; yield out()
            if r: resp = r
        vid = resp["id"]
        log.append(f"✅ Video subido: https://youtu.be/{vid}")
        tp = _prep_thumb(thumb)
        if thumb and not tp:
            log.append("⚠️ Miniatura omitida (debe ser JPG/PNG ≤ 2 MB).")
        elif tp:
            try:
                h = dict(_auth(alias)); h["Content-Type"] = mimetypes.guess_type(tp)[0] or "image/jpeg"
                with open(tp, "rb") as fh:
                    rt = requests.post("https://www.googleapis.com/upload/youtube/v3/thumbnails/set",
                                       params={"videoId": vid, "uploadType": "media"},
                                       headers=h, data=fh.read(), timeout=120)
                log.append("🖼️ Miniatura aplicada." if rt.status_code == 200
                           else f"⚠️ Miniatura no aplicada: {_yt_err(rt)}")
            except Exception as e:
                log.append(f"⚠️ Miniatura no aplicada: {e}")
        pl = (playlist or "").strip()
        if pl:
            try:
                rp = requests.post(f"{YT_API}/playlistItems", params={"part": "snippet"},
                                   headers=_auth(alias), timeout=60,
                                   json={"snippet": {"playlistId": pl,
                                                     "resourceId": {"kind": "youtube#video", "videoId": vid}}})
                log.append("📚 Añadido a la lista de reproducción." if rp.status_code == 200
                           else f"⚠️ No se añadió a la lista: {_yt_err(rp)}")
            except Exception as e:
                log.append(f"⚠️ No se añadió a la lista: {e}")
        st = resp.get("status", {})
        if st.get("privacyStatus") == "private" and privacy != "private":
            log.append("⚠️ YouTube dejó el video en PRIVADO: el proyecto de API aún no está auditado "
                       "(restricción de YouTube para proyectos nuevos). Ver notas.")
        cfg = _load_cfg()
        cfg["history"].append({"t": time.strftime("%Y-%m-%d %H:%M"), "channel": alias,
                               "title": title, "id": vid, "privacy": st.get("privacyStatus", privacy)})
        cfg["history"] = cfg["history"][-30:]
        _save_cfg(cfg)
        yield out()
    except Exception as e:
        log.append(f"❌ {e}")
        yield out()

# ───────────────────────── Canales (OAuth dispositivo) ─────────────────────────
def dev_start(alias, cid, csec, state):
    alias, cid, csec = (alias or "").strip(), (cid or "").strip(), (csec or "").strip()
    if not (alias and cid and csec):
        raise gr.Error("Rellena alias, Client ID y Client Secret.")
    r = requests.post("https://oauth2.googleapis.com/device/code",
                      data={"client_id": cid, "scope": OAUTH_SCOPE}, timeout=30)
    j = r.json()
    if "device_code" not in j:
        raise gr.Error(f"Google rechazó la solicitud: {j.get('error')} {j.get('error_description', '')}. "
                       "¿El cliente es de tipo «TV y dispositivos de entrada limitada»?")
    state = {"alias": alias, "cid": cid, "csec": csec, "device_code": j["device_code"],
             "interval": int(j.get("interval", 5)), "exp": time.time() + int(j.get("expires_in", 1800))}
    md = (f"### Autoriza el canal «{alias}»\n"
          f"1. Abre **{j.get('verification_url', 'https://www.google.com/device')}**\n"
          f"2. Introduce el código: **`{j['user_code']}`**\n"
          f"3. Elige la cuenta **y el canal correcto** (si tienes canales de marca) y acepta.\n"
          f"4. Vuelve aquí y pulsa **✅ Ya autoricé**.")
    return md, state

def _channels_md():
    cfg = _load_cfg()
    if not cfg["channels"]: return "_Aún no hay canales conectados._"
    return "\n".join(f"- **{a}** → {c.get('channel_title') or 'sin verificar'} "
                     f"`{c.get('channel_id', '')}`" for a, c in cfg["channels"].items())

def dev_finish(state):
    if not state: raise gr.Error("Primero pulsa «Pedir código».")
    interval, tok = state["interval"], None
    while time.time() < state["exp"]:
        j = requests.post("https://oauth2.googleapis.com/token", data={
            "client_id": state["cid"], "client_secret": state["csec"],
            "device_code": state["device_code"],
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code"}, timeout=30).json()
        if "access_token" in j: tok = j; break
        err = j.get("error")
        if err == "authorization_pending": time.sleep(interval); continue
        if err == "slow_down": interval += 5; time.sleep(interval); continue
        raise gr.Error(f"Autorización fallida: {err} {j.get('error_description', '')}")
    if not tok: raise gr.Error("El código expiró. Pide uno nuevo.")
    if not tok.get("refresh_token"):
        raise gr.Error("Google no devolvió refresh_token. Revoca el acceso de la app en tu cuenta y repite.")
    cfg = _load_cfg()
    alias = state["alias"]
    old = cfg["channels"].get(alias, {})
    cfg["channels"][alias] = {"client_id": state["cid"], "client_secret": state["csec"],
                              "refresh_token": tok["refresh_token"],
                              "channel_title": "", "channel_id": "",
                              "defaults": old.get("defaults", {})}
    cfg["last_client"] = {"client_id": state["cid"], "client_secret": state["csec"]}
    cfg["last_channel"] = alias
    _save_cfg(cfg); _TOK.pop(alias, None)
    msg = f"✅ Canal «{alias}» conectado."
    try:
        info = _channel_info(alias)
        if info:
            cfg = _load_cfg()
            cfg["channels"][alias]["channel_title"] = info["title"]
            cfg["channels"][alias]["channel_id"] = info["id"]
            _save_cfg(cfg)
            msg = f"✅ Canal «{alias}» conectado → **{info['title']}**"
        else:
            msg += " ⚠️ La cuenta elegida no tiene canal de YouTube."
    except Exception as e:
        msg += f" ⚠️ No se pudo leer el canal: {e}"
    return (msg, gr.Dropdown(choices=_ch_choices(), value=alias),
            gr.Dropdown(choices=_ch_choices(), value=alias), _channels_md(), None)

def verify_channel(alias):
    if not alias: raise gr.Error("Elige un canal.")
    info = _channel_info(alias)
    if not info: return "⚠️ Token válido pero la cuenta no tiene canal."
    cfg = _load_cfg()
    cfg["channels"][alias]["channel_title"] = info["title"]
    cfg["channels"][alias]["channel_id"] = info["id"]
    _save_cfg(cfg)
    return f"✅ «{alias}» funciona → {info['title']} ({info['id']})"

def delete_channel(alias, confirm):
    if not alias: raise gr.Error("Elige un canal.")
    if not confirm: raise gr.Error("Marca la casilla de confirmación.")
    cfg = _load_cfg()
    cfg["channels"].pop(alias, None)
    if cfg["last_channel"] == alias: cfg["last_channel"] = ""
    _save_cfg(cfg); _TOK.pop(alias, None)
    ch = _ch_choices()
    first = ch[0][1] if ch else None
    return (f"🗑️ Canal «{alias}» eliminado.", gr.Dropdown(choices=ch, value=first),
            gr.Dropdown(choices=ch, value=first), _channels_md())

# ───────────────────────── Perfiles por canal ─────────────────────────
def on_channel_change(ch):
    cfg = _load_cfg()
    if ch and ch in cfg["channels"]:
        cfg["last_channel"] = ch; _save_cfg(cfg)
    p = _profile(ch)
    models = list(_xk_models().keys())
    p["ai_model"] = p["ai_model"] if p["ai_model"] in models else (models[0] if models else None)
    return [p[k] for k in PROF_KEYS]

def save_defaults(ch, *vals):
    if not ch or ch not in _load_cfg()["channels"]:
        raise gr.Error("Elige un canal primero.")
    d = dict(zip(PROF_KEYS, vals))
    d["n_titles"] = int(d.get("n_titles") or 6)
    cfg = _load_cfg()
    cfg["channels"][ch]["defaults"] = d
    _save_cfg(cfg)
    return f"💾 Predeterminados guardados para «{ch}» (se autorrellenarán en los próximos videos)."

def load_playlists(ch, current):
    if not ch: raise gr.Error("Elige un canal.")
    r = requests.get(f"{YT_API}/playlists", params={"part": "snippet", "mine": "true", "maxResults": 50},
                     headers=_auth(ch), timeout=30)
    if r.status_code != 200: raise gr.Error(_yt_err(r))
    items = [("— ninguna —", "")] + [(i["snippet"]["title"], i["id"]) for i in r.json().get("items", [])]
    return gr.Dropdown(choices=items, value=current or "")

# ───────────────────────── UI ─────────────────────────
CATEGORIES = [("Cine y animación", "1"), ("Autos", "2"), ("Música", "10"), ("Mascotas", "15"),
              ("Deportes", "17"), ("Viajes", "19"), ("Videojuegos", "20"), ("Gente y blogs", "22"),
              ("Comedia", "23"), ("Entretenimiento", "24"), ("Noticias", "25"),
              ("Cómo hacer / estilo", "26"), ("Educación", "27"), ("Ciencia y tecnología", "28")]
LANGS = [("Español", "es"), ("English", "en"), ("Português", "pt"), ("Français", "fr"),
         ("Deutsch", "de"), ("Italiano", "it"), ("日本語", "ja"), ("한국어", "ko")]

_cfg0 = _load_cfg()
_ch0 = _ch_choices()
_last0 = _cfg0["last_channel"] if _cfg0["last_channel"] in _cfg0["channels"] else (_ch0[0][1] if _ch0 else None)
_models0 = list(_xk_models().keys())

with gr.Blocks(title="Subida a YouTube") as demo_yt:
    gr.Markdown("# 📺 Subida a YouTube — multi-canal + xKiro")
    with gr.Tabs():
        # ───────── TAB SUBIR ─────────
        with gr.Tab("📤 Subir video"):
            with gr.Row():
                ch_dd = gr.Dropdown(choices=_ch_choices(), value=_last0, label="Canal", scale=3)
            gr.Markdown("### 1) Video")
            with gr.Row():
                vid_dd = gr.Dropdown(choices=_scan_videos(), value=None, label="Videos encontrados en el servidor", scale=4)
                vid_ref = gr.Button("🔄", scale=1)
            with gr.Row():
                vid_man = gr.Textbox(label="…o ruta manual", placeholder="/kaggle/working/.../video.mp4")
                vid_up = gr.File(label="…o subir archivo (lento por el túnel)", type="filepath", file_types=["video"])
            vid_info = gr.Textbox(label="Info del video", interactive=False, lines=2)

            gr.Markdown("### 2) Texto con xKiro")
            with gr.Accordion("🤖 Generar títulos, descripción y tags", open=True):
                prompt_tb = gr.Textbox(label="Prompt / idea del video", lines=4,
                                       placeholder="Ej: Final de Got Talent con cantante tímida que sorprende a los jueces con una balada…")
                ctx_tb = gr.Textbox(label="Contexto del canal (se guarda por canal)", lines=2,
                                    placeholder="Tono, público objetivo, estilo, palabras clave fijas…")
                with gr.Row():
                    model_dd = gr.Dropdown(choices=_models0, value=(_models0[0] if _models0 else None),
                                           label="Modelo xKiro", allow_custom_value=True)
                    ntitles_sl = gr.Slider(3, 10, value=6, step=1, label="Nº de títulos")
                gen_btn = gr.Button("✨ Generar con xKiro", variant="primary")
                ai_status = gr.Markdown()
                titles_rd = gr.Radio(choices=[], label="Títulos sugeridos (elige uno → pasa al campo Título)")
            title_tb = gr.Textbox(label="Título (máx. 100)", max_lines=2)
            desc_tb = gr.Textbox(label="Descripción (máx. 5000 bytes)", lines=12)
            tags_tb = gr.Textbox(label="Etiquetas (separadas por coma)", lines=2)

            gr.Markdown("### 3) Ajustes de publicación")
            with gr.Row():
                privacy_rd = gr.Radio([("Privado", "private"), ("No listado", "unlisted"), ("Público", "public")],
                                      value="private", label="Visibilidad")
                category_dd = gr.Dropdown(CATEGORIES, value="22", label="Categoría")
                lang_dd = gr.Dropdown(LANGS, value="es", label="Idioma", allow_custom_value=True)
            with gr.Row():
                kids_cb = gr.Checkbox(label="Hecho para niños", value=False)
                synth_cb = gr.Checkbox(label="Contenido alterado/sintético (IA)", value=True)
                embed_cb = gr.Checkbox(label="Permitir insertar", value=True)
                stats_cb = gr.Checkbox(label="Estadísticas públicas", value=True)
                notify_cb = gr.Checkbox(label="Notificar a suscriptores", value=False)
            license_rd = gr.Radio([("YouTube estándar", "youtube"), ("Creative Commons", "creativeCommon")],
                                  value="youtube", label="Licencia")
            with gr.Row():
                playlist_dd = gr.Dropdown(choices=[("— ninguna —", "")], value="", allow_custom_value=True,
                                          label="Lista de reproducción (ID)", scale=4)
                pl_btn = gr.Button("🔄 Cargar listas del canal", scale=1)
            with gr.Row():
                sched_tb = gr.Textbox(label="Programar (opcional) AAAA-MM-DD HH:MM", placeholder="2026-10-15 18:30")
                tz_tb = gr.Textbox(label="Zona horaria", value="UTC", placeholder="America/New_York")
            footer_tb = gr.Textbox(label="Pie fijo de descripción (se añade al generar con xKiro)", lines=3,
                                   placeholder="Redes, enlaces, avisos legales…")

            gr.Markdown("### 4) Miniatura (opcional)")
            with gr.Row():
                thumb_img = gr.Image(label="Miniatura (JPG/PNG ≤ 2 MB)", type="filepath", height=220)
                with gr.Column():
                    frame_sec = gr.Number(value=3, label="Segundo del frame a extraer", precision=1)
                    frame_btn = gr.Button("🎞️ Extraer frame del video")

            with gr.Row():
                save_btn = gr.Button("💾 Guardar ajustes como predeterminados del canal")
                up_btn = gr.Button("🚀 SUBIR A YOUTUBE", variant="primary")
            save_msg = gr.Markdown()
            up_status = gr.Textbox(label="Estado de la subida", lines=8, interactive=False)
            gr.Markdown("**Últimas subidas**")
            hist_md = gr.Markdown(_history_md())

        # ───────── TAB CANALES ─────────
        with gr.Tab("🔑 Canales / APIs"):
            with gr.Accordion("📘 Cómo obtener Client ID / Secret (una vez por proyecto)", open=False):
                gr.Markdown(
                    "1. console.cloud.google.com → crea un proyecto → **APIs y servicios → Biblioteca** → activa **YouTube Data API v3**.\n"
                    "2. **Pantalla de consentimiento OAuth** → pásala a **«En producción»** (en «Pruebas» el token caduca a los 7 días).\n"
                    "3. **Credenciales → Crear → ID de cliente OAuth** → tipo **«TV y dispositivos de entrada limitada»**.\n"
                    "4. Copia Client ID y Client Secret aquí. Puedes reutilizar el mismo cliente para varios canales; "
                    "al autorizar, Google te deja elegir el canal.")
            with gr.Row():
                new_alias = gr.Textbox(label="Alias del canal", placeholder="Mi canal principal")
                new_cid = gr.Textbox(label="Client ID", value=_cfg0["last_client"].get("client_id", ""))
                new_csec = gr.Textbox(label="Client Secret", type="password",
                                      value=_cfg0["last_client"].get("client_secret", ""))
            dev_state = gr.State(None)
            start_btn = gr.Button("1️⃣ Pedir código de autorización")
            dev_md = gr.Markdown()
            finish_btn = gr.Button("2️⃣ ✅ Ya autoricé → guardar canal", variant="primary")
            conn_msg = gr.Markdown()
            gr.Markdown("### Canales guardados")
            channels_md = gr.Markdown(_channels_md())
            with gr.Row():
                manage_dd = gr.Dropdown(choices=_ch_choices(), value=_last0, label="Canal")
                verify_btn = gr.Button("🔍 Verificar")
                del_cb = gr.Checkbox(label="Confirmo borrar", value=False)
                del_btn = gr.Button("🗑️ Eliminar canal")
            manage_msg = gr.Markdown()

    # ───────── Cableado ─────────
    PROF_COMPS = [privacy_rd, category_dd, lang_dd, tags_tb, footer_tb, kids_cb, synth_cb, embed_cb,
                  stats_cb, license_rd, notify_cb, playlist_dd, ctx_tb, tz_tb, model_dd, ntitles_sl]

    ch_dd.change(on_channel_change, ch_dd, PROF_COMPS)
    demo_yt.load(on_channel_change, ch_dd, PROF_COMPS)
    vid_ref.click(lambda: gr.Dropdown(choices=_scan_videos(), value=None), None, vid_dd)
    for _c in (vid_dd, vid_man, vid_up):
        _c.change(video_info, [vid_dd, vid_man, vid_up], vid_info)
    gen_btn.click(ai_generate, [prompt_tb, ctx_tb, model_dd, ntitles_sl, tags_tb, footer_tb, lang_dd],
                  [titles_rd, desc_tb, tags_tb, ai_status])
    titles_rd.change(pick_title, titles_rd, title_tb)
    save_btn.click(save_defaults, [ch_dd] + PROF_COMPS, save_msg)
    frame_btn.click(extract_frame, [vid_dd, vid_man, vid_up, frame_sec], thumb_img)
    pl_btn.click(load_playlists, [ch_dd, playlist_dd], playlist_dd)
    up_btn.click(do_upload,
                 [ch_dd, vid_dd, vid_man, vid_up, title_tb, desc_tb, tags_tb, privacy_rd, category_dd,
                  lang_dd, kids_cb, synth_cb, embed_cb, stats_cb, license_rd, notify_cb, playlist_dd,
                  sched_tb, tz_tb, thumb_img],
                 [up_status, hist_md])
    start_btn.click(dev_start, [new_alias, new_cid, new_csec, dev_state], [dev_md, dev_state])
    finish_btn.click(dev_finish, dev_state, [conn_msg, ch_dd, manage_dd, channels_md, dev_state])
    verify_btn.click(verify_channel, manage_dd, manage_msg)
    del_btn.click(delete_channel, [manage_dd, del_cb], [manage_msg, ch_dd, manage_dd, channels_md])

try:
    demo_yt.close()   # si re-ejecutas la celda, cierra solo ESTA UI (no las otras)
except Exception:
    pass
demo_yt.queue().launch(share=True, prevent_thread_lock=True, show_error=True)