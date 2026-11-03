# ═══ 🛑 DETENER LTX — mata el subproceso de generación (no toca nada más) ═══
import os, time, signal, psutil, json

_yo = os.getpid()
_objetivo = []
for p in psutil.process_iter(["pid", "cmdline"]):
    try:
        cmd = " ".join(p.info["cmdline"] or [])
        if ("run_ltx25_generate.py" in cmd or "run_ltx23_generate.py" in cmd) and p.info["pid"] != _yo:
            _objetivo.append(p)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

if not _objetivo:
    print("ℹ️ No hay ningún subproceso LTX en ejecución.")
else:
    # incluir hijos (por si lanzó procesos auxiliares)
    todos = {}
    for p in _objetivo:
        todos[p.pid] = p
        try:
            for c in p.children(recursive=True):
                todos[c.pid] = c
        except psutil.NoSuchProcess:
            pass

    print(f"🛑 Deteniendo {len(todos)} proceso(s): {sorted(todos)}")
    for p in todos.values():
        try: p.send_signal(signal.SIGTERM)
        except psutil.NoSuchProcess: pass
    _, vivos = psutil.wait_procs(list(todos.values()), timeout=8)
    for p in vivos:
        try: p.kill()
        except psutil.NoSuchProcess: pass
    psutil.wait_procs(vivos, timeout=5)

    # dejar el archivo de progreso coherente para la UI
    try:
        with open("/tmp/ltx25_progress.json", "w") as f:
            json.dump({"status": "🛑 Detenido manualmente"}, f)
    except Exception:
        pass

    time.sleep(1)
    try:
        import subprocess
        print(subprocess.run("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader",
                             shell=True, capture_output=True, text=True).stdout.strip(), "(VRAM en uso)")
    except Exception:
        pass
    print("✅ LTX detenido.")