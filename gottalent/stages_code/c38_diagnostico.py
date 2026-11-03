import os, subprocess, shutil, psutil, importlib.metadata as md, socket, glob

def hdr(t): print(f"\n── {t} " + "─"*(50-len(t)))

hdr("RAM / CPU")
vm = psutil.virtual_memory()
print(f"RAM: {vm.used/1024**3:.1f} / {vm.total/1024**3:.1f} GB ({vm.percent}%)  |  CPU: {psutil.cpu_percent(interval=1)}%")

hdr("DISCO")
for p in ["/kaggle/working", "/tmp"]:
    u = shutil.disk_usage(p)
    print(f"{p}: libre {u.free/1024**3:.1f} GB de {u.total/1024**3:.1f} GB")

hdr("TOP PROCESOS (RAM)")
procs = []
for p in psutil.process_iter(["pid", "name", "cmdline", "memory_info", "cpu_percent"]):
    try:
        procs.append((p.info["memory_info"].rss/1024**3, p.info["pid"], " ".join(p.info["cmdline"] or [])[:110]))
    except Exception:
        pass
for rss, pid, cmd in sorted(procs, reverse=True)[:8]:
    print(f"{rss:5.1f} GB  pid {pid:<7} {cmd}")

hdr("PROCESOS QUE PUEDEN BLOQUEAR")
out = subprocess.run("ps aux | grep -E 'run_ltx25|qwen3_tts_worker|hf_transfer|huggingface|git clone|pip ' | grep -v grep",
                     shell=True, capture_output=True, text=True).stdout
print(out.strip() or "ninguno")

hdr("GRADIO / TÚNEL")
print("gradio:", md.version("gradio"))
frpc = glob.glob(os.path.expanduser("~/.cache/huggingface/gradio/frpc/*")) + glob.glob("/usr/local/lib/python3*/dist-packages/gradio/frpc_*")
print("frpc (túnel share=True):", frpc or "NO descargado todavía")
for host in ["huggingface.co", "api.gradio.app", "cdn-media.huggingface.co"]:
    try:
        socket.create_connection((host, 443), timeout=5).close(); print(f"✅ {host}")
    except Exception as e:
        print(f"❌ {host}: {e}")

hdr("LOG")
lg = glob.glob("/kaggle/working/**/run.log", recursive=True)
for f in lg[:3]:
    print(f"{f}: {os.path.getsize(f)/1024**2:.2f} MB")

hdr("HF_TRANSFER")
print("HF_HUB_ENABLE_HF_TRANSFER =", os.environ.get("HF_HUB_ENABLE_HF_TRANSFER"))
try: print("hf_transfer instalado:", md.version("hf_transfer"))
except Exception: print("hf_transfer instalado: no")