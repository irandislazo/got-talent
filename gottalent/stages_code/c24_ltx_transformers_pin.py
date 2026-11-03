import subprocess, sys, os

TARGET = "/kaggle/tmp/ltx_hf_compat"
TF_VER = "5.14.1"
os.makedirs(TARGET, exist_ok=True)

def check():
    env = os.environ.copy()
    env["PYTHONPATH"] = TARGET + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run(
        [sys.executable, "-c",
         "import transformers as t; print(t.__version__); from mmgp import offload, quant_router"],
        capture_output=True, text=True, env=env, timeout=900)
    return r.returncode == 0, ((r.stdout or "") + (r.stderr or ""))

ok, out = check()
if ok and out.strip().splitlines()[0] == TF_VER:
    print(f"✅ Ya fijado: transformers {TF_VER} para el subproceso LTX")
else:
    intentos = [
        ("sin dependencias", ["--no-deps", f"transformers=={TF_VER}"]),
        ("con dependencias", [f"transformers=={TF_VER}", "huggingface_hub>=1.5.0,<2.0.0"]),
    ]
    for etiqueta, specs in intentos:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
                            "--no-cache-dir", "--target", TARGET] + specs,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"❌ pip ({etiqueta}):", (r.stderr or r.stdout)[-600:]); continue
        ok, out = check()
        if ok:
            print(f"✅ transformers {out.strip().splitlines()[0]} aislado ({etiqueta})"); break
        print(f"⚠ Import falló ({etiqueta}):\n{out[-800:]}")
    else:
        print("❌ No se pudo fijar. Pégame el último mensaje de arriba.")