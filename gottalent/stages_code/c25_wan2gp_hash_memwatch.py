import os, subprocess, hashlib
W = "/kaggle/working/Wan2GP"
r = subprocess.run(["git","-C",W,"rev-parse","--short=10","HEAD"], capture_output=True, text=True)
print("Commit Wan2GP:", r.stdout.strip() or "sin .git", "(validado: 0e58385fbd)")
esperado = {"wgp.py":"a33acb7b9a", "requirements.txt":"23a261e975", "models/ltx2/ltx2_handler.py":"f65f7785f3"}
for rel, exp in esperado.items():
    p = os.path.join(W, rel)
    got = hashlib.md5(open(p,"rb").read()).hexdigest()[:10] if os.path.exists(p) else "NO EXISTE"
    print(f"{'✅' if got==exp else '⚠️'} {rel}: {got} (esperado {exp})")
print(subprocess.run("pip show mmgp | sed -n 1,2p", shell=True, capture_output=True, text=True).stdout)

# Monitor de RAM desacoplado → /kaggle/working/mem_watch.log (cada 3 s)
open("/tmp/memwatch.sh","w").write('''#!/bin/bash
while true; do
  echo "$(date +%T) $(free -m | awk 'NR==2{print "used="$3"MB avail="$7"MB"}') | $(ps -eo rss,comm --sort=-rss | sed -n 2,3p | tr '\\n' ' ')" >> /kaggle/working/mem_watch.log
  sleep 3
done
''')
if subprocess.run(["pgrep","-f","memwatch.sh"], capture_output=True).returncode != 0:
    subprocess.Popen(["bash","/tmp/memwatch.sh"], start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("Monitor activo → /kaggle/working/mem_watch.log")