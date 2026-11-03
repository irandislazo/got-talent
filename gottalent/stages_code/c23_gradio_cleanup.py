import subprocess, time
import gradio as gr

gr.close_all()                                   # cierra todos los servidores Gradio del kernel
time.sleep(2)
subprocess.run("pkill -f frpc_linux_amd64", shell=True)   # elimina túneles que hayan quedado
time.sleep(1)

print(subprocess.run("ss -ltnp | grep -E ':786[0-9]' || echo 'puertos 786x libres ✅'",
                     shell=True, capture_output=True, text=True).stdout)
print(subprocess.run("ps aux | grep frpc | grep -v grep || echo 'sin túneles ✅'",
                     shell=True, capture_output=True, text=True).stdout)