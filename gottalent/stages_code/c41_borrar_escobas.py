
#Eliminar escebas de jueses en los dialogos

import glob, os
base = "/kaggle/working/VideoAutomationProject/videos/20261002_023053_mi_plantilla_1_niña"
for etapa in ("etapa2_dialogos", "etapa4_dialogos"):
    for d in sorted(glob.glob(f"{base}/{etapa}/frag_*_juez_*")):
        for f in ("clip.mp4", "clip.sig", "clip.fallback"):
            p = os.path.join(d, f)
            if os.path.exists(p):
                print("borrando", p, os.path.getsize(p)//1024, "KB")
                os.remove(p)