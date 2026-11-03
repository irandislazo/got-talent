import requests, time, os

def probar(base, mb=5):
    base = base.rstrip("/")
    try:
        t = time.time(); requests.get(f"{base}/config", timeout=60); rtt = time.time() - t
        data = os.urandom(mb * 1024 * 1024)
        t = time.time()
        r = requests.post(f"{base}/gradio_api/upload", files=[("files", ("test.bin", data))], timeout=300)
        up = time.time() - t
        path = r.json()[0]
        t = time.time()
        d = requests.get(f"{base}/gradio_api/file={path}", timeout=300)
        down = time.time() - t
        print(f"{base[:48]:<49} latencia {rtt:5.2f}s | subida {mb/up:6.2f} MB/s | bajada {len(d.content)/1024**2/down:6.2f} MB/s")
    except Exception as e:
        print(f"{base[:48]:<49} ❌ {type(e).__name__}: {str(e)[:120]}")

probar(demo.local_url)
if getattr(demo, "share_url", None):
    probar(demo.share_url)