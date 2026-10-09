"""Reference-audio capture tool — record in the browser OR drop a file, get a
16 kHz mono WAV via ffmpeg.

Produces the reference clip used by the Finnish TTS benchmark for
  - Chatterbox voice cloning  (--ref-wav)
  - SQUIM naturalness MOS      (--mos-ref)

Usage (from api/):
  uv run python bench/make_ref.py            # → http://localhost:36399 , writes bench/ref_fi.wav
  uv run python bench/make_ref.py --out bench/my_ref.wav --port 36399

Then open the URL, record from the mic (or choose any audio file), and click
"Lähetä". The server runs ffmpeg and writes the 16k-mono WAV. No extra deps —
stdlib only — but ffmpeg must be on PATH.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

OUT_PATH = Path("bench/ref_fi.wav")  # overridden in main()

PAGE = """<!doctype html>
<html lang="fi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Speechos — referenssiääni</title>
<style>
  body{font-family:system-ui,sans-serif;max-width:640px;margin:2rem auto;padding:0 1rem;color:#111}
  h1{font-size:1.3rem} button{font-size:1rem;padding:.6rem 1rem;margin:.25rem .25rem 0 0;cursor:pointer}
  .row{margin:1rem 0;padding:1rem;border:1px solid #ddd;border-radius:8px}
  #status{white-space:pre-wrap;font-family:monospace;background:#f6f6f6;padding:.75rem;border-radius:6px;min-height:2.5rem}
  audio{width:100%;margin-top:.5rem}
  small{color:#666}
</style></head><body>
<h1>Referenssiääninäyte → 16 kHz mono WAV</h1>
<p><small>Tuottaa tiedoston: <code>__OUT__</code>. Puhu n. 10–20 s selkeää suomea.</small></p>

<div class="row">
  <b>1) Nauhoita mikistä</b><br>
  <button id="rec">● Aloita nauhoitus</button>
  <button id="stop" disabled>■ Lopeta</button>
  <audio id="preview" controls></audio>
</div>

<div class="row">
  <b>2) …tai valitse äänitiedosto</b> (mp3, wav, m4a, ogg…)<br>
  <input id="file" type="file" accept="audio/*">
</div>

<div class="row">
  <button id="send" disabled>⬆ Lähetä &amp; muunna (ffmpeg)</button>
  <div id="status">Valmiina.</div>
  <audio id="result" controls style="display:none"></audio>
</div>

<script>
let blob=null, mr=null, chunks=[];
const $=id=>document.getElementById(id);
function setBlob(b,label){blob=b; $("send").disabled=!b; if(b){$("preview").src=URL.createObjectURL(b);} $("status").textContent=label;}

$("rec").onclick=async()=>{
  const stream=await navigator.mediaDevices.getUserMedia({audio:true});
  mr=new MediaRecorder(stream); chunks=[];
  mr.ondataavailable=e=>chunks.push(e.data);
  mr.onstop=()=>{const b=new Blob(chunks,{type:mr.mimeType||"audio/webm"});
    stream.getTracks().forEach(t=>t.stop()); setBlob(b,"Nauhoitus valmis. Voit lähettää.");};
  mr.start(); $("rec").disabled=true; $("stop").disabled=false; $("status").textContent="Nauhoitetaan…";
};
$("stop").onclick=()=>{mr&&mr.stop(); $("rec").disabled=false; $("stop").disabled=true;};
$("file").onchange=e=>{const f=e.target.files[0]; if(f) setBlob(f,"Tiedosto valittu: "+f.name);};

$("send").onclick=async()=>{
  if(!blob) return;
  $("send").disabled=true; $("status").textContent="Lähetetään ja ajetaan ffmpeg…";
  try{
    const r=await fetch("/upload",{method:"POST",headers:{"Content-Type":blob.type||"application/octet-stream"},body:blob});
    const j=await r.json();
    if(j.ok){$("status").textContent="✅ Valmis: "+j.path+"  ("+j.seconds+" s, "+j.sample_rate+" Hz, "+j.channels+" kanava)";
      const res=$("result"); res.style.display="block"; res.src="/ref?t="+Date.now();}
    else{$("status").textContent="❌ Virhe: "+j.error;}
  }catch(err){$("status").textContent="❌ "+err;}
  $("send").disabled=false;
};
</script></body></html>"""


def _ffprobe(path: Path) -> dict:
    """Return {seconds, sample_rate, channels} if ffprobe is available, else {}."""
    if not shutil.which("ffprobe"):
        return {}
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate,channels:format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout
        info = json.loads(out)
        st = (info.get("streams") or [{}])[0]
        dur = float(info.get("format", {}).get("duration", 0.0))
        return {
            "seconds": round(dur, 1),
            "sample_rate": int(st.get("sample_rate", 0)),
            "channels": int(st.get("channels", 0)),
        }
    except Exception:
        return {}


class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # quieter console
        pass

    def do_GET(self):
        if self.path.startswith("/ref"):
            if not OUT_PATH.exists():
                self._json(404, {"error": "no output yet"})
                return
            data = OUT_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        html = PAGE.replace("__OUT__", str(OUT_PATH)).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def do_POST(self):
        if self.path != "/upload":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            self._json(400, {"error": "empty body"})
            return
        raw = self.rfile.read(length)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        try:
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_path, "-ac", "1", "-ar", "16000", str(OUT_PATH)],
                capture_output=True, text=True,
            )
            if proc.returncode != 0:
                self._json(500, {"error": "ffmpeg failed: " + proc.stderr[-400:]})
                return
            info = _ffprobe(OUT_PATH)
            print(f"  wrote {OUT_PATH}  {info or '(ffprobe unavailable)'}", flush=True)
            self._json(200, {"ok": True, "path": str(OUT_PATH),
                             "seconds": info.get("seconds", "?"),
                             "sample_rate": info.get("sample_rate", 16000),
                             "channels": info.get("channels", 1)})
        finally:
            Path(tmp_path).unlink(missing_ok=True)


def main() -> None:
    global OUT_PATH
    ap = argparse.ArgumentParser(description="Browser reference-audio capture -> 16k mono WAV")
    ap.add_argument("--out", default="bench/ref_fi.wav", help="output WAV path")
    ap.add_argument("--port", type=int, default=36399)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found on PATH. Install it (apt-get install ffmpeg / winget install ffmpeg).")

    OUT_PATH = Path(args.out)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Reference-audio tool: http://{args.host}:{args.port}")
    print(f"Output will be written to: {OUT_PATH.resolve()}")
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
