"""Blinded listening page for the Finnish TTS clips saved by tts_fi.py --save-audio.

Layout expected (gitignored, never committed):
  <dir>/sentences.json            the 10 listening sentences
  <dir>/<engine>/<NN>.wav         one clip per engine per sentence

Writes:
  <dir>/index.html   per sentence, every engine's clip in a random order labelled A, B, C, ...
                     with a 1–5 rating per clip; ratings stay in the browser (localStorage) and
                     "Download ratings" saves ratings.json. No engine names in the page.
  <dir>/key.json     label → engine per sentence (do not open before rating)

Unblind afterwards:
  uv run python bench/make_listen_page.py --unblind bench/listen/ratings.json
→ mean rating per engine (+ the n of rated clips), printed and written to <dir>/unblinded.md.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import statistics
import string
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent


def build(listen_dir: Path, seed: int) -> None:
    sentences = json.loads((listen_dir / "sentences.json").read_text(encoding="utf-8"))
    engines = sorted(d.name for d in listen_dir.iterdir() if d.is_dir())
    rng = random.Random(seed)
    key: dict[str, dict[str, str]] = {}
    blocks = []
    for i, sent in enumerate(sentences):
        clips = [e for e in engines if (listen_dir / e / f"{i:02d}.wav").exists()]
        rng.shuffle(clips)
        key[str(i)] = {}
        rows = []
        for label, eng in zip(string.ascii_uppercase, clips):
            key[str(i)][label] = eng
            name = f"s{i}-{label}"
            radios = "".join(
                f'<label><input type="radio" name="{name}" value="{v}">{v}</label>' for v in range(1, 6))
            rows.append(f'<div class="clip"><span class="lab">{label}</span>'
                        f'<audio controls preload="none" src="{html.escape(eng)}/{i:02d}.wav"></audio>'
                        f'<span class="rate">{radios}</span></div>')
        blocks.append(f'<section><h2>{i + 1}. “{html.escape(sent)}”</h2>{"".join(rows)}</section>')

    page = f"""<!doctype html>
<html lang="fi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Suomi-TTS kuuntelu</title>
<style>
:root{{--bg:#fafaf7;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#e3e2dc;--acc:#2d5bd7}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--fg:#ecebe6;--muted:#9a9993;--card:#1f1f1d;--line:#33332f;--acc:#7c9dff}}}}
body{{background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif;margin:0 auto;max-width:860px;padding:16px}}
h1{{font-size:1.4rem}} h2{{font-size:1.05rem;margin:0 0 .6rem}}
section{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:14px 0}}
.clip{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:4px 0}}
.lab{{font-weight:700;width:1.5rem}} audio{{height:34px;flex:1 1 260px;min-width:0}}
.rate label{{margin-right:6px;color:var(--muted)}} .note{{color:var(--muted);font-size:.9rem}}
button{{background:var(--acc);color:#fff;border:0;border-radius:6px;padding:8px 14px;font-size:1rem;cursor:pointer}}
</style></head><body>
<h1>Suomi-TTS sokkokuuntelu</h1>
<p class="note">Jokaisella lauseella on samat moottorit satunnaisessa järjestyksessä (A, B, …); kirjain ei
tarkoita samaa moottoria eri lauseissa. Arvioi luonnollisuus 1 (robotti/virheellinen) – 5 (kuin ihminen).
Arviot tallentuvat selaimeen. Lopuksi lataa ratings.json ja aja <code>make_listen_page.py --unblind</code>.
Älä avaa key.json:ia ennen arviointia.</p>
{"".join(blocks)}
<p><button id="dl">Lataa ratings.json</button> <span class="note" id="cnt"></span></p>
<script>
const K='speechos-fi-listen';
let st={{}};try{{st=JSON.parse(localStorage.getItem(K)||'{{}}')}}catch(e){{}}
function upd(){{document.getElementById('cnt').textContent=Object.keys(st).length+' arviota';}}
document.querySelectorAll('input[type=radio]').forEach(r=>{{
  if(st[r.name]==r.value)r.checked=true;
  r.addEventListener('change',()=>{{st[r.name]=Number(r.value);try{{localStorage.setItem(K,JSON.stringify(st))}}catch(e){{}};upd();}});
}});
document.getElementById('dl').onclick=()=>{{
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(st,null,1)],{{type:'application/json'}}));
  a.download='ratings.json';a.click();}};
upd();
</script></body></html>
"""
    (listen_dir / "index.html").write_text(page, encoding="utf-8")
    (listen_dir / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"{len(sentences)} sentences x {len(engines)} engines -> {listen_dir / 'index.html'}  (key: key.json)")


def unblind(listen_dir: Path, ratings_path: Path) -> None:
    key = json.loads((listen_dir / "key.json").read_text(encoding="utf-8"))
    ratings = json.loads(ratings_path.read_text(encoding="utf-8"))
    per_engine: dict[str, list[int]] = {}
    for name, val in ratings.items():
        sent, label = name[1:].split("-")
        eng = key.get(sent, {}).get(label)
        if eng:
            per_engine.setdefault(eng, []).append(int(val))
    rows = sorted(per_engine.items(), key=lambda kv: -statistics.mean(kv[1]))
    md = "| Engine | mean rating | n |\n|---|---:|---:|\n" + "\n".join(
        f"| {e} | {statistics.mean(v):.2f} | {len(v)} |" for e, v in rows) + "\n"
    (listen_dir / "unblinded.md").write_text(md, encoding="utf-8")
    print(md)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build / unblind the Finnish TTS listening page")
    ap.add_argument("--dir", default=str(BENCH_DIR / "listen"))
    ap.add_argument("--seed", type=int, default=73104, help="shuffle seed for the blinded order")
    ap.add_argument("--unblind", default=None, help="ratings.json downloaded from the page")
    args = ap.parse_args()
    d = Path(args.dir)
    if args.unblind:
        unblind(d, Path(args.unblind))
    else:
        build(d, args.seed)


if __name__ == "__main__":
    main()
