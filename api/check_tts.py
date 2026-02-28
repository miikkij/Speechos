import httpx
r = httpx.get("http://localhost:36300/api/system/model-options")
print(f"Status: {r.status_code}")
print(f"Content: {r.text[:200]}")
if r.status_code == 200:
    tts = r.json()["options"]["tts"]
    print(f"\n{len(tts)} TTS options:")
    for t in tts:
        print(f"  {t['engine']}: {t['label']}")
