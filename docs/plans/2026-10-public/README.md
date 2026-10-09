# Speechos — lokakuun 2026 päivitys (julkinen vaihe)

Kaksi itsenäistä Claude Code -sessiota, jotka voi ajaa rinnakkain omissa git worktreeissään `main`ista.

| Sessio | Tiedosto | Aihe | Worktree / branch |
| --- | --- | --- | --- |
| A | [session-a-core.md](session-a-core.md) | Ydin-API, web UI, testit, CI, Docker, arkkitehtuuridokut | `../Speechos-a` / `next/a-core` |
| B | [session-b-fi-models.md](session-b-fi-models.md) | FI-benchmark 2.0: uudet suomen STT- ja TTS-mallit | `../Speechos-b` / `next/b-fi-models` |

## Lähtötila

`main` sisältää commitin `87b6507`: suomen STT/TTS-benchmark (`api/bench/`), monikielinen synteesi ja `bench`-extra.
Lukitut tulokset 19.6.2026: STT faster-whisper large-v3 FLEURS-fi WER 7,9 %; TTS Piper `fi_FI-harri-medium` CER 3,5 %.

## Tiedostojen omistajuus

| Polku | Omistaja |
| --- | --- |
| `api/src/**` (paitsi alla mainitut uudet tiedostot), `api/tests/**`, `api/test_*.py`, `web/**`, `docker/**`, `.github/**`, `docs/architecture/**`, `README.md`, `CLAUDE.md`, `start.*`, `api/pyproject.toml`:n `dev`-extra ja ydinriippuvuudet | A |
| `api/bench/**`, `docs/research/**`, uudet moottorimoduulit `api/src/engines_fi/**`, `api/pyproject.toml`:n `bench*`-extrat | B |

`api/uv.lock` generoidaan uudelleen mergen yhteydessä (`uv lock`), sitä ei yhdistetä käsin.

## Merge

A ja B mergetään `main`iin kumpi tahansa ensin. Mergen jälkeen: `cd api && uv lock && uv run pytest`, `cd web && pnpm build`.
B:n mahdollinen moottorirekisteröinti `api/src/models.py`:hin tehdään mergen yhteydessä B:n handoff-ohjeen mukaan.
Ei pusheja eikä tageja ilman erillistä pyyntöä.
