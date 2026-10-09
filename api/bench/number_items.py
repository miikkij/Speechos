"""Key-item accuracy for the number/abbreviation sentences (bench/sentences_numbers_fi.txt).

Round-trip CER is the wrong tool for number reading: a correct verbalization ("neljäkymmentäyhdeksän euroa
yhdeksänkymmentä senttiä") comes back from the STT judge as "49 euroa 90 senttiä", which matches neither the
written input ("49,90 euroa") nor its spelled-out form. Instead, each sentence lists the items a correct reading
must contain in the judge transcript (alternatives with "|"), and we count how many are present.

    uv run python bench/number_items.py bench/results/tts_fi_results_v3-numbers.json \
                                        bench/results/tts_fi_results_v3-numbers-norm.json
"""

from __future__ import annotations

import json
import sys

# Same order as sentences_numbers_fi.txt.
KEYS = [
    ["24", "maaliskuuta"],
    ["49", "90", "euroa"],
    ["mannerheimintie", "12", "helsinki"],
    ["esim|esimerkiksi", "tts", "mallille"],
    ["asiakaspalvelu", "example", "fi"],
    ["7", "45", "9|yhdeksän"],
    ["3", "5", "prosenttia", "28", "euroa"],
    ["1995"],
    ["15", "minuuttia", "jne|edelleen"],
    ["040", "123", "4567", "16"],
]


def present(hyp: str, item: str) -> bool:
    toks = hyp.replace("12a", "12 a").split()
    return any(alt in toks for alt in item.split("|"))


def main() -> None:
    print("| Results | Engine | key items | sentences fully right |\n|---|---|---:|---:|")
    for path in sys.argv[1:]:
        data = json.load(open(path, encoding="utf-8"))
        for r in data["results"]:
            if "details" not in r:
                continue
            hit = tot = full = n = 0
            for x in r["details"]:
                keys = KEYS[x["i"]]
                h = sum(present(x["hyp"], it) for it in keys)
                hit, tot, full, n = hit + h, tot + len(keys), full + (h == len(keys)), n + 1
            tag = "normalize_fi" if data["params"].get("normalize_fi") else "raw text"
            print(f"| {tag} | {r['engine']} | {hit}/{tot} = {hit / tot:.0%} | {full}/{n} |")


if __name__ == "__main__":
    main()
