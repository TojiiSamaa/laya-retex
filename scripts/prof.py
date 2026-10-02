"""Étape 1 — le « prof » : un LLM local (Ollama) étiquette tes textes gratuitement.

    python prof.py entree.jsonl sortie.jsonl [modele=qwen2.5:14b] [paralleles=4]

entree.jsonl : une ligne par texte, {"txt": "..."}.
sortie.jsonl : {"txt": ..., "cat": ..., "niveau": ...} ; les réponses illisibles sont écartées (on ne garde que du propre).
Adapte PROMPT, CATS et niveau() à ta propre décision.
"""
import json, re, sys, urllib.request, concurrent.futures as cf

OLLAMA = "http://127.0.0.1:11434/v1/chat/completions"
CATS = ["outil", "tuto", "actu", "autre"]
PROMPT = ("Tu classes une fiche de veille technique. Fiche :\n---\n%s\n---\n"
          'Réponds UNIQUEMENT un JSON : {"score": <0-10 intérêt réel>, "cat": "<' + "|".join(CATS) + '>"}')


def niveau(score): return "faible" if score <= 2 else "moyen" if score <= 5 else "fort"


def etiqueter(txt, modele):
    req = urllib.request.Request(OLLAMA, json.dumps({"model": modele, "temperature": 0.2, "max_tokens": 200,
          "messages": [{"role": "user", "content": PROMPT % txt[:6000]}]}).encode(), headers={"content-type": "application/json"})
    rep = json.loads(urllib.request.urlopen(req, timeout=300).read())["choices"][0]["message"]["content"]
    try:
        j = json.loads(re.search(r"\{[\s\S]*\}", rep).group(0))
        cat = str(j["cat"]).strip().lower()
        return {"txt": txt, "cat": cat, "niveau": niveau(float(j["score"]))} if cat in CATS else None
    except Exception:
        return None  # « parse failed » : jeté, pas deviné


def main():
    src, dst = sys.argv[1], sys.argv[2]
    modele = sys.argv[3] if len(sys.argv) > 3 else "qwen2.5:14b"
    par = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    textes = [json.loads(l)["txt"] for l in open(src, encoding="utf-8") if l.strip()]
    with cf.ThreadPoolExecutor(par) as ex, open(dst, "w", encoding="utf-8") as out:
        ok = 0
        for r in ex.map(lambda t: etiqueter(t, modele), textes):
            if r: out.write(json.dumps(r, ensure_ascii=False) + "\n"); ok += 1
    print(f"{ok}/{len(textes)} textes étiquetés proprement -> {dst}")


if __name__ == "__main__":
    main()
