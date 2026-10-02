"""Recherche hybride locale sur un dossier de notes/code : mots-clés (BM25) + sens (embeddings), fusionnés par rangs (RRF).
C'est ce qui a battu Laya sur la recherche documentaire.

    python recherche_hybride.py <dossier> "ta question" [k=5]

Embeddings : bge-m3 via Ollama (`ollama pull bge-m3`). Sans Ollama, repli automatique sur les mots-clés seuls.
Vecteurs mis en cache dans <dossier>/.hybride-cache.npz (recalculés si la liste des fichiers change).
"""
import json, math, os, re, sys, urllib.request
from collections import Counter

EXT = (".md", ".txt", ".py", ".js", ".mjs", ".ts", ".tsx", ".go", ".rs", ".cs")
SKIP = re.compile(r"(^|[\\/])(\.|node_modules|dist|build|__pycache__)")
MOT = re.compile(r"[a-zà-ÿ0-9_]{3,}")


def lire(dossier):
    docs = {}
    for d, _, fs in os.walk(dossier):
        for f in fs:
            p = os.path.join(d, f); r = os.path.relpath(p, dossier)
            if f.endswith(EXT) and not SKIP.search(r):
                try: docs[r] = open(p, encoding="utf-8", errors="replace").read()
                except OSError: pass
    return docs


def carte(chemin, txt): return "%s\n%s" % (chemin, re.sub(r"\s+", " ", txt)[:1500])


def bm25(docs, q, k):
    ids = list(docs); tf = [Counter(MOT.findall((i + " " + docs[i]).lower())) for i in ids]
    lg = [sum(t.values()) or 1 for t in tf]; moy = sum(lg) / len(lg)
    df = Counter(w for t in tf for w in t); n = len(ids)
    idf = {w: math.log(1 + (n - c + .5) / (c + .5)) for w, c in df.items()}
    ws = set(MOT.findall(q.lower()))
    sc = [(sum(idf[w] * t[w] * 2.2 / (t[w] + 1.2 * (.25 + .75 * lg[i] / moy)) for w in ws if w in t), ids[i]) for i, t in enumerate(tf)]
    return [i for s, i in sorted(sc, reverse=True)[:k] if s]


def embed(textes):
    import numpy as np
    out = []
    for i in range(0, len(textes), 32):
        req = urllib.request.Request("http://127.0.0.1:11434/api/embed", json.dumps({"model": "bge-m3", "input": textes[i:i + 32]}).encode())
        out += json.loads(urllib.request.urlopen(req, timeout=300).read())["embeddings"]
    v = np.array(out, dtype=np.float32); return v / np.linalg.norm(v, axis=1, keepdims=True)


def dense(dossier, docs, q, k):
    import numpy as np
    ids = list(docs); cache = os.path.join(dossier, ".hybride-cache.npz")
    if os.path.exists(cache) and list(np.load(cache)["ids"]) == ids: E = np.load(cache)["E"]
    else: E = embed([carte(i, docs[i]) for i in ids]); np.savez(cache, E=E, ids=np.array(ids))
    return [ids[j] for j in np.argsort(-(E @ embed([q])[0]))[:k]]


def chercher(dossier, q, k=5):
    docs = lire(dossier)
    listes = [bm25(docs, q, 100)]
    try: listes.append(dense(dossier, docs, q, 100))
    except Exception as e: print(f"(embeddings indisponibles, mots-clés seuls : {e.__class__.__name__})", file=sys.stderr)
    s = Counter()
    for liste in listes:
        for rang, d in enumerate(liste): s[d] += 1 / (60 + rang)  # RRF : on fusionne des RANGS, pas des scores incomparables
    return [d for d, _ in s.most_common(k)]


if __name__ == "__main__":
    for r in chercher(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 5): print(r)
