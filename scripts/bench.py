"""Étape 3 — le duel honnête : prof contre élève, sur les MÊMES fiches d'examen, l'un APRÈS l'autre
(jamais les deux en même temps sur la carte graphique, sinon on mesure la bagarre pour la VRAM).

    python bench.py etiquettes.jsonl <modele_eleve> [prof=qwen2.5:14b] [n=300]

Référence = l'étiquette d'origine du prof. On mesure la justesse ET le débit réel (fiches/heure).
"""
import json, sys, time, urllib.request, concurrent.futures as cf
import torch, laya
import eleve, prof

data, modele = sys.argv[1], sys.argv[2]
PROF = sys.argv[3] if len(sys.argv) > 3 else "qwen2.5:14b"
N = int(sys.argv[4]) if len(sys.argv) > 4 else 300
rows = eleve.charger(data)[1][:N]


def run_prof(par=4):
    etiqueter = lambda r: prof.etiqueter(r["txt"], PROF) or {"cat": None, "niveau": None}
    etiqueter(rows[0])  # chargement du modèle hors chrono
    t = time.time()
    with cf.ThreadPoolExecutor(par) as ex: out = [(o["cat"], o["niveau"]) for o in ex.map(etiqueter, rows)]
    dt = time.time() - t
    try:  # libère la VRAM avant le tour de l'élève
        urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:11434/api/generate",
                               json.dumps({"model": PROF, "keep_alive": 0}).encode()), timeout=60).read()
    except Exception: pass
    return out, dt


def run_eleve(dev):
    eleve.DEV = dev
    a = laya.load(modele, device=dev); a.model.eval()
    enc = eleve.encoder(a, rows); B = 32 if dev == "cuda" else 8  # les lots, c'est tout le ×100
    eleve.fwd(a, enc[:2])  # chauffe
    t = time.time(); preds = []
    with torch.no_grad():
        for i in range(0, len(enc), B):
            batch = enc[i:i + B]; lg, _ = eleve.fwd(a, batch)
            preds += [eleve.QS[it[0]["k"]][1][int(lg[j].argmax())] for j, it in enumerate(batch)]
    dt = time.time() - t; del a; torch.cuda.empty_cache()
    return [(preds[2 * i], preds[2 * i + 1]) for i in range(len(rows))], dt


print(f"{len(rows)} fiches d'examen, jamais vues par l'élève\n")
duels = [(f"prof {PROF} (4 en parallèle)", run_prof), ("élève Laya, CPU (lots de 8)", lambda: run_eleve("cpu"))]
if torch.cuda.is_available(): duels.insert(1, ("élève Laya, GPU (lots de 32)", lambda: run_eleve("cuda")))
for nom, fn in duels:
    out, dt = fn()
    c = 100 * sum(o[0] == r["cat"] for o, r in zip(out, rows)) / len(rows)
    n = 100 * sum(o[1] == r["niveau"] for o, r in zip(out, rows)) / len(rows)
    print(f"{nom:34} catégorie {c:5.1f} %  niveau {n:5.1f} %  |  {1000 * dt / len(rows):6.0f} ms/fiche = {3600 * len(rows) / dt:9.0f} fiches/h", flush=True)
