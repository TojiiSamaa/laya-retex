"""Étape 2 — l'« élève » : Laya apprend à imiter le prof, puis décide ~100× plus vite (GPU, par lots).

    python eleve.py train etiquettes.jsonl <sortie> [epoques=3]   # découpe 90/10, entraîne, sauvegarde, examine
    python eleve.py eval  etiquettes.jsonl <modele>               # examine un modèle déjà entraîné

etiquettes.jsonl = sortie de prof.py. L'examen porte sur les 10 % jamais vus à l'entraînement.
Mesures : accord avec le prof, classe majoritaire (le minimum à battre), part où Laya se dit sûr (>= 0,8)
et sa justesse dans ce cas (= est-il bien calibré ?).
Testé avec laya==0.3.20 (API interne laya.common).
"""
import glob, json, os, random, shutil, sys, time
from collections import Counter
import torch, laya
from laya.common import build_sequence, collate_items, DecisionModel
from prof import CATS

BASE = "convaiinnovations/laya-multilingual"  # le checkpoint multilingue (le checkpoint anglais est faible en français)
NIV = ["faible", "moyen", "fort"]
QS = {
    "cat": ({"t": "choice", "ins": "Dans quelle catégorie ranger cette fiche ?", "crit": {c: None for c in CATS}}, CATS),
    "niveau": ({"t": "choice", "ins": "Quel est l'intérêt réel de cette fiche ?", "crit": {n: None for n in NIV}}, NIV),
}
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def charger(p):
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    random.Random(0).shuffle(rows); n = len(rows) // 10
    return rows[n:], rows[:n]  # entraînement, examen


def encoder(a, rows):
    H = a.cfg.get("head_max_len", 256)
    out = []
    for r in rows:
        for k, (q, labels) in QS.items():
            ids, mk = build_sequence(a.tok, r["txt"], q, 512, H)
            out.append([{"ids": ids, "markers": mk, "qtype": 0, "label": labels.index(r[k]), "k": k}])
    return out


def fwd(a, batch):
    b = {k: v.to(DEV) for k, v in collate_items(batch, a.tok.pad_token_id).items() if torch.is_tensor(v)}
    with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
        lg, _ = DecisionModel.forward(a.model, b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
    return lg.float(), b["label"]


@torch.no_grad()
def examiner(a, rows):
    a.model.eval(); enc = encoder(a, rows); res = {k: Counter() for k in QS}
    for i in range(0, len(enc), 64):
        batch = enc[i:i + 64]; lg, y = fwd(a, batch); p = torch.softmax(lg, -1)
        for j, it in enumerate(batch):
            r = res[it[0]["k"]]; ok = int(p[j].argmax()) == int(y[j])
            r["n"] += 1; r["ok"] += ok
            if float(p[j].max()) >= 0.8: r["sur"] += 1; r["sur_ok"] += ok
    return {k: {"accord_avec_prof": round(r["ok"] / r["n"], 3),
                "classe_majoritaire": round(Counter(x[k] for x in rows).most_common(1)[0][1] / len(rows), 3),
                "se_dit_sur": round(r["sur"] / r["n"], 3),
                "justesse_quand_sur": round(r["sur_ok"] / max(1, r["sur"]), 3)} for k, r in res.items()}


def main():
    mode, data, cible = sys.argv[1], sys.argv[2], sys.argv[3]
    train, test = charger(data)
    if mode == "eval":
        print(json.dumps(examiner(laya.load(cible, device=DEV), test), ensure_ascii=False, indent=1)); return

    a = laya.load(BASE, device=DEV); random.seed(0)
    ep = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    enc = encoder(a, train); B, LR = 16, 3e-5
    opt = torch.optim.AdamW(a.model.parameters(), lr=LR, weight_decay=0.01)
    steps = ep * (len(enc) // B)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.06)
    k, t0 = 0, time.time()
    for e in range(ep):
        a.model.train(); random.shuffle(enc)
        for i in range(0, len(enc) - B + 1, B):
            lg, y = fwd(a, enc[i:i + B])
            loss = torch.nn.functional.cross_entropy(lg, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(a.model.parameters(), 1.0); opt.step(); sch.step(); k += 1
            if k % 200 == 0: print(f"époque {e} {k}/{steps} perte {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)

    # sauvegarde AVANT l'examen : une heure de GPU ne doit pas se perdre sur une erreur d'examen
    snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--%s/snapshots/*" % BASE.replace("/", "--")))[0]
    shutil.copytree(snap, cible, dirs_exist_ok=True)
    from safetensors.torch import save_file
    save_file({n: v.detach().cpu().contiguous() for n, v in a.model.state_dict().items()}, os.path.join(cible, "model.safetensors"))
    r = examiner(a, test); json.dump(r, open(os.path.join(cible, "eval.json"), "w"))
    print(json.dumps(r, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
