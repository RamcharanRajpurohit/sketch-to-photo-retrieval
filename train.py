"""Step 2: train the projection head on the cached CLIP features and evaluate retrieval.
Usage: python train.py [backbone] [instance|category|both] [temperature]   (seconds on a laptop CPU)"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from sbir import BACKBONE, Head, split_photos

backbone = sys.argv[1] if len(sys.argv) > 1 else BACKBONE
target = sys.argv[2] if len(sys.argv) > 2 else "both"  # positives: same photo (instance), same category, or both
# CLIP starts training at 0.07 and ends at 0.01 (logit_scale 100); validation picked 0.07 (README, experiments)
tau = float(sys.argv[3]) if len(sys.argv) > 3 else 0.07
run = f"{backbone}_{target}_tau{tau}"
torch.manual_seed(0)
d = torch.load(f"features_{backbone}.pt")
P, S, labels, pair = d["photo_feats"], d["sketch_feats"], d["labels"], d["pair"]
split = split_photos(len(P))  # test sketches come from photos never used in training


def subset(k):
    """Photos of split k, their sketches, and each sketch's photo position inside that gallery."""
    pi = (split == k).nonzero().squeeze(1)
    si = (split[pair] == k).nonzero().squeeze(1)
    pos = torch.full((len(P),), -1)
    pos[pi] = torch.arange(len(pi))
    return pi, si, pos[pair[si]]


@torch.no_grad()
def evaluate(head, k):
    """Category level: Acc@1, P@10, mAP (same category = correct). Instance level: exact photo in top-1/top-10."""
    pi, si, true = subset(k)
    s, p = (head(S[si], "sketch"), head(P[pi], "photo")) if head else (S[si], P[pi])
    order = (s @ p.T).argsort(dim=1, descending=True)
    rel = (labels[pi][order] == labels[pair[si]][:, None]).float()
    ap = (rel.cumsum(1) / torch.arange(1, len(pi) + 1) * rel).sum(1) / rel.sum(1)
    rank = (order == true[:, None]).float().argmax(1)
    return {"cat_acc@1": rel[:, 0].mean().item(), "cat_P@10": rel[:, :10].mean().item(), "cat_mAP": ap.mean().item(),
            "inst_acc@1": (rank < 1).float().mean().item(), "inst_acc@10": (rank < 10).float().mean().item()}


def info_nce(logits, ids):
    """CLIP-style contrastive loss; every column with the same id as the row is a positive."""
    same = (ids[:, None] == ids[None]).float()  # several sketches in a batch can share a photo / category
    same = same / same.sum(1, keepdim=True)
    return (F.cross_entropy(logits, same) + F.cross_entropy(logits.T, same)) / 2


head = Head(P.shape[1])
opt = torch.optim.Adam(head.parameters(), lr=1e-3)
train_sketches = subset(0)[1]
history, best = [], -1
for epoch in range(30):
    total = 0
    for b in torch.randperm(len(train_sketches)).split(256):
        si = train_sketches[b]
        pi = pair[si]
        logits = head(S[si], "sketch") @ head(P[pi], "photo").T / tau
        loss = sum(info_nce(logits, ids) for ids in {"instance": [pi], "category": [labels[pi]],
                                                      "both": [pi, labels[pi]]}[target])
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(b)
    val = evaluate(head, 1)
    history.append({"epoch": epoch + 1, "loss": total / len(train_sketches), **val})
    if val["cat_mAP"] > best:  # keep the epoch with the best validation mAP
        best = val["cat_mAP"]
        torch.save(head.state_dict(), f"head_{run}.pt")

head.load_state_dict(torch.load(f"head_{run}.pt"))
result = {"zero_shot_test": evaluate(None, 2), "trained_test": evaluate(head, 2),
          "best_epoch": max(history, key=lambda h: h["cat_mAP"])["epoch"], "history": history}
print(run, json.dumps({k: result[k] for k in ("zero_shot_test", "trained_test", "best_epoch")}, indent=1))

Path("results").mkdir(exist_ok=True)
results = json.loads(Path("results/results.json").read_text()) if Path("results/results.json").exists() else {}
results[run] = result
Path("results/results.json").write_text(json.dumps(results, indent=1))
