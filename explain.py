"""Step 3: qualitative examples and a check that the stroke attribution is faithful.
Usage: python explain.py"""
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image

from sbir import BACKBONE, CATEGORIES, load_system, split_photos, stroke_attribution

encoder, preprocess, d, gallery = load_system()
labels, pair = d["labels"], d["pair"]
test = (split_photos(len(labels))[pair] == 2).nonzero().squeeze(1).tolist()  # sketches of test photos
random.seed(0)
np.random.seed(0)


@torch.no_grad()
def embed(imgs):
    return F.normalize(encoder(torch.stack([preprocess(img) for img in imgs])), dim=-1)


def load_sketch(i):
    return Image.open(d["sketches"][i]).convert("RGB").resize((224, 224))  # same grid as the heatmap


# 1) Qualitative figure: sketch | stroke attribution | heatmap | top-5 photos (green = right category, * = exact photo)
rows = random.sample(test, 6)
fig, ax = plt.subplots(len(rows), 8, figsize=(16, 2.3 * len(rows)))
for r, i in enumerate(rows):
    sketch = load_sketch(i)
    top = (gallery @ embed([sketch])[0]).topk(5).indices.tolist()
    heat, strokes = stroke_attribution(encoder, preprocess, sketch, gallery[top[0]])
    cam = show_cam_on_image(np.asarray(sketch, dtype=np.float32) / 255, heat, use_rgb=True)
    for c, img in enumerate([sketch, strokes, cam] + [Image.open(d["photos"][j]) for j in top]):
        ax[r, c].imshow(img)
        ax[r, c].set_xticks([]), ax[r, c].set_yticks([])
    for c, j in enumerate(top, 3):
        plt.setp(ax[r, c].spines.values(), color="green" if labels[j] == labels[pair[i]] else "red", linewidth=4)
        ax[r, c].set_title(CATEGORIES[labels[j]] + (" *" if j == pair[i] else ""), fontsize=9)
    ax[r, 0].set_ylabel(CATEGORIES[labels[pair[i]]])
for c, t in enumerate(["sketch", "stroke attribution", "Grad-CAM (top-1)"]):
    ax[0, c].set_title(t, fontsize=9)
fig.tight_layout()
fig.savefig(f"results/qualitative_{BACKBONE}.png", dpi=110)

# 2) Deletion curve (RISE, Petsiuk et al. 2018, arXiv 1806.07421): erase stroke pixels from most to least important
#    and track the similarity to the matched photo. A faithful map makes it fall faster than a random order (lower AUC).
#    Erasing ink to white keeps a sketch a sketch, so the erased images stay close to the data (cf. Gomez et al. 2022).
fractions = np.linspace(0, 1, 11)
curves = {"Grad-CAM order": [], "random order": []}
for i in random.sample(test, 200):
    sketch = load_sketch(i)
    photo = gallery[(gallery @ embed([sketch])[0]).argmax()]
    heat, _ = stroke_attribution(encoder, preprocess, sketch, photo)
    ys, xs = (np.array(sketch.convert("L")) < 200).nonzero()  # stroke pixels
    for name, order in (("Grad-CAM order", heat[ys, xs].argsort()[::-1]), ("random order", np.random.permutation(len(ys)))):
        steps = []
        for f in fractions:
            erased, k = np.array(sketch), order[:int(f * len(ys))]
            erased[ys[k], xs[k]] = 255
            steps.append(Image.fromarray(erased))
        curves[name].append((embed(steps) @ photo).tolist())
auc = {name: round(float(np.trapezoid(np.mean(c, 0), fractions)), 4) for name, c in curves.items()}
print("deletion AUC (lower = more faithful):", auc)
results = json.loads(Path("results/results.json").read_text())
results[f"deletion_auc_{BACKBONE}"] = auc
Path("results/results.json").write_text(json.dumps(results, indent=1))
