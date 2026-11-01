"""Step 1: encode every photo and sketch once with frozen CLIP and save the features.
Usage: python embed.py [backbone]   (run once; it is the only slow step)"""
import sys

import torch
import torch.nn.functional as F
from PIL import Image

from sbir import BACKBONE, list_images, load_clip

backbone = sys.argv[1] if len(sys.argv) > 1 else BACKBONE
model, preprocess = load_clip(backbone)
photos, labels, sketches, pair = list_images()


@torch.no_grad()
def encode(paths, batch=256):
    feats = []
    for i in range(0, len(paths), batch):
        x = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[i:i + batch]])
        feats.append(F.normalize(model.encode_image(x), dim=-1))
        print(f"{backbone}: {i + len(x)}/{len(paths)}", flush=True)
    return torch.cat(feats)


torch.save(dict(photos=photos, labels=labels, sketches=sketches, pair=pair,
                photo_feats=encode(photos), sketch_feats=encode(sketches)), f"features_{backbone}.pt")
