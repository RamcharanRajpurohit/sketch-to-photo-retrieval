"""Shared core: data list, frozen CLIP encoder, projection head, Grad-CAM stroke attribution."""
from glob import glob
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image, ImageFilter
from pytorch_grad_cam import GradCAM

BACKBONE = "ViT-B-32"  # chosen on validation: 5x more faithful stroke attribution than ViT-B-16 (README)
DATA = "data/sketchy"
CATEGORIES = ["airplane", "apple", "bicycle", "butterfly", "car_(sedan)", "cat", "chair", "cup", "dog", "fish",
              "flower", "guitar", "hat", "horse", "mushroom", "sailboat", "scissors", "shoe", "teapot", "umbrella"]


def list_images():
    """Photo paths, sketch paths, photo labels, and for every sketch the index of the photo it was drawn from."""
    photos, labels, sketches, pair = [], [], [], []
    for c, cat in enumerate(CATEGORIES):
        index = {}
        for p in sorted(glob(f"{DATA}/photo/tx_000000000000/{cat}/*.jpg")):
            index[Path(p).stem] = len(photos)
            photos.append(p)
            labels.append(c)
        for s in sorted(glob(f"{DATA}/sketch/tx_000000000000/{cat}/*.png")):
            sketches.append(s)
            pair.append(index[Path(s).stem.rsplit("-", 1)[0]])  # "n02691156_58-3" was drawn from photo "n02691156_58"
    return photos, torch.tensor(labels), sketches, torch.tensor(pair)


def split_photos(n):
    """Random 70/10/20 split by photo (0 train, 1 val, 2 test); a photo's sketches follow it."""
    return torch.bucketize(torch.rand(n, generator=torch.Generator().manual_seed(0)), torch.tensor([0.7, 0.8]))


def load_clip(backbone=BACKBONE):
    # OpenAI's weights were trained with QuickGELU; open_clip's plain names use GELU unless forced (open_clip #771)
    model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained="openai", force_quick_gelu=True)
    model.eval().requires_grad_(False)  # CLIP stays frozen; only the head is trained
    return model, preprocess


class Head(nn.Module):
    """One 512x512 linear map per domain, started at identity, so before training it is plain CLIP."""

    def __init__(self, dim=512):
        super().__init__()
        self.sketch, self.photo = nn.Linear(dim, dim), nn.Linear(dim, dim)
        for layer in (self.sketch, self.photo):
            nn.init.eye_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, domain):
        return F.normalize(getattr(self, domain)(x), dim=-1)


def load_system(run=f"{BACKBONE}_both_tau0.07"):
    """Frozen CLIP + trained head, the cached features, and all photos embedded once as the search gallery."""
    clip, preprocess = load_clip()
    head = Head()
    head.load_state_dict(torch.load(f"head_{run}.pt"))
    data = torch.load(f"features_{BACKBONE}.pt")
    with torch.no_grad():
        gallery = head(data["photo_feats"], "photo")
    return nn.Sequential(clip.visual, head.sketch).eval(), preprocess, data, gallery  # encoder: sketch -> embedding


def vit_reshape(tokens):
    """(N, 1+h*w, D) ViT tokens -> (N, D, h, w) map, dropping the CLS token (pytorch-grad-cam ViT recipe)."""
    n = int((tokens.shape[1] - 1) ** 0.5)
    return tokens[:, 1:].reshape(tokens.shape[0], n, n, -1).permute(0, 3, 1, 2)


def stroke_attribution(encoder, preprocess, sketch, photo_emb):
    """Grad-CAM heatmap of how much each region of the sketch raised its similarity to `photo_emb`.
    Returns (heatmap 224x224 in [0,1], image with every stroke pixel coloured by that heat)."""
    layer = encoder[0].transformer.resblocks[-1].ln_1  # last block: chosen on validation (README, 4.3)
    x = preprocess(sketch).unsqueeze(0).requires_grad_()  # CLIP is frozen, so gradients must start at the input
    with torch.enable_grad(), GradCAM(model=encoder, target_layers=[layer], reshape_transform=vit_reshape) as cam:
        # target = cosine similarity to one photo (pytorch-grad-cam "pixel attribution for embeddings")
        heat = cam(input_tensor=x, targets=[lambda e: F.cosine_similarity(e, photo_emb, dim=0)])[0]
    gray = sketch.convert("L").resize((224, 224)).filter(ImageFilter.MinFilter(3))  # thicken thin strokes
    strokes = np.array(gray) < 128
    h = heat[strokes]  # rescale over strokes only: ViT heat often peaks on empty background patches
    out = np.ones((224, 224, 3))
    out[strokes] = colormaps["jet"]((h - h.min()) / (h.max() - h.min() + 1e-8))[:, :3]
    return heat, Image.fromarray((out * 255).astype(np.uint8))
