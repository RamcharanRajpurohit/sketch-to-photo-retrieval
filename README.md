# Sketch-to-photo retrieval with stroke attribution

Draw a sketch and the system finds matching real photos. It also colours **your strokes** by how much each one
drove the match (Grad-CAM), so you can see what the model "looked at", edit the sketch, and search again.

```
sketch ─► CLIP ViT-B/32 image encoder (frozen) ─► sketch head (512x512 linear) ─┐
                                                                                ├─► cosine similarity ─► top-5 photos
photo  ─► CLIP ViT-B/32 image encoder (frozen) ─► photo head  (512x512 linear) ─┘
                │
                └─ Grad-CAM: gradient of "similarity to the chosen photo" w.r.t. the last ViT block
                   ─► 7x7 heatmap ─► upsampled and painted onto the stroke pixels only
```

## 1. Files (run in this order)

| file | what it does | time |
|---|---|---|
| `sbir.py` | shared code: 20 categories, data list, photo split, CLIP loader, the head, Grad-CAM stroke attribution | – |
| `embed.py` | encodes the 2,000 photos and 12,176 sketches once with frozen CLIP, saves `features_<backbone>.pt` | ~5 min on the server |
| `train.py` | trains the head on the saved features (contrastive loss), evaluates on the test split | ~20 s on the laptop |
| `explain.py` | qualitative figure + deletion test (is the attribution faithful?) | ~4 min on the server |
| `app.py` | Gradio demo: draw, search, see stroke attribution, edit, search again | 0.34 s per search on the laptop |

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -p .venv open_clip_torch grad-cam gradio matplotlib spaces
# data: Sketchy (Sangkloy et al. 2016), Hugging Face mirror DrRORAL/sketchy-dataset, extracted to data/sketchy/
.venv/bin/python embed.py                     # ViT-B-32 (default); RN50 and ViT-B-16 were run for the comparison
.venv/bin/python train.py                     # = train.py ViT-B-32 both 0.07  (backbone, loss, temperature)
.venv/bin/python explain.py
.venv/bin/python app.py                       # open http://127.0.0.1:7860, pick the brush, draw, Search
```

Only `embed.py` and `explain.py` are heavy (CLIP over thousands of images), so they ran on the CPU server
(the IITJ CPU server, 128 cores). Training and the app run on the laptop CPU.

**Live demo (free):** https://huggingface.co/spaces/AaravDB/sketch-to-photo-retrieval, a Hugging Face Space on
ZeroGPU (the only free Gradio hardware for free accounts). `search` is wrapped in `@spaces.GPU(duration=10)`
because ZeroGPU charges the declared duration (default 60 s) against each visitor's daily quota (2 min logged out,
5 min logged in). Click-to-explain runs on CPU because Gradio's click event can't be passed into the GPU worker.
The Space holds the photos as one zip that a 3-line `start.py` unpacks at startup. The demo sleeps after 48 h
without visitors, and the first visit after that takes about a minute to wake it.

## 2. Data

* **Sketchy** (Sangkloy et al., 2016): 125 categories, 12,500 photos, 75,471 sketches. Every sketch was drawn from
  one specific photo, so we know its category **and** its exact photo.
* **20 easy-to-draw categories** (airplane, apple, bicycle, butterfly, car, cat, chair, cup, dog, fish, flower,
  guitar, hat, horse, mushroom, sailboat, scissors, shoe, teapot, umbrella): 2,000 photos, 12,176 sketches.
* **Split by photo**, 70 / 10 / 20 (random, seed 0): train 1,397 photos (8,496 sketches), val 204 (1,261),
  test 399 (2,419). A test sketch's photo is never seen in training. **Every choice was made on val**; test is
  reported once.

## 3. Method

1. **Frozen CLIP ViT-B/32** (OpenAI weights) embeds sketches and photos into 512-d. OpenAI's weights were trained
   with QuickGELU, and open_clip's plain model names use GELU unless `force_quick_gelu=True` (open_clip issue #771),
   so we set it.
2. **Projection head**: one 512x512 linear layer for sketches and one for photos, both **initialised to the
   identity**, so before training the system is exactly zero-shot CLIP (same idea as the linear adapter of ProLIP,
   arXiv 2410.05270). Only these 525 k parameters are trained (the CLIP image encoder has 88 M), on cached features, in seconds.
3. **Loss: CLIP-style InfoNCE.** In a batch of 256 sketches and their photos, each sketch must pick its positives
   among all photos (softmax over cosine similarity / temperature), in both directions. Two kinds of positives are
   added together: *instance* (the photo it was drawn from) + *category* (any photo of the same class).
4. **Training**: Adam, lr 1e-3, temperature 0.07, 30 epochs, keep the epoch with the best **val mAP**.
5. **Retrieval**: cosine similarity between the sketch and all 2,000 photos, top 5.
6. **Stroke attribution (Grad-CAM)**: target = cosine similarity between the sketch and the chosen photo
   ("pixel attribution for embeddings", pytorch-grad-cam). Layer = `ln_1` of the last ViT block; its 49 patch tokens
   are reshaped to 7x7 (the library's ViT recipe). The map is upsampled to 224x224 and **painted on stroke pixels
   only**, rescaled over the strokes. Why only strokes: ViTs park global information in empty background patches
   (Darcet et al., *Vision Transformers Need Registers*, ICLR 2024), so the raw map often peaks on blank paper
   (see the Grad-CAM column of `results/qualitative_ViT-B-32.png`, apple and sailboat rows).
7. **Refinement loop (app)**: draw → Search → red strokes = drove the match, blue = barely mattered → add or erase
   strokes → Search again. Clicking a result explains that photo. A history table shows the top-1 category,
   similarity, and the rank of the first photo of the category you say you are drawing.

## 4. Results

Metrics (standard in SBIR papers):
* **Category level**: Acc@1 (top-1 photo has the right category), P@10, **mAP@all** (the main SBIR metric).
  Papers on full Sketchy use P@200, but our test gallery has only ~20 photos per class, so we use P@10.
* **Instance level** (fine-grained SBIR): **Acc@1 / Acc@10** = % of sketches whose own photo is in the top 1 / 10.

### 4.1 Main result (test: 2,419 sketches vs 399 photos)

| backbone | | cat Acc@1 | cat P@10 | cat mAP@all | inst Acc@1 | inst Acc@10 |
|---|---|---|---|---|---|---|
| RN50 | zero-shot CLIP | 49.8 | 39.0 | 0.347 | 3.7 | 22.3 |
| RN50 | + trained head | 89.8 | 88.2 | 0.872 | 13.3 | 68.4 |
| ViT-B/16 | zero-shot CLIP | 68.8 | 50.1 | 0.416 | 5.5 | 31.3 |
| ViT-B/16 | + trained head | 94.0 | 93.2 | 0.928 | 17.2 | 72.1 |
| ViT-B/32 | zero-shot CLIP | 56.0 | 46.9 | 0.402 | 5.7 | 28.7 |
| **ViT-B/32** | **+ trained head (final)** | **92.5** | **91.4** | **0.903** | **17.7** | **72.0** |

* Training only the small head more than doubles mAP (0.40 → 0.90): CLIP knows the objects, but puts sketches and
  photos in different parts of its space; the head aligns them.
* **Why ViT-B/32 and not ViT-B/16?** ViT-B/16 retrieves a bit better (val mAP 0.925 vs 0.901), but its stroke
  attribution is almost random on val (deletion AUC only 1.6 % below random, vs 6.8 % for ViT-B/32, section 4.3),
  and it is 5x slower (1.7 s vs 0.34 s per search on the laptop). Attribution is the point of this project, so
  ViT-B/32. RN50 is worse than both on val (mAP 0.876).

### 4.2 Loss and temperature (decided on val, ViT-B/32)

| loss positives | temperature | val mAP | val inst Acc@10 | best epoch |
|---|---|---|---|---|
| instance only | 0.07 | 0.857 | 87.1 | 1 |
| instance only | 0.01 | 0.811 | 86.7 | 2 |
| category only | 0.07 | **0.920** | 82.6 | 13 |
| category only | 0.01 | 0.909 | 81.8 | 24 |
| **instance + category** | **0.07** | 0.901 | **89.0** | 5 |
| instance + category | 0.01 | 0.881 | 84.9 | 29 |

* **Instance only** overfits after epoch 1: train loss keeps falling (3.11 → 0.73) while val mAP falls
  (0.857 → 0.663 at epoch 30), because it pushes apart photos of the same class.
* **Category only** has the best mAP but is bad at finding the exact photo.
* **Both** has the best sum (mAP + inst Acc@10 = 1.790 vs 1.747 and 1.727) → final loss.
* **Temperature**: CLIP starts training at 0.07 and ends at 0.01 (its learned `logit_scale` is 100, read from the
  weights). Val prefers 0.07 in every row.
* **Tried and dropped**: ProLIP's regulariser λ‖W − I‖² with λ = 1/N (arXiv 2410.05270, Eq. 8). With ~430
  training sketches per class it changed nothing (val mAP 0.898 with it vs 0.901 without for "both"; 0.909 vs
  0.920 for "category"): it was designed for ≤16 examples per class. Early stopping on val is enough here.

### 4.3 Is the stroke attribution faithful? Deletion test

Following RISE (Petsiuk et al., arXiv 1806.07421): erase stroke pixels from most to least important (Grad-CAM order)
in 10 % steps and track the similarity to the matched photo; compare with erasing in random order. A faithful map
makes the similarity fall faster → **lower area under the curve (AUC)**. Erasing ink to white keeps a sketch a
sketch, so the erased images stay close to real sketches (the main criticism of deletion metrics is
out-of-distribution masked images, Gomez et al., arXiv 2201.13291).

**Choosing the Grad-CAM layer (val, 100 sketches).** For ViTs the layer is a real choice, not a given, so we
measured it:

| backbone | random | last block | block −2 | block −4 | flat (useless) maps, last / −2 / −4 |
|---|---|---|---|---|---|
| ViT-B/16 | 0.507 | 0.503 | 0.501 | 0.499 | – |
| **ViT-B/32** | 0.485 | **0.452** | 0.458 | 0.447 | **0** / 2 / 3 of 60 |

Block −4 is only 0.005 better than the last block but gives a flat map (all strokes the same colour) on 5 % of
sketches, so we use the **last block**, the library default.

**Test (200 sketches)**: ViT-B/32 AUC **0.449** with Grad-CAM order vs **0.476** random (5.6 % lower → faithful);
ViT-B/16 0.495 vs 0.500 (barely better than random).

### 4.4 Qualitative results and the refinement loop

* `results/qualitative_ViT-B-32.png`: 6 random test sketches → stroke attribution, raw Grad-CAM, top-5 photos
  (green = right category, `*` = the exact photo). The model looks at the parts that define the object: the
  horse's **mane**, the scissors' **blade tips**, the cup's **handle side**, the **top of the sail**.
* Refinement in the real app (screenshots `results/app_*.png`): a base drawing, then one edit.

| goal | attempt 1 | edit | attempt 2 | what the attribution shows |
|---|---|---|---|---|
| mushroom | cap only → top-1 **hat**, first mushroom at rank 49 | add a stem | **mushroom**, rank 1 (sim 0.52 → 0.62) | the stem is red, the cap blue |
| fish | oval only → top-1 **hat**, first fish at rank 492 | add tail + eye | **fish**, rank 1 (sim 0.56 → 0.58) | the eye is red |

## 5. Limitations

* 20 categories and a 2,000-photo gallery (full Sketchy has 125 categories).
* Exact-photo retrieval is still hard (17.7 % Acc@1): the head is linear and CLIP is frozen.
* Grad-CAM on ViT-B/32 is coarse (7x7 patches of 32 px) and is only modestly better than random erasing.
* The refinement loop was shown on our own drawings, not with a user study.

## 6. Viva cheat-sheet (short answers)

* **Why CLIP?** Trained on 400 M image–text pairs, it already knows what a cup or a horse looks like, even roughly.
  We only need to align sketches with photos.
* **Why freeze CLIP and train a head?** Small data (8.5 k sketches) and a CPU. The head has 525 k parameters and
  trains in seconds on cached features; fine-tuning CLIP's 88 M would overfit and need a GPU.
* **Why identity initialisation?** At epoch 0 the system is zero-shot CLIP, so training starts from a good point.
* **What is InfoNCE?** A softmax classification: "which photo in this batch belongs to my sketch?" Cross-entropy over
  cosine similarities divided by a temperature, both ways (sketch→photo and photo→sketch), like CLIP.
* **What is temperature?** It sharpens the softmax; smaller = only the hardest negatives matter. We tried CLIP's
  start (0.07) and end (0.01) values; val picked 0.07.
* **Why instance + category positives?** Instance alone forgets categories, category alone forgets the exact photo;
  together is best on val.
* **What is Grad-CAM here?** Take the similarity between the sketch and one photo, back-propagate it to the last ViT
  block, weight each of the 7x7 patch features by its average gradient, sum, ReLU. Bright = patches that raised the
  similarity.
* **Why paint only the strokes?** The question is "which strokes mattered"; and ViT heat often lands on blank paper
  (register-like background tokens).
* **How do you know the explanation is right?** Deletion test: erasing the strokes Grad-CAM calls important lowers
  the similarity faster than erasing random strokes (AUC 0.449 vs 0.476).
* **Why ViT-B/32 if ViT-B/16 retrieves better?** Its attribution was nearly random and it is 5x slower; the
  attribution is the novelty.
* **mAP@all?** For each sketch, average the precision at every rank where a right-category photo appears, then
  average over sketches. 1.0 = all right photos ranked first.
* **Why split by photo?** So no test photo (or another sketch of it) is seen in training: no leakage.
* **Why the server?** Only for the one-time CLIP passes over thousands of images (laptop CPU: ~1–8 images/s).

## 7. References

* Sangkloy et al., *The Sketchy Database: Learning to Retrieve Badly Drawn Bunnies*, ACM TOG 2016
  (HF mirror `DrRORAL/sketchy-dataset`).
* Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), ICML 2021,
  arXiv 2103.00020; OpenCLIP https://github.com/mlfoundations/open_clip (QuickGELU: issue #771).
* Fahes et al., *CLIP's Visual Embedding Projector is a Few-shot Cornucopia* (ProLIP), WACV 2026, arXiv 2410.05270.
* Selvaraju et al., *Grad-CAM*, arXiv 1610.02391 (IJCV); pytorch-grad-cam https://jacobgil.github.io/pytorch-gradcam-book
* Darcet et al., *Vision Transformers Need Registers*, ICLR 2024, arXiv 2309.16588.
* Petsiuk et al., *RISE*, arXiv 1806.07421; Gomez et al., *Metrics for saliency map evaluation*, arXiv 2201.13291.
* SBIR metrics: mAP@all and P@K (e.g. arXiv 2401.04860, 2404.18695); Acc@1/Acc@10 for fine-grained SBIR
  (e.g. arXiv 2210.10486).
