"""Step 4: interactive demo. Draw a sketch -> top-5 photos + which strokes drove the match.
Edit the sketch and search again; the history table shows whether the edit helped. Usage: python app.py"""
import spaces  # Hugging Face ZeroGPU hosting (free); a no-op anywhere else. Must come before torch.
import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image

from sbir import CATEGORIES, load_system, stroke_attribution

encoder, preprocess, d, gallery = load_system()  # gallery = all 2000 photos, embedded once
photos, labels = d["photos"], d["labels"]


@spaces.GPU(duration=10)  # a call takes ~1 s; ZeroGPU budgets the declared duration (default 60 s) against the daily quota
def search(canvas, goal, history):
    if canvas is None or canvas["composite"] is None:
        raise gr.Error("Pick the brush on the left of the canvas and draw something first.")
    sketch = Image.alpha_composite(Image.new("RGBA", canvas["composite"].size, "white"),
                                   canvas["composite"].convert("RGBA")).convert("RGB")  # strokes on white
    with torch.no_grad():
        sims = gallery @ F.normalize(encoder(preprocess(sketch).unsqueeze(0)), dim=-1)[0]
    order = sims.argsort(descending=True)
    top = order[:5].tolist()
    goal_rank = (labels[order] == CATEGORIES.index(goal)).nonzero()[0].item() + 1 if goal else "-"
    history = history + [[len(history) + 1, CATEGORIES[labels[top[0]]], round(sims[top[0]].item(), 3), goal_rank]]
    results = [(photos[i], f"{CATEGORIES[labels[i]]}  {sims[i]:.2f}") for i in top]
    strokes = stroke_attribution(encoder, preprocess, sketch, gallery[top[0]])[1]
    return results, strokes, sketch, top, history, history


def select(evt: gr.SelectData, sketch, top):  # explain the clicked photo; SelectData can't be pickled into the GPU worker
    return explain(sketch, top[evt.index])


@spaces.GPU(duration=10)  # Grad-CAM (backward) must not run in the main process: it breaks later ZeroGPU workers
def explain(sketch, i):
    return stroke_attribution(encoder, preprocess, sketch, gallery[i])[1]


with gr.Blocks(title="Sketch to photo retrieval") as demo:
    gr.Markdown("### Sketch → photo search\nPick the brush ✎, draw, press **Search**. Red strokes mattered most; "
                "tap a result to explain it.")
    sketch_state, top_state, history_state = gr.State(), gr.State(), gr.State([])
    with gr.Row():
        with gr.Column():
            pad = gr.Sketchpad(type="pil", canvas_size=(400, 400), height=420, label="Your sketch",
                               brush=gr.Brush(colors=["#000000"], color_mode="fixed", default_size=5))
            button = gr.Button("Search", variant="primary")  # right under the canvas: visible on a phone without scrolling
            goal = gr.Dropdown(CATEGORIES, value=None, label="What are you drawing? (optional, tracks progress)")
        with gr.Column():
            results = gr.Gallery(label="Top-5 photos (click one to explain it)", columns=5, height=230)
            strokes = gr.Image(label="Stroke attribution", height=260)
            table = gr.Dataframe(headers=["attempt", "top-1 category", "top-1 similarity", "rank of first goal photo"],
                                 label="Refinement history")
    button.click(search, [pad, goal, history_state], [results, strokes, sketch_state, top_state, history_state, table],
                 scroll_to_output=True)  # on a phone the results sit below the canvas
    results.select(select, [sketch_state, top_state], strokes)

if __name__ == "__main__":
    demo.launch()
