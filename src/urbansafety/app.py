"""Démonstration Gradio : recherche, analyse et description de vidéos."""
from __future__ import annotations

from pathlib import Path


def _thumbnail(video_path: Path, size=(320, 180)):
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 3)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return Image.new("RGB", size, (40, 40, 40))
    return Image.fromarray(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), size))


def _timeline(scores, duration: float, threshold: float = 0.5):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    t = np.linspace(0, duration, len(scores))
    fig = plt.figure(figsize=(12, 4))
    plt.plot(t, scores, linewidth=2)
    plt.fill_between(t, 0, scores, alpha=0.3)
    plt.axhline(threshold, color="red", linestyle="--", label="seuil")
    plt.xlabel("temps (s)")
    plt.ylabel("score d'anomalie")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return fig


def build_demo(system):
    import cv2
    import gradio as gr

    def search(query, k):
        if not query.strip():
            return [], "Saisissez une requête.", None
        hits = system.search_videos(query, int(k))
        gallery = [(_thumbnail(Path(h["path"])), f"#{i} {h['score']:.3f}\n{Path(h['path']).stem[:25]}")
                   for i, h in enumerate(hits, 1)]
        best = next((h["path"] for h in hits if Path(h["path"]).exists()), None)
        return gallery, f"« {query} » : {len(hits)} vidéos", best

    def analyze(video):
        if video is None:
            return "Aucune vidéo fournie.", None
        cap = cv2.VideoCapture(str(video))
        duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / (cap.get(cv2.CAP_PROP_FPS) or 30)
        cap.release()
        r = system.analyze(system.features(Path(video)))
        lines = [f"Vidéo : {Path(video).name}", f"Durée : {duration:.1f} s, {len(r['scores'])} segments",
                 f"Verdict : {'ANOMALIE DÉTECTÉE' if r['anomaly'] else 'COMPORTEMENT NORMAL'}",
                 f"Score max {r['max']:.4f}, moyen {r['mean']:.4f}"]
        if r["anomaly"]:
            lines.append(f"Pic : segment {r['peak_segment']} (~{r['peak_segment'] / len(r['scores']) * duration:.1f} s)")
            if r["description"]:
                lines.append(f"Description la plus proche : {r['description'][:200]}")
        return "\n".join(lines), _timeline(r["scores"], duration)

    def describe(video):
        if video is None:
            return "Aucune vidéo fournie."
        hits = system.describe(Path(video))
        return "\n\n".join(f"{i}. (similarité {h['score']:.3f}) {h['text']}" for i, h in enumerate(hits, 1))

    with gr.Blocks(title="UrbanSafety") as demo:
        gr.Markdown("# UrbanSafety\nDétection d'anomalies et recherche sémantique en vidéosurveillance "
                    "(OpenCLIP, FAISS, BiLSTM en MIL)")
        with gr.Tab("Recherche"):
            query = gr.Textbox(label="Requête", placeholder="A person stealing a bag, car accident...")
            k = gr.Slider(1, 10, value=5, step=1, label="Nombre de résultats")
            button = gr.Button("Rechercher", variant="primary")
            summary = gr.Textbox(label="Résumé")
            gallery = gr.Gallery(columns=3, allow_preview=False)
            player = gr.Video(label="Meilleur résultat")
            button.click(search, [query, k], [gallery, summary, player])
        with gr.Tab("Analyse"):
            video_in = gr.Video(label="Vidéo")
            gr.Button("Analyser", variant="primary").click(
                analyze, video_in, [gr.Textbox(label="Rapport", lines=6), gr.Plot(label="Scores temporels")])
        with gr.Tab("Description"):
            video_desc = gr.Video(label="Vidéo")
            gr.Button("Décrire", variant="primary").click(describe, video_desc, gr.Textbox(label="Descriptions", lines=10))
    return demo
