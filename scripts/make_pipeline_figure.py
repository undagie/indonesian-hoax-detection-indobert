"""Render the pipeline and proposed-model architecture figure."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)


C_INPUT = ("#DCE6F2", "#3C3C3C")     # data / input
C_ENC = ("#C9D8EC", "#3C3C3C")       # encoder utama
C_MID = ("#ECECEC", "#3C3C3C")       # intermediate
C_OUT = ("#D8E4D0", "#3C3C3C")       # output / akhir
C_PROP = ("#F6D9A8", "#3C3C3C")      # proposed / loss (tebal)
TEXT = "#1A1A1A"
SUBT = "#444444"
ARROW = "#3C3C3C"

X_L, X_R = 0.05, 0.95
CX = (X_L + X_R) / 2
W = X_R - X_L

GAP_ARROW = 0.008
GAP_SECTION = 0.022
H_LABEL = 0.018
LINE_H = 0.0125
PAD_BOX = 0.012
BOX_PAD = 0.003


@dataclass
class Cursor:
    y: float

    def take(self, dy: float) -> float:
        top = self.y
        self.y -= dy
        return top


def box(ax, cur: Cursor, title: str, body: list[str], colors,
        *, fontsize=7.1, title_size=7.3, lh=1.25, highlight=False):
    """Draw a labelled box (bold title line, regular body)."""
    n_lines = 1 + len(body)
    h = n_lines * LINE_H + PAD_BOX
    top = cur.take(h)
    face, edge = colors
    patch = FancyBboxPatch(
        (X_L, top - h), W, h,
        boxstyle=f"round,pad={BOX_PAD},rounding_size=0.020",
        linewidth=1.6 if highlight else 1.0,
        facecolor=face, edgecolor=edge, zorder=2,
    )
    ax.add_patch(patch)
    title_y = top - PAD_BOX / 2 - LINE_H / 2
    ax.text(CX, title_y, title,
            ha="center", va="center",
            fontsize=title_size, color=TEXT, weight="bold", zorder=3)
    for i, line in enumerate(body, start=1):
        ly = title_y - i * LINE_H
        ax.text(CX, ly, line,
                ha="center", va="center",
                fontsize=fontsize, color=TEXT, linespacing=lh, zorder=3)
    return top, top - h


def arrow(ax, cur: Cursor, *, dashed=False):
    top = cur.take(GAP_ARROW + 0.004)
    arr = FancyArrowPatch(
        (CX, top), (CX, top - GAP_ARROW),
        arrowstyle="-|>", mutation_scale=9,
        linewidth=1.0, color=ARROW,
        linestyle="--" if dashed else "-",
        zorder=1,
    )
    ax.add_patch(arr)


def section(ax, cur: Cursor, text: str):
    cur.take(GAP_SECTION)
    top = cur.take(H_LABEL)
    ax.text(X_L, top - H_LABEL / 2, text,
            ha="left", va="center",
            fontsize=7.4, color=TEXT, weight="bold")


def render(out_path_noext: str) -> list[str]:
    fig = plt.figure(figsize=(3.4, 11.6), dpi=600)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(CX, 0.985,
            "Pipeline dan Arsitektur Model Usulan",
            ha="center", va="top", fontsize=10.0, weight="bold", color=TEXT)
    ax.text(CX, 0.962,
            "Deteksi berita palsu berbahasa Indonesia berbasis IndoBERT-base\n"
            "dengan augmentasi data dan penanganan ketidakseimbangan kelas",
            ha="center", va="top", fontsize=7.1, color=SUBT, linespacing=1.25)

    cur = Cursor(y=0.928)

    section(ax, cur, "Tahap 1.  Penyiapan Data")
    box(ax, cur,
        "Korpus Kaggle indonesiafalsenews",
        ["Data_latih.csv, 4231 artikel berlabel"], C_INPUT)
    arrow(ax, cur)
    box(ax, cur,
        "Pra-pemrosesan",
        ["gabung judul dengan narasi,",
         "normalisasi, hapus URL, HTML"], C_INPUT)
    arrow(ax, cur)
    box(ax, cur,
        "Pembagian stratified",
        ["latih 70, validasi 10, uji 20 persen"], C_INPUT)

    section(ax, cur, "Tahap 2.  Augmentasi Data (kelas minoritas)")
    arrow(ax, cur)
    box(ax, cur,
        "Augmentasi pada subset latih",
        ["EDA, back-translation ID-EN,",
         "kontekstual berbasis IndoBERT MLM",
         "rasio 1,0, prob 0,15, minoritas saja"], C_MID)

    section(ax, cur, "Tahap 3.  Penanganan Ketidakseimbangan")
    arrow(ax, cur)
    box(ax, cur,
        "Strategi yang dievaluasi",
        ["class weight, weighted sampler,",
         "random over / under sampling,",
         "SMOTE pada embedding,",
         "focal, class-balanced, LDAM loss"], C_MID)

    section(ax, cur, "Tahap 4.  Model Usulan - IndoBERT-base")

    arrow(ax, cur)
    model_top, _ = box(ax, cur,
        "Tokenisasi WordPiece IndoBERT",
        ["panjang maksimum 384 token"], C_INPUT)

    arrow(ax, cur)
    box(ax, cur,
        "Embedding",
        ["token + segmen + posisi"], C_MID)

    arrow(ax, cur)
    box(ax, cur,
        "Encoder IndoBERT-base",
        ["12 lapisan transformer,",
         "dimensi tersembunyi 768,",
         "110 juta parameter"], C_ENC)

    arrow(ax, cur)
    box(ax, cur,
        "Representasi token CLS",
        ["vektor konteks 768 dimensi"], C_MID)

    arrow(ax, cur)
    box(ax, cur,
        "Classification head",
        ["dropout 0,1, linear 768 ke 2,",
         "softmax"], C_MID)

    arrow(ax, cur)
    _, sm_bot = box(ax, cur,
        "Probabilitas kelas keluaran",
        ["hoaks atau fakta"], C_OUT)

    arrow(ax, cur)
    box(ax, cur,
        "Fungsi kerugian gabungan",
        ["class weighting dikalikan focal loss"], C_PROP, highlight=True)

    section(ax, cur, "Tahap 5.  Pelatihan")
    arrow(ax, cur)
    box(ax, cur,
        "Optimisasi multi-seed",
        ["AdamW, lr 2e-5, weight decay 0,01",
         "warmup linear 10 persen + decay linear",
         "AMP, grad-clip 1,0,",
         "early-stop pada F1-macro",
         "seed 7, 42, 101, 1234, 1137"], C_MID)

    section(ax, cur, "Tahap 6.  Evaluasi dan Pelaporan")
    arrow(ax, cur)
    box(ax, cur,
        "Metrik dan uji signifikansi",
        ["accuracy, precision, recall,",
         "F1-macro, ROC-AUC, PR-AUC,",
         "bootstrap 95 persen CI,",
         "uji McNemar dan paired t-test"], C_OUT)

    os.makedirs(os.path.dirname(out_path_noext), exist_ok=True)
    png = out_path_noext + ".png"           # 600 DPI (HD default)
    png_hd = out_path_noext + "_hd.png"     # 1200 DPI (ultra-HD)
    pdf = out_path_noext + ".pdf"           # vektor
    svg = out_path_noext + ".svg"           # vektor (edit)
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png_hd, dpi=1200, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return [png, png_hd, pdf, svg]


if __name__ == "__main__":
    out = os.path.join(REPO, "paper", "figures", "gambar3_pipeline_model")
    for p in render(out):
        print(f"[ok] saved: {p}")
