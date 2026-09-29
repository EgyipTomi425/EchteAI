"""Shared figure style (validated categorical palette, light print surface, thin marks)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Categorical slots 1-4 in bar order FP32, FP16, INT8, FP8 (validated: adjacent CVD dE >= 9.1, normal-vision dE >= 22.9 on the light
# surface; slots 3-4 are below 3:1 contrast, so every chart using them is direct-labelled or has a
# table counterpart in the manuscript).
PRECISION_COLORS = {"fp32": "#2a78d6", "fp16": "#eb6834", "int8": "#1baf7a", "int8fp32": "#e87ba4",
                    "fp8": "#eda100"}
PRECISION_LABELS = {"fp32": "FP32", "fp16": "FP16", "int8": "INT8", "int8fp32": "INT8 (FP32 fallback)",
                    "fp8": "FP8"}
MODEL_LABELS = {"frcnn_r50_fpn": "Faster R-CNN R50-FPN", "yolov10s": "YOLOv10-S", "yolov10x": "YOLOv10-X",
                "efficientnet_b0": "EfficientNet-B0", "densenet121": "DenseNet-121"}
# Detector identity where precision colours are not in use: categorical slots 7, 8, 6 (violet, red,
# green). Validated all-pairs on the light surface; the green-red CVD distance (7.2) is in the 6-8 band,
# so every chart using these colours also encodes the model by marker and line style.
MODEL_COLORS = {"frcnn_r50_fpn": "#4a3aa7", "yolov10s": "#e34948", "yolov10x": "#008300"}
MODEL_MARKERS = {"frcnn_r50_fpn": ("o", "-"), "yolov10s": ("s", "--"), "yolov10x": ("^", ":")}
SEQUENTIAL = "Blues"
BLUE_700 = "#0d366b"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2, "lines.markersize": 5,
        "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 8,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
