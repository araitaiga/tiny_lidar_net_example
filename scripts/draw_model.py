"""Generate an architecture diagram of TinyLidarNet (in the style of NVIDIA PilotNet / the paper's Fig.3)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


# -------- Color palette (academic blue-based) --------
C_INPUT = "#E8F1FA"
C_INPUT_EDGE = "#1F4E79"
C_CONV = "#4F81BD"
C_CONV_EDGE = "#1F4E79"
C_FLAT = "#9E9E9E"
C_FLAT_EDGE = "#424242"
C_FC = "#F4B183"
C_FC_EDGE = "#B45F06"
C_OUT = "#C00000"
C_TEXT = "#1F3864"


def draw_conv_block(ax, x_center, y_center, width, height, depth, label_top, label_bottom):
    """Draw a Conv feature map in a 3D-like style."""
    # Side and top parallelograms that represent depth
    half_w = width / 2
    half_h = height / 2
    # Front face
    front = mpatches.Rectangle(
        (x_center - half_w, y_center - half_h),
        width,
        height,
        facecolor=C_CONV,
        edgecolor=C_CONV_EDGE,
        linewidth=1.4,
        zorder=3,
    )
    ax.add_patch(front)
    # Top face (parallelogram)
    top = Polygon(
        [
            (x_center - half_w, y_center + half_h),
            (x_center - half_w + depth, y_center + half_h + depth * 0.5),
            (x_center + half_w + depth, y_center + half_h + depth * 0.5),
            (x_center + half_w, y_center + half_h),
        ],
        facecolor="#7FA9D6",
        edgecolor=C_CONV_EDGE,
        linewidth=1.2,
        zorder=2,
    )
    ax.add_patch(top)
    # Side face
    side = Polygon(
        [
            (x_center + half_w, y_center + half_h),
            (x_center + half_w + depth, y_center + half_h + depth * 0.5),
            (x_center + half_w + depth, y_center - half_h + depth * 0.5),
            (x_center + half_w, y_center - half_h),
        ],
        facecolor="#3A6FA5",
        edgecolor=C_CONV_EDGE,
        linewidth=1.2,
        zorder=2,
    )
    ax.add_patch(side)

    # Labels
    ax.text(
        x_center,
        y_center + half_h + depth * 0.5 + 0.35,
        label_top,
        ha="center",
        va="bottom",
        fontsize=9,
        color=C_TEXT,
        fontweight="bold",
    )
    ax.text(
        x_center,
        y_center - half_h - 0.35,
        label_bottom,
        ha="center",
        va="top",
        fontsize=8,
        color=C_TEXT,
    )


def draw_fc_block(ax, x_center, y_center, height, neurons_label, role_label):
    """Draw a fully-connected layer as a tall rectangle."""
    width = 0.55
    rect = FancyBboxPatch(
        (x_center - width / 2, y_center - height / 2),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=C_FC,
        edgecolor=C_FC_EDGE,
        linewidth=1.3,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(
        x_center,
        y_center,
        neurons_label,
        ha="center",
        va="center",
        fontsize=9,
        color="#1A1A1A",
        fontweight="bold",
        rotation=90,
    )
    ax.text(
        x_center,
        y_center - height / 2 - 0.35,
        role_label,
        ha="center",
        va="top",
        fontsize=8,
        color=C_TEXT,
    )


def draw_arrow(ax, x_start, x_end, y, text=None):
    arr = FancyArrowPatch(
        (x_start, y),
        (x_end, y),
        arrowstyle="-|>",
        mutation_scale=14,
        color="#404040",
        linewidth=1.2,
        zorder=1,
    )
    ax.add_patch(arr)
    if text is not None:
        ax.text(
            (x_start + x_end) / 2,
            y + 0.18,
            text,
            ha="center",
            va="bottom",
            fontsize=7,
            color="#404040",
            style="italic",
        )


def main(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(18, 8.5), dpi=200)
    ax.set_xlim(-0.2, 19.5)
    ax.set_ylim(-1.2, 9.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # ------------- Title -------------
    ax.text(
        9.5,
        9.0,
        "TinyLidarNet : 1D CNN Architecture",
        ha="center",
        va="center",
        fontsize=16,
        color=C_TEXT,
        fontweight="bold",
    )
    ax.text(
        9.5,
        8.45,
        "Input : 1081-point LiDAR scan   →   Output : (Steering, Speed)",
        ha="center",
        va="center",
        fontsize=11,
        color=C_TEXT,
    )

    y_main = 3.8

    # ------------- Input (LiDAR scan) -------------
    input_x = 0.9
    input_w = 0.5
    input_h = 4.8
    rect = mpatches.Rectangle(
        (input_x - input_w / 2, y_main - input_h / 2),
        input_w,
        input_h,
        facecolor=C_INPUT,
        edgecolor=C_INPUT_EDGE,
        linewidth=1.5,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(
        input_x,
        y_main,
        "1 × 1081",
        ha="center",
        va="center",
        fontsize=10,
        color=C_TEXT,
        fontweight="bold",
        rotation=90,
    )
    ax.text(
        input_x,
        y_main + input_h / 2 + 0.35,
        "Input",
        ha="center",
        va="bottom",
        fontsize=10,
        color=C_TEXT,
        fontweight="bold",
    )
    ax.text(
        input_x,
        y_main - input_h / 2 - 0.35,
        "LiDAR Scan\n(1081 ranges)",
        ha="center",
        va="top",
        fontsize=8,
        color=C_TEXT,
    )

    # ------------- Conv layers -------------
    # (x_center, width_visual (time-axis shrinkage), height_visual (channels), depth)
    # Tuned so the time-axis shrinkage and channel growth are visually conveyed
    conv_specs = [
        # name, x, time_visual_w, ch_visual_h, depth, top_label, bottom_label
        ("Conv1", 3.0, 1.3, 2.4, 0.45,
         "Conv1d  1 → 24",
         "k = 10,  s = 4\n24 × 268"),
        ("Conv2", 5.5, 0.95, 3.2, 0.55,
         "Conv1d  24 → 36",
         "k = 8,  s = 4\n36 × 66"),
        ("Conv3", 7.7, 0.7, 4.0, 0.65,
         "Conv1d  36 → 48",
         "k = 4,  s = 2\n48 × 32"),
        ("Conv4", 9.7, 0.55, 4.8, 0.7,
         "Conv1d  48 → 64",
         "k = 3,  s = 1\n64 × 30"),
        ("Conv5", 11.5, 0.5, 4.8, 0.7,
         "Conv1d  64 → 64",
         "k = 3,  s = 1\n64 × 28"),
    ]

    prev_right = input_x + input_w / 2
    for name, cx, w, h, d, ltop, lbot in conv_specs:
        # Arrow
        draw_arrow(ax, prev_right + 0.05, cx - w / 2 - 0.05, y_main, text="ReLU")
        draw_conv_block(ax, cx, y_main, w, h, d, ltop, lbot)
        prev_right = cx + w / 2 + d  # extends to the right by the depth amount

    # ------------- Flatten -------------
    flat_x = 13.3
    flat_w = 0.35
    flat_h = 5.2
    draw_arrow(ax, prev_right + 0.05, flat_x - flat_w / 2 - 0.05, y_main)
    rect = mpatches.Rectangle(
        (flat_x - flat_w / 2, y_main - flat_h / 2),
        flat_w,
        flat_h,
        facecolor=C_FLAT,
        edgecolor=C_FLAT_EDGE,
        linewidth=1.3,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(
        flat_x,
        y_main,
        "Flatten",
        ha="center",
        va="center",
        fontsize=9,
        color="white",
        fontweight="bold",
        rotation=90,
    )
    ax.text(
        flat_x,
        y_main + flat_h / 2 + 0.35,
        "1792",
        ha="center",
        va="bottom",
        fontsize=9,
        color=C_TEXT,
        fontweight="bold",
    )
    ax.text(
        flat_x,
        y_main - flat_h / 2 - 0.35,
        "Flatten\n64 × 28 = 1792",
        ha="center",
        va="top",
        fontsize=8,
        color=C_TEXT,
    )

    # ------------- FC layers -------------
    # height is set with the log of the neuron count in mind for readability
    fc_specs = [
        # x, height, label, role
        (14.3, 4.6, "100", "FC1\n1792 → 100\nReLU + Dropout(0.2)"),
        (15.4, 3.2, "50", "FC2\n100 → 50\nReLU + Dropout(0.2)"),
        (16.4, 1.8, "10", "FC3\n50 → 10\nReLU"),
        (17.4, 0.9, "2", "FC4\n10 → 2\ntanh"),
    ]

    prev_right = flat_x + flat_w / 2
    for cx, h, neurons, role in fc_specs:
        draw_arrow(ax, prev_right + 0.05, cx - 0.55 / 2 - 0.05, y_main)
        draw_fc_block(ax, cx, y_main, h, neurons, role)
        prev_right = cx + 0.55 / 2

    # ------------- Output -------------
    out_x = 18.4
    draw_arrow(ax, prev_right + 0.05, out_x - 0.25, y_main)
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (out_x - 0.25, y_main - 0.9),
            0.5,
            1.8,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=C_OUT,
            edgecolor="#660000",
            linewidth=1.3,
            zorder=3,
        )
    )
    ax.text(
        out_x,
        y_main + 0.35,
        "Steer",
        ha="center",
        va="center",
        fontsize=9,
        color="white",
        fontweight="bold",
    )
    ax.text(
        out_x,
        y_main - 0.35,
        "Speed",
        ha="center",
        va="center",
        fontsize=9,
        color="white",
        fontweight="bold",
    )
    ax.text(
        out_x,
        y_main + 1.1,
        "Output",
        ha="center",
        va="bottom",
        fontsize=10,
        color=C_TEXT,
        fontweight="bold",
    )
    ax.text(
        out_x,
        y_main - 1.0,
        "tanh ∈ [-1, 1]\n(2 values)",
        ha="center",
        va="top",
        fontsize=8,
        color=C_TEXT,
    )

    # ------------- Legend -------------
    legend_y = -0.4
    legend_items = [
        (C_INPUT, C_INPUT_EDGE, "Input"),
        (C_CONV, C_CONV_EDGE, "Conv1d + ReLU"),
        (C_FLAT, C_FLAT_EDGE, "Flatten"),
        (C_FC, C_FC_EDGE, "Fully Connected"),
        (C_OUT, "#660000", "Output (tanh)"),
    ]
    lx = 2.5
    for fc, ec, label in legend_items:
        ax.add_patch(
            mpatches.Rectangle(
                (lx, legend_y - 0.18),
                0.35,
                0.35,
                facecolor=fc,
                edgecolor=ec,
                linewidth=1.2,
            )
        )
        ax.text(
            lx + 0.45,
            legend_y,
            label,
            ha="left",
            va="center",
            fontsize=9,
            color=C_TEXT,
        )
        lx += 2.8

    # ------------- Annotation -------------
    ax.text(
        9.5,
        -1.05,
        "k : kernel size,   s : stride.   Feature-map labels show  (channels) × (length).   Dropout(p=0.2) is applied after FC1 and FC2.",
        ha="center",
        va="center",
        fontsize=9,
        color="#404040",
        style="italic",
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    main(here / "outputs" / "tiny_lidar_net_architecture.png")
