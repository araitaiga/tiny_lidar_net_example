"""Plot the shape of the tanh function together with its output range [-1, 1] and save it as an image.

This figure corresponds to the explanation in the article where tanh is used in the final layer to bound the output to [-1, 1].
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "tanh.png"


def main() -> None:
    x = np.linspace(-5.0, 5.0, 500)
    y = np.tanh(x)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, color="#1f77b4", linewidth=2, label=r"$y = \tanh(x)$")

    # Asymptotes indicating the output range [-1, 1]
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.axhline(-1.0, color="gray", linestyle="--", linewidth=1)

    # Axes through the origin
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("tanh")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
