"""Huber損失の形状を複数の delta について描画し、画像として保存する。

trainer.py では nn.HuberLoss(delta=1.0) を使用しており、この図は delta が
損失関数の形状（誤差が小さい領域では二乗、大きい領域では線形）に与える影響を示す。
比較のため二乗誤差（delta -> infinity に相当）も併せて描画する。
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "huber.png"

DELTAS = [0.5, 1.0, 2.0]


def huber_loss(error: np.ndarray, delta: float) -> np.ndarray:
    """要素ごとの Huber 損失を返す。

    |error| <= delta では 0.5 * error^2、それ以外では delta * (|error| - 0.5 * delta)。
    """
    abs_error = np.abs(error)
    quadratic = 0.5 * abs_error**2
    linear = delta * (abs_error - 0.5 * delta)
    return np.where(abs_error <= delta, quadratic, linear)


def main() -> None:
    x = np.linspace(-5.0, 5.0, 500)

    fig, ax = plt.subplots(figsize=(6, 4))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for delta, color in zip(DELTAS, colors):
        y = huber_loss(x, delta)
        ax.plot(x, y, color=color, linewidth=2, label=rf"$\delta = {delta}$")

    # 比較用の二乗誤差（delta -> infinity に相当）
    ax.plot(
        x,
        0.5 * x**2,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=r"$\frac{1}{2}x^2$ (squared error)",
    )

    # 原点を通る軸
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.5, 8)
    ax.set_xlabel("error")
    ax.set_ylabel("loss")
    ax.set_title("Huber loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center")

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
