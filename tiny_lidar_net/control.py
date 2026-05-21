"""車両制御値の型定義。"""

from typing import NamedTuple

import numpy as np

# モデル最終層 tanh の出力域 [-1, 1] と物理値を対応させる正規化定数。
# 学習時はこの値でラベルを [-1, 1] に正規化、推論時は逆変換で物理値へ戻す。
MAX_STEERING = 0.5  # [rad]
MAX_SPEED = 5.0     # [m/s]


class Control(NamedTuple):
    """車両制御値（ステアリング角と速度）。

    並び順の規約:
        - 学習データ (.npz) / モデル出力テンソル: [steering, speed]
        - robosim2d ``sim.step()`` の action 配列:  [speed, steering]

    並び順の入れ替えミスを防ぐため、変換は ``to_array()`` / ``to_action()``
    に集約する。
    """

    steering: float
    speed: float

    def to_array(self) -> np.ndarray:
        """学習データ保存用 ``[steering, speed]`` 配列に変換。"""
        return np.array([self.steering, self.speed], dtype=np.float32)

    def to_action(self) -> np.ndarray:
        """robosim2d ``sim.step()`` 用 ``[speed, steering]`` 配列に変換。"""
        return np.array([self.speed, self.steering], dtype=np.float32)

    def to_normalized(self) -> np.ndarray:
        """tanh 出力域 [-1, 1] に正規化した ``[steering, speed]`` 配列。"""
        return np.array(
            [self.steering / MAX_STEERING, self.speed / MAX_SPEED],
            dtype=np.float32,
        )

    @classmethod
    def from_array(cls, arr) -> "Control":
        """学習データ ``[steering, speed]`` 配列から復元。"""
        return cls(steering=float(arr[0]), speed=float(arr[1]))

    @classmethod
    def from_normalized(cls, arr) -> "Control":
        """tanh 出力 ``[steering, speed]`` (∈ [-1, 1]) を物理値へ逆正規化。"""
        return cls(
            steering=float(arr[0]) * MAX_STEERING,
            speed=float(arr[1]) * MAX_SPEED,
        )
