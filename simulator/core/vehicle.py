"""車両モデル（Bicycle Model）"""

from dataclasses import dataclass

import numpy as np


@dataclass
class VehicleState:
    """車両の状態

    Attributes:
        x: X座標
        y: Y座標
        yaw: 向き（ラジアン、X軸正方向が0、反時計回りが正）
    """

    x: float
    y: float
    yaw: float

    def to_array(self) -> np.ndarray:
        """numpy配列に変換"""
        return np.array([self.x, self.y, self.yaw])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "VehicleState":
        """numpy配列から作成"""
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))


@dataclass
class VehicleControl:
    """車両への制御入力

    Attributes:
        steering_angle: ステアリング角（ラジアン、左が正）
        speed: 速度（m/s）
    """

    steering_angle: float
    speed: float

    def clip(self, max_steering: float, max_speed: float) -> "VehicleControl":
        """制御値をクリップ"""
        return VehicleControl(
            steering_angle=np.clip(
                self.steering_angle, -max_steering, max_steering),
            speed=np.clip(self.speed, -max_speed, max_speed),
        )


class Vehicle:
    """車両モデル（Bicycle Model）

    シンプルなBicycle Modelを使用して車両の動きをシミュレート。

    Attributes:
        state: 現在の車両状態
        wheelbase: ホイールベース（前輪と後輪の距離）
        width: 車両の幅
        length: 車両の長さ
        max_steering: 最大ステアリング角（ラジアン）
        max_speed: 最大速度（m/s）
    """

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        yaw: float = 0.0,
        wheelbase: float = 2.5,
        width: float = 1.8,
        length: float = 4.0,
        max_steering: float = np.pi / 4,
        max_speed: float = 10.0,
    ):
        self.state = VehicleState(x, y, yaw)
        self.wheelbase = wheelbase
        self.width = width
        self.length = length
        self.max_steering = max_steering
        self.max_speed = max_speed

    @property
    def collision_radius(self) -> float:
        """衝突判定用の半径"""
        return max(self.width, self.length) / 2

    @property
    def x(self) -> float:
        return self.state.x

    @property
    def y(self) -> float:
        return self.state.y

    @property
    def yaw(self) -> float:
        return self.state.yaw

    def step(self, control: VehicleControl, dt: float) -> VehicleState:
        """制御入力に基づいて車両状態を更新

        Bicycle Modelによる運動学モデル:
        - dx/dt = v * cos(yaw)
        - dy/dt = v * sin(yaw)
        - dyaw/dt = v / L * tan(steering_angle)
        """
        ctrl = control.clip(self.max_steering, self.max_speed)

        v = ctrl.speed
        delta = ctrl.steering_angle

        dx = v * np.cos(self.state.yaw)
        dy = v * np.sin(self.state.yaw)
        dyaw = v / self.wheelbase * np.tan(delta)

        new_x = self.state.x + dx * dt
        new_y = self.state.y + dy * dt
        new_yaw = self.state.yaw + dyaw * dt
        new_yaw = np.arctan2(np.sin(new_yaw), np.cos(new_yaw))

        self.state = VehicleState(new_x, new_y, new_yaw)
        return self.state

    def compute_next_state(self, control: VehicleControl, dt: float) -> VehicleState:
        """次の状態を計算（状態は更新しない）"""
        ctrl = control.clip(self.max_steering, self.max_speed)

        v = ctrl.speed
        delta = ctrl.steering_angle

        dx = v * np.cos(self.state.yaw)
        dy = v * np.sin(self.state.yaw)
        dyaw = v / self.wheelbase * np.tan(delta)

        new_x = self.state.x + dx * dt
        new_y = self.state.y + dy * dt
        new_yaw = self.state.yaw + dyaw * dt
        new_yaw = np.arctan2(np.sin(new_yaw), np.cos(new_yaw))

        return VehicleState(new_x, new_y, new_yaw)

    def set_state(self, x: float, y: float, yaw: float) -> None:
        """車両状態を設定"""
        self.state = VehicleState(x, y, yaw)

    def get_corners(self) -> np.ndarray:
        """車両の4隅の座標を取得"""
        cos_yaw = np.cos(self.state.yaw)
        sin_yaw = np.sin(self.state.yaw)

        hl, hw = self.length / 2, self.width / 2
        local_corners = np.array([
            [-hl, -hw],
            [hl, -hw],
            [hl, hw],
            [-hl, hw],
        ])

        rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        corners = local_corners @ rot.T + \
            np.array([self.state.x, self.state.y])
        return corners
