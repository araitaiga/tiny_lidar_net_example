"""シミュレーター統合モジュール"""

from dataclasses import dataclass
from dataclasses import field
import json
from pathlib import Path

import numpy as np

from simulator.core.lidar import Lidar2D
from simulator.core.vehicle import Vehicle
from simulator.core.vehicle import VehicleControl
from simulator.core.vehicle import VehicleState
from simulator.core.world import World


@dataclass
class SimulationStep:
    """シミュレーションの1ステップの記録"""

    timestamp: float
    state: VehicleState
    control: VehicleControl
    lidar_ranges: np.ndarray
    collision: bool


@dataclass
class SimulationRecord:
    """シミュレーション記録（学習データ用）"""

    steps: list[SimulationStep] = field(default_factory=list)
    world_width: float = 100.0
    world_height: float = 100.0

    def to_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """学習データ形式に変換

        Returns:
            (X, y): X=入力データ, y=出力データ（制御入力）
        """
        if not self.steps:
            return np.array([]), np.array([])

        X_list = []
        y_list = []

        for step in self.steps:
            state_arr = step.state.to_array()
            x_data = np.concatenate([step.lidar_ranges, state_arr])
            X_list.append(x_data)

            y_data = np.array(
                [step.control.steering_angle, step.control.speed])
            y_list.append(y_data)

        return np.array(X_list), np.array(y_list)


class Simulator:
    """自動運転シミュレーター

    ワールド、車両、LiDARを統合したシミュレーション環境。
    """

    def __init__(
        self,
        world: World | None = None,
        vehicle: Vehicle | None = None,
        lidar: Lidar2D | None = None,
        dt: float = 0.1,
    ):
        self.world = world or World()
        self.vehicle = vehicle or Vehicle()
        self.lidar = lidar or Lidar2D()
        self.dt = dt
        self.time = 0.0

        # 初期位置を保存（reset() で復元するため）
        self._initial_x = self.vehicle.x
        self._initial_y = self.vehicle.y
        self._initial_yaw = self.vehicle.yaw

        self._record = SimulationRecord(
            world_width=self.world.width,
            world_height=self.world.height,
        )
        self._recording = False

    def reset(
        self,
        vehicle_x: float | None = None,
        vehicle_y: float | None = None,
        vehicle_yaw: float | None = None,
    ) -> np.ndarray:
        """シミュレーションをリセット

        引数なしの場合は初期位置に復帰する。
        """
        self.time = 0.0

        x = vehicle_x if vehicle_x is not None else self._initial_x
        y = vehicle_y if vehicle_y is not None else self._initial_y
        yaw = vehicle_yaw if vehicle_yaw is not None else self._initial_yaw
        self.vehicle.set_state(x, y, yaw)

        self._record = SimulationRecord(
            world_width=self.world.width,
            world_height=self.world.height,
        )

        return self.get_lidar_scan()

    def step(self, control: VehicleControl) -> tuple[np.ndarray, bool, dict]:
        """シミュレーションを1ステップ進める"""
        next_state = self.vehicle.compute_next_state(control, self.dt)

        collision = self.world.is_collision(
            next_state.x,
            next_state.y,
            self.vehicle.collision_radius,
        )

        if not collision:
            self.vehicle.step(control, self.dt)

        self.time += self.dt
        lidar_scan = self.get_lidar_scan()

        if self._recording:
            step_record = SimulationStep(
                timestamp=self.time,
                state=VehicleState(
                    self.vehicle.x, self.vehicle.y, self.vehicle.yaw),
                control=control,
                lidar_ranges=lidar_scan.copy(),
                collision=collision,
            )
            self._record.steps.append(step_record)

        info = {
            "time": self.time,
            "x": self.vehicle.x,
            "y": self.vehicle.y,
            "yaw": self.vehicle.yaw,
        }

        return lidar_scan, collision, info

    def get_lidar_scan(self) -> np.ndarray:
        """現在位置でのLiDARスキャンを取得"""
        edges = self.world.get_all_edges()
        return self.lidar.scan(
            self.vehicle.x,
            self.vehicle.y,
            self.vehicle.yaw,
            edges,
        )

    def start_recording(self) -> None:
        """シミュレーション記録を開始"""
        self._recording = True
        self._record = SimulationRecord(
            world_width=self.world.width,
            world_height=self.world.height,
        )

    def stop_recording(self) -> SimulationRecord:
        """シミュレーション記録を停止して返す"""
        self._recording = False
        return self._record

    @classmethod
    def from_file(
        cls,
        world_file: str,
        vehicle_x: float | None = None,
        vehicle_y: float | None = None,
        vehicle_yaw: float | None = None,
    ) -> "Simulator":
        """ワールド設定ファイルからシミュレーターを作成"""
        with open(Path(world_file), "r", encoding="utf-8") as f:
            data = json.load(f)

        world = World.load(world_file)
        vehicle_start = data.get("vehicle_start", {})

        x = vehicle_x if vehicle_x is not None else vehicle_start.get(
            "x", world.width / 2)
        y = vehicle_y if vehicle_y is not None else vehicle_start.get(
            "y", world.height / 2)
        yaw = vehicle_yaw if vehicle_yaw is not None else vehicle_start.get(
            "yaw", 0.0)

        vehicle = Vehicle(x=x, y=y, yaw=yaw)
        lidar = Lidar2D(num_rays=1081)

        return cls(world=world, vehicle=vehicle, lidar=lidar)
