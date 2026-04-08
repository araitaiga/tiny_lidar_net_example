"""自動運転シミュレーターパッケージ

モジュール構成:
    core/  - シミュレーターのコアコンポーネント
    viz/   - 可視化（matplotlib必須）
    ml/    - 機械学習（PyTorch必須）
"""

# コアモジュール（常に利用可能）
from simulator.core import (
    Lidar2D,
    Obstacle,
    SimulationRecord,
    SimulationStep,
    Simulator,
    Vehicle,
    VehicleControl,
    VehicleState,
    World,
)

# 可視化モジュール（matplotlibが必要）
try:
    from simulator.viz import (
        AutoController,
        ManualController,
        RealtimeVisualizer,
    )

    HAS_VISUALIZER = True
except ImportError:
    HAS_VISUALIZER = False

# 機械学習モジュール（PyTorchが必要）
try:
    from simulator.ml import (
        DrivingNet,
        LidarDataset,
        Trainer,
        train_from_file,
    )

    HAS_ML = True
except ImportError:
    HAS_ML = False

__all__ = [
    # Core
    "World",
    "Obstacle",
    "Vehicle",
    "VehicleState",
    "VehicleControl",
    "Lidar2D",
    "Simulator",
    "SimulationStep",
    "SimulationRecord",
    # Visualization
    "RealtimeVisualizer",
    "ManualController",
    "AutoController",
    # Machine Learning
    "DrivingNet",
    "LidarDataset",
    "Trainer",
    "train_from_file",
]
