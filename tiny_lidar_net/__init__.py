"""TinyLidarNet: a 1D CNN that regresses control values from a LiDAR scan."""

from tiny_lidar_net.control import Control
from tiny_lidar_net.dataset import LidarDataset
from tiny_lidar_net.model import TinyLidarNet
from tiny_lidar_net.trainer import Trainer, train_from_file

# Simulator time per step [s]
DT = 0.1

__all__ = [
    "DT",
    "Control",
    "LidarDataset",
    "TinyLidarNet",
    "Trainer",
    "train_from_file",
]
