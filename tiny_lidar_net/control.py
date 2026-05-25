"""Type definitions for vehicle control values."""

from typing import NamedTuple

import numpy as np

# Normalization constants that map the tanh output range [-1, 1] of the model's
# final layer to physical values. During training, labels are normalized into
# [-1, 1] by these values; during inference, they are de-normalized back to physical values.
MAX_STEERING = 0.5  # [rad]
MAX_SPEED = 5.0     # [m/s]


class Control(NamedTuple):
    """Vehicle control values (steering angle and speed).

    Ordering conventions:
        - Training data (.npz) / model output tensor: [steering, speed]
        - robosim2d ``sim.step()`` action array:       [speed, steering]

    Use ``to_array()`` / ``to_action()`` for conversions, so the ordering
    is centralized in this class and cannot be mixed up.
    """

    steering: float
    speed: float

    def to_array(self) -> np.ndarray:
        """Convert to a ``[steering, speed]`` array used for training-data storage."""
        return np.array([self.steering, self.speed], dtype=np.float32)

    def to_action(self) -> np.ndarray:
        """Convert to a ``[speed, steering]`` array used by robosim2d ``sim.step()``."""
        return np.array([self.speed, self.steering], dtype=np.float32)

    @classmethod
    def from_normalized(cls, arr) -> "Control":
        """De-normalize a tanh output ``[steering, speed]`` (∈ [-1, 1]) to physical values."""
        return cls(
            steering=float(arr[0]) * MAX_STEERING,
            speed=float(arr[1]) * MAX_SPEED,
        )
