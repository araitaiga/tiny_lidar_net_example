"""Type definitions for vehicle control values."""

from typing import NamedTuple

import numpy as np

# Normalization constants that map the tanh output range [-1, 1] of the model's
# final layer to physical values. During training, labels are normalized into
# [-1, 1] by these values; during inference, they are de-normalized back to physical values.
MAX_STEERING = 0.5  # [rad]
MAX_SPEED = 5.0     # [m/s]


def normalize_labels(labels: np.ndarray) -> np.ndarray:
    """Normalize physical ``[steering, speed]`` labels into the tanh range ``[-1, 1]``.

    Accepts either a single sample ``(2,)`` or a batch ``(N, 2)``; the last axis is
    treated as ``[steering, speed]``. Returns a new float32 array.
    """
    out = labels.astype(np.float32).copy()
    out[..., 0] /= MAX_STEERING
    out[..., 1] /= MAX_SPEED
    return out


def denormalize_labels(labels: np.ndarray) -> np.ndarray:
    """De-normalize ``[steering, speed]`` values in ``[-1, 1]`` back to physical values.

    Inverse of :func:`normalize_labels`. Accepts ``(2,)`` or ``(N, 2)``; the last axis
    is treated as ``[steering, speed]``. Returns a new float32 array.
    """
    out = labels.astype(np.float32).copy()
    out[..., 0] *= MAX_STEERING
    out[..., 1] *= MAX_SPEED
    return out


class Control(NamedTuple):
    """Vehicle control values (steering angle and speed).

    Ordering conventions:
        - Training data (.npz) / model output tensor: [steering, speed]
        - robosim2d ``sim.step()`` action array:       [speed, steering]

    Use ``to_training_label()`` / ``to_robot_action()`` for conversions, so the ordering
    is centralized in this class and cannot be mixed up.
    """

    steering: float
    speed: float

    def to_training_label(self) -> np.ndarray:
        """Convert to a ``[steering, speed]`` label array stored in the training-data ``.npz``."""
        return np.array([self.steering, self.speed], dtype=np.float32)

    def to_robot_action(self) -> np.ndarray:
        """Convert to a ``[speed, steering]`` action array consumed by robosim2d ``sim.step()``."""
        return np.array([self.speed, self.steering], dtype=np.float32)

    @classmethod
    def from_model_output(cls, arr) -> "Control":
        """De-normalize a model tanh output ``[steering, speed]`` (∈ [-1, 1]) to physical values."""
        steering, speed = denormalize_labels(np.asarray(arr))
        return cls(steering=float(steering), speed=float(speed))
