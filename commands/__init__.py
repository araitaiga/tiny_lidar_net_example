"""CLIコマンドモジュール"""

from commands.autodrive import run_autodrive
from commands.collect import run_collect
from commands.train import run_train

__all__ = [
    "run_collect",
    "run_train",
    "run_autodrive",
]
