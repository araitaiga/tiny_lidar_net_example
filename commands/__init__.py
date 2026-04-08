"""CLIコマンドモジュール"""

from commands.auto import run_auto
from commands.manual import run_manual
from commands.train import run_train

__all__ = [
    "run_manual",
    "run_train",
    "run_auto",
]
