"""モデル学習コマンド"""

from pathlib import Path


def run_train(
    data_files: list[str],
    model_output: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    save_best: str | None = None,
) -> None:
    """モデル学習を実行

    Args:
        data_files: 学習データファイルのリスト
        model_output: モデル出力ファイル
        epochs: エポック数
        batch_size: バッチサイズ
        learning_rate: 学習率
    """
    from simulator import train_from_file

    # ファイル存在チェック
    missing_files = [f for f in data_files if not Path(f).exists()]
    if missing_files:
        print("エラー: 以下のファイルが見つかりません:")
        for f in missing_files:
            print(f"  {f}")
        return

    train_from_file(
        data_files=data_files,
        model_output=model_output,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        plot_loss=True,
        save_best=save_best,
    )
