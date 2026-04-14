# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Use Japanese language for all responses unless explicitly asked otherwise.


## プロジェクト概要

tiny-lidar-net学習用の最小2Dシミュレーター。tiny-lidar-net-playgroundからの教育目的の抽出版。
静的環境のみを対象とし、教師あり学習（CNN: DrivingNet）に焦点を当てる。
シミュレーション環境にはIR-SIM（ir-sim）を使用。

Python 3.11+、パッケージマネージャ: uv、MLフレームワーク: PyTorch、シミュレーター: IR-SIM。

## よく使うコマンド

```bash
# インストール（uv使用）
uv pip install -e ".[all]"       # 全機能（ir-sim + torch）
uv pip install -e ".[ml]"        # コア + MLのみ

# CLIコマンド（すべてmain.py経由、出力は outputs/ ディレクトリ）
uv run python main.py manual -w worlds/circuit.yaml -o outputs/training_data.npz   # 手動操作でデータ収集
uv run python main.py train -d outputs/data.npz -o outputs/model.pth -e 100       # CNN学習
uv run python main.py auto -w worlds/circuit.yaml -m outputs/model.pth             # CNN自動運転
uv run python main.py list                                                          # ワールド一覧

# テスト（テストスイートは未作成）
uv run pytest
```

## アーキテクチャ

### モジュール構成

```
main.py              # argparse CLIエントリーポイント、commands/へディスパッチ
outputs/             # 出力ファイル（モデル .pth、学習データ .npz、プロット .png）
commands/            # CLIサブコマンド（manual, train, auto）
  manual.py          # IR-SIM環境を作成、キーボード操作でデータ収集
  train.py           # CNN学習（NPZファイルからモデル生成）
  auto.py            # IR-SIM環境を作成、学習済みモデルで自動運転
simulator/
  ml/                # 機械学習（torch必須）
    model.py         # DrivingNet: 1D CNN（1081入力 → ステアリング + 速度出力）
    dataset.py       # LidarDataset: NPZファイルをPyTorch Datasetとして読み込み
    trainer.py       # 学習ループ（train/val分割、MSE損失）
worlds/              # ワールド設定YAML（IR-SIM形式）
```

### 主要な設計方針

- **IR-SIMによるシミュレーション**: 車両物理（Ackermannモデル）、LiDARセンサー、衝突判定、可視化はすべてIR-SIMが担当。独自のシミュレーターコードは持たない。
- **静的環境のみ**: ワールドは静的矩形障害物と境界壁のみ。
- **データ形式**: 学習データは`.npz`で保存。配列は`lidar` (N, 1081)と`control` (N, 2)。複数NPZファイルを結合して学習可能。

### データフロー

手動運転: キーボード → action [speed, steering] → env.step() → (lidar, control)ペア → outputs/*.npz
CNN学習: outputs/*.npz → LidarDataset → Trainer → outputs/*.pth + outputs/*.png
CNN自動運転: LiDARスキャン → DrivingNet.predict() → action → env.step()

### 規約

- ワールド設定は`worlds/`ディレクトリのYAML（IR-SIM形式）
- モデルファイル`.pth`（PyTorch state_dict）と学習データ`.npz`はgitignore対象
- プロジェクトのドキュメント（README、コミットメッセージ、docstring）は日本語、コード識別子は英語
