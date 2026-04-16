# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Use Japanese language for all responses unless explicitly asked otherwise.


## プロジェクト概要

tiny-lidar-net学習用の最小2Dシミュレーター。tiny-lidar-net-playgroundからの教育目的の抽出版。
静的環境のみを対象とし、教師あり学習（CNN: DrivingNet）に焦点を当てる。
シミュレーション環境にはrobosim2d（軽量2Dロボットシミュレーター）を使用。

Python 3.11+、パッケージマネージャ: uv、MLフレームワーク: PyTorch、シミュレーター: robosim2d。

## よく使うコマンド

```bash
# インストール（uv使用）
uv pip install -e ".[all]"       # 全機能（robosim2d + matplotlib + torch）
uv pip install -e ".[ml]"        # コア + MLのみ

# CLIコマンド（すべてmain.py経由、出力は outputs/ ディレクトリ）
uv run python main.py manual -w worlds/circuit -o outputs/training_data.npz   # 手動操作でデータ収集
uv run python main.py train -d outputs/data.npz -o outputs/model.pth -e 100  # CNN学習
uv run python main.py auto -w worlds/circuit -m outputs/model.pth            # CNN自動運転
uv run python main.py list                                                    # ワールド一覧

# テスト（テストスイートは未作成）
uv run pytest
```

## アーキテクチャ

### モジュール構成

```
main.py              # argparse CLIエントリーポイント、commands/へディスパッチ
outputs/             # 出力ファイル（モデル .pth、学習データ .npz、プロット .png）
commands/            # CLIサブコマンド（manual, train, auto）
  manual.py          # robosim2d環境を作成、キーボード操作でデータ収集
  train.py           # CNN学習（NPZファイルからモデル生成）
  auto.py            # robosim2d環境を作成、学習済みモデルで自動運転
simulator/
  ml/                # 機械学習（torch必須）
    model.py         # DrivingNet: 1D CNN（1081入力 → ステアリング + 速度出力）
    dataset.py       # LidarDataset: NPZファイルをPyTorch Datasetとして読み込み
    trainer.py       # 学習ループ（train/val分割、MSE損失）
worlds/              # ワールド設定（ディレクトリごとにrobot.yaml + world.yamlを格納）
  circuit/           # サーキットコース
    robot.yaml       # ロボット設定（kinematics, shape, sensors）
    world.yaml       # ワールド設定（サイズ, 障害物）
  simple/            # シンプルな障害物配置
    robot.yaml
    world.yaml
```

### 主要な設計方針

- **robosim2dによるシミュレーション**: 車両物理（Ackermann/DiffDriveモデル）、LiDARセンサー、衝突判定、可視化はrobosim2dが担当。独自のシミュレーターコードは持たない。
- **静的環境のみ**: ワールドは静的矩形障害物と境界壁のみ。
- **データ形式**: 学習データは`.npz`で保存。配列は`lidar` (N, 1081)と`control` (N, 2)。複数NPZファイルを結合して学習可能。

### データフロー

手動運転: キーボード → action [speed, steering] → sim.step() → (lidar, control)ペア → outputs/*.npz
CNN学習: outputs/*.npz → LidarDataset → Trainer → outputs/*.pth + outputs/*.png
CNN自動運転: LiDARスキャン → DrivingNet.predict() → action → sim.step()

### 規約

- ワールド設定は`worlds/`ディレクトリ配下のサブディレクトリ（robot.yaml + world.yaml）
- モデルファイル`.pth`（PyTorch state_dict）と学習データ`.npz`はgitignore対象
- プロジェクトのドキュメント（README、コミットメッセージ、docstring）は日本語、コード識別子は英語
