# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Use Japanese language for all responses unless explicitly asked otherwise.


## プロジェクト概要

**TinyLiDARNet** を解説するための最小2Dシミュレーターサンプル。tiny-lidar-net-playgroundからの教育目的の抽出版。
静的環境のみを対象とし、教師あり学習（1D CNN: TinyLiDARNet）に焦点を当てる。
シミュレーション環境にはrobosim2d（軽量2Dロボットシミュレーター）を使用。

Python 3.11+、パッケージマネージャ: uv、MLフレームワーク: PyTorch、シミュレーター: robosim2d。

## よく使うコマンド

```bash
# インストール（uv使用）
uv pip install -e .                # numpy/robosim2d/matplotlib/torch をすべて取得

# CLIコマンド（すべてmain.py経由、出力は outputs/ ディレクトリ）
uv run python main.py collect   -w worlds/circuit -o outputs/training_data.npz          # 手動運転で教師データ収集
uv run python main.py train     -d outputs/data.npz -o outputs/tiny_lidar_net.pth -e 100  # CNN学習
uv run python main.py autodrive -w worlds/circuit -m outputs/tiny_lidar_net.pth         # TinyLiDARNetで自動運転
uv run python main.py evaluate  -m outputs/tiny_lidar_net.pth                           # 複数ワールド×CW/CCWでheadless評価
uv run python main.py list                                                              # ワールド一覧

# テスト（テストスイートは未作成）
uv run pytest
```

## アーキテクチャ

### モジュール構成

```
main.py              # argparse CLIエントリーポイント、commands/へディスパッチ
outputs/             # 出力ファイル（モデル .pth、学習データ .npz、プロット .png）
commands/            # CLIサブコマンド（collect, train, autodrive, evaluate）
  collect.py         # robosim2d環境を作成、キーボードの手動運転で教師データ収集
  train.py           # CNN学習（NPZファイルからモデル生成）
  autodrive.py       # robosim2d環境を作成、学習済みモデルで自動運転
  evaluate.py        # 複数world×複数開始位置でheadless走行し走行距離・速度・ステアリングRMSを集計
tiny_lidar_net/      # TinyLiDARNet パッケージ（torch必須）
  control.py         # Control: ステアリング・速度を保持する NamedTuple
  model.py           # TinyLiDARNet: 1D CNN（1081入力 → ステアリング + 速度出力）
  dataset.py         # LidarDataset: NPZファイルをPyTorch Datasetとして読み込み
  trainer.py         # 学習ループ（train/val分割、MSE損失）
worlds/              # ワールド設定（ディレクトリごとにrobot.yaml + world.yaml + 任意のeval.yamlを格納）
  circuit/           # サーキットコース
    robot.yaml       # ロボット設定（kinematics, shape, sensors）
    world.yaml       # ワールド設定（サイズ, 障害物）
    eval.yaml        # evaluate コマンド用スタート位置リスト（任意、CW/CCW 両方向）
  simple/            # シンプルな障害物配置
    robot.yaml
    world.yaml
```

### 主要な設計方針

- **robosim2dによるシミュレーション**: 車両物理（Ackermann/DiffDriveモデル）、LiDARセンサー、衝突判定、可視化はrobosim2dが担当。独自のシミュレーターコードは持たない。
- **静的環境のみ**: ワールドは静的矩形障害物と境界壁のみ。
- **データ形式**: 学習データは`.npz`で保存。配列は`lidar` (N, 1081)と`control` (N, 2)。複数NPZファイルを結合して学習可能。
- **Control 型による並び順固定**: `tiny_lidar_net.Control` (NamedTuple) で `[steering, speed]` ↔ `[speed, steering]` の変換を集約。学習データ・モデル出力は `[steering, speed]`、robosim2d の action は `[speed, steering]`。
- **説明用コード**: 教育目的のため、複雑な torch optional ガードや動的分岐は持たず、シンプルさと可読性を優先する。

### データフロー

collect:   キーボード → Control(steering, speed) → sim.step() → (lidar, control)ペア → outputs/*.npz
train:     outputs/*.npz → LidarDataset → Trainer → outputs/*.pth + outputs/*.png
autodrive: LiDARスキャン → TinyLiDARNet.predict() → Control → sim.step()
evaluate:  各world × 各eval.yaml start → headlessループ → 走行距離/速度/StRMS集計 + collision/stuck/survived 分類

`collect` では `speed=0 and steering=0` の停止フレームは記録しない（ラベル分布の偏り防止）。

### 規約

- ワールド設定は`worlds/`ディレクトリ配下のサブディレクトリ（robot.yaml + world.yaml）
- モデルファイル`.pth`（PyTorch state_dict）と学習データ`.npz`はgitignore対象
- プロジェクトのドキュメント（README、コミットメッセージ、docstring）は日本語、コード識別子は英語
