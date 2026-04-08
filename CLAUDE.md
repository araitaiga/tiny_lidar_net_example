# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Use Japanese language for all responses unless explicitly asked otherwise.


## プロジェクト概要

tiny-lidar-net学習用の最小2Dシミュレーター。tiny-lidar-net-playgroundからの教育目的の抽出版。
静的環境のみを対象とし、教師あり学習（CNN: DrivingNet）に焦点を当てる。

Python 3.11+、パッケージマネージャ: uv、MLフレームワーク: PyTorch。

## よく使うコマンド

```bash
# インストール（uv使用）
uv pip install -e ".[all]"       # 全機能（numpy + matplotlib + torch）
uv pip install -e ".[viz]"       # コア + 可視化のみ
uv pip install -e ".[ml]"        # コア + MLのみ

# CLIコマンド（すべてmain.py経由、出力は outputs/ ディレクトリ）
uv run python main.py manual -w worlds/circuit.json -o outputs/training_data.npz   # 手動操作でデータ収集
uv run python main.py train -d outputs/data.npz -o outputs/model.pth -e 100       # CNN学習
uv run python main.py auto -w worlds/circuit.json -m outputs/model.pth             # CNN自動運転
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
simulator/
  core/              # シミュレーションエンジン（numpyのみ、オプション依存なし）
    world.py         # ワールド・矩形障害物、JSON読み込み、衝突判定・辺抽出
    vehicle.py       # Bicycle Model車両運動学（VehicleState, VehicleControl dataclass）
    lidar.py         # 360度2D LiDAR（1081本レイ、ベクトル化レイ-セグメント交差判定）
    simulator.py     # World+Vehicle+Lidarの統合、ステップループ・記録管理
  viz/               # 可視化レイヤー（matplotlib必須）
    realtime.py      # リアルタイム描画基底クラス（ワールド・障害物・車両・LiDAR点群）
    manual.py        # キーボード操作コントローラ + NPZ記録
    auto.py          # CNN推論可視化
  ml/                # 機械学習（torch必須）
    model.py         # DrivingNet: 1D CNN（1081入力 → ステアリング + 速度出力）
    dataset.py       # LidarDataset: NPZファイルをPyTorch Datasetとして読み込み
    trainer.py       # 学習ループ（train/val分割、MSE損失）
worlds/              # ワールド設定JSON（width, height, obstacles, vehicle_start）
```

### 主要な設計方針

- **オプション依存の分離**: `simulator/core/`はnumpyのみ使用。vizとmlのimportは`simulator/__init__.py`でtry/exceptガードされ、ImportError時にインストール手順を提示する。
- **静的環境のみ**: 動的障害物（DynamicObstacle）は非対応。ワールドは静的矩形障害物と境界壁のみ。
- **ベクトル化LiDAR**: `lidar.py`は1081本すべてのレイ-セグメント交差をNumPyブロードキャスティングで一括計算（レイ×辺の行列演算）。Pythonループなし。
- **データ形式**: 学習データは`.npz`で保存。配列は`lidar` (N, 1081)と`control` (N, 2)。複数NPZファイルを結合して学習可能。

### データフロー

手動運転: キーボード → VehicleControl → Simulator.step() → (lidar, control)ペア → outputs/*.npz
CNN学習: outputs/*.npz → LidarDataset → Trainer → outputs/*.pth + outputs/*.png
CNN自動運転: LiDARスキャン → DrivingNet.predict() → VehicleControl → Simulator.step()

### 規約

- 状態オブジェクトにはdataclassを使用: `VehicleState(x, y, yaw)`, `VehicleControl(steering_angle, speed)`
- ワールド設定は`worlds/`ディレクトリのJSON
- モデルファイル`.pth`（PyTorch state_dict）と学習データ`.npz`はgitignore対象
- プロジェクトのドキュメント（README、コミットメッセージ、docstring）は日本語、コード識別子は英語
