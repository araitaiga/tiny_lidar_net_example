# tiny-lidar-net-example

2D LiDARシミュレーションと教師あり学習による自動運転の最小構成サンプル。
**TinyLiDARNet**（1D CNN）の動作を解説するための学習用コードです。

シミュレーション環境に [robosim2d](../robosim2d/) を使用し、
キーボード操作でのデータ収集 → CNN学習 → 自動運転の一連の流れを体験できる。

## 構成

```
tiny-lidar-net-example/
├── main.py                  # CLIエントリーポイント
├── commands/
│   ├── collect.py           # 手動運転で教師データを収集
│   ├── train.py             # CNNモデル学習
│   └── autodrive.py         # 学習済みモデルで自動運転
├── tiny_lidar_net/          # TinyLiDARNet パッケージ
│   ├── control.py           # Control（NamedTuple）
│   ├── model.py             # TinyLiDARNet（1D CNN）
│   ├── dataset.py           # LidarDataset（NPZローダー）
│   └── trainer.py           # 学習ループ
├── worlds/
│   ├── circuit/             # 100×60 のサーキットコース
│   │   ├── robot.yaml       # ロボット設定
│   │   └── world.yaml       # ワールド設定
│   └── simple/              # 50×50 の障害物ワールド
│       ├── robot.yaml
│       └── world.yaml
├── outputs/                 # 出力先（モデル・学習データ・プロット）
└── pyproject.toml
```

### 各コンポーネントの役割

| コンポーネント | 説明 |
|---|---|
| **robosim2d** | 2Dシミュレーション環境。車両物理（Ackermann/DiffDriveモデル）、LiDARセンサー（1081本レイ）、衝突判定、描画をすべて担当 |
| **commands/** | CLIサブコマンド。robosim2dの環境を作成・操作する |
| **tiny_lidar_net/** | PyTorchベースのCNN（TinyLiDARNet）と学習パイプライン。シミュレーターには依存しない |
| **worlds/** | ワールド定義ディレクトリ（各ワールドにrobot.yaml + world.yamlを格納） |

## セットアップ

Python 3.11以上が必要。

```bash
# uv を使用する場合（推奨）
uv pip install -e .

# pip を使用する場合
pip install -e .
```

依存（numpy / robosim2d / matplotlib / torch）はすべて主依存として一括インストールされる。

## 使い方

3つのステップで自動運転モデルを構築する。

### 1. 教師データ収集（`collect`）

キーボードで車両を手動運転し、LiDARスキャンと操作入力のペアを記録する。

```bash
python main.py collect -w worlds/circuit -o outputs/training_data.npz
```

シミュレーターウィンドウが開き、車両とLiDARが表示される。

**操作方法:**

| キー | 操作 |
|---|---|
| `W` / `↑` | 加速 |
| `S` / `↓` | 減速 |
| `A` / `←` | 左ステアリング |
| `D` / `→` | 右ステアリング |
| `Space` | ブレーキ（速度0） |
| `Q` / `Esc` | 終了して保存 |

終了時に `.npz` ファイルが保存される（`lidar: (N, 1081)`, `control: (N, 2)`）。
`speed=0` かつ `steering=0` のフレームは記録しない（ラベル分布の偏りを防ぐため）。

### 2. モデル学習（`train`）

収集したデータで TinyLiDARNet を学習する。

```bash
python main.py train -d outputs/training_data.npz -o outputs/tiny_lidar_net.pth -e 100
```

複数のデータファイルを結合して学習することも可能:

```bash
python main.py train -d outputs/data1.npz outputs/data2.npz -o outputs/tiny_lidar_net.pth
```

**主なオプション:**

| オプション | デフォルト | 説明 |
|---|---|---|
| `-d` | (必須) | 学習データファイル（複数指定可） |
| `-o` | `outputs/tiny_lidar_net.pth` | モデル出力先 |
| `-e` | 100 | エポック数 |
| `-b` | 32 | バッチサイズ |
| `--lr` | 0.001 | 学習率 |

学習完了後、モデル（`.pth`）と損失曲線（`.png`）が出力される。

### 3. 自動運転（`autodrive`）

学習済みモデルを使って車両を自動制御する。

```bash
python main.py autodrive -w worlds/circuit -m outputs/tiny_lidar_net.pth
```

LiDARスキャンをモデルに入力し、予測されたステアリングと速度で車両を制御する。
衝突するかウィンドウを閉じると終了する。

### その他

```bash
# 利用可能なワールド一覧
python main.py list
```

## ワールド定義

ワールドは `worlds/` ディレクトリ配下にサブディレクトリとして定義する。
各サブディレクトリに `robot.yaml`（ロボット設定）と `world.yaml`（ワールド設定）を配置する。

**robot.yaml:**
```yaml
kinematics: {name: 'acker'}                                      # Ackermann（自転車）モデル
shape: {name: 'rectangle', length: 4.0, width: 1.8, wheelbase: 2.5}
state: [5.0, 5.0, 0.0, 0.0]                                     # [x, y, yaw, steering]
vel_max: [10.0, 0.7854]                                          # [最大速度, 最大ステアリング角]
sensors:
  - type: 'lidar2d'
    number: 1081           # レイ数
    range_max: 30.0        # 最大検出距離 [m]
    angle_range: 6.28318   # 360度
```

**world.yaml:**
```yaml
world:
  height: 50.0
  width: 50.0

obstacle:
  - shape: {name: 'rectangle', length: 4.0, width: 4.0}
    state: [15.0, 15.0, 0]    # [x, y, angle]
```

## データフロー

```
collect (手動運転で教師データ収集)
  キーボード → Control(steering, speed) → sim.step() → LiDARスキャン記録 → .npz

train (モデル学習)
  .npz → LidarDataset → TinyLiDARNet (1D CNN) → .pth

autodrive (自動運転)
  LiDARスキャン → TinyLiDARNet.predict() → Control → sim.step()
```

### Control 型と並び順の規約

`tiny_lidar_net.Control` は `NamedTuple` で、ステアリング角と速度を保持する:

| 用途 | 並び順 |
|---|---|
| 学習データ (`.npz`) / モデル出力テンソル | `[steering, speed]` |
| robosim2d `sim.step()` の action 配列 | `[speed, steering]` |

並び順の混乱を避けるため、変換は必ず `Control.to_array()` / `Control.to_action()` を使う。

## TinyLiDARNet アーキテクチャ

LiDARスキャン（1081次元）からステアリング角と速度を出力する1D CNN。

```
入力: (batch, 1, 1081)
  ↓ Conv1d(1→24,  k=10, s=4)   → 268
  ↓ Conv1d(24→36, k=8,  s=4)   → 66
  ↓ Conv1d(36→48, k=4,  s=2)   → 32
  ↓ Conv1d(48→64, k=3,  s=1)   → 30
  ↓ Conv1d(64→64, k=3,  s=1)   → 28
  ↓ Flatten                     → 1792
  ↓ FC(1792→100) + ReLU + Dropout
  ↓ FC(100→50)   + ReLU + Dropout
  ↓ FC(50→10)    + ReLU
  ↓ FC(10→2)
出力: (batch, 2) = [steering_angle, speed]
```


# メモ

公式実装

https://github.com/CSL-KU/TinyLidarNet/blob/main/train.py

通常層Activation: ReLU
最終出力層Activation: tanh  
Dropout: なし  
Optimizer: Adam(5e-5)  
loss_function: huber
batch_size: 64, epoch: 20

### 学習データ
curved_circuit  
outputs/training_data_curved_circuit.npz  
