---
title: "TinyLidarNet 論文解説と PyTorch + 2D シミュレータによる再現実装"
emoji: "🏎️"
type: "tech"
topics: ["機械学習", "深層学習", "ロボティクス", "lidar", "pytorch"]
published: false
---

> 元論文: [TinyLidarNet: 2D LiDAR-based End-to-End Deep Learning Model for F1TENTH Autonomous Racing (Zarrar et al., 2024)](https://arxiv.org/abs/2410.07447)
> 公式実装: <https://github.com/CSL-KU/TinyLidarNet>
> 本記事の再現実装: （自リポジトリのURLを後で挿入）

## はじめに

本記事では、2D LiDAR の距離配列を入力として、ステアリング角と速度を直接出力する End-to-End モデル「TinyLidarNet」を取り上げます。前半では論文の内容を整理し、後半では PyTorch と 2D シミュレータ (robosim2d) を用いた最小構成の自作実装について解説します。

TinyLidarNet は F1TENTH Autonomous Grand Prix で 3 位を獲得した軽量モデルで、INT8 量子化を行えば MCU 上でもリアルタイム推論が可能とされています。公式実装は TensorFlow/Keras + ROS + 実機 F1TENTH の構成になっており、再現には実機環境が必要です。本記事ではこの公式実装の中身には踏み込まず、論文に準拠した最小構成の自作実装を題材として、1D CNN がどのように動作するかを順に確認していきます。

## 1. TinyLidarNet とは

### 1.1 何をするモデルか

TinyLidarNet の入出力と構成は次のようになっています。

- 入力: 2D LiDAR の 1D 距離配列（1081 点）
- 出力: ステアリング角と速度の 2 値
- 構成: PilotNet を参考にした 9 層 CNN（5 層の 1D Conv + 4 層の FC）
- 学習: 人間の運転を模倣する Behavior Cloning（手動運転およそ 5 分、約 1.2 万サンプル）

カメラ映像ではなく LiDAR スキャンを直接ニューラルネットワークに入力し、操舵と速度の制御指令をワンショットで生成する、というのが特徴になります。

### 1.2 背景: End-to-End 自動運転と 2D LiDAR

End-to-End 自動運転は、センサ入力から制御出力までを単一のニューラルネットワークで写像する手法です。先行研究として、ALVINN [2] を起点に NVIDIA の PilotNet / DAVE-2 [3] と続く、カメラ画像から操舵を推論するモデルが代表例として広く知られています。一方で、F1TENTH のような小型自律走行のコミュニティでは、LiDAR を入力に用いた End-to-End 手法も継続的に試されてきました。

LiDAR を使う動機としては、カメラと比べて入力データ量が少なく、照明条件にも影響を受けにくいという点が挙げられます。ただし論文 [1] では、F1TENTH 領域における従来の LiDAR End-to-End 手法の多くが MLP ベースであり、高速走行や未訓練コースでの性能に課題があったと整理されています。

TinyLidarNet はこの構図に対して、入力処理を 1D Conv に置き換えるという比較的単純な変更を加えたモデルです。論文 [1] では、2D LiDAR スキャンには角度方向の空間構造が含まれており、1D CNN は局所パターンを共通フィルタで抽出できるため、こうした特徴を表現しやすいと整理されています。MLP のように各角度点を独立に重み付けする場合、コース形状が少し変わるだけで全体の特徴量が大きく変化してしまいやすいのに対して、1D CNN は「左に壁、右に通路」「正面に折れ曲がりがある」といった局所構造を、比較的少ない学習データでも捉えやすいと考えられます。論文 [1] では、この性質が未訓練コースに対する汎化性能の向上に寄与したと述べられています。

### 1.3 アーキテクチャ

ネットワーク全体は 5 層の 1D 畳み込みと 4 層の全結合からなります。論文 [1] の Figure 3 に対応する構成を表にまとめると、次のようになります。

| 層 | カーネル / stride | 出力ch | 出力長 |
|---|---|---|---|
| Conv1 | k=10, s=4 | 24 | 268 |
| Conv2 | k=8, s=4  | 36 | 66 |
| Conv3 | k=4, s=2  | 48 | 32 |
| Conv4 | k=3, s=1  | 64 | 30 |
| Conv5 | k=3, s=1  | 64 | 28 |
| FC1   | —         | 100 | — |
| FC2   | —         | 50  | — |
| FC3   | —         | 10  | — |
| FC4 (出力) | —     | 2 (steering, speed) | — |

- 総パラメータ数: 220,686
- 推論時の演算量: 約 1.5M MACs
- 入力ダウンサンプルによる派生モデル: L (1081 点) / M (541 点) / S (271 点)

PilotNet (約 250K パラメータ、約 27M MACs) と比較すると、パラメータ数は同程度ですが、入力次元の低さに起因して演算量は 2 桁ほど少なく抑えられているとされます。

### 1.4 論文で報告されている評価結果

論文 [1] の主要な結果を整理すると、次のようになります。

**大会実績**

第 12 回 F1TENTH Autonomous Grand Prix において、13 チーム中 3 位を獲得したと報告されています。

**未訓練コースでの汎化**

- シミュレーション 4 トラック (GYM / Austin / Moscow / Spielberg) において、TinyLidarNet 系列 (L/M/S) はいずれも完走率 100% を達成したと報告されています
- 比較対象の MLP256 はほとんどのケースで完走できなかったとされています
- 実機の未訓練トラックでも、TinyLidarNet のみが完走したと報告されています

**INT8 量子化後の推論レイテンシ**

| プラットフォーム | TinyLidarNetL | TinyLidarNetM | TinyLidarNetS |
|---|---|---|---|
| Jetson Xavier NX | <1 ms | <1 ms | <1 ms |
| ESP32-S3 (Xtensa LX7) | 16 ms | 8 ms | 4 ms |
| Raspberry Pi Pico (Cortex-M0+) | 196 ms | 91 ms | 36 ms |

40 Hz (周期 25 ms) の制御を想定すると、ESP32-S3 では L 版でも余裕があり、Raspberry Pi Pico では S 版に限れば 20 Hz 程度の制御に間に合う水準とされています。

**副次的な観察**

論文 [1] では、静的障害物のみで学習したモデルが、競技中に動的な対戦車両を追い越す挙動を示したと報告されています。これは移動する車両をネットワークが壁や静止物として扱った結果ではないかと推測されていますが、限定的な条件下での観察であり、一般的な動的環境への対応を意味するわけではない点には注意が必要です。

---

## 2. 自作実装で動かす（PyTorch + 2D シミュレータ）

公式実装は TensorFlow/Keras + ROS + 実機 F1TENTH の構成になっているため、環境構築のコストが大きめです。本記事では論文のアーキテクチャと学習設定に準拠した最小構成を、PyTorch + robosim2d (軽量 2D シミュレータ) で組み直しました。コード量を抑えて、モデル定義や学習ループの中身を順に追いやすい構成にしています。

### 2.1 公式実装と自作実装のスタック差

| 項目 | 公式 | 自作 (本記事) |
|---|---|---|
| フレームワーク | TensorFlow / Keras | PyTorch |
| 環境 | 実機 F1TENTH + ROS Noetic | 2D シミュレータ (robosim2d) + キーボード操作 |
| データ形式 | rosbag (.bag) | NumPy 配列 (.npz) |
| 量子化 | TFLite int8 / fp32 | なし（解説目的のため省略） |

### 2.2 リポジトリ構成

```
tiny_lidar_net_example/
├── main.py                  # CLI エントリ (collect / train / autodrive / evaluate)
├── tiny_lidar_net/
│   ├── model.py             # 1D CNN 本体（論文に準拠）
│   ├── control.py           # ステア・速度の正規化と並び順を集約
│   ├── dataset.py           # NPZ ファイルを PyTorch Dataset 化
│   └── trainer.py           # Behavior Cloning 学習ループ
├── commands/                # サブコマンド (collect / train / autodrive / evaluate)
├── worlds/                  # robosim2d 用コース定義 (YAML)
└── outputs/                 # 出力先 (モデル .pth / 学習データ .npz / プロット .png)
```

学習データ収集 (`collect`)、学習 (`train`)、自動走行 (`autodrive`)、複数ワールドでの評価 (`evaluate`) の 4 つを `main.py` のサブコマンドとして提供しています。

### 2.3 モデル定義 (`tiny_lidar_net/model.py`)

論文の 9 層構成をそのまま PyTorch で組んだのが以下のコードです。

```python
class TinyLidarNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_length = 1081  # 入力長は 1081 に固定 (Lモデル)

        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3, stride=1)

        self.flatten = nn.Flatten()
        # 入力長 1081 のとき Conv5 通過後の時間軸長は 28
        flat_len = 64 * 28  # = 1792
        self.fc1 = nn.Linear(flat_len, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x
```

実装上のポイントは次の 3 点になります。

**Conv の kernel/stride を論文の表とそのまま一致させている**

5 層の 1D Conv は、論文 [1] の Figure 3 と同じカーネル長・ストライドを指定しています。

**入力長を 1081 に固定している**

本実装では入力する LiDAR 点群のサイズを 1081 (Lモデル) に固定しており、Conv5 通過後の時間軸長 28 から、FC1 の入力サイズを `flat_len = 64 * 28 = 1792` として直接与えています。公式 Keras 実装は入力 shape から FC1 の入力サイズを自動計算する性質を持ち、L (1081) / M (541) / S (271) の派生モデルに対応していますが、本実装ではこれを `nn.LazyLinear` などで再現することはせず、簡単のため省略しました。

**最終層に `tanh` を入れて出力域を [-1, 1] に揃えている**

ステアリングと速度はいずれも有界な物理量であり、公式実装と同様に `tanh` で出力域を [-1, 1] に押し込めています。ラベル側も同じレンジに正規化することで、損失計算と推論時の逆変換を素直に書けるようにしています。なお、本実装では小規模な学習データでの過学習を抑える目的で、FC1/FC2 後に `Dropout(0.2)` を加えています。

### 2.4 出力値の正規化 (`tiny_lidar_net/control.py`)

ステアリングと速度の値は、物理単位 (rad / m·s⁻¹) と `tanh` の出力域 ([-1, 1]) の間を双方向に行き来します。また、学習データとシミュレータ API の間で「ステアリングと速度の並び順」も食い違います。これらを 1 か所にまとめておくために、`Control` という NamedTuple を用意しました。

```python
# モデル最終層 tanh の出力域 [-1, 1] と物理値を対応させる正規化定数。
MAX_STEERING = 0.5  # [rad]
MAX_SPEED = 5.0     # [m/s]


class Control(NamedTuple):
    """車両制御値（ステアリング角と速度）。

    並び順の規約:
        - 学習データ (.npz) / モデル出力テンソル: [steering, speed]
        - robosim2d ``sim.step()`` の action 配列:  [speed, steering]
    """

    steering: float
    speed: float

    def to_training_label(self) -> np.ndarray:
        return np.array([self.steering, self.speed], dtype=np.float32)

    def to_robot_action(self) -> np.ndarray:
        return np.array([self.speed, self.steering], dtype=np.float32)

    def to_normalized(self) -> np.ndarray:
        return np.array(
            [self.steering / MAX_STEERING, self.speed / MAX_SPEED],
            dtype=np.float32,
        )

    @classmethod
    def from_model_output(cls, arr) -> "Control":
        return cls(
            steering=float(arr[0]) * MAX_STEERING,
            speed=float(arr[1]) * MAX_SPEED,
        )
```

このように 1 か所に集約しておくと、`tanh` 出力と物理値の往復、`[steering, speed]` と `[speed, steering]` の並びの違いを取り違えにくくなります。

### 2.5 学習ループ (`tiny_lidar_net/trainer.py`)

学習ループは、論文に合わせた Huber loss と Adam の単純な組み合わせで構成しています。

```python
class Trainer:
    """TinyLidarNet を Huber 損失で学習するシンプルな Trainer。

    公式実装に準拠したハイパラ:
        Optimizer: Adam(lr=5e-5),  Loss: HuberLoss(δ=1.0),
        Batch: 64,  Epoch: 20
    """

    def __init__(self, model, learning_rate=5e-5, device=None):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.HuberLoss(delta=1.0)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.train_losses, self.val_losses = [], []

    def _train_one_epoch(self, loader, n_samples):
        self.model.train()
        total_loss = 0.0
        for lidar, control in loader:
            lidar = lidar.to(self.device)
            control = control.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(lidar)
            loss = self.criterion(output, control)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * lidar.size(0)
        return total_loss / n_samples
```

実装上のポイントは次のとおりです。

- 損失関数には `nn.HuberLoss(delta=1.0)` を採用し、論文の記述に合わせています
- 最適化は `Adam(lr=5e-5)`、`batch_size=64` で、公式実装と同一の設定です
- 学習データは `random_split` で train/val に分割するシンプルな構成にしています

データセットは NPZ ファイル (`lidar` と `control` の 2 配列を持つ形式) を `LidarDataset` で読み込み、`DataLoader` 経由でミニバッチに分割する、ごく標準的な構成です。

---

## 3. デモ・実験結果と考察

ここでは、学習データの構成（ワールド数・走行方向）を変えながら、自作実装の汎化挙動を `evaluate` コマンドで定量的に確認していきます。単一ワールド・単一方向の学習データから始めて、徐々に多様性を増やす形で挙動の変化を観察します。

### 3.1 評価方法

汎化性能を定量的に確認するために、`main.py evaluate` で複数ワールド × 複数初期位置の組み合わせをヘッドレスで走行させ、その結果を集計しました。

- 各ワールドの `eval.yaml` に CW / CCW 双方の初期位置を合計 4 点ずつ定義しており、計 6 ワールド × 4 スタートを試行しています
- 各試行は次の 3 種類に分類しています
  - `collision`: 障害物・壁に衝突して停止
  - `stuck`: 一定時間ほとんど進まない、または平均速度がしきい値未満
  - `survived`: 制限ステップまで衝突せず走行を継続
- 集計指標は完走率 (`Success%`)、平均走行距離 (`MeanDist`)、平均速度 (`MeanAvgV`) を確認しています

### 3.2 単一ワールド・単一方向で学習した場合

最初に、特定のワールドを特定の方向で走行した教師データのみで学習させた場合の挙動を確認します。

#### 3.2.1 学習設定

- 学習データ
  - ワールド: `worlds/curved_circuit`
  - 進行方向: 反時計回り (CCW) のみ
  - サンプル数: 約 6,200 samples
- モデル: `outputs/network/net_curved_circuit_only_ccw.pth`

#### 3.2.2 学習データと同じ条件での走行

学習時と同じワールド・同じ初期位置・同じ進行方向で再生したところ、半周程度であれば走行を継続できる挙動が確認できました。

- 成功時のデモ: `docs/zenn/Screencast from 2026-05-20 10-43-06.webm` を埋め込む予定です
- 曲率の異なる区間でステアリングと速度が変化する様子が観察できます

#### 3.2.3 初期位置・ワールドを変えた場合の挙動

一方、学習データに含まれていない初期位置や別のワールドで走行させると、走行が継続できなくなる挙動が観察されました。

- 失敗時のデモ: [curved_training_collision.gif](./curved_training_collision.gif) を埋め込む予定です

`evaluate` の集計結果は次のようになりました。6 ワールド × 4 スタートのすべての試行で衝突が観察されています。学習対象である `curved_circuit` についても、`eval.yaml` で定義した初期位置は訓練データの開始位置とは一致しないため、衝突を免れていません。

```text
World                     Success%  Stuck%  Coll%  MeanDist  MeanAvgV
---------------------------------------------------------------------
chicane_circuit                 0%      0%   100%      28.3      1.31
circuit                         0%      0%   100%      27.8      1.42
curved_circuit                  0%      0%   100%      55.3      2.30
diagonal_chicane_circuit        0%      0%   100%      31.4      1.41
double_island_circuit           0%      0%   100%      46.2      1.57
maze                            0%      0%   100%      36.8      1.15
```

- 学習対象の `curved_circuit` でも全試行が衝突しており、初期位置を変えるだけで走行が破綻しやすい様子が確認できました
- 学習データが「特定の形状を、特定の向きで走る」という 1 通りのパターンに偏っているため、特徴量が初期位置や曲率の符号に過学習している可能性が考えられます

### 3.3 複数ワールド・両方向で学習した場合

続いて、学習データに別ワールドと逆方向の走行を加え、走行パターンの多様性を増やした場合の挙動を確認します。

#### 3.3.1 学習設定

- 学習データ
  - ワールド: `worlds/curved_circuit`、`worlds/chicane_circuit`
  - 進行方向: 時計回り (CW) と反時計回り (CCW) の両方
  - サンプル数: 約 24,800 samples
- モデル: `outputs/network/net_chicane_curved.pth`

#### 3.3.2 評価結果

```text
World                     Success%  Stuck%  Coll%  MeanDist  MeanAvgV
---------------------------------------------------------------------
chicane_circuit                 0%     25%    75%     110.7      1.34
circuit                        50%     50%     0%    1745.6      2.56
curved_circuit                  0%     25%    75%     100.1      1.81
diagonal_chicane_circuit        0%     50%    50%      97.2      1.32
double_island_circuit           0%     50%    50%     462.9      1.74
maze                            0%     25%    75%      65.1      1.68
```

- 学習対象の `curved_circuit` と `chicane_circuit` では、衝突率が 100% から 75% に下がり、`stuck` で停止する試行が現れました
- 未学習ワールドの `circuit` では、4 試行中 2 試行で制限ステップまで走行を継続し、平均走行距離も 1,745.6 m と他ワールドを大きく上回りました
- `diagonal_chicane_circuit` や `double_island_circuit` でも衝突率が下がり、走行距離が伸びる傾向が見られました
- 通路幅が狭く形状が大きく異なる `maze` でも `stuck` の試行が現れており、衝突直前で減速・停止する挙動が増えている様子が観察されました

#### 3.3.3 2 モデルの比較

完走率と平均走行距離を 2 モデルで並べると、複数ワールド × 両方向の学習データを用いた場合の方が、全ワールドにわたって挙動が改善する傾向が見られます。

| World | only_ccw: Success% / MeanDist | chicane+curved (CW+CCW): Success% / MeanDist |
| --- | --- | --- |
| chicane_circuit | 0% / 28.3 m | 0% / 110.7 m |
| circuit | 0% / 27.8 m | 50% / 1745.6 m |
| curved_circuit | 0% / 55.3 m | 0% / 100.1 m |
| diagonal_chicane_circuit | 0% / 31.4 m | 0% / 97.2 m |
| double_island_circuit | 0% / 46.2 m | 0% / 462.9 m |
| maze | 0% / 36.8 m | 0% / 65.1 m |

### 3.4 学習曲線

下図は、曲線サーキットで作成した約 10 分ほどの教師データを用いて、100 epoch 学習させた際の Huber loss の推移です。

![学習曲線](./tiny_lidar_net_curved_circuit_epoch100.png)

train / val loss はいずれも数 epoch で大きく低下し、その後はゆっくりと収束していく挙動が見られました。論文 [1] の epoch 数 (20) よりも長く回してみたところ、それ以降も val loss はゆるやかに下がり続ける様子が観察できます。

ただし 3.2 で確認した通り、損失そのものが収束していても、初期位置や別ワールドへの汎化性能とは必ずしも一致しない点には注意が必要です。

---

## 4. まとめ

本記事では、2D LiDAR + 1D CNN + Behavior Cloning という比較的素朴な組み合わせで構成された TinyLidarNet を、論文と自作実装の両面から整理してきました。論文 [1] では、F1TENTH の競技と未訓練コースの両方において、従来の MLP ベース手法よりも良好な結果が報告されており、パラメータ数は約 220K と軽量で、INT8 量子化を行えば MCU 上でもリアルタイム推論が可能とされています。

自作の PyTorch + 2D シミュレータ実装で得られた観察結果は、次のように整理できます。

- 約 220K パラメータと数分〜十数分程度の手動運転データでも、学習データと一致する条件下では 2D シミュレータ上での走行が成立しました
- 単一ワールド・単一方向の学習データでは、学習対象のワールドであっても初期位置を変えるだけで走行が破綻し、Behavior Cloning が学習データの分布に対して敏感である様子が確認できました
- 学習データに逆方向の走行や別ワールドの走行を加えると、未学習ワールドでも走行が継続する試行が現れ、データ多様性が汎化に寄与する傾向が見られました
- 一方で、`maze` のように通路幅や形状が大きく異なるワールドでは依然として衝突が支配的であり、本実装の範囲では 2 ワールド程度の学習データではサンプル外への一般化に限界があるように見えます
- 本実装は静的環境のみを対象としており、動的障害物への対応や Sim2Real (シミュレータと実機のギャップ) の検証は行っていません
- 論文 [1] で報告された「動的車両を壁とみなして追い越す」現象は限定条件下での観察結果であり、本実装では再現も一般化の検証も行っていません

論文 [1] の主要な主張である「1D Conv による角度方向特徴の抽出が、MLP と比べて有利に働く」という観察については、自作の小規模な再現でも、データ多様性を確保した場合に未学習ワールドへの汎化が現れるなど、方向性として矛盾しない結果が得られたと考えています。

今後の発展としては、MLP 版との同条件での比較、PyTorch → ONNX → TFLite への量子化、動的障害物への拡張、Sim2Real などが考えられます。

---

## 参考文献

[1] Mohammed Misbah Zarrar, Qitao Weng, Bakhbyergyen Yerjan, Ahmet Soyyigit, Heechul Yun. "TinyLidarNet: 2D LiDAR-based End-to-End Deep Learning Model for F1TENTH Autonomous Racing." arXiv:2410.07447, 2024. <https://arxiv.org/abs/2410.07447>

[2] Dean A. Pomerleau. "ALVINN: An Autonomous Land Vehicle in a Neural Network." Advances in Neural Information Processing Systems 1 (NIPS 1988), pp. 305-313, 1989. <https://proceedings.neurips.cc/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html>

[3] Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D. Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, Xin Zhang, Jake Zhao, Karol Zieba. "End to End Learning for Self-Driving Cars." arXiv:1604.07316, 2016. <https://arxiv.org/abs/1604.07316>
