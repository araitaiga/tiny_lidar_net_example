## はじめに

本記事では、2D LiDAR を入力、ステアリング角と速度を出力とする End-to-End モデルである、TinyLidarNet [1] を取り上げます。TinyLidarNet は F1TENTH Autonomous Grand Prix で 3 位を獲得した軽量モデルで、INT8 量子化を行えば MCU 上でもリアルタイム推論が可能とされています。
前半では論文の内容を整理し、後半では PyTorch と 2D シミュレータを用いた小規模構成の実装を用いて具体的な実装や動作を解説します。

## 1. TinyLidarNet

### 1.1 概要

End-to-End 自動運転は、センサ入力から制御出力までを単一のニューラルネットワークで写像する手法です。こうした End-to-End モデルのネットワークとして有名なものは、ALVINN [2] や NVIDIA の PilotNet / DAVE-2 [3] などがあります。これらのモデルの多くは、自動車の自動運転向けに、カメラ画像から操舵を推論するものでした。その一方、TinyLidarNet は LiDAR を入力とする End-to-End モデルです。

LiDAR がカメラと比較して優れている点は、入力データ量が少ないこと、照明条件に影響を受けにくいことなどが挙げられます。過去にも LiDAR を入力とする End-to-End モデルは存在しましたが、その多くが MLP (全結合層の積層) ベースであり、性能に課題があったと主張されています。

そこで、TinyLidarNet は 5 層の 1D Conv (畳み込み層) + 4 層の FC (全結合層) からなるモデルを提案しています。入力に対する処理を 1D Conv に置き換えることで、2D LiDAR データに含まれる空間構造を抽出しやすくなり、論文中ではこの性質が未訓練コースに対する汎化性能の向上に寄与したと述べられています。

- 入力: 2D LiDAR の 1D 距離配列（1081 点）
- 出力: ステアリング角と速度の 2 値
- 総パラメータ数: 約 22 万パラメータと少数
- 目的: 人間の運転操作を模倣する Behavior Cloning
- 学習データ: 手動運転およそ 5 分、約 1.2 万サンプルと少量

<デモ動画>

## 1.2 アーキテクチャ

ネットワーク構造は以下です。

![tiny_lidar_net_architecture](./outputs/tiny_lidar_net_arch.png)

- 総パラメータ数: 220,686
- 入力ダウンサンプルによる派生モデル: L (1081 点) / M (541 点) / S (271 点)

## 1.3 論文で報告されている評価

論文で主張されている主要な評価をまとめると、次のようになります。

**大会実績**

第 12 回 F1TENTH Autonomous Grand Prix において、13 チーム中 3 位を獲得しました。

**未訓練コースでの汎化**

- シミュレーション 4 トラックにおいて、TinyLidarNet 系列 (L/M/S) はいずれも完走率 100% を達成
- 比較対象の MLP256 はほとんどのケースで完走できなかった
- 実機の未訓練トラックでも、TinyLidarNet のみが完走

**INT8 量子化後の推論レイテンシ**

| プラットフォーム | TinyLidarNetL | TinyLidarNetM | TinyLidarNetS |
|---|---|---|---|
| Jetson Xavier NX | <1 ms | <1 ms | <1 ms |
| ESP32-S3 (Xtensa LX7) | 16 ms | 8 ms | 4 ms |
| Raspberry Pi Pico (Cortex-M0+) | 196 ms | 91 ms | 36 ms |

40 Hz (周期 25 ms) の制御を想定すると、ESP32-S3 では L 版でも余裕があり、Raspberry Pi Pico では S 版に限れば 20 Hz 程度の制御に間に合う水準とされています。

**副次的な性能**

静的障害物環境でのみで学習したモデルが、動的な他車両を追い越す挙動を示したと報告されています。これは移動する車両をネットワークが壁や静止物として扱った結果ではないかと推測されています。

## 2. TinyLidarNetの実装

公式実装は TensorFlow/Keras + ROS の構成になっているため、動作させるまでの環境構築のコストが大きめです。本記事では、論文のアーキテクチャに準拠した小規模構成を、PyTorch + 自作の軽量 2D シミュレータ (robosim2d) で実装しました。コード量を抑えて、本質的なモデル定義や学習方法を追いやすい実装を意識しています。
なお、2D シミュレータについては、当初 IR-SIM (<https://github.com/hanruihua/ir-sim>) や F1TENTH Gym (<https://github.com/f1tenth/f1tenth_gym>) の利用を検討しました。手元の環境では 1081 点規模の 2D LiDAR をシミュレートする際の処理コストが大きかったため、本記事では軽量化を目的に自作した robosim2d を用いています。TinyLidarNet の理解が主目的であるため、シミュレータの選定自体は本質ではありません。

以下、本実装を具体的なコードと共に解説します。

- TinyLidarNet 実装: <https://github.com/araitaiga/tiny_lidar_net_example>

- 2D シミュレータ: <https://github.com/araitaiga/robosim2d>

### 2.1 リポジトリ構成

```
tiny_lidar_net_example/
├── main.py                  # CLI エントリ (collect / train / autodrive / evaluate)
├── tiny_lidar_net/
│   ├── model.py             # 1D CNN 本体（論文に準拠）
│   ├── control.py           # ステア・速度の正規化と並び順を集約
│   ├── dataset.py           # NPZ ファイルを PyTorch Dataset 化
│   └── trainer.py           # Behavior Cloning 学習ループ
├── commands/                # サブコマンド (collect / train / autodrive / evaluate)
└── worlds/                  # robosim2d 用コース定義 (YAML)
```

学習データ収集 (`collect`)、学習 (`train`)、自動走行 (`autodrive`)、複数ワールドでの評価 (`evaluate`) の 4 つを `main.py` のサブコマンドとしています。

### 2.2 モデル定義 (`tiny_lidar_net/model.py`)

```python
class TinyLidarNet(nn.Module):
    def __init__(self, input_length: int = 1081):
        super().__init__()
        self.input_length = input_length

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x
```

畳み込み層 + ReLU による変換を5層、全結合層 + ReLUによる変換を4層重ねた、非常にシンプルなモデルです。
最終層には `tanh` が入っています。 tanh は入力を [-1, 1] の範囲に揃える、S字型の活性化関数です。

![tanh](./outputs/tanh.png)

ステアリングと速度はいずれも有界な物理量 [最小制御値, 最大制御値] のため、-1 --> 最小制御値、 1 --> 最大制御値 のようにマッピングすることで、ニューラルネットワークの出力が物理量の範囲を超えないようにすることができます。損失計算の際には、教師データ側も教師制御値を同じレンジ [-1, 1] に正規化する必要があります。

なお、上記の実装ではInputするLiDAR点群のサイズを1081に固定していますが、公式実装ではこの入力長が変更可能にしており、L (1081) / M (541) / S (271) といった派生モデルに対応しています。これをPyTorchで実現するためには、全結合層にnn.LazyLinearを使用するなどの方法がありますが、今回は簡単のため省略します。

### 2.3 出力値の正規化 (`tiny_lidar_net/control.py`)

上述のとおり、ステアリングと速度の値は、物理単位 (rad / m·s⁻¹) と `tanh` の出力域 ([-1, 1]) の間を双方向に変換する必要があります。また、学習データとシミュレータ API の間で制御形式が若干異なります。これらの知識を 1 か所にまとめておくためのクラスです。

```python
# モデル最終層 tanh の出力域 [-1, 1] と物理値を対応させる正規化定数
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

    # 学習データ (.npz) に格納するラベル配列 [steering, speed]
    def to_training_label(self) -> np.ndarray:
        return np.array([self.steering, self.speed], dtype=np.float32)

    # robosim2d sim.step() に渡す action 配列 [speed, steering]
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

### 2.4 学習ループ (`tiny_lidar_net/trainer.py`)

```python
class Trainer:
    """TinyLidarNet を Huber 損失で学習する Trainer。

    公式実装に準拠したハイパーパラメータ:
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

学習ループは、論文に合わせた Huber loss と Adam の組み合わせで構成しています。
実装上のポイントは次のとおりです。

- 損失関数には `nn.HuberLoss(delta=1.0)` を採用し、論文の記述に合わせています
- 最適化は `Adam(lr=5e-5)`、`batch_size=64` で、公式実装と同一の設定です
- 学習データは `random_split` で train/val に分割するシンプルな構成にしています

データセットは NPZ ファイル (`lidar` と `control` の 2 配列を持つ形式) を `LidarDataset` で読み込み、`DataLoader` 経由でミニバッチに分割する、ごく標準的な構成です。

## 3. デモ・実験

ここでは、学習データの多様性とTinyLidarNetの汎化性能を確認していきます。単一ワールド・単一方向の学習データでのモデルと、多様性を増やした学習データでのモデルの挙動の変化を観察します。

### 3.1 評価方法

### 3.2 単一ワールド・単一方向の学習データの場合

まず、特定のワールドを特定の方向で走行した教師データのみで学習させた場合の挙動を確認します。

- 学習データ
  - ワールド: `worlds/curved_circuit`
  - 進行方向: 反時計回り (CCW) のみ
  - サンプル数: 約 6,200 samples
- モデル: `outputs/network/net_curved_circuit_only_ccw.pth`

学習時と同じワールド・同じ初期位置・同じ進行方向で再生したところ、半周程度であれば走行を継続できる挙動が確認できました。  

(3倍速)  
![outputs/contents/only_curved_ccw.gif](tiny_lidar_net_example/outputs/contents/only_curved_ccw.gif)

一方、学習データに含まれていない初期位置や別のワールドで走行させると、それらがシンプルな環境であっても走行が継続できなくなる挙動が観察されました。

TinyLidarNetは自己方位に対する周囲形状を元に、自己方位前方側のフリーな領域へ進むような制御指令値を学習していると考えられます。そのため、初期位置を変えるだけで、ロボットがサーキットコースを時計回りに走行するか、反時計回りに走行するかが自然に切り替わります。
しかし、このケースでは学習データが反時計回りでのみ収集されているので、制御指令値に反時計回りのバイアスがかかっており、大局的には道が進行方向右側にカーブしているような状況でも、細かい障害物の周囲を局所的に反時計回りに回ろうとする挙動が確認されました。  

![outputs/contents/curved_training_ccw_moving_overtraining.gif](tiny_lidar_net_example/outputs/contents/curved_training_ccw_moving_overtraining.gif)


複数のワールド (学習時のワールドを含む) と複数の初期状態 (位置、方位) で、壁に衝突するかスタックするまでの平均走行距離と速度を測定した結果が以下です。  
10000 ステップの間 衝突 or スタックせずに走行を継続した場合、Successとみなしています。  


| World | Trials | Success% | Stuck% | Coll% | MeanDist | MeanAvgV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chicane_circuit | 4 | 0% | 0% | 100% | 28.3 | 1.31 |
| circuit | 4 | 0% | 0% | 100% | 27.8 | 1.42 |
| curved_circuit | 4 | 0% | 0% | 100% | 55.3 | 2.30 |
| diagonal_chicane_circuit | 4 | 0% | 0% | 100% | 31.4 | 1.41 |
| double_island_circuit | 4 | 0% | 0% | 100% | 46.2 | 1.57 |
| maze | 4 | 0% | 0% | 100% | 36.8 | 1.15 |

- 全ワールド、特に学習対象の `curved_circuit` でも全試行が衝突しており、初期状態を変えるだけで走行が破綻しやすい様子が確認できました
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

| World | Trials | Success% | Stuck% | Coll% | MeanDist | MeanAvgV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chicane_circuit | 4 | 0% | 25% | 75% | 110.7 | 1.34 |
| circuit | 4 | 50% | 50% | 0% | 1745.6 | 2.56 |
| curved_circuit | 4 | 0% | 25% | 75% | 100.1 | 1.81 |
| diagonal_chicane_circuit | 4 | 0% | 50% | 50% | 97.2 | 1.32 |
| double_island_circuit | 4 | 0% | 50% | 50% | 462.9 | 1.74 |
| maze | 4 | 0% | 25% | 75% | 65.1 | 1.68 |

- 全てのワールドで、衝突率が 100% から下がり、一方で `stuck` により停止する試行が現れました
- 未学習/シンプルなワールドの `circuit` では、4 試行中 2 試行で走行を継続し、平均走行距離も 1,745.6 m と他ワールドを大きく上回りました
- `diagonal_chicane_circuit` や `double_island_circuit` でも衝突率が下がり、走行距離が伸びる傾向が見られました


2モデルの比較のまとめです  


| World | only_ccw: Success% / MeanDist | chicane+curved (CW+CCW): Success% / MeanDist |
| --- | --- | --- |
| chicane_circuit | 0% / 28.3 m | 0% / 110.7 m |
| circuit | 0% / 27.8 m | 50% / 1745.6 m |
| curved_circuit | 0% / 55.3 m | 0% / 100.1 m |
| diagonal_chicane_circuit | 0% / 31.4 m | 0% / 97.2 m |
| double_island_circuit | 0% / 46.2 m | 0% / 462.9 m |
| maze | 0% / 36.8 m | 0% / 65.1 m |
