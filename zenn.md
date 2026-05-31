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

---

# Outline

## 0. はじめに

- 本記事の目的: 2D LiDAR の距離配列を入力として、ステアリング角と速度を直接出力する End-to-End モデル「TinyLidarNet」を、論文の解説と自作の PyTorch 実装の二段構成で説明する
- TinyLidarNet は F1TENTH の大会で 3 位を獲得した軽量モデルで、MCU でもリアルタイム推論できる点が特徴
- 公式実装 (TensorFlow/Keras + ROS + 実機 F1TENTH) は構成の参照先として扱うのみで、本記事では中身に踏み込まない。代わりに、最小構成の自作実装 (PyTorch + 2D シミュレータ) の主要なコードを解説する
- 自作実装はミニマルな構成にしているため、コードを追いながら 1D CNN の動作を確認しやすい

---

## 1. TinyLidarNet とは

> 背景・アーキテクチャ・強みを 1 章にまとめる

### 1.1 何をするモデルか
- 入力: 2D LiDAR の 1D 距離配列（1081 点）
- 出力: ステアリング角と速度の 2 値
- 構成: PilotNet を参考にした 9 層 CNN（5 1D Conv + 4 FC）
- 学習: 人間の運転を模倣する Behavior Cloning（手動運転 5 分・約 1.2 万サンプル）

### 1.2 背景: End-to-End 自動運転と 2D LiDAR
- 先行研究として、ALVINN (1989) → NVIDIA PilotNet / DAVE-2 (2016) と続く、カメラ画像から操舵を推論する E2E モデルがある
- 2D LiDAR を使う利点: データ量が少なく、照明条件にも影響されにくい
- F1TENTH 領域の従来の LiDAR E2E は MLP ベースで、高速走行や未訓練コースでの性能に課題があったとされる
- TinyLidarNet では入力処理を 1D Conv に変更
  - 2D LiDAR scanには空間的構造の情報が含まれており、1D CNN はそうした特徴を表現しやすい
  - 論文では、この性質が未訓練コースに対する汎化性能の向上につながったと述べられている

### 1.3 アーキテクチャ
- PilotNet（2D Conv ベース）と TinyLidarNet（1D Conv ベース）の対比図を入れる予定

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

- パラメータ数: 220,686、推論時の演算量: 約 1.5M MACs
- 入力ダウンサンプルによる派生モデル: L (1081点) / M (541点) / S (271点)

### 1.4 論文で報告されている評価結果
- 大会実績: 第12回 F1TENTH Autonomous Grand Prix で 13 チーム中 3 位
- 未訓練コースでの汎化:
  - シミュレーション 4 トラック (GYM / Austin / Moscow / Spielberg) で、TinyLidarNet は完走率 100% を達成。比較対象の MLP256 はほとんどのケースで完走できなかった
  - 実機の未訓練トラックでも同様に TinyLidarNet のみが完走
- INT8 量子化後の推論レイテンシ:
  - Jetson Xavier NX: <1 ms
  - ESP32-S3: 4–16 ms
  - Raspberry Pi Pico: 36–196 ms（S 版なら 20Hz 制御に間に合う水準）
- 副次的な観察: 静的障害物のみで学習した重みでも、競技では動的な対戦車両を追い越す挙動が確認された（移動する車両を壁や静止物として扱った結果と推測されている）

---

## 2. 自作実装で動かす（PyTorch + 2D シミュレータ）

> 公式実装は TF/Keras + ROS + 実機 F1TENTH の構成で、環境を整えるコストが大きい。本記事では PyTorch + robosim2d を使った最小構成の自作実装で、モデルの中身に集中できるようにする

### 2.1 公式実装と自作実装のスタック差

| 項目 | 公式 | 自作 (本記事) |
|---|---|---|
| フレームワーク | TensorFlow / Keras | PyTorch |
| 環境 | 実機 F1TENTH + ROS Noetic | 2D シミュレータ (robosim2d) + キーボード操作 |
| 量子化 | TFLite int8 | なし（解説目的のため省略） |

### 2.2 リポジトリ構成
```
tiny_lidar_net_example/
├── main.py                  # CLI エントリ (collect / train / autodrive)
├── tiny_lidar_net/
│   ├── model.py             # 1D CNN 本体（論文に準拠）
│   ├── control.py           # ステア・速度の正規化と並び順を集約
│   ├── dataset.py           # NPZ ファイルを PyTorch Dataset 化
│   └── trainer.py           # Behavior Cloning 学習ループ
├── commands/                # サブコマンド (collect / train / autodrive)
└── worlds/                  # robosim2d 用コース定義 (YAML)
```

### 2.3 モデル定義（`tiny_lidar_net/model.py`）
- 引用するコード: `__init__` の Conv/FC 定義、`forward`、`predict`
- 解説ポイント:
  - Conv の kernel/stride は論文の表とそのまま一致させている (`_CONV_PARAMS`)
  - FC1 の入力サイズは入力長から動的に算出（公式 Keras の遅延構築と等価な挙動を PyTorch で再現）
  - 最終層に `tanh` を入れ、出力域を [-1, 1] に揃えている。（公式実装と同様）

### 2.4 出力値の正規化（`tiny_lidar_net/control.py`）
- 引用するコード: `MAX_STEERING`, `MAX_SPEED`, `Control` NamedTuple, `to_normalized` / `from_model_output`
- 解説ポイント:
  - 物理値と `tanh` 出力 [-1, 1] の双方向変換を 1 か所に集約し、取り違えを防いでいる

### 2.5 学習ループ（`tiny_lidar_net/trainer.py`）
- 引用するコード: `Trainer.__init__`（Huber + Adam の設定）と `_train_one_epoch` の中心部分
- 解説ポイント:
  - 損失関数は `nn.HuberLoss(delta=1.0)`（論文に合わせている）
  - `random_split` で train/val を分割するシンプルな構成
  - ハイパーパラメータは公式実装と同じ値: `Adam(lr=5e-5)`, `batch=64`, `epoch=20`

---

## 3. デモ・実験結果と考察

> 学習データの構成（ワールド数・走行方向）を変えながら、汎化挙動を `evaluate` コマンドで定量的に確認する。
> 単一ワールド・単一方向の学習データから始めて、徐々に多様性を増やす形で比較する。

### 3.1 評価方法

- `main.py evaluate` を用いて、複数ワールド × 複数初期位置の組み合わせで走行を行う
- 各ワールドの `eval.yaml` には CW / CCW 双方の初期位置 (合計 4 点) を定義しており、計 6 ワールド × 4 スタートを試行する
- 試行結果は次の 3 種類に分類する
  - `collision`: 障害物・壁に衝突して停止
  - `stuck`: 一定時間ほとんど進まない、または平均速度がしきい値未満
  - `survived`: 制限ステップまで衝突せず走行を継続
- 集計指標として完走率 (`Success%`)、平均走行距離 (`MeanDist`)、平均速度 (`MeanAvgV`) を確認する

### 3.2 単一ワールド・単一方向で学習した場合

#### 3.2.1 学習設定

- 学習データ:
  - ワールド: `worlds/curved_circuit`
  - 進行方向: 反時計回り (CCW) のみ
  - サンプル数: 約6200 samples
- モデル: `outputs/network/net_curved_circuit_only_ccw.pth`

#### 3.2.2 学習データと同じ条件での走行

学習時と同じワールド・同じ初期位置・同じ進行方向で再生すると、約半周にわたって走行を継続できる。

- 成功時のデモ: `outputs/Screencast from 2026-05-20 10-43-06.webm` を埋め込む予定
- 曲率の変化への追従挙動が確認できる

#### 3.2.3 初期位置・ワールドを変えた場合の挙動

学習データに含まれていない初期位置や別のワールドで走行させると、走行が継続しなくなる。

- 失敗時のデモ: [outputs/contents/curved_training_collision.gif](tiny_lidar_net_example/outputs/contents/curved_training_collision.gif) を埋め込む予定

`evaluate` の集計結果は次の通りで、6 ワールド × 4 スタートのすべてが衝突となった。`curved_circuit` でも `eval.yaml` の初期位置は訓練データの開始位置と一致しないため、衝突を免れていない。

```text
World                      Trials  Success%  Stuck%  Coll%  MeanDist  MeanAvgV
--------------------------------------------------------------------------------
chicane_circuit                 4        0%      0%   100%      28.3      1.31
circuit                         4        0%      0%   100%      27.8      1.42
curved_circuit                  4        0%      0%   100%      55.3      2.30
diagonal_chicane_circuit        4        0%      0%   100%      31.4      1.41
double_island_circuit           4        0%      0%   100%      46.2      1.57
maze                            4        0%      0%   100%      36.8      1.15
```

- 学習対象の `curved_circuit` であっても全試行が衝突しており、初期位置を変えるだけで走行が破綻している
- 学習データが「特定の形状を、特定の向きで走る」という 1 通りのパターンに偏っているため、特徴量が初期位置や曲率の符号を過学習していると考えられる

### 3.3 複数ワールド・両方向で学習した場合

#### 3.3.1 学習設定

学習データに別ワールドと逆方向の走行を加え、走行パターンの多様性を増やす。

- 学習データ:
  - ワールド: `worlds/curved_circuit`、`worlds/chicane_circuit`
  - 進行方向: 時計回り (CW)、反時計回り (CCW)
  - サンプル数: 約24800 samples
- モデル: `outputs/network/net_chicane_curved.pth`

#### 3.3.2 評価結果

```text
World                      Trials  Success%  Stuck%  Coll%  MeanDist  MeanAvgV
--------------------------------------------------------------------------------
chicane_circuit                 4        0%     25%    75%     110.7      1.34
circuit                         4       50%     50%     0%    1745.6      2.56
curved_circuit                  4        0%     25%    75%     100.1      1.81
diagonal_chicane_circuit        4        0%     50%    50%      97.2      1.32
double_island_circuit           4        0%     50%    50%     462.9      1.74
maze                            4        0%     25%    75%      65.1      1.68
```

- 学習対象の `curved_circuit` と `chicane_circuit` では、衝突率が 100% から 75% に下がり、`stuck` で停止する試行が現れた
- 未学習ワールドの `circuit` では 4 試行中 2 試行で制限ステップまで走行を継続し、平均走行距離も 1745.6 m と他ワールドを大きく上回った
- `diagonal_chicane_circuit` や `double_island_circuit` でも衝突率が下がり、走行距離が伸びる傾向が見られた
- 通路幅が狭く形状が大きく異なる `maze` でも `stuck` の試行が現れ、衝突直前で減速・停止する挙動が増えている

#### 3.3.3 2 モデルの比較

完走率と平均走行距離を 2 モデルで並べると、複数ワールド × 両方向の学習データで全ワールドにわたって挙動が改善している。

| World | only_ccw: Success% / MeanDist | chicane+curved (CW+CCW): Success% / MeanDist |
| --- | --- | --- |
| chicane_circuit | 0% / 28.3 m | 0% / 110.7 m |
| circuit | 0% / 27.8 m | 50% / 1745.6 m |
| curved_circuit | 0% / 55.3 m | 0% / 100.1 m |
| diagonal_chicane_circuit | 0% / 31.4 m | 0% / 97.2 m |
| double_island_circuit | 0% / 46.2 m | 0% / 462.9 m |
| maze | 0% / 36.8 m | 0% / 65.1 m |

### 3.4 学習曲線

- [outputs/tiny_lidar_net_curved_circuit_epoch100.png](tiny_lidar_net_example/outputs/tiny_lidar_net_curved_circuit_epoch100.png) を貼る
- train / val loss の推移、Huber 損失の収束の様子について触れる
- 単一ワールド・単一方向の学習データでは、val loss は 20 epoch を超えても下がり続ける
- 損失そのものが収束していても、走行可能な範囲の汎化性能とは別物であることを 3.2 / 3.3 の結果と合わせて確認する

### 3.5 考察

- 約 220K パラメータと 25000 サンプル程度の手動運転データでも、学習データと一致する条件下では自作シミュレータ上で走行が成立した
- 単一ワールド・単一方向の学習データでは、学習対象のワールドであっても初期位置を変えるだけで走行が破綻し、Behavior Cloning が学習データの分布に対して敏感であることが確認できた
- 学習データに逆方向の走行や別ワールドのデータを加えると、未学習ワールドでも走行が継続する試行が現れ、データ多様性が汎化に寄与することが確認できた
- 一方で、`maze` のように通路幅・形状が大きく異なるワールドでは依然として衝突が支配的であり、本実装の範囲では 2 ワールド程度の学習データではサンプル外への一般化に限界がある
- 本実装の範囲: 静的環境のみを対象とし、動的障害物への対応や Sim2Real は検証していない
- 論文中で報告された「壁とみなして動的障害物をかわす」現象については、本実装での再現や一般化の検証は行っていない


---

## 付録（書くか未定）

- A. 環境構築 (`uv pip install -e .`) と CLI の使い方
- B. 公式 (TF/Keras) と自作 (PyTorch) の対応表（Conv パラメータ・損失・最適化）
- C. 参考文献
