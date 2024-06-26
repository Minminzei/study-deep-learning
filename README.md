#  Study Classification
クラス分類の勉強会用のリポジトリです。

## ゴール
- [猫 犬 馬]の3値分類を行う
- validationでの正解率が95%を超えること
- test画像を100%の正解率で分類できること
- 画像はそれぞれ`cat`:1668枚、`dog`:4863枚、`horse`:2623枚

## 環境構築
[初期構築はこちらを参考](https://github.com/Minminzei/study-deep-learning/blob/main/BUILD.md)

```bash
# venv起動
source venv/bin/activate
```

## 1. 機械学習
`y = w1x1 + w2x2 + ... b`となるy,xの関係を探し出す。

```bash
# y = 2X + 1
X = [0, 1, 3, 5, 6]
Y = [1, 3, 7, 11, 13]
```
上のようなデータを教師ありデータセットといい、このような関係になるweightとbiasを学習する。出力値と正解との誤差がなくなるようにweight, biasを試行していき、ネットワークが出力した値と正解データYとの乖離度をなくすように学習が進んでいく。

#### 学習の流れ
![学習の流れ](https://developers.google.com/static/machine-learning/crash-course/images/GradientDescentDiagram.svg)

```bash
[1回目]w = 1, b = 10で試すと
x = 1のとき
Y = 1 * 1 + 10 = 11, 正解は3なので+8の誤差がある

[2回目]w = 1.2, b = 3で試すと出力は4.2、正解は3なので+1.2の誤差がある...

# [regression.py]を実行してみる
python regression.py

# 50 epoch回すとy=2x+1の近いモデルが生成された
# >  y = 2.012694835662842x + 0.9413922429084778
```

## 2. ニューラルネットワーク
### 2-1. 基本構造
- 入力層-隠れ層-出力層という構造になる。隠れ層が複数あるとディープニューラルネットワークと呼ばれる
- 各ニューロンが回帰式を持ち、活性化関数を経て出力される.`y = active(w1x1 + w2x2 + .... + b)`
- 活性化関数によって非線形問題を解けるようになる。[参考](https://qiita.com/masatomix/items/42b322a8db61e5b4d65f)
- ニューラルネットワークの学習をディープラーニングといい、機械学習の一つに分類される

#### ニューラルネットワーク
![ニューラルネットワーク](https://developers.google.com/static/machine-learning/crash-course/images/activation.svg)

#### 線形分離可能
![非線形分類：XOR問題](https://developers.google.com/machine-learning/crash-course/images/LinearProblem1.png)

#### 非非線形分類：XOR問題
![非線形分類：XOR問題](https://developers.google.com/machine-learning/crash-course/images/LinearProblem2.png)


## 2-2. CNN(convolutional nueral network)
- 畳み込み層とプーリング層を繰り返して、画像の特徴を抽出していく(表現学習)
- 抽出された特徴を全結合層に渡して、分類問題などを解く(分類ヘッダ)

![CNN](https://developers.google.com/static/machine-learning/practica/image-classification/images/cnn_architecture.svg)

```bash
[original.pyのCNNアーキテクチャを確認する]
python main.py summary -m original
```

## 3. ディープラーニング

#### 3-1. 用語
学習では<b>データセット</b>をいくつかのグループ(<b>バッチ</b>)に分けてweightを更新するが、1グループに含まれるデータセットの数を<b>バッチサイズ</b>、生成されたバッチがwを1度更新することを<b>イテレーション</b>、すべてのバッチが1度更新することを<b>エポック</b>と呼ぶ。<br />
`Ex. 5000枚のデータセットを、バッチサイズ：100枚でバッチ化すると50個のグループが作成され、wの学習は50イテレーション実行され、50イテレーション完了すると学習回数:1エポックとなる。`

#### 3-2. 損失の計算
教師あり学習はYが連続値をとる<b>回帰</b>(Ex. 立地から家賃を予測する)とそれ以外の<b>分類</b>の2つにカテゴライズされる。
回帰における損失は予想値Yと正解値の四則演算をベースに算出できるが、分類における損失は次の値から算出する。

例). 猫か犬かを判定する2値分類タスク
|    |   |
|--------|------|
| <b>真陽性（TP）</b><br>猫を猫と分類できた  | <b>偽陽性（FP）</b><br>猫を犬と分類した  |
| <b>偽陰性（FN）</b><br>犬を猫を分類した  | <b>真陰性（TN）</b><br>犬を犬と分類できた  |

|   値   | 説明  |　式  |
|--------|------|------|
| 正解率(accuracy) | 全データ中、正しく予測できた割合 | TP + TN / TP + TN + FP + FN |
| 適合率(precision) | 予測が陽性の中で、実際に陽性だった割合 | TP / TP + FP |
| 再現率(recall) | 実際の陽性の中で、正しく陽性と予測できた割合 | TP / TP + FN |
| F値(F measure) | 適合率と再現率を調和平均したもの | 2 * precision * recall / precision + recall|

#### 3-3. オプティマイザー
誤差を最小にするwの探索はアルゴリズム的に行われるが、これを勾配降下法と呼ぶ。勾配降下法は損失の傾きが0になるようなwを探索するが、どの程度動かすかは学習率で決める。学習率が小さすぎると局所最適に陥ってしまうが、大きすぎると収束しなくなる。

#### 勾配降下法
![勾配降下法](https://developers.google.com/static/machine-learning/crash-course/images/GradientDescentGradientStep.svg)

- Adamは学習率を動的に変更することでこれらの問題に対処する
- オプティマイザーとしては確率的勾配降下法（SGD）などいろいろあるよ [こちら参考](https://qiita.com/omiita/items/1735c1d048fe5f611f80)

#### 3-4. ハイパーパラメーターチューニング
機会学習ではパラメータのweightを試行していくが、ネットワークや学習方法全体の設定値によって学習の質が変わってくる。この設定値をハイパーパラメーターとよび人間が手動で設定する。ハイパーパラメーターチューニングではパフォーマンスが良さそうなハイパーパラメーターの組み合わせを探索する。代表的なハイパーパラメーターは次のもの。
- バッチサイズ
- 学習率(learning rate)
- 損失関数
- オプティマイザー

#### 3-5. 学習
```bash
[original.pyを学習する]
python main.py train -m original

[学習結果で推論する]
python main.py predict -m original
```

## 4. 過学習への対応
ニューラルネットワークの目的は汎化能力(generalization)を高めること。＝訓練データの予測ではなく、学習に使用していないネットワークにとって未知のデータから正確に予測すること。訓練誤差が小さいのにもかかわらずテスト誤差が大きくなるとき、ネットワークは過学習(over fitting)を起こしている。

#### 過学習
![過学習](https://github.com/Minminzei/faceswap-sample/assets/3320542/785a6ab7-a81f-4667-b808-2848ffbc3ce5)

過学習への対策の第一歩はデータセットを訓練データ、検証データ、テストデータに分割すること。8:1:1に分けることが多い。
他にも過学習が置き始めたら学習をストップする早期終了を行うことで過学習に対応する。

|  名称  | 説明  |
|--------|------|
| 訓練データ | 学習(train)に使うデータセット |
| 検証データ | 検証(validation)に使うデータセット |
| テストデータ | 推論(predict)に使うデータセット。汎化能力を測る |
| 早期終了 | 訓練時のコールバック関数で一定の条件で学習を終了させる |

```bash
model.fit(
    # 訓練データ
    train_data,
    # 検証データ 3つのバッチグループに対して検証を行う
    validation_data=validation_data,
    validation_steps=3,
    epochs=100,
    # 早期終了
    callbacks=[keras.callbacks.EarlyStopping(patience=3)],
)
```

モデルアーキテクチャにドロップアウト層や正則化を加えることで過学習が起きにくいモデルを作ることができる。公開されているモデルの多くにはこれらが組み込まれている。

#### ドロップアウト
![ドロップアウト](https://github.com/takumiohym/practical-ml-vision-book-ja/blob/main/images/ch02/fig02-22.png?raw=true)

```bash
model = keras.models.Sequential([
    keras.layers.Input(shape=([1])),
    keras.layers.Dense(
        units=100,
        activation=keras.activations.relu,
        # L2正則化
        kernel_regularizer=keras.regularizers.l2(0.001),
    ),
    # ドロップアウト
    keras.layers.Dropout(0.2),
    ...
])
```

## 5. モデル中心のアプローチ
#### 5-1. 学習のタイプ
モデルを一から作るのは非効率なのですでにあるモデルを使う。どのように使うかによって3つの学習種別がある。

|  学習タイプ    |  アーキテクチャ |　訓練済みWeight | 学習対象のWeight| 1ラベルあたりの必要枚数 |
|--------|------|------|------|-----|
| 0からの学習 | 使う | 使わない |　すべて | 5000枚以上 |
| ファインチューニング | 使う | 使う |　すべて | 1000-5000枚|
| 転移学習 | 使う | 使う |一部 | 1000枚以下|

必要なラベル付きデータセットの数は`0からの学習 > ファインチューニング > 転移学習`の順になる。
＊必要枚数はあくまで参考。

#### 5-2. 代表的なモデルアーキテクチャ

|  モデル名    |  特徴 | パラメーター数 | 論文URL |
|--------|------|------| -----|
| AlexNet | CVにCNNを適用した走り。relu関数を活性化関数に使うことで勾配消失をに強い深層NNを構築した | 370万 | [url](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)|
| ResNet(50) | AlexNetをより深くしたアーキテクチャ。層を増やすと勾配消失問題が発生するが、スキップ接続というアイディアで50-100層からのCNNを構築した | 2300万 | [url](https://arxiv.org/abs/1512.03385) |
| EfficientNet(B4) | Compound Coefficientというアイディアで少ないパラメーターで高いパフォーマンスを出したすごいアーキテクチャ。転移学習がやりやすいと評判。 | 2000万 | [url](https://arxiv.org/abs/1905.11946) |

```bash
# ResNet50を使用する 
keras.applications.ResNet50(
    # 使いたい訓練済みWeight。0から学習したい場合はNoneを指定
    weights="imagenet",
    input_shape=[SIZE, SIZE, 3],
)
```

#### 5-3. モデルの比較
レポジトリでは次の3つのモデルアーキテクチャを使って性能を比較した。
訓練済みのWeightは使わず、100 epochずつ学習し、[猫 犬 馬]の3値クラス分類を解かせた。

1. Original: 独自組んだCNNアーキテクチャ
2. ResNet
3. EfficientNet

|  モデル名  |  検証データの正解率 | テストデータの正解率 |
|--------|------|------| 
| Original |83% | 44%|
| ResNet |82% | 44%|
| EfficientNet | 83% | 77%|

```bash
# 学習
python main.py train -m {original, resnet, efficientnet} -t {fromzero, transfer_learning, fine_tuning}
# 推論
python main.py predict -m {original, resnet, efficientnet} -t {fromzero, transfer_learning, fine_tuning}

# オプション
python main.py --help

usage: main.py [-h] [-m MODEL] [-t TYPE] function_name
positional arguments:
  function_name         実行するメソッド名{summary, train, predict}
optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        学習モデル{original, resnet, efficientnet}
  -t TYPE, --type TYPE  学習タイプ{fromzero, transfer_learning, fine_tuning}
```

#### 5-4. 転移学習、ファインチューニング
ResNetとEfficientNetを対象に転移学習、ファインチューニングを行い、zero学習とパフォーマンスを比較した

|  モデル  |  学習タイプ  |  検証データの正解率 | テストデータの正解率 | エポック |
|--------|------|------| ------| --| 
| Original | 0からの学習 | 83% | 44%| 100 |
| ResNet | 0からの学習 | 82% | 44%| 100 |
| ResNet | ファインチューニング |70-90% | 88%| 50 |
| ResNet | 転移学習 |97% | 77%| 30 |
| EfficientNet | 0からの学習 | 83% | 77%| 100 |
| EfficientNet | ファインチューニング |95% | 77%| 50 |
| EfficientNet | 転移学習 | 99% | 100%| 30 |

##### 雑感
- 検証データの正解率は表記の+-10%の幅で安定しなかった。
- ファインチューニング、転移学習ともに10 epoch程度で収束した
- 転移学習が最も早く学習できた
- EfficientNetがパフォーマンス、学習スピードともに良さげ。また評判通り転移学習がやりやすいと感じた。
- デバッグが難しい。データセットの生成ロジックを変える前は何時間学習しても損失が減らなかったが、変更後は10-100エポックで高いパフォーマンスを出せるようになった
- callback関数の設定は必須だと思った。ほしい機能は次のもの
  - 学習精度がnエポックにわたって収斂した場合、学習を終了する
  - 検証データの正解率が悪化し続けるなど、過学習が確認できたら早期終了する
  - nエポックごとにモデルを保存する。＊セッションが切れて学習が無駄になることや、学習を終えたいときにインターバルごとに保存されていると助かる

## 6. データ中心のアプローチ
#### 6-1. 特徴量エンジニアリングと画像拡張
汎化能力を高めるもう一つのアプローとして、質の高いデータセットをたくさん集めモデルが学習しやすい形に変換して食べさせるデータ中心のアプローチがあるよ。大切な領域なので今後掘り下げていくよ