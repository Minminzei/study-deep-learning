#  Study Classification
クラス分類の勉強会用のリポジトリです。

## ゴール
- validationでの正解率が95%を超えること
- test画像を100%の正解率で分類できること
- 画像はそれぞれ`cat`:1668枚、`dog`:4863枚、`horse`:2623枚です。

## 環境構築
[初期構築はこちらを参考](https://github.com/Minminzei/study-deep-learning/blob/main/README.md)

```bash
# venv起動
source venv/bin/activate
```

## 1. 機械学習
`y = w1x1 + w2x2 + ... b`となるy,xの関係を探し出す。

```bash
# y = 2X + 1となる関係
X = [0, 1, 3, 5, 6]
Y = [1, 3, 7, 11, 13]
```
上のようなデータを教師ありデータセットといい、このような関係になるweightとbiasを学習する。出力値と正解との誤差がなくなるようにweight, biasを試行していき、ネットワークが出力した値と正解データYとの乖離度をなくすように学習が進んでいく(誤差逆伝播法：backpropagation)。

#### 学習と誤差逆伝播法
![学習の流れ](https://developers.google.com/static/machine-learning/crash-course/images/GradientDescentDiagram.svg)

```bash
[1回目]w = 1, b = 10で試すと
x = 1のとき
Y = 1 * 1 + 10 = 11, 正解は3なので+8の誤差がある

[2回目]w = 1.2, b = 3で試すと出力は4.2、正解は3なので+1.2の誤差がある...

# [regression.py]を実行してみる
python regression.py
```

## 2. ニューラルネットワーク
### 2-1. 基本構造
- 入力層-隠れ層-出力層という構造になる。隠れ層が複数あるとディープニューラルネットワークと呼ばれる
- 各ニューロンが回帰式を持ち、活性化関数を経て出力される.`y = active(w1x1 + w2x2 + .... + b)`
- 活性化関数によって非線形問題を解けるようになる。[参考](https://qiita.com/masatomix/items/42b322a8db61e5b4d65f)
- ニューラルネットワークの学習をディープラーニングといい、機械学習の一つに分類される

#### ニューラルネットワーク
![ニューラルネットワーク](https://developers.google.com/static/machine-learning/crash-course/images/activation.svg)

#### 非線形分類：XOR問題
![非線形分類：XOR問題](https://developers.google.com/machine-learning/crash-course/images/FeatureCrosses1.png)


## 2-2. CNN(convolutional nueral network)
- 畳み込み層とプーリング層を繰り返して、画像の特徴を抽出していく(表現学習)
- 抽出された特徴を全結合層に渡して、分類問題などを解く

![CNN](https://developers.google.com/static/machine-learning/practica/image-classification/images/cnn_architecture.svg)

```bash
[original.pyのCNNアーキテクチャを確認する]
python main.py summary -m original
```

## 3. ディープラーニング

#### 3-1. 用語
学習ではデータセットをいくつかのグループ(バッチ)に分けてweightを更新するが、1グループに含まれるデータセットの数をバッチサイズ、生成されたバッチがwを1度更新することをイテレーション、すべてのバッチが1度更新することをエポックと呼ぶ。
Ex. 5000枚のデータセットを、バッチサイズ：100枚でバッチ化すると50個のグループが作成され、wの学習は50イテレーション実行され、50イテレーション完了すると学習回数:1エポックとなる。

#### 3-2. 分類モデルにおける評価
教師あり学習はYが連続値をとる回帰(Ex. 立地から家賃を予測する)とそれ以外の分類の2つにカテゴライズされる。
回帰における損失は予想値Yと正解値の四則演算をベースに算出できるが、分類における損失は次の値から算出する。

例). 猫か犬かを判定する2値分類タスク
|    |   |
|--------|------|
| 真陽性（TP）<br>猫を猫と分類できた  | 偽陽性（FP）<br>猫を犬と分類した  |
| 偽陰性（FN）<br>犬を猫を分類した  | 真陰性（TN）<br>犬を犬と分類できた  |

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
機会学習ではパラメータのweightを試行していくが、ネットワークや学習方法自体の設定値によって学習の質が変わってくる。この設定値をハイパーパラメーターとよび、ハイパーパラメーターチューニングではパフォーマンスが良さそうなハイパーパラメーターの組み合わせを探索する。代表的なハイパーパラメーターは次のもの。
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
ニューラルネットワークの目的は汎化(generalization)能力を高めること。＝訓練データに予想ではなく、学習に使用していない未知のデータから正確に予想すること。

#### 過学習
![過学習](https://github.com/Minminzei/faceswap-sample/assets/3320542/785a6ab7-a81f-4667-b808-2848ffbc3ce5)

過学習への対策の第一歩はデータセットを訓練データ、検証データ、テストデータに分割すること。8:1:1に分けることが多い。

|      |   |
|--------|------|
| 訓練データ | 学習(train)に使うデータセット |
| 検証データ | 検証(validation)に使うデータセット |
| テストデータ | 推論(predict)に使うデータセット。汎化能力を測る |

- 他にもモデルアーキテクチャにドロップアウト層や正則化を加えることで過学習が起きにくいモデルを作れる。
- 訓練時にcallbacksで過学習が置き始めたら学習を終了させるなどの早期終了を追加できる

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

model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
    loss=keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)

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


モデルアーキテクチャ：モデルの構造。

#### 5-2. 転移学習、ファインチューニング
学習済みweight
転移学習
ファインチューニング

#### 5-3. 代表的なモデルアーキテクチャ

- AlexNet
- ResNet
- EfficientNet

## 6. データ中心のアプローチ
#### 6-1. 特徴量エンジニアリングと画像拡張
