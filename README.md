#  Study Classification
クラス分類の勉強会用のリポジトリです。

## ゴール
- validationでの正解率が95%を超えること
- test画像を100%の正解率で分類できること

## このリポジトリで対象とすること
1. モデルのフロムスクラッチでの学習
1. 既存モデルの選択
1. 既存モデルの転移学習
1. 既存モデルのファインチューニング
1. 特徴量エンジニアリング/画像拡張

## 画像について
学習用の画像は`resources/animals`にフォルダごとに配置しています。
フォルダ名はそれぞれクラス名になります。画像数はそれぞれ`cat`:1668枚、`dog`:4863枚、`horse`:2623枚で不均一になっています。

## 事前準備
```bash
# 1. kaggleセットアップ
# Settings > API > Create New Tokenででkaggle.jsonを発行
https://www.kaggle.com/settings
# 権限設定
mv kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# 2. opensslセットアップ
brew install openssl@1.1
```

## 初回構築

```bash
# venv初期化
python3 -m venv venv

# venv起動
source venv/bin/activate

# pip更新
python -m pip install -U pip

# パッケージインストール
pip install --no-cache-dir -r requirements.txt

# datasetsをダウンロード
sh bin/download.sh
```

## 起動
```bash
# venv起動
source venv/bin/activate
```

## フロムスクラッチでの学習
```bash
python main.py train
```

## 転移学習
```bash
python main.py transfer_learning
```

## モデルの推論
```bash
python main.py predict
```
