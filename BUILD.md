# 初期構築

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

## venv環境を構築

```bash
# venv初期化
python3 -m venv venv

# venv起動
source venv/bin/activate

# pip更新
python -m pip install -U pip

# パッケージインストール(MAC)
pip install --no-cache-dir -r requirements/macos.txt

# datasetsをダウンロード
sh bin/download.sh
```

## GPUサーバーでの初期構築
```bash
git clone https://github.com/Minminzei/study-deep-learning.git

export KAGGLE_USERNAME=kaggleのユーザー名
export KAGGLE_KEY=kaggleのAPIキー

cd study-deep-learning

sh bin/gpu_setup.sh
sh bin/download.sh
```

## Google Colabでの初期構築
```bash

!git clone https://github.com/Minminzei/study-deep-learning.git
%cd /content/study-deep-learning

%env KAGGLE_USERNAME=kaggleのユーザー名
%env KAGGLE_KEY=kaggleのAPIキー

!sh bin/colab_setup.sh
!sh bin/download.sh
```