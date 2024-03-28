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

# パッケージインストール
pip install --no-cache-dir -r requirements.txt

# datasetsをダウンロード
sh bin/download.sh
```
