#!/bin/sh

# データセットを取得
echo "Download dataset animals10"
# kaggle datasets download alessiocorrado99/animals10 -p resources
# unzip resources/animals10.zip -d resources/animals10

# # 使うanimal画像を移動
# mv resources/animals10/raw-img/cane/ resources/animals10/dog
# mv resources/animals10/raw-img/cavallo/ resources/animals10/horse
# mv resources/animals10/raw-img/gatto/ resources/animals10/cat

rm -rf resources/animals10/raw-img resources/animals10/translate.py
