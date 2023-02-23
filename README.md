# OpenCVで画像処理の基本を実装したプログラム

## 実行方法
1. gitからダウンロードする
```shell
git clone https://github.com/sola2riku0217/opencv_practice
```
2. haarcascadeファイルをOpenCV公式サイトからダウンロードして所望の位置におく<br>
[OpenCV公式](https://opencv.org/releases/)のSourcesからダウンロード。<br>
dataフォルダの中のhaarcascadesフォルダをopencv_practiceフォルダ直下に置く。この時にhaarcascade_frontalface_default.xmlファイルがあることを確認しておく。

3.　main.pyを実行する
```pythoｎ
python main.py
```
* 引数の説明<br>

|  コマンド          　|  結果                 |
| ----               | ----                  |
|  main.py  　　　　　 |  デフォルトカメラで実行　 |
|  main.py カメラ番号  |  カメラ番号のカメラで実行 |
|  main.py パス       |  動画ファイルで実行      |

-----



## haarcascadeファイルについて

[こちらのURL](https://github.com/opencv/opencv/tree/master/data/haarcascades)から拝借したらうまく動かなかったので[OpenCV公式](https://opencv.org/releases/)のSourcesからダウンロードしました。<br>
【参考】
https://qiita.com/s-kajioka/items/b9207812fc968161f78b
