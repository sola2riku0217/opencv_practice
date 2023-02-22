import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import dlib

CASCADE_PATH = "./haarcascades/"
CASCADE = cv2.CascadeClassifier(CASCADE_PATH + "haarcascade_frontalface_default.xml")

# LEARNED_MODEL_PATH = "./learned-models/"
# PREDICTOR = dlib.shape_predictor(LEARNED_MODEL_PATH + "dlib_face_recognition_resnet_model_v1.dat")

##　顔検出の関数
def face_position(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = CASCADE.detectMultiScale(gray,minSize=(100,100))
    
    for x,y,w,h in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        
    return image

## 動的背景差分の関数
fgbg = cv2.createBackgroundSubtractorMOG2(history=60,detectShadows=False)
def back_subtract(image,th):
    kernel = np.ones((th,th),np.uint8)
    fgmask = fgbg.apply(image)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN,kernel)
    return fgmask

## フレーム間差分の関数
def frame_subtract(image,image_pre):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray_pre = cv2.cvtColor(image_pre,cv2.COLOR_BGR2GRAY)
    dst = cv2.absdiff(gray,gray_pre)
   
    return dst

## トラックバーのコールバック関数
def trackbar(val):
    print("trackber1 : ",val)
    
def main():
    ###############################
    # コマンドライン引数を受け取ることができる
    # なし　：　デフォルトカメラ画像
    # 数字　：　カメラ番号
    # パス　：　動画ファイル
    ###############################
    flg = 0 # モード切り替えのためのフラグ
    args = sys.argv
    if len(args) == 1:
        cap = cv2.VideoCapture(0)
    elif args[1].isdigit():
        cap = cv2.VideoCapture(int(args[1]))
    elif os.path.exists(args[1]):
        try:
            cap = cv2.VideoCapture(args[1])
        except:
            print("ファイルの場所を確認してください")
    else:
        print("コマンドライン引数を確認してください")
    
    cv2.namedWindow("window1")
    cv2.namedWindow("window2")
    ##############################
    # GUIの設定
    #　　　　　・　トラックバー
    #
    ##############################
    cv2.createTrackbar("trackbar","window2",1, 10, trackbar)
    
    
    ###############################
    # 1キー　：　顔検出
    # 2キー　：　動的背景差分
    # 3キー　：　フレーム間差分
    # 4キー　：　オプティカルフロー
    ###############################   
    while(cap.isOpened()):
        ret, frame = cap.read()
        th = cv2.getTrackbarPos("trackbar","window2")
        
        
        if flg == 0:
            frame_out = frame.copy()
        if flg == 1:
            frame_out = face_position(frame)
        if flg == 2:
            frame_out = back_subtract(frame, th)
        if flg == 3:
            frame_out = frame_subtract(frame, frame_pre) 
            frame_pre = frame.copy()
               
        cv2.imshow("window1", frame)
        cv2.imshow("window2", frame_out)
        
   
        k = cv2.waitKey(1)
        if k == ord("q"):
            break
        if k == ord("1"):
            flg = 1
        if k == ord("2"):
            flg = 2
        if k == ord("3"):
            frame_pre = frame.copy()
            flg = 3


if __name__ == "__main__":
    
    main()

