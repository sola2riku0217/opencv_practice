import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import dlib
import particlefilter as pf

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

## オプティカルフロー
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def opticalFlow(image,image_pre):
    image2 = image.copy()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray_pre = cv2.cvtColor(image_pre,cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(gray_pre,mask=None,**feature_params)
    mask = np.zeros_like(image2)
    
    p1,st,err = cv2.calcOpticalFlowPyrLK(gray_pre,gray,p0,None,**lk_params)
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        e = a - (c - a) * 2
        f = b - (d - b) * 2
       
        mask = cv2.arrowedLine(mask,(int(a),int(b)),(int(e),int(f)),(255,255,255),2)
        image2 = cv2.circle(image2,(int(a),int(b)),5,(255,0,0),-1)
        img = cv2.add(image2,mask)
    return img


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
    cv2.moveWindow("window1",0,0)
    cv2.namedWindow("window2")
    cv2.moveWindow("window2",0,300)
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
    # 5キー　：　パーティクルフィルター
    # 6キー　：　エッジ検出
    # 7キー　：　残像
    # 8キー　：　幽体離脱
    # 9キー　：　前景抽出
    ###############################   
    frame_pre = None
    frame_out = None
    filter = 0.1
    while(cap.isOpened()):
        
        ret, frame = cap.read()
        th = cv2.getTrackbarPos("trackbar","window2")
        
        if flg == 0:
            frame_out = frame.copy()
        if flg == 1:
            frame_in = frame.copy()
            frame_out = face_position(frame_in)
        if flg == 2:
            frame_out = back_subtract(frame, th)
        if flg == 3:
            frame_out = frame_subtract(frame, frame_pre)       
        if flg == 4:
            frame_out = opticalFlow(frame, frame_pre)  
        if flg == 5 :
            ## 関数実行
            frame_out,pos = pf.particle_filter(frame,pos)
        if flg == 6 :
            frame_out = cv2.Canny(frame,threshold1=100,threshold2=200)
        if flg == 7 :
            th = cv2.getTrackbarPos("trackbar","window2") * 0.1
            bufimg = cv2.addWeighted(src1=bufimg,alpha=1-th,src2=frame_pre,beta=th,gamma=0)
            frame_out = bufimg.copy()
        if flg == 8 :
            frame_out = cv2.addWeighted(bufimg, 0.5, frame,0.5,0) 
            
        cv2.imshow("window1", frame)
        cv2.imshow("window2", frame_out)
        
   
        frame_pre = frame.copy()
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
        if k == ord("4"):
            frame_pre = frame.copy()
            flg = 4
        if k == ord("5"):
            pos = pf.initialize(frame,N=300)
            flg = 5
        if k == ord("6"):
            flg = 6
        if k == ord("7"):
            bufimg = frame.copy()
            flg = 7
        if k == ord("8"):
            bufimg = frame.copy()
            flg = 8


if __name__ == "__main__":
    
    main()

