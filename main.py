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


def face_position(gray_image):
    faces = CASCADE.detectMultiScale(gray_image,minSize=(100,100))
    return faces

def facemark(gray_image):
    face_roi = face_position(gray_image)
    landmark = []
    
    for face in faces_roi:
        detector = dlib.get_frontal_face_detector()
        rects = detector(gray_image,1)
        landmarks = []
        
        for rect in rects:
            landmarks.append(numpy.array([[p.x,p.y] for p in PREDICTOR(gray_image.rect).parts()]))
     
    return landmarks
    
    
def main():
    ###############################
    # コマンドライン引数を受け取ることができる
    # なし　：　デフォルトカメラ画像
    # 数字　：　カメラ番号
    # パス　：　動画ファイル
    ###############################
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
    
    ###############################
    # １キー　：　動的背景差分
    # ２キー　：　フレーム間差分
    # ３キー　：　オプティカルフロー
    ###############################   
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_position(gray)
        
        for x,y,w,h in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
               
        cv2.imshow("Window1", frame)
        
        k = cv2.waitKey(1)
        if k == ord("q"):
            break


if __name__ == "__main__":
    
    main()

