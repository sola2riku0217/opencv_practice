import cv2
import numpy as np

## 追跡対象の色の範囲（Hueの範囲）
def is_target(roi):
    return ((roi <= 30) | (roi >= 150))

## ラベリングして最も面積の大きい物体を持ってくる
def max_moment_point(mask):
    label = cv2.connectedComponentsWithStats(mask)
    data = np.delete(label[2],0,0)
    center = np.delete(label[3],0,0)
    moment = data[:,4]
    max_index = np.argmax(moment)
    
    return center[max_index]

def initialize(image,N):
    ## HSVに変換してS,V成分を２値化する
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]
    ret, s = cv2.threshold(hsv[:,:,1],0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ret, v = cv2.threshold(hsv[:,:,2],0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    h[(s==0)|(v==0)] = 100
    
    mask = h.copy()
    mask[is_target(mask) == False] = 0
    x,y = max_moment_point(mask)
    w = calc_likelihood(x,y,image)
    ps = np.ndarray((N,3),dtype=np.float32)
    ps[:] = [x,y,w]
    return ps
            
def resampling(ps):
    ws = ps[:,2].cumsum()
    last_w = ws[ws.shape[0] - 1]
    new_ps = np.empty(ps.shape)
    for i in range(ps.shape[0]):
        w = np.random.rand() * last_w
        new_ps[i] = ps[(ws > w).argmax()]
        new_ps[i,2] = 1.0
    
    return new_ps

def predict_position(ps, var = 13.0):
    ps[:,0] += np.random.randn((ps.shape[0])) * var
    ps[:,1] += np.random.randn((ps.shape[0])) * var

## 尤度を算出する
def calc_likelihood(x,y,img,w=30,h=30):
    x1,y1 = max(0,x-w/2), max(0,y-h/2)
    x2,y2 = min(img.shape[1], x+w/2), min(img.shape[0],y+h/2)
    x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
    roi = img[y1:y2,x1:x2]
    
    ## 矩形領域中に含まれる追跡対象の色の存在率を尤度として計算する
    count = roi[is_target(roi)].size
    if count > 0:
        result = float(count) / img.size
    else:
        result = 0.0001
    return result
    
    
def calc_weight(ps,img):
    for i in range(ps.shape[0]):
        ps[i][2] = calc_likelihood(ps[i,0],ps[i,1],img)
        
    ps[:,2] *= ps.shape[0] / ps[:,2].sum()
    
def observer(ps,img):
    
    calc_weight(ps,img)
    
    x = (ps[:,0] * ps[:,2]).sum()
    y = (ps[:,1] * ps[:,2]).sum()
    
    result = (x,y) / ps[:,2].sum()
    return result
    
# pos = None
def particle_filter(image,pos):
    
    ## HSVに変換してS,V成分を２値化する
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]
    ret, s = cv2.threshold(hsv[:,:,1],0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ret, v = cv2.threshold(hsv[:,:,2],0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    h[(s==0)|(v==0)] = 100
    
    # パーティクルフィルタ実装
    pos = resampling(pos) ## リサンプリング
    predict_position(pos) ## 推定
    x,y = observer(pos,h) ## 観測
    
    ## 画面内に入っているパーティクルのみ持ってくる
    pos1 = pos[(pos[:,0] >= 0) & (pos[:,0] < image.shape[1]) & (pos[:,1] >= 0) & (pos[:,1] < image.shape[0])]
    
    ## パーティクルを塗りつぶす
    for i in range(pos1.shape[0]):
        image[int(pos1[i,1]), int(pos1[i,0])] = [0,0,200]
    
    ## パーティクルの中身を赤で囲む
    cv2.rectangle(image, (int(x-20),int(y-20)),(int(x+20),int(y+20)),(0,0,200),5)
    
    return image, pos



