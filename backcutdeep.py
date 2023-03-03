import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import transforms


def backcut(image):
    img = cv2.imread(image)
    img = img[:,:,::-1]
    h, w, _ = img.shape
    img = cv2.resize(img,(320,320))
    
    device = torch.device("cuda:0",if torch.cuda.is_available() else "cpu")
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    model = model.to(device)
    model.eval()
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0).to(device)