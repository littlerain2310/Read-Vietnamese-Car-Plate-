from typing import Text
import pytesseract
import cv2
from lib_detection import load_model, detect_lp, im2single
from skimage import measure
from imutils import perspective
import imutils
from skimage.filters import threshold_local
import numpy as np
import re
import sys
import glob
from matplotlib import pyplot as plt
from utils import segmantation,calculating_IOU,sort_contours,image_files_from_folder
import os



input_dir = 'OCR_VN/gt'
txts = image_files_from_folder(input_dir)
# Load model LP detection
wpod_net_path = "wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)

for txt_file in txts:
    f = open(txt_file, "r")
    label = str(f.read())
    files = [c for c in label]
    filename = txt_file.split('/')[-1]
    filename = filename.split('.')[0]
    img_path = 'plateVN/{}.jpg'.format(filename)
    Ivehicle = cv2.imread(img_path)
    # Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
    Dmax = 608
    Dmin = 288

    # Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
    ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)

    _ , LpImg, lp_type,_,score = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)

    x= 0
    if (len(LpImg)):

        n=0
        for Img in LpImg:
            # Chuyen doi anh bien so
            Img = cv2.convertScaleAbs(Img, alpha=(255.0))
            
            height = Img.shape[0]
            width = Img.shape[1]
            Img = Img[5:height-5,10:width -10]
            
            # cv2.imshow('crop',Img)
            # cv2.waitKey(0)
            # Img = LpImg[0]
            if (lp_type == 2):
                # cv2.imshow('anh chua cat',Img)
                # print("Bien so VUONG")
                height = Img.shape[0]
                width = Img.shape[1]
                height_cutoff = height // 2
                Img1= Img[:height_cutoff,:]
                Img2= Img[height_cutoff:,:]
                Img12 = np.hstack((Img1, Img2))             
            else:
                # print("Bien so DAI")
                Img12 = Img
            # cv2.imwrite(path,Img)
            origin = Img12
            
            _,_,_,for_CNN,chars = segmantation(Img12)
            for_CNN = np.array([c for c in for_CNN], dtype="float32")
            if len(for_CNN) <= 2:
                continue
            detach_image_file = zip(chars,files)
            i=0
            for char,file in detach_image_file:
                i+=1
                path = 'data_for_CNN/{}'.format(file)
                cv2.imwrite(os.path.join(path , '{}_{}.jpg'.format(filename,i)),char)