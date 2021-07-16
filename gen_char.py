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
from utils import image_files_from_folder
import glob
from matplotlib import pyplot as plt



def segmantation(Img12,filename):
    # kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
    # cv2.imshow('not closing',binary)
    # cv2.waitKey(0)
    img = Img12
    gray = cv2.cvtColor( Img12, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(19,19),0)
    # cv2.imshow("Anh bien so sau chuyen xam", gray)
    # Ap dung threshold de phan tach so va nen
    binary = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    cv2.THRESH_BINARY,11,2)
    binary = cv2.bitwise_not(binary)
            
    kernel = np.ones((5,5),np.uint8)
    # dilation = cv2.dilate(binary,kernel,iterations = 1)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    height, width = binary.shape
    binary_inverse = cv2.bitwise_not(closing)
    # cv2.drawContours(img, contours, -1, (0,255,0), 3)
    # print('width :'+str(width) +'height : '+ str(height))
    # cv2.imshow('closing',binary)
    # cv2.waitKey(0)
    # loop over our contours
    plate_num = ''
    number = 0
    pts = ''
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        crop = img[y:y+h,x:x+w]
        # cv2.imshow('binary',crop)
        # cv2.waitKey(0)
        ratio = h / float(w)
        area = h * w
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # approximate the contour
        if (width / float(w) < 30) and (float(h) / height > 0.8) and (ratio > 1.0) and(area > 20) :
            # print('x1 : {},y1 :{},x2 :{},y2 :{}'.format(x,y,x+w,y+h))
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            pts +='character 0.9 {} {} {} {}'.format(x,y,x+w,y+h)
            pts +='\n'
    with open('/home/long/Study/AI/Evaluation/mAP/input/detection-results/{}.txt'.format(filename), 'w') as f:
        f.write('{}'.format(pts))
        f.close()  
	

# Đường dẫn ảnh, các bạn đổi tên file tại đây để thử nhé
input_dir = sys.argv[1]
img_files = image_files_from_folder(input_dir)

for img_path in img_files:
    # create figure
    
    # setting values to rows and column variables
    rows = 2
    columns = 2
    # Đọc file ảnh đầu vào
    Ivehicle = cv2.imread(img_path)
    # Adds a subplot at the 1st position
    filename = img_path.split('/')[-1]
    filename = filename.split('.')[0]
    segmantation(Ivehicle,filename)

