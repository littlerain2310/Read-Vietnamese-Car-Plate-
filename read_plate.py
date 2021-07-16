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
from utils import image_files_from_folder,segmantation
import glob
from matplotlib import pyplot as plt

# Dinh nghia cac ky tu tren bien so
char_list =  '0123456789ABCDEFGHKLMNPRSTUVXYZ'

# Ham fine tune bien so, loai bo cac ki tu khong hop ly
def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString
def sort_contours(cnts):

    reverse = False
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts

    

# Đường dẫn ảnh, các bạn đổi tên file tại đây để thử nhé
input_dir = sys.argv[1]
img_files = image_files_from_folder(input_dir)

# Load model LP detection
wpod_net_path = "wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)


# model = E2E()
print(len(img_files))
for img_path in img_files:
    # create figure
    fig = plt.figure(figsize=(10, 7))
    
    # setting values to rows and column variables
    rows = 2
    columns = 2
    # Đọc file ảnh đầu vào
    Ivehicle = cv2.imread(img_path)
    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)
    filename = img_path.split('/')[-1]
    filename = filename.split('.')[0]
    # showing image
    plt.imshow(Ivehicle)
    plt.axis('off')
    plt.title("original")
    # cv2.imshow('input',Ivehicle)
    # Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
    Dmax = 608
    Dmin = 288

    # Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
    ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)

    _ , LpImg, lp_type,_,_ = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)

    x= 0
    if (len(LpImg)):
        for Img in LpImg:
            # Chuyen doi anh bien so
            Img = cv2.convertScaleAbs(Img, alpha=(255.0))
            
            height = Img.shape[0]
            width = Img.shape[1]
            Img = Img[15:height-15,10:width -10]
            fig.add_subplot(rows, columns, 2)
            # showing image
            plt.imshow(Img)
            plt.axis('off')
            plt.title("cropped")
            # cv2.imshow('crop',Img)
            # cv2.waitKey(0)
            # Img = LpImg[0]
            if (lp_type == 2):
                # cv2.imshow('anh chua cat',Img)
                print("Bien so VUONG")
                height = Img.shape[0]
                width = Img.shape[1]
                height_cutoff = height // 2
                Img1= Img[:height_cutoff,:]
                Img2= Img[height_cutoff:,:]
                Img12 = np.hstack((Img1, Img2))             
            else:
                print("Bien so DAI")
                Img12 = Img
        
            img_draw_char,plate = segmantation(Img12,filename)
            # plate =sorted_Roi(contours,binary)
            # cv2.drawContours(binary, contours, -1, (0,0,0), 3)
            
            fig.add_subplot(rows, columns, 3)
    
            # # showing image
            # plt.imshow(binary)
            # plt.axis('off')
            # plt.title("binary")
            # Nhan dien bien so. Cau hinh --psm 7 la de nhan dien 1 line only
            
            # draw_label(Ivehicle,fine_tune(text))
            # Hien thi anh va luu anh ra file output.png
            # fig.add_subplot(rows, columns, 4)
            # print("Plate la :  {}".format(plate))
            # showing image
            plt.imshow(img_draw_char)
            plt.axis('off')
            plt.title("{}".format(plate))
            # cv2.imshow("Anh input", Ivehicle)
            # cv2.imwrite("output.png",Ivehicle)
            x+=30
            # cv2.waitKey(0)

    plt.show()
    cv2.destroyAllWindows()