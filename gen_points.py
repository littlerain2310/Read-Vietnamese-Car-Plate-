from gen_ocr import segmantation
import pytesseract
import cv2
from lib_detection import load_model, detect_lp, im2single
import numpy as np
import re
import sys
from utils import image_files_from_folder,segmantation
import glob

# Dinh nghia cac ky tu tren bien so
char_list =  '0123456789ABCDEFGHKLMNPRSTUVXYZ'

# Ham fine tune bien so, loai bo cac ki tu khong hop ly
def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString


    

# Đường dẫn ảnh, các bạn đổi tên file tại đây để thử nhé
input_dir = sys.argv[1]
img_files = image_files_from_folder(input_dir)

# Load model LP detection
wpod_net_path = "wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)



for img_path in img_files:
    # Đọc file ảnh đầu vào
    Ivehicle = cv2.imread(img_path)
    # cv2.imshow('input',Ivehicle)



    # Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
    Dmax = 608
    Dmin = 288

    # Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
    ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)

    _, _, _,points,confidence = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)
    filename = img_path.split('/')[-1]
    filename = filename.split('.')[0]

    # segmantation(Ivehicle,filename)
    # print(filename )
    pts = ''
    for point in points:
        assert(point.shape[0] == 2 and point.shape[1] == 4)
        pts += 'license '
        pts += str(confidence) + ' '
        for i in range(4):
            pt1 = tuple(point[:,i].astype(int).tolist())
            pts += str(pt1[0])+ ' ' +str(pt1[1])+' '
        pts +='\n'
    with open('./detection/{}.txt'.format(filename), 'w') as f:
        f.write('{}'.format(pts))
        f.close()  
    
    