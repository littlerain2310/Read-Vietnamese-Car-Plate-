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

def segmantation_(Img12,filename):
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
    # contours,_ = sort_contours(contours)
    height, width = binary.shape
    center_height =height / 2
    binary_inverse = cv2.bitwise_not(closing)

    # loop over our contours
    char = []
    plate_num = ''
    number = 0
    pts = ''
    candidate = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        crop = img[y:y+h,x:x+w]
        # cv2.imshow('binary',crop)
        # cv2.waitKey(0)
        ratio = w / float(h)
        solidity = cv2.contourArea(c) / float(w * h)
        area = h * w
        area_ratio = area / float(width * height)
        height_ratio = h / float(height)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # approximate the contour
        if (area_ratio >0.03)and (w / float(width) <0.25) and (float(h) / height > 0.3) and (w / float(width) >0.03) :
            candidate.append(c)
            # # print('x1 : {},y1 :{},x2 :{},y2 :{}'.format(x,y,x+w,y+h))
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # number +=1
            # crop = binary[y:y+h,x:x+w]
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # # print(crop.shape)
            # crop=prepare(crop)
            # # print(crop.shape)
            # char.append(crop)
    
    candidate_rm_inner = []
    #get rid of inner contours
    n = len(candidate)
    # print(n)
    for i in range(n):
        for j in range(i+1, n):
            # print(j)
            # print(len(candidate))
            ov = calculating_IOU(candidate[i],candidate[j])
            if ov > 0.3:
                area1 = cv2.contourArea(candidate[i])
                area2 = cv2.contourArea(candidate[j])
                if area1 > area2 :
                    candidate_rm_inner.append(candidate[j])
                    # print(ov)
                else:
                    # print(ov)
                    candidate_rm_inner.append(candidate[i])
    for c in candidate_rm_inner:
        try:
            candidate.remove(c)
        except:
            pass
    #sort by area
    areaArray = []
    truly_contour =[]
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        areaArray.append(area)
    #first sort the array by area
    sorteddata = sorted(zip(areaArray, candidate), key=lambda x: x[0], reverse=True)
    print(len(sorteddata))
    pts =''
    try:
        for n in range(1,9):
            Ndlargestcontour = sorteddata[n-1][1]
            truly_contour.append(Ndlargestcontour)
            #draw it
            x, y, w, h = cv2.boundingRect(Ndlargestcontour)
            # print('x1 : {},y1 :{},x2 :{},y2 :{}'.format(x,y,x+w,y+h))

            pts +='character 0.9 {} {} {} {}'.format(x,y,x+w,y+h)
            pts +='\n'
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    except:
        for n in range(len(sorteddata)):
            Ndlargestcontour = sorteddata[n-1][1]
            truly_contour.append(Ndlargestcontour)
            #draw it
            x, y, w, h = cv2.boundingRect(Ndlargestcontour)
            # print('x1 : {},y1 :{},x2 :{},y2 :{}'.format(x,y,x+w,y+h))

            pts +='character 0.9 {} {} {} {}'.format(x,y,x+w,y+h)
            pts +='\n'
    boxes = sort_contours(truly_contour,center_height)
	
    for b in boxes:
        x,y,w,h = b
        print('x1 : {},y1 :{},x2 :{},y2 :{}'.format(x,y,x+w,y+h))
    # with open('/home/long/Study/AI/Evaluation/mAP/input/detection-results/{}.txt'.format(filename), 'w') as f:
    #     f.write('{}'.format(pts))
    #     f.close()  


# # Đường dẫn ảnh, các bạn đổi tên file tại đây để thử nhé
# input_dir = sys.argv[1]
# img_files = image_files_from_folder(input_dir)
# print(len(img_files))
# for img_path in img_files:
#     # create figure
    
#     # setting values to rows and column variables
#     rows = 2
#     columns = 2
#     # Đọc file ảnh đầu vào
#     Ivehicle = cv2.imread(img_path)
#     # Adds a subplot at the 1st position
#     filename = img_path.split('/')[-1]
#     filename = filename.split('.')[0]
#     # segmantation_(Ivehicle,filename)
#     closing,img_draw_char,_,boxes = segmantation(Ivehicle,filename)
#     pts =''
#     # cv2.imshow('close',closing)
#     # cv2.imshow('draw',img_draw_char)
#     # cv2.waitKey(0)
#     for b in boxes:
#         x,y,w,h = b
#         pts +='character 0.9 {} {} {} {}'.format(x,y,x+w,y+h)
#         pts +='\n'
#     with open('/home/long/Study/AI/Evaluation/mAP/input/detection-results/{}.txt'.format(filename), 'w') as f:
#         f.write('{}'.format(pts))
#         f.close()
#     # cv2.imshow('img',img)
#     # cv2.waitKey(0)
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
            
            _,_,_,for_CNN,chars = segmantation(Img12,filename)
            for_CNN = np.array([c for c in for_CNN], dtype="float32")
            if len(for_CNN) <= 2:
                continue
            detach_image_file = zip(chars,files)
            i=0
            for char,file in detach_image_file:
                i+=1
                path = 'data_for_CNN/{}'.format(file)
                cv2.imwrite(os.path.join(path , '{}_{}.jpg'.format(filename,i)),char)