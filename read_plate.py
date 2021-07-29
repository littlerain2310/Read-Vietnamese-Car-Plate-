from typing import Text
from model import CNN_Model
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
import os
# Dinh nghia cac ky tu tren bien so
char_list =  '0123456789ABCDEFGHKLMNPRSTUVXYZ'


class_names =  ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']



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
# try:
#     os.mkdir(os.path.join(input_dir, "transform(0.9+)"))
# except:
#     None
recogChar = CNN_Model().model
recogChar.load_weights('good2.h5')

# model = E2E()
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

    path = input_dir + '/transform(0.9+)/' + filename + '.jpg'
    # showing image
    plt.imshow(Ivehicle)
    plt.axis('off')
    plt.title("original")
    # cv2.imshow('input',Ivehicle)
    # cv2.waitKey(0)
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
            Img = Img[5:height-5,10:width -10]
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
            closing,img_draw_char,chars = segmantation(Img12,filename)
            fig.add_subplot(rows, columns, 3)
            # showing image
            plt.imshow(closing)
            plt.axis('off')
            plt.title("binary")
            print('Co {} ky tu'.format(len(chars)))            
            chars = np.array([c for c in chars], dtype="float32")
            preds = recogChar.predict(chars)
            result =[]
            for pred,char in zip(preds,chars): 	      
                # find the index of the label with the largest corresponding
                # probability, then extract the probability and label
                i = np.argmax(pred)
                prob = pred[i]
                label = class_names[i]

                # draw the prediction on the image
                print(f"Predict >> {label} - {prob * 100:.2f}%")
                if(prob * 100>55):
                    result.append(label)
                # cv2.imshow("character",char)
                # cv2.waitKey(0)
            # plate =sorted_Roi(contours,binary)
            # cv2.drawContours(binary, contours, -1, (0,0,0), 3)
            plate = ''

            fig.add_subplot(rows, columns, 4)
            for i in result:
                    clean_text = re.sub('[\W_]+', '', i)
                    clean_text = clean_text.upper()
                    plate += clean_text     
            # showing image
            plt.imshow(img_draw_char)
            plt.axis('off')
            plt.title("{}".format(plate))
            # cv2.imshow("Anh input", Ivehicle)
            # cv2.imwrite("output.png",Ivehicle)
            x+=30
            cv2.waitKey(0)

    plt.show()
    cv2.destroyAllWindows()