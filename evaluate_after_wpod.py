from typing import Text
import cv2
from lib_detection import load_model, detect_lp, im2single
from skimage.filters import threshold_local
import sys
from utils import image_files_from_folder,segmantation
from model import CNN_Model
import numpy as np
import re

class_names =  ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


# Đường dẫn ảnh, các bạn đổi tên file tại đây để thử nhé
input_dir = sys.argv[1]
img_files = image_files_from_folder(input_dir)

recogChar = CNN_Model().model
recogChar.load_weights('new2.h5')

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
    _,img_draw_char,chars,_=segmantation(Ivehicle,filename)
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    chars = np.array([c for c in chars], dtype="float32")
    if len(chars) <=2:
        continue
    preds = recogChar.predict(chars)
    result =[]
    for (pred) in (preds): 	
        # find the index of the label with the largest corresponding
        # probability, then extract the probability and label
        i = np.argmax(pred)
        prob = pred[i]
        label = class_names[i]

        # draw the prediction on the image
        # print(f"Predict >> {label} - {prob * 100:.2f}%")
        result.append(label)
    # plate =sorted_Roi(contours,binary)
    # cv2.drawContours(binary, contours, -1, (0,0,0), 3)
    platenum = ''
    for i in result:
            clean_text = re.sub('[\W_]+', '', i)
            platenum += clean_text  
    # print(platenum)
    # cv2.imshow('after all',img_draw_char)
    # cv2.waitKey(0)
    with open('./{}/detection_ocr/{}.txt'.format(input_dir,filename), 'w') as f:
        f.write('{}'.format(platenum))
        f.close()   
