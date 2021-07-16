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
import re

# Dinh nghia cac ky tu tren bien so
char_list =  '0123456789ABCDEFGHKLMNPRSTUVXYZ'

# Ham fine tune bien so, loai bo cac ki tu khong hop ly
def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString
def read_whole_line(binary):
    platenum =' '
    # Nhan dien bien so. Cau hinh --psm 7 la de nhan dien 1 line only
    text = pytesseract.image_to_string(binary, lang="eng", config="--psm 7")
    cleanString = re.sub('\W+','', text )
    cleanString = cleanString.upper()
    # print(cleanString.upper())
    platenum += cleanString + '\n'    
    return platenum
    


# Đường dẫn ảnh, các bạn đổi tên file tại đây để thử nhé
input_dir = sys.argv[1]
img_files = image_files_from_folder(input_dir)

# Load model LP detection
wpod_net_path = "wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)

# model = E2E()

for img_path in img_files:
    # Đọc file ảnh đầu vào
    Ivehicle = cv2.imread(img_path)
    # cv2.imshow('input',Ivehicle)
    filename = img_path.split('/')[-1]
    filename = filename.split('.')[0]


    # Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
    Dmax = 608
    Dmin = 288

    # Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
    ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)

    _ , LpImg, lp_type,_,_ = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)


    platenum =''
    x= 0
    if (len(LpImg)):
        for Img in LpImg:
            # Chuyen doi anh bien so
            Img = cv2.convertScaleAbs(Img, alpha=(255.0))
            height = Img.shape[0]
            width = Img.shape[1]
            Img = Img[15:height-15,10:width -10]
            # Img = LpImg[0]
            if (lp_type == 2):
                # cv2.imshow('anh chua cat',Img)
                # print("Bien so VUONG")
                height = Img.shape[0]
                width = Img.shape[1]
                height_cutoff = height // 2
                Img1= Img[:height_cutoff,:]
                Img2= Img[height_cutoff:,:]
                # segmentation(Img1)
                # segmentation(Img2)
                Img12 = np.hstack((Img1, Img2))
            else:
                # print("Bien so DAI")
                Img12 = Img       
             
            _,plate = segmantation(Img12,filename)
            # print(plate)
            platenum +=plate
            platenum+='\n'
    with open('./ocr/detection/{}.txt'.format(filename), 'w') as f:
        f.write('{}'.format(platenum))
        f.close()  
            

    cv2.destroyAllWindows()