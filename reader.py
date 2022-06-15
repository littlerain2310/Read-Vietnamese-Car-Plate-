
from lib_detection import load_model, detect_lp, im2single
from model import CNN_Model
import cv2
import numpy as np
import matplotlib.pyplot as plt
from detect_number import Segmentation

class Reader:
    """
    The class take image contains a car to read its plate number
    """
    def __init__(self) :
        """
        Take 1 argument is the image
        """
        # load all the models
        wpod_net_path = "wpod-net_update1.json"
        self.wpod_net = load_model(wpod_net_path)
        self.recogChar = CNN_Model().model
        self.recogChar.load_weights('CNN.h5')
        self.characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.class_names =  ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.result = ''
    def getImg(self,img_path):
        self.Ivehicle = cv2.imread(img_path)
    def readplate(self):
        """
        This one will read out the number of plate 
        """

        # The maximum and minimum size of an image
        Dmax = 608
        Dmin = 288

        # find the min bound
        ratio = float(max(self.Ivehicle.shape[:2])) / min(self.Ivehicle.shape[:2])
        side = int(ratio * Dmin)
        bound_dim = min(side, Dmax)

        _ , LpImg, lp_type,_,score = detect_lp(self.wpod_net, im2single(self.Ivehicle), bound_dim, lp_threshold=0.5)
        if (len(LpImg)):
            for Img in LpImg:
                # Chuyen doi anh bien so
                Img = cv2.convertScaleAbs(Img, alpha=(255.0))
                
                height = Img.shape[0]
                width = Img.shape[1]
                Img = Img[5:height-5,10:width -10]
                # Divide 2 line plate into 2 parts
                if (lp_type == 2):
                    height = Img.shape[0]
                    width = Img.shape[1]
                    height_cutoff = height // 2
                    Img1= Img[:height_cutoff,:]
                    Img2= Img[height_cutoff:,:]
                    Img12 = np.hstack((Img1, Img2))             
                else:
                    Img12 = Img
                origin = Img12.copy()
                
                segment = Segmentation(Img12)
                closing,img_draw_char,chars,boxes_char,char_raw = segment.segmenation()
                chars = np.array([c for c in chars], dtype="float32")
                if len(chars) <= 2:
                    return
                self.origin = origin
                self.closing = closing
                self.img_draw_char = img_draw_char
                self.boxes_char = boxes_char
                self.char_raw =char_raw
            #    Classify after cutting number out of plate
                dic = {}
                for i,c in enumerate(self.characters):
                    dic[i] = c
                preds = self.recogChar.predict(chars)
                result =''
                for (pred) in (preds): 	
                    # find the index of the label with the largest corresponding
                    # probability, then extract the probability and label
                    i = np.argmax(pred)
                    label = self.class_names[i]
                    result+=label
                self.result += result
    
    def display(self):
        fig = plt.figure(figsize=(10, 7))
        rows = 2
        columns = 2

        # origin image
        fig.add_subplot(rows, columns, 1)
        plt.imshow(self.Ivehicle)
        plt.axis('off')
        plt.title("original")

        # after cropping
        fig.add_subplot(rows, columns, 2)
        plt.imshow(self.origin)
        plt.axis('off')
        plt.title("cropped ")

        # adjust image to binary
        fig.add_subplot(rows, columns, 3)
        plt.imshow(self.closing)
        plt.axis('off')
        plt.title("binary")

        # final result
        fig.add_subplot(rows, columns, 4)
        plt.imshow(self.img_draw_char)
        plt.axis('off')
        plt.title("{}".format(self.result))
        self.result = ''
        plt.show()
    def reset(self):
        self.result = ''


