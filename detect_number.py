from utils import *
import imutils
class Segmentation():
    def __init__(self,image):
        self.origin = image
        self.image = self.origin.copy()
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = gray.shape
        self.center_height = self.height / 2
    def find_height(self,avg_height = None):
        '''find avarage height of contours'''
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(19,19),0)
        # Ap dung threshold de phan tach so va nen
        binary = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
        cv2.THRESH_BINARY,11,2)
        self.binary = cv2.bitwise_not(binary)
                
        kernel = np.ones((5,5),np.uint8)
        if avg_height is None:
            self.closing = cv2.morphologyEx(self.binary, cv2.MORPH_CLOSE, kernel)
        contours = cv2.findContours(self.closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        # contours,_ = sort_contours(contours)
        height, width = self.binary.shape
       
        candidate = []
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            
            area = h * w
            area_ratio = area / float(width * height)
            if avg_height is None :
                # approximate the contour
                if(area_ratio >0.03)and (w / float(width) <0.25) and (float(h) / height > 0.3) \
                and (w / float(width) >0.03) :
                    # print(area)
                    candidate.append(c)
            else:
                height_truly = abs(h - avg_height)
                height_ratio = h / float(height)
                # approximate the contour with avg_height
                if   (w / float(width) <0.25) and (height_truly <= 16) and \
                (w / float(width) >0.03) and (height_ratio >0.3) :

                    candidate.append(c)
            
        candidate_rm_inner = []
        #get rid of inner contours
        n = len(candidate)
        for i in range(n):
            for j in range(i+1, n):
                ov = calculating_IOU(candidate[i],candidate[j])
                if ov > 0.3:
                    area1 = cv2.contourArea(candidate[i])
                    area2 = cv2.contourArea(candidate[j])
                    if area1 > area2 :
                        candidate_rm_inner.append(candidate[j])
                    else:
                        candidate_rm_inner.append(candidate[i])
        for c in candidate_rm_inner:
            try:
                candidate.remove(c)
            except:
                pass
        
        #sort by area
        areaArray = []
        truly_contour = []
        for i, c in enumerate(candidate):
            area = cv2.contourArea(c)
            areaArray.append(area)
        #first sort the array by area
        sorteddata = sorted(zip(areaArray, candidate), key=lambda x: x[0], reverse=True)
        # print(len(sorteddata))
        if len(sorteddata) >7:
            for n in range(1,9):
                Ndlargestcontour = sorteddata[n-1][1]
                truly_contour.append(Ndlargestcontour)
                
        else:
            for n in range(len(sorteddata)):
                Ndlargestcontour = sorteddata[n-1][1]
                truly_contour.append(Ndlargestcontour)
                
        avg_height = cluster_height(truly_contour)
        return avg_height,truly_contour
    def segmenation(self):
        #find average height first time
        avg_height,_ = self.find_height()
        # classify type plate
        closing = self.closing
        binary = self.binary
        two_line = False
        if avg_height <= self.center_height:
            two_line = True
        if two_line:
            height_cutoff = self.height // 2
            
            self.closing = separate(closing,height_cutoff)
            
            self.binary = separate(binary,height_cutoff)
            
            self.image = separate(self.image,height_cutoff)
        #segment character based on avg_height
        _,truly_contour = self.find_height(avg_height)
        height_2, width_2 = self.closing.shape
        center_height = height_2 / 2

        boundingBoxes = [cv2.boundingRect(c) for c in truly_contour]
        #sortbyX
        boxes = sorted(boundingBoxes,key = lambda b :b[0])
        boxes_4_eval = []
        char_raw = []
        if two_line:
            first_line = boxes[:3]
            line_two = boxes[3:]
            for b in first_line:
                x,y,w,h = b
                boxes_4_eval.append(b)
                
            for b in line_two:
                x,y,w,h = b
                x = x - self.width
                y = y+height_cutoff
                b= x,y,w,h
                boxes_4_eval.append(b)
        else:
            boxes_4_eval = boxes
            for b in boxes:
                x,y,w,h = b
        char = []
        for b in boxes:
            x,y,w,h = b
            
            crop = self.binary[y:y+h,x:x+w]
            crop=prepare(crop)
            char.append(crop)
            crop_img = self.origin[y:y+h,x:x+w]
            char_raw.append(crop_img)
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (0,255 , 0), 2)
        
        return self.closing,self.image,char,boxes_4_eval,char_raw