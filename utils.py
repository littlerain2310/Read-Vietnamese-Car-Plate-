
import os
import re
import fnmatch
import sys
from glob import glob
import numpy as np
import cv2
from imutils import perspective
import imutils
import pytesseract

def image_files_from_folder(folder,upper=True):
	extensions = ['jpg','jpeg','png']
	img_files  = []
	for ext in extensions:
		img_files += glob('%s/*.%s' % (folder,ext))
		if upper:
			img_files += glob('%s/*.%s' % (folder,ext.upper()))
	return img_files
def text_files_from_folder(folder,upper=True):
	extensions = ['txt']
	txt_file  = []
	for ext in extensions:
		txt_file += glob('%s/*.%s' % (folder,ext))
		if upper:
			txt_file += glob('%s/*.%s' % (folder,ext.upper()))
	return txt_file
def draw_losangle(I,pts,color=(1.,0.,255.),thickness=3):
	assert(pts.shape[0] == 2 and pts.shape[1] == 4)

	for i in range(4):
		pt1 = tuple(pts[:,i].astype(int).tolist())
		pt2 = tuple(pts[:,(i+1)%4].astype(int).tolist())
		cv2.line(I,pt1,pt2,color,thickness)
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
		if (width / float(w) < 30) and (float(h) / height > 0.5) and (ratio > 1.0) and(area > 20) :
            # print('x1 : {},y1 :{},x2 :{},y2 :{}'.format(x,y,x+w,y+h))
			# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
			number +=1
			crop = binary_inverse[y:y+h,x:x+w]
            # cv2.imwrite("./tmp_/{}_{}.png".format(filename,number), crop)   
			crop = cv2.copyMakeBorder(crop, 100, 100, 100, 100, cv2.BORDER_CONSTANT,value = (255,0,0))
			# cv2.imshow('binary',crop)
			# cv2.waitKey(0)
			try:
				text = pytesseract.image_to_string(crop, config='--psm 10')
				# clean tesseract text by removing any unwanted blank spaces
				clean_text = re.sub('[\W_]+', '', text)
				clean_text = clean_text.upper()
				plate_num += clean_text
			except: 
				text = None
	# cv2.imshow('Pix',img)
	# cv2.waitKey(0)
	return img,plate_num       
