
import os
import re
import fnmatch
import sys
from glob import glob
import tensorflow as tf
import numpy as np
import cv2
from imutils import perspective
import imutils
import pytesseract


def json_files_from_folder(folder,upper=True):
	extensions = ['json']
	img_files  = []
	for ext in extensions:
		img_files += glob('%s/*.%s' % (folder,ext))
		if upper:
			img_files += glob('%s/*.%s' % (folder,ext.upper()))
	return img_files

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

def calculating_IOU(contour1,contour2):
	x1,y1,w1,h1 = cv2.boundingRect(contour1)
	box1 = [x1,y1,x1+w1,y1+h1]
	# print(box1)
	x2,y2,w2,h2 = cv2.boundingRect(contour2)
	box2 = [x2,y2,x2+w2,y2+h2]
	# print(box2)
	bi = [max(box1[0],box2[0]), max(box1[1],box2[1]), min(box1[2],box2[2]), min(box1[3],box2[3])]
	iw = bi[2] - bi[0] + 1
	# print(iw)
	ih = bi[3] - bi[1] + 1
	ov=0
	if iw > 0 and ih > 0:
		# compute overlap (IoU) = area of intersection / area of union
		ua = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1) + (box2[2] - box2[0]
						+ 1) * (box2[3] - box2[1] + 1) - iw * ih
		ov = iw * ih / ua
	return ov
def draw_losangle(I,pts,color=(1.,0.,255.),thickness=3):
	assert(pts.shape[0] == 2 and pts.shape[1] == 4)

	for i in range(4):
		pt1 = tuple(pts[:,i].astype(int).tolist())
		pt2 = tuple(pts[:,(i+1)%4].astype(int).tolist())
		cv2.line(I,pt1,pt2,color,thickness)
def sort_contours(contours,center_height, x_axis_sort='LEFT_TO_RIGHT', y_axis_sort='TOP_TO_BOTTOM'):
	# print(contours[0][0].shape)
    # initialize the reverse flag
	x_reverse = False
	y_reverse = False
	if x_axis_sort == 'RIGHT_TO_LEFT':
		x_reverse = True
	if y_axis_sort == 'BOTTOM_TO_TOP':
		y_reverse = True
	boundingBoxes = [cv2.boundingRect(c) for c in contours]
	char = []
	sortedByy = sorted(boundingBoxes,key = lambda b :b[1])

	# print('char3th :{} and center:{}'.format(sortedByy[2][3],center_height))
	try:
		if (sortedByy[2][3]<center_height) :
			#sorted by x and y
			first_line = sortedByy[0:3]
			second_line = sortedByy[3:]
			first_line = sorted(first_line,key=lambda b:b[0])
			second_line = sorted(second_line,key=lambda b:b[0])
			for first in first_line:
				char.append(first)
			for second in second_line:
				char.append(second)
		else:
			#sortedbyx
			char = sorted(boundingBoxes,key=lambda b:b[0])
	except:
		char = sorted(boundingBoxes,key=lambda b:b[0])
	# # sorting on x-axis 
	# sortedByX = zip(*sorted(zip(contours, boundingBoxes),
	# key=lambda b:b[1][0], reverse=x_reverse))

	# # sorting on y-axis 
	# (contours, boundingBoxes) = zip(*sorted(zip(*sortedByX),
	# key=lambda b:b[1][1], reverse=y_reverse))
	# # return the list of sorted contours and bounding boxes
	return char

def segmantation(Img12,filename):
	# kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	# thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
	# cv2.imshow('not closing',binary)
	# cv2.waitKey(0)
	img = Img12
	# cv2.imshow('not closing',img)
	# cv2.waitKey(0)
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
	# contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
	# contours,_ = sort_contours(contours)
	height, width = binary.shape
	center_height = height / 2
	binary_inverse = cv2.bitwise_not(closing)
	#sort by area
	# areaArray = []
	# for i, c in enumerate(contours):
	# 	area = cv2.contourArea(c)
	# 	areaArray.append(area)
	# #first sort the array by area
	# sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)

	# cv2.drawContours(img, contours, -1, (255,0,0), 3)
	# print('width :'+str(width) +'height : '+ str(height))
	# cv2.imshow('closing',binary)
	# cv2.waitKey(0)
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
		ratio = h / float(w)
		solidity = cv2.contourArea(c) / float(w * h)
		area = h * w
		area_ratio = area / float(width * height)
		height_ratio = h / float(height)
		# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
		# approximate the contour
		if(area_ratio >0.03)and (w / float(width) <0.25) and (float(h) / height > 0.3) and (w / float(width) >0.03) :
			# print(area)
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
		# print('a')
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
			#draw it
			# x, y, w, h = cv2.boundingRect(Ndlargestcontour)
			# crop = binary[y:y+h,x:x+w]
			# crop=prepare(crop)
			# char.append(crop)
			# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
	else:
		for n in range(len(sorteddata)):
			Ndlargestcontour = sorteddata[n-1][1]
			truly_contour.append(Ndlargestcontour)
			#draw it
			# x, y, w, h = cv2.boundingRect(Ndlargestcontour)
			# crop = binary[y:y+h,x:x+w]
			# crop=prepare(crop)
			# char.append(crop)
			# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
	
	boxes = sort_contours(truly_contour,center_height)
	
	for b in boxes:
		x,y,w,h = b
		# print('x :{}, y : {}'.format(x,y))
		crop = binary[y:y+h,x:x+w]
		crop=prepare(crop)
		char.append(crop)
		cv2.rectangle(img, (x, y), (x + w, y + h), (0,255 , 0), 2)
	# char =sorted(char,key= lambda x : x[0])
	# char =sorted(char,key= lambda x : x[1])
	# cv2.imshow('Pix',img)
	# cv2.waitKey(0)
	return closing,img,char   

def prepare(img):
    (tH, tW) = img.shape
    dX = int(max(0, 32 - tW) / 2.0)
    dY = int(max(0, 32 - tH) / 2.0)  
    # pad the image and force 32x32 dimensions
    padded = cv2.copyMakeBorder(img, top=dY, bottom=dY, left=dX, right=dX, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    padded = cv2.resize(img, (32, 32))
    # prepare the padded image for classification via our
    # handwriting OCR model
    padded = padded.astype("float32") / 255.0
    padded = np.expand_dims(padded, axis=-1)
    # padded = tf.expand_dims(padded, 0) 
    return padded