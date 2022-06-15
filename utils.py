from glob import glob
import numpy as np
import cv2



def fix_dimension(img): 
  new_img = np.zeros((28,28,3))
  for i in range(3):
    new_img[:,:,i] = img
  return new_img

def prepare(img):
	(tH, tW) = img.shape
	dX = int(max(0, 28 - tW) / 2.0)
	dY = int(max(0, 28 - tH) / 2.0)  
	# pad the image and force 28x28 dimensions
	padded = cv2.copyMakeBorder(img, top=dY, bottom=dY, left=dX, right=dX, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
	padded = cv2.resize(img, (28, 28))

	img = fix_dimension(padded)
	
	
	return img

def json_files_from_folder(folder,upper=True):
	extensions = ['json']
	img_files  = []
	for ext in extensions:
		img_files += glob('%s/*.%s' % (folder,ext))
		if upper:
			img_files += glob('%s/*.%s' % (folder,ext.upper()))
	return img_files

def image_files_from_folder(folder,upper=True):
	extensions = ['jpg','jpeg','png','txt']
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
def group_height (contours):
	boundingBoxes = [cv2.boundingRect(c) for c in contours]
	heights = [b[3] for b in boundingBoxes ]
	group_height = []
	prev = None
	for height in heights:
		if not prev or height - prev <= 15:
			group_height.append(height)
		else:
			yield group_height
			group_height = [height]
		prev = height
	if group_height:
		yield group_height
def cluster_height(contuors):
	avg_height = 0
	for common_height in group_height(contuors):
		if len(common_height) > 3:
			avg_height = sum(common_height) / len(common_height)
	return avg_height

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

	return char

	
def separate(img,height):
	Img1= img[:height,:]
	Img2= img[height:,:]
	img = np.hstack((Img1, Img2)) 
	return img


