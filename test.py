from numpy import character
from model import CNN_Model
import sys
from utils import *
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image


class_names =  ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


recogChar = CNN_Model().model
recogChar.load_weights('new.h5')

# Đường dẫn ảnh, các bạn đổi tên file tại đây để thử nhé

def fix_dimension(img): 
  new_img = np.zeros((28,28,3))
  for i in range(3):
    new_img[:,:,i] = img
  return new_img

def prepare(img):
    gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(19,19),0)
    # cv2.imshow("Anh bien so sau chuyen xam", gray)
    # Ap dung threshold de phan tach so va nen
    binary = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    cv2.THRESH_BINARY,11,2)
    binary = cv2.bitwise_not(binary)
    # print(binary.shape)    
    # img = image.img_to_array(img, dtype='uint8')
    # thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
	# cv2.THRESH_BINARY,11,2)
    (tH, tW) = binary.shape
    dX = int(max(0, 28 - tW) / 2.0)
    dY = int(max(0, 28 - tH) / 2.0)  
    # pad the image and force 28x28 dimensions
    padded = cv2.copyMakeBorder(binary, top=dY, bottom=dY, left=dX, right=dX, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    padded = cv2.resize(binary, (28, 28))
    # prepare the padded image for classification via our
    # handwriting OCR model
    # padded = padded.astype("float32") / 255.0
    # padded = np.expand_dims(padded, axis=-1)
    # padded = tf.expand_dims(padded, 0) 
    img = fix_dimension(padded)
    img = img.reshape(1,28,28,3)
    return img

input_dir = sys.argv[1]
img_files = image_files_from_folder(input_dir)
# model = E2E()
for img_path in img_files:
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i,c in enumerate(characters):
        dic[i] = c
    # setting values to rows and column variables
    rows = 2
    columns = 2
    # Đọc file ảnh đầu vào
    Ivehicle = cv2.imread(img_path)
    # Adds a subplot at the 1st position
    filename = img_path.split('/')[-1]
    filename = filename.split('.')[0]
    # image_ready = prepare(Ivehicle)
    image_ready = prepare(Ivehicle)
    # print(image_ready.shape)
    predictions = recogChar.predict(image_ready)
    score = tf.nn.softmax(predictions[0])
    # predictions = recogChar.predict_classes(image_ready)[0]
    # character = dic[predictions]
    print(character)
    # cv2.imshow('img',Ivehicle)
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    # cv2.waitKey(0)