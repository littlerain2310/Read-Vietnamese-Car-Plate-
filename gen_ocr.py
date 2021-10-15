import sys
from reader import Reader
from utils import image_files_from_folder

    


# Đường dẫn ảnh, các bạn đổi tên file tại đây để thử nhé
input_dir = sys.argv[1]
img_files = image_files_from_folder(input_dir)


read = Reader()

for img_path in img_files:
    # Đọc file ảnh đầu vào   
    filename = img_path.split('/')[-1]
    filename = filename.split('.')[0]
    read.getImg(img_path)
    read.readplate()
    plate = read.result
           
    with open('./OCR_VN/detection/{}.txt'.format(filename), 'w') as f:
        f.write('{}'.format(plate))
        f.close()  
    read.reset()
            

