from utils import image_files_from_folder


img_files = image_files_from_folder('./after_wpod/0.9+')
print(len(img_files))

for img_path in img_files:
    # Đọc file ảnh đầu vào
    # cv2.imshow('input',Ivehicle)
    filename = img_path.split('/')[-1]
    filename = filename.split('.')[0]
    with open('./after_wpod/0.9+/{}.txt'.format(filename), 'w') as f:
        f.close()