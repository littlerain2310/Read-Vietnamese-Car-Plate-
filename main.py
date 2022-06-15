import argparse
from reader import Reader
from utils import image_files_from_folder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str)
    parser.add_argument('--dir', type=str,default='./test')
    args = parser.parse_args()
    read = Reader()
    if args.image is None:
        # take input dir and read each file sequencely
        input_dir = args.dir
        img_files = image_files_from_folder(input_dir)
        #StartReading
        for img_path in img_files:
            read.getImg(img_path)
            read.readplate()
            read.display()
    else:
        img_path = args.image
        read.getImg(img_path)
        read.readplate()
        read.display()
