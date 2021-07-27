from utils import text_files_from_folder
import asrtoolkit
from fuzzywuzzy import fuzz
from utils import text_files_from_folder
import sys


str1 = 'acb'
str2 = 'abcd'

acc = 0.0

gt_dir = sys.argv[1]
dt_dir = sys.argv[2]
ground_truths = text_files_from_folder(gt_dir)
detections = text_files_from_folder(dt_dir)

ground_truth_list = []

for ground_truth in ground_truths:
    filename = ground_truth.split('/')[-1]
    filename = filename.split('.')[0]
    f = open(ground_truth, "r")
    gt =str(f.read())
    detectionfile = open('{}/{}.txt'.format(dt_dir,filename), "r")
    detect = str(detectionfile.read())
    acc_file = 100.0 - float(asrtoolkit.cer(gt,detect))
    acc += acc_file
    # print(gt)
acc = acc /(len(ground_truths))
# acc = acc /60
print(acc)
# acc = 100.0 - float(asrtoolkit.cer(str1,str2))
# print(acc)
# print(fuzz.ratio(str1,str2 ))