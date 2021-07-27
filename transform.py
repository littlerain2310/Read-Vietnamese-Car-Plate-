import json
from glob import glob
from utils import json_files_from_folder

def getPoints(filename):
    f = open(filename,)
   
    # returns JSON object as 
    # a dictionary
    data = json.load(f)

    count = 0
    filename = filename.split('/')[-1]
    filename = filename.split('.')[0]
    points = ''
    for y in data['shapes']:
        
        points += str(y['label'])
        points += ' '
        for i in y['points']:
            points +=str(i[0]) +' '+ str(i[1])
            points += ' ' 
          
        points +='\n'  
        # if(count > 4):
        #     print(filename)
        #     raise Exception("Sorry, no numbers below zero")
        # count +=1
    # if(count>1):
    #     print(filename)
    with open('/home/long/Study/AI/Evaluation/Wpod-evaluation/mAP/input/ground-truth/{}.txt'.format(filename), 'w') as f:
        f.write('{}'.format(points))
        f.close()


files = json_files_from_folder('./gt_plate_VN')
for json_file in files:
    getPoints(json_file)