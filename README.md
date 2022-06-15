<h1><b>ALPR</b></h1>

<h1>Set Up</h1>
pip install -r requirements.txt


<h1>Run</h1>
The project consists of 3 steps :<br>
- cut and transform plate (Wpod-net) <b>lib_detection.py</b><br>
- segment the number from plate (opencv2) <b>detect_number.py</b><br>
- regconize number (CNN) <b>model.py</b>

<br>
Run inference for a folder <br>
python main.py --dir yourfolder <br> <yourimage folder>
or single image <br>
python main.py --image yourimage <br><br> 


To test the program just run <br>
  <i>python main.py</i>
  
For example:
  - python main.py --dir ./test
  - python main.py --image test/images.jpeg
  
<h1>Result </h1>  

  ![Figure_1](https://user-images.githubusercontent.com/56443812/137426501-3303170b-e2ee-4ce9-9490-6d79782cd90c.png)
  
