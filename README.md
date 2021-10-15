<h1><b>ALPR</b></h1>

<h2>Environments</h2>
pip install -r requirements.txt


<h2>Run</h2>
The project consists of 3 steps :
- cut and transform plate (Wpod-net)
- segment the number (opencv2)
- regconize number (CNN)


Run 
python read_plate.py --dir yourfolder --noDir False <yourimage folder>
or 
python read_plate.py --image yourimage --noDir True
They will read out the plate number

To test the program just run 
  python reader.py
  
For example:
  - python reader.py --dir ./test --noDir False
  - python reader.py --image test/images.jpeg --noDir True

  
  
  ![Figure_1](https://user-images.githubusercontent.com/56443812/137426501-3303170b-e2ee-4ce9-9490-6d79782cd90c.png)
  
