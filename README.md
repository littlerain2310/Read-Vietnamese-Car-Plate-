The project consists of 3 steps :
- cut and transform plate (Wpod-net)
- segment the number (opencv2)
- regconize number (CNN)


Run 
python read_plate.py --dir yourfolder --noDir False <yourimage folder>
or 
python read_plate.py --image yourimage --noDir True
They will read out the plate number
  
For example:
  - python read_plate.py --dir ./test --noDir False
  - python read_plate.py --image test/images.jpeg --noDir True

  
  
