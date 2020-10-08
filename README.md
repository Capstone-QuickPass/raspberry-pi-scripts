# raspberry-pi-scripts


## Version 0.0.1

Using TensorFlow Lite Object Detection Model to detect the existence of a person.
#### Functionalities:
- Draws a Region of Interest (ROI) box around a potential person and displays a likelihood score.
- If the likelihood score > 0.60 (60%), a photo is taken and stored in the base project directory.
#### How To Run:
Make sure numpy, picamera and pillow are installed
Install Tensorflow Lite:

-`curl -O http://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip -d ${DATA_DIR}
rm coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip`

Start Program:

-`python3 detect_picamera.py`



 
