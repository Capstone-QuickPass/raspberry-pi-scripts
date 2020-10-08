# python3
#
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import re

from annotation import Annotator

import numpy as np
import picamera

from PIL import Image
from tflite_runtime.interpreter import Interpreter

width = 1080
height = 720





def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold, annotator, camera):
  """Returns a list of detection results, each a dictionary of object info."""
  results = []

      
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all output details
  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))
  
  
  for i in range(count):
    if scores[i] >= threshold and classes[i]==0.0:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      ymin, xmin, ymax, xmax = result['bounding_box']
      annotator.bounding_box([int(xmin * width), int(ymin * height), int(xmax * width), int(ymax * height)])  
      if result['class_id'] == 0:
        annotator.text([int(xmin * width),  int(ymin * height)],'%s\n%.2f' % ('Person', result['score']))
      results.append(result)
  print(results)

  
  return results


def main():
  count = 0

  interpreter = Interpreter('detect.tflite')
  interpreter.allocate_tensors()
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  with picamera.PiCamera(
      resolution=(width, height), framerate=30) as camera:
    camera.start_preview()
    try:
      stream = io.BytesIO()
      annotator = Annotator(camera)
      for _ in camera.capture_continuous(
          stream, format='jpeg', use_video_port=True):
        stream.seek(0)
        image = Image.open(stream).convert('RGB').resize(
            (input_width, input_height), Image.ANTIALIAS)
        annotator.clear()
        results = detect_objects(interpreter, image, 0.6, annotator, camera)
        
        annotator.update()
        if len(results) == 1:
            camera.capture('person'+str(count)+'.jpeg')
            count += 1
        

        stream.seek(0)
        stream.truncate()

    finally:
      camera.stop_preview()


if __name__ == '__main__':
  main()
