#!/usr/bin/env python3 

'''
Hi Mickey. 
This is a people detector; it uses a open CV which has a lot of capability to do real-time object detection with a camera. 
They have a built in ML Algorithm to detect people, so this is a try at that. 
If it works, I will replicate it in javascript.
'''

# import the necessary packages
import numpy as np
import cv2
import time
import os.path
from matplotlib import pyplot as plt
 
# initialize the HOG descriptor/person detector
imageFiles = ['whitecollar.jpeg', 'warehouse.jpeg', 'warehouse2.png', 'warehouse3.jpg', 'warehouse4.png']
videoFiles = ['warehouse.mp4']

class PeopleFinder:
  hog = cv2.HOGDescriptor()
  hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

  def __init__(self, filePath):
    # ditch if we can't find the pic
    if not os.path.exists(filePath):
      return

    self.filePath = filePath
    self.image = cv2.imread(self.filePath)
    cv2.startWindowThread() 

  def showOriginalPic(self):
    cv2.imshow("Output", self.image)
    cv2.waitKey(0)
  
  def showDetection(self):
    self.boxedImage = self.image
    boxes, weights  = self.hog.detectMultiScale(self.image)
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    for (xA, yA, xB, yB) in boxes:
      # display the detected boxes in the colour picture
      cv2.rectangle(self.boxedImage , (xA, yA), (xB, yB), (0, 255, 0), 2)
    
    cv2.imshow("Output", self.boxedImage)
    cv2.waitKey(0)
  
  def saveOutput(self, filePath):
    cv2.imwrite(filePath, self.boxedImage)

def videoEx():
  hog = cv2.HOGDescriptor()
  hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

   # one off for video
  cap = cv2.VideoCapture("../inputs/warehouse.mp4")

  # the output will be written to output.avi
  out = cv2.VideoWriter(
      'output.avi',
      cv2.VideoWriter_fourcc(*'MJPG'),
      15.,
      (640,480))
  
  while(True):
      # Capture frame-by-frame
      ret, frame = cap.read()
  
      # resizing for faster detection
      frame = cv2.resize(frame, (640, 480))
      # using a greyscale picture, also for faster detection
      gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
  
      # detect people in the image
      # returns the bounding boxes for the detected objects
      boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )
  
      boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
  
      for (xA, yA, xB, yB) in boxes:
          # display the detected boxes in the colour picture
          cv2.rectangle(frame, (xA, yA), (xB, yB),
                            (0, 255, 0), 2)
      
      # Write the output video 
      out.write(frame.astype('uint8'))
      # Display the resulting frame
      cv2.imshow('frame',frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  
  # When everything done, release the capture
  cap.release()
  # and release the output
  out.release()

if __name__ == "__main__":
  videoEx()

  '''
  for image in imageFiles:
    inputImage = "../inputs/" + image
    outputImage = "../outputs/" + image
    ex = PeopleFinder(inputImage)
    ex.showDetection()
    ex.saveOutput(outputImage)
  '''

 