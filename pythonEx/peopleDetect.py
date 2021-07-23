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

if __name__ == "__main__":
  for image in imageFiles:
    inputImage = "../inputs/" + image
    outputImage = "../outputs/" + image
    ex = PeopleFinder(inputImage)
    ex.showDetection()
    ex.saveOutput(outputImage)