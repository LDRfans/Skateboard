import cv2
import torch
import json
import os
import numpy as np

LABEL = "testing"
dirPath = "../COCO_3classes/"+LABEL
# dirPath = "../reduced dataset/flying skateboard"
outPath = '../out'

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).autoshape()  # for PIL/cv2/np inputs and NMS

# Read Image List
imgList = sorted(os.listdir(dirPath))
# fo = open("../reduced dataset/legal_coco.txt", "r")
# imgList = fo.readlines()
# print(imgList)
# exit()

# Images
imgs = []
filePathList = []
for imgName in imgList:
    imgName = imgName.strip('\n')
    if imgName[0] == '.' or imgName.split('.')[-1] != 'jpg':
        continue
    filePath = dirPath+'/'+imgName
    imgs.append(cv2.imread(filePath)[:, :, ::-1])
    # filePathList.append(filePath)
    filePathList.append(imgName)

# Inference
results = model(imgs, size=640)  # includes NMS

# Results (images)
# results.print()  # print results to screen
# results.show()  # display results
# results.save()  # save as results1.jpg, results2.jpg... etc.

# Data
output = {}
for imgIdx in range(len(results.xyxy)):
    xyxy = results.xyxy[imgIdx]
    height = imgs[imgIdx].shape[0]
    width = imgs[imgIdx].shape[1]
    imageOutput = {}
    # init
    personConfidence = 0
    skateboardConfidence = 0
    personBoundingBox = {'x1': 0.0, 'y1': 0.0, 'x2': 0.0, 'y2': 0.0, 'confidence': 0.0}
    skateboardBoundingBox = {'x1': 0.0, 'y1': 0.0, 'x2': 0.0, 'y2': 0.0, 'confidence': 0.0}
    # for all objects
    for i in range(xyxy.shape[0]):
        entry = xyxy[i].numpy().tolist()
        classId = entry[5]
        if classId == 0:   # person
            if personConfidence < entry[4]:
                personConfidence = entry[4]
                personBoundingBox = {'x1': entry[0]/width, 'y1': entry[1]/height, 'x2': entry[2]/width, 'y2': entry[3]/height, 'confidence': entry[4]}
        if classId == 36:  # skateboard
            if skateboardConfidence < entry[4]:
                skateboardConfidence = entry[4]
                skateboardBoundingBox = {'x1': entry[0]/width, 'y1': entry[1]/height, 'x2': entry[2]/width, 'y2': entry[3]/height, 'confidence': entry[4]}
    imageOutput['person'] = personBoundingBox
    imageOutput['skateboard'] = skateboardBoundingBox
    output[filePathList[imgIdx]] = imageOutput

# Write .json
fileName = "../out/YOLO_"+LABEL+".json"
with open(fileName, "w", encoding='utf-8') as jsonFile:
    json.dump(output, jsonFile)


#          x1 (pixels)  y1 (pixels)  x2 (pixels)  y2 (pixels)   confidence        class
# tensor([[7.47613e+02, 4.01168e+01, 1.14978e+03, 7.12016e+02, 8.71210e-01, 0.00000e+00],
#         [1.17464e+02, 1.96875e+02, 1.00145e+03, 7.11802e+02, 8.08795e-01, 0.00000e+00],
#         [4.23969e+02, 4.30401e+02, 5.16833e+02, 7.20000e+02, 7.77376e-01, 2.70000e+01],
#         [9.81310e+02, 3.10712e+02, 1.03111e+03, 4.19273e+02, 2.86850e-01, 2.70000e+01]])
