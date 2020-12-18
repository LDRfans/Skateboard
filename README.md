# Skateboard
Computer Vision 2020 Course Project


openpose[CMU]:https://github.com/CMU-Perceptual-Computing-Lab/openpose



## TODO

- 高子淇：跑通openpose
- 赵乘风：跑通openpose，整理数据集
- 李德润：搜集平面与物体关系检测的相关方法
- 刘芊渝：探索糊不糊方法
- 张晨杨：OpenCV 物体检测（coco dataset）



## Dev log

### Ziqi's part

### 12/12

- Running openpose successfully;

- Uploading the result on reduced dataset in `pose` , with json and jpg format for each image.


### 12/16

- Parse json files from openpose;

- Calculate 4 Angles and show the result.

### 12/18

- Process pose data into a csv file, with label mapping being { 0:flying, 1:sliding, 2:static}