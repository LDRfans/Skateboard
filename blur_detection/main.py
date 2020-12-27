# import blurDtt
import blurDtt2
import cv2
# import numpy as np
import matplotlib.pyplot as plt

STATIC = './dataset/TOYdataset/static/TOYstatic.txt'
SLIDING = './dataset/TOYdataset/sliding/TOYsliding.txt'
OVERHEAD = './dataset/TOYdataset/overhead/TOYoverhead.txt'

SET = ['STATIC', 'SLIDING', 'OVERHEAD']

def load_picture():
    imgPathList = []
    with open(OVERHEAD,'r') as f:
        paths = f.readlines()
    for p in paths:
        p = p.strip()
        imgPathList.append(p)
    return imgPathList


if __name__ == "__main__":
    detail, blur, index, part = 0, 0, 1, 0
    minList1, minList2 = [], []
    maxlist = []
    wholeList = []
    bllurlist = []
    partblurList = []
    print('%%%%% OVERHEAD')
    for p in load_picture():
        wholeVar, min1, min2, rmMax = blurDtt2.patchLaplace(p)
        minList1.append(min1)
        minList2.append(min2)
        maxlist.append(rmMax)
        wholeList.append(wholeVar)
        result =  blurDtt2.inference(wholeVar, min1, min2, rmMax, index)
        if result == 0:
            detail += 1
        elif result == 1:
            blur += 1
            bllurlist.append(index)
        else:
            part += 1
            partblurList.append(index)
        index += 1
    print("blur ----", blur, "detail ----", detail, "part ----", part) 
    print("blur is : ", bllurlist) 
    print("part blur is : ", partblurList) 
    print("static :", (detail+blur)/(index-1), "\n moving :",part/(index-1))


    # fig = plt.figure()
    # ax1 = fig.add_subplot(1,1,1)
    # ax1.plot(minList1, color = 'b', label = 'min1')
    # # ax2 = ax1.twinx()
    # # ax3 = ax1.twinx()
    # ax1.plot(minList2, color = 'r', label = 'min2')
    # plt.savefig('sliding.png')