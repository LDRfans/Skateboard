import json
import numpy as np
from math import sqrt, pi
import os
import re
import csv

Body25Kp = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"]

"""
    Kp For reference:
        {0, “Nose”},      {1, “Neck”},   {2, “RShoulder”}, {3, “RElbow”},     {4, “RWrist”},
        {5, “LShoulder”}, {6, “LElbow”}, {7, “LWrist”},    {8, “MidHip”},     {9, “RHip”},
        {10, “RKnee”},    {11, “RAnkle”},{12, “LHip”},     {13, “LKnee”},     {14, “LAnkle”},
        {15, “REye”},     {16, “LEye”},  {17, “REar”},     {18, “LEar”},      {19, “LBigToe”},
        {20, “LSmallToe”},{21, “LHeel”}, {22, “RBigToe”},  {23, “RSmallToe”}, {24, “RHeel”},
"""

NOHUMAN = 0
LACKKP = 1
FlyingAvg = [138.8811921632241, 142.2911862962904, 88.91668435883095, 90.02673982106752, 150.4544595129767, 145.43553591321907, 116.73161248462749, 103.71046752557818]
SlidingAvg = [124.49742389801821, 124.20766153812566, 141.5521921015001, 137.74095218264895, 147.0786284501213, 139.4531926227165, 97.09056700644855, 93.83422144988009]
StaticAvg = [110.41301588840834, 108.88675673505043, 145.58438761917873, 145.46713221548964, 140.3103182492076, 142.2842076131475, 96.94728274168779, 99.11966207655712]
Avg = [FlyingAvg, SlidingAvg, StaticAvg]

def ReadFolder(case):
    if int(case) == 0:
        path = "./pose/coco/flying/json"
    elif int(case) == 1:
        path = "./pose/coco/sliding/json"
    elif int(case) == 2:
        path = "./pose/coco/static/json"
    else:
        raise Exception("Input should be in {flying, sliding, static}")
    
    Files = os.listdir(path)

    if len(Files) > 100:
        Files = Files[:100]

    NumLackKp = 0
    NumNoHuman = 0
    Result = [] # need to specify format
    ProperImg = [] # a list of proper image
    for i in Files:
        try:
            Kp, _ = ParseJson(path + '/' + i)
        except:
            NumNoHuman += 1
            Result.append(NOHUMAN)
            continue
        try:
            Angle = Pose2Angle(Kp)
            Result.append(Angle)
            RePath = i.split("_")[0] + '.jpg'
            ProperImg.append(RePath)
        except:
            NumLackKp += 1
            Result.append(LACKKP)

    print("Total Number of images from {} is {};\nNumber of images with a lack of keypoints is {};\nNumber of images with no human / more than 1 detected is {}. ".format(case, len(Files), NumLackKp, NumNoHuman))
    ProperNum = len(Files) - NumLackKp - NumNoHuman
    LShoulder = [i[0] for i in Result if i not in [NOHUMAN, LACKKP]]
    RShoulder = [i[1] for i in Result if i not in [NOHUMAN, LACKKP]]
    LKnee = [i[2] for i in Result if i not in [NOHUMAN, LACKKP]]
    RKnee = [i[3] for i in Result if i not in [NOHUMAN, LACKKP]]
    print("avg degree: ", np.mean(LShoulder), np.mean(RShoulder), np.mean(LKnee), np.mean(RKnee))
    print("std: ", np.std(LShoulder), np.std(RShoulder), np.std(LKnee), np.std(RKnee))


    # write the proper images' name into a file
    # with open("legal.txt", "w+") as f:
    #    for img in ProperImg:
    #        f.write(img + "\n")

def ReadFolder1(case):
    if int(case) == 0:
        path = "./pose/coco/flying/json"
    elif int(case) == 1:
        path = "./pose/coco/sliding/json"
    elif int(case) == 2:
        path = "./pose/coco/static/json"
    else:
        raise Exception("Input should be in {flying, sliding, static}")
    
    Files = os.listdir(path)

    NumLackKp = 0
    NumNoHuman = 0
    Result = [] # need to specify format
    ProperImg = [] # a list of proper image
    for i in Files:
        try:
            Kp, _ = ParseJson(path + '/' + i)
        except:
            NumNoHuman += 1
            Result.append([NOHUMAN]*8 + [int(case)])
            continue
        try:
            Angle = Pose2Angle(Kp, int(case))
            Result.append(list(Angle) +[int(case)])
        except:
            NumLackKp += 1
            Result.append([LACKKP]*8 +[int(case)])

    print("Total Number of images from {} is {};\nNumber of images with a lack of keypoints is {};\nNumber of images with no human / more than 1 detected is {}. ".format(case, len(Files), NumLackKp, NumNoHuman))
    ProperNum = len(Files) - NumLackKp - NumNoHuman
    return Result


def CalculateAvg(Result):
    LShoulder = [i[0] for i in Result if i not in [NOHUMAN, LACKKP]]
    RShoulder = [i[1] for i in Result if i not in [NOHUMAN, LACKKP]]
    LKnee = [i[2] for i in Result if i not in [NOHUMAN, LACKKP]]
    RKnee = [i[3] for i in Result if i not in [NOHUMAN, LACKKP]]
    print("avg degree: ", np.mean(LShoulder), np.mean(RShoulder), np.mean(LKnee), np.mean(RKnee))
    LShoulder = [i[4] for i in Result if i not in [NOHUMAN, LACKKP]]
    RShoulder = [i[5] for i in Result if i not in [NOHUMAN, LACKKP]]
    LKnee = [i[6] for i in Result if i not in [NOHUMAN, LACKKP]]
    RKnee = [i[7] for i in Result if i not in [NOHUMAN, LACKKP]]
    print("avg degree: ", np.mean(LShoulder), np.mean(RShoulder), np.mean(LKnee), np.mean(RKnee))



def WriteCsv(header=["LShoulder", "RShoulder", "LKnee", "RKnee", "LElbow", "RElbow", "LHip", "RHip", "Label"], data=[], fileName="pose/coco_pose.csv"):
    try:
        with open(fileName,"a", newline='') as f:
            writer = csv.writer(f)
            if data[0][-1] == 0:
                writer.writerow(header)
            writer.writerows(data)
    except:
        Exception("Please check your input parameters! They are invalid:(")

        


def ParseJson(path="./pose/flying/json/img01_keypoints.json"): 
    '''
        Input: a path string
        Output:  
            Kp: a dict of absolute position of each keypoint
            C: a dict of confidence level of each keypoint
            value is none if the keypoint is not detected (C[i] == 0)
    '''
    with open(path,"r") as f:
        Dict = json.load(f)
        if len(Dict["people"]) != 1: # no human / too many detected
            return None
        Pose = Dict['people'][0]['pose_keypoints_2d']
        Mask = [0, 1, 2]
        Kp = { Body25Kp[i]: (Pose[3*i], Pose[3*i+1]) if Pose[3*i+2] != 0 else None for i in range(25) }
        C =  { Body25Kp[i]: Pose[3*i+2] if Pose[3*i+2] != 0 else None for i in range(25) }
        return Kp, C


def Pose2Angle(Kp, case):
    '''
        Input: Kp
        Output: 8 angles 
    '''
    try:
        LShoulder = CalculateAngle(Kp["LElbow"], Kp["LShoulder"], Kp["Neck"])
    except:
        LShoulder = Avg[case][0]
    try:
        RShoulder = CalculateAngle(Kp["RElbow"], Kp["RShoulder"], Kp["Neck"])
    except:
        RShoulder = Avg[case][1]

    try:
        LKnee = CalculateAngle(Kp["LHip"], Kp["LKnee"], Kp["LAnkle"])
    except:
        LKnee = Avg[case][2]
    try:
        RKnee = CalculateAngle(Kp["RHip"], Kp["RKnee"], Kp["RAnkle"])
    except:
        RKnee = Avg[case][3]

    try:
        LElbow = CalculateAngle(Kp["LShoulder"], Kp["LElbow"], Kp["LWrist"])
    except:
        LElbow = Avg[case][4]
    try:
       RElbow = CalculateAngle(Kp["RShoulder"], Kp["RElbow"], Kp["RWrist"])
    except:
        RElbow = Avg[case][5]

    try:
        LHip = CalculateAngle(Kp["MidHip"], Kp["LHip"], Kp["LKnee"])
    except:
        LHip = Avg[case][6]
    try:
        RHip = CalculateAngle(Kp["MidHip"], Kp["RHip"], Kp["RKnee"])
    except:
        RHip = Avg[case][7]

    return (LShoulder, RShoulder, LKnee, RKnee, LElbow, RElbow, LHip, RHip)



def CalculateAngle(A, B, C):
    '''
        Input: 3 2-d points (ABC)
        Output：<ABC in degree
    ''' 
    a = np.array(A)
    b = np.array(B)
    c = np.array(C)
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / ( np.linalg.norm(ba) * np.linalg.norm(bc) )
    angle = np.arccos(cos)*180/pi
    return angle



if __name__ == "__main__":
    while True:
        case = input("We are dealing with coco now. Choose one from { 0(flying), 1(sliding), 2(static)}!\n")
        Result = ReadFolder1(case)
        WriteCsv(data=Result)