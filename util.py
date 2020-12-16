import json
import numpy as np
from math import sqrt, pi
import os
import re

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

def ReadFolder(case):
    if int(case) == 0:
        path = "./pose/flying/json"
    elif int(case) == 1:
        path = "./pose/sliding/json"
    elif int(case) == 2:
        path = "./pose/static/json"
    elif int(case) == 3:
        path = "./pose/coco/json"
    else:
        raise Exception("Input should be in {flying, sliding, static, coco}")
    
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
    # TBA : write the result into a file
    with open("legal.txt", "w+") as f:
        for img in ProperImg:
            f.write(img + "\n")


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


def Pose2Angle(Kp):
    '''
        Input: Kp
        Output: 4 angles 
    '''
    LShoulder = CalculateAngle(Kp["LElbow"], Kp["LShoulder"], Kp["Neck"])
    RShoulder = CalculateAngle(Kp["RElbow"], Kp["RShoulder"], Kp["Neck"])
    LKnee = CalculateAngle(Kp["LHip"], Kp["LKnee"], Kp["LAnkle"])
    RKnee = CalculateAngle(Kp["RHip"], Kp["RKnee"], Kp["RAnkle"])
    return (LShoulder, RShoulder, LKnee, RKnee)



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
        case = input("Choose one from { 0(flying), 1(sliding), 2(static), 3(coco) }:\n")
        ReadFolder(case)