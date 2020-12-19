    try:
        LShoulder = CalculateAngle(Kp["LElbow"], Kp["LShoulder"], Kp["Neck"])
    except:
        raise Exception("LShoulder Nan")

    try:
        RShoulder = CalculateAngle(Kp["RElbow"], Kp["RShoulder"], Kp["Neck"])
    except:
        raise Exception("RShoulder Nan")

    try:
        LKnee = CalculateAngle(Kp["LHip"], Kp["LKnee"], Kp["LAnkle"])
    except:
        raise Exception("LKnee Nan")    
    try:
        RKnee = CalculateAngle(Kp["RHip"], Kp["RKnee"], Kp["RAnkle"])
    except:
        raise Exception("RKnee Nan")

    try:
        LElbow = CalculateAngle(Kp["LShoulder"], Kp["LElbow"], Kp["LWrist"])
    except:
        raise Exception("LElbow Nan")
    try:
       RElbow = CalculateAngle(Kp["RShoulder"], Kp["RElbow"], Kp["RWrist"])
    except:
        raise Exception("RElbow Nan")

    try:
        LHip = CalculateAngle(Kp["MidHip"], Kp["LHip"], Kp["LKnee"])
    except:
        raise Exception("LHip Nan")
    try:
        RHip = CalculateAngle(Kp["MidHip"], Kp["RHip"], Kp["RKnee"])