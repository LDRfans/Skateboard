import cv2
import numpy as np

PATCH = 8
WHOLE_THRESHOLD = 200
MAXLIMIT = 0.868
MINLIMIT = 0.955

def patchLaplace(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shape = gray.shape
    hStep, wStep = shape[0]//PATCH, shape[1]//PATCH
    grayGauss = cv2.GaussianBlur(gray,(3,3),0) # smooth noise
    patchlist = []
    y = 0
    while(y+hStep <= shape[0]):
        x = 0
        while(x+wStep <= shape[1]):
            patchVar = cv2.Laplacian(grayGauss[y:y+hStep, x:x+wStep], cv2.CV_64F).var()
            patchlist.append(patchVar)
            x += wStep
        y += hStep

    max = np.max(patchlist)
    min = np.min(patchlist)
    patchNoMax = patchlist.copy()
    patchNoMin = patchlist.copy()
    patchNoMax.remove(max)
    patchNoMin.remove(min)
    
    wholeVar = np.var(patchlist)
    varNoMax = np.var(patchNoMax)
    varNoMin = np.var(patchNoMin)
    # print("picture {:d} var :".format(i), wholeVar, varNoMax/wholeVar, varNoMin/wholeVar)

    rmMin = varNoMin/wholeVar
    rmMax = varNoMax/wholeVar

    return inference(wholeVar, rmMin, rmMax)



def inference(var, min, max):
    """ return 0: detailed;   
        return 1: whole image is blur;   
        return 2: partial blur, partial detailed.   
    """
    print("var, min, max :",var, min, max)
    if var < WHOLE_THRESHOLD:
        if max > MAXLIMIT and  min < MINLIMIT:
            "whole maybe really blur"
            return 1
        else:
            "partial blur"
            return 2
    elif max > MAXLIMIT and min > MINLIMIT:
        "whole is detailed"
        return 0
    else:
        return 2
        