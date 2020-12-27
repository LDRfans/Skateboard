import cv2
import numpy as np

TOLERANCE = 30
PATCH = 7
WHOLE_THRESHOLD = 140
BLUR_BOUND = 70
MAXLIMIT = 0.87
# MAXLIMIT = 0.868
# MINLIMIT = 0.955



def directEdge(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mm = cv2.Laplacian(gray, cv2.CV_64F).var()
    if mm < TOLERANCE:
        print("$$$ DEBUG : Contrast Enhancement !")
        print(mm)
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(20,20))
        gray = clahe.apply(gray)
        print(cv2.Laplacian(gray, cv2.CV_64F).var())

    xDirection = cv2.Sobel(gray, -1, 1, 0).var()
    yDirection = cv2.Sobel(gray, -1, 0, 1).var()
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)
    x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
    y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)




def patchLaplace(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    shape = gray.shape
    hStep, wStep = shape[0]//PATCH, shape[1]//PATCH
    grayGauss = cv2.GaussianBlur(gray,(3,3),0) # smooth noise
    if cv2.Laplacian(grayGauss, cv2.CV_64F).var() < TOLERANCE:
        # print("$$$ DEBUG : Contrast Enhancement !")
        clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(15,15))
        grayGauss = clahe.apply(grayGauss)

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
    min1 = np.min(patchlist)
    patchNoMax = patchlist.copy()
    patchNoMin = patchlist.copy()
    patchNoMax.remove(max)
    patchNoMin.remove(min1)
    min2 = np.min(patchNoMin)
    # print("*** min =", min1, min2)
    
    originVar = np.var(patchlist)
    varNoMax = np.var(patchNoMax)

    rmMax = varNoMax/originVar
    wholeVar = cv2.Laplacian(grayGauss, cv2.CV_64F).var()

    return (wholeVar, min1, min2, rmMax)



def inference(var, min1, min2, max, index):
    """ return 0: detailed;   
        return 1: whole image is blur;   
        return 2: partial blur, partial detailed.   
    """
    # print("var, maxRate :",var, max)
  
    if var < WHOLE_THRESHOLD:
        if var < BLUR_BOUND:
            "blur" # shake hands :(
            return 1
        if min1 < 1: # background is white(empty)
            "whole is detailed"
            return 0           
        if min1 < 3.5 and min2 < 4: # background is nearly empty (sky, wall)
            "whole is detailed"
            return 0
        if min1 < 38 and min2 < 38:
            "partial blur"
            return 2 
        if max > 0.95:
            "whole is detailed"
            return 0 
        else:
            "partial blur"
            return 2 
            print("### index:", index)
    elif max < MAXLIMIT:     # var > WHOLE_THRESHOLD:
        "partial blur"
        return 2 
    else:   # max > MAXLIMIT and  var > WHOLE_THRESHOLD:
        "whole is detailed"
        return 0

