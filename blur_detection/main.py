import blurDtt

# PATH = pass

def bunkLoadImg():
    imgList = []
    with open(PATH,'r') as f:
        paths = f.readlines()
    for p in paths:
        # remove linebreak at end of path
        p = p.strip()
        imgList.append(cv2.imread(p))
    return imgList
    


if __name__ == "__main__":
    imgPath = input("Enter test image path:")
    result = blurDtt.patchLaplace(imgPath)

    if result == 0:
        print("  @@@@@@@  Detailed !")
    elif result == 1:
        print("  @@@@@@@  Blur !")
    else:
        print("  @@@@@@@  Blur also detailed")