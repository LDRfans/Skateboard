import shutil


fo = open("./imgList.txt", "r")
imgList = fo.readlines()

for name in imgList:
    imgName = name.rstrip('\n')

    source = '%s/%s' % ('../reduced dataset/COCO_skateboard', imgName)
    target = '../reduced dataset/useful_COCO'

    # Copy
    try:
        shutil.copy(source, target)
    except IOError as e:
        print("Unable to copy file. %s" % e)
    except:
        print("Unexpected error:", sys.exc_info())
