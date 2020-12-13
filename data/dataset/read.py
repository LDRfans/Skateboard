import cv2

filepath = 'static/1.jpg'

img = cv2.imread(filepath, cv2.IMREAD_COLOR)
# print(img)
# exit()
cv2.imshow(filepath, img)
cv2.waitKey(0)
cv2.destroyAllWindows()
