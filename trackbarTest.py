import numpy as np
import cv2 as cv


def trackbarCallBack(position):
    print(position)

img = np.zeros((300, 512, 3), np.uint8)
cv.namedWindow("test")

cv.createTrackbar("B", "test", 0,255, trackbarCallBack)
cv.createTrackbar("G", "test", 0,255, trackbarCallBack)
cv.createTrackbar("R", "test", 0,255, trackbarCallBack)

while(1):
    cv.imshow("test", img)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()