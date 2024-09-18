import cv2 as cv
import numpy as np
import CameraObject

camera = CameraObject.RealSense2()
camera.setupCamera()
camera.setRemoveBackgroundThreshold(1)
align = camera.getAlign()

title_window = "Filter Image"

cv.namedWindow(title_window, cv.WINDOW_NORMAL)



try:
    while True:
        frames = camera.pipeline.wait_for_frames()
        alignedFrames = align.process(frames)

        alignedDepthFrame = camera.getDepthFrame(alignedFrames)
        colorFrame = camera.getColorFrame(alignedFrames)

        if not alignedDepthFrame or not colorFrame:
            continue

        depthImage = camera.getDataFromFrame(alignedDepthFrame)
        colorImage = camera.getDataFromFrame(colorFrame)
        
        depthImage3D = np.dstack((depthImage, depthImage, depthImage))  
        bgRemoved = camera.removeObject(depthImage3D, colorImage)

        # convert the color image to HSV
        hsvImage = cv.cvtColor(bgRemoved, cv.COLOR_BGR2HSV)

        # define range of blue color in HSV
        lower_blue = np.array([110,50,50])
        upper_blue = np.array([130,255,255])
        # Threshold the HSV image to get only blue colors
        mask = cv.inRange(hsvImage, lower_blue, upper_blue)
        # Bitwise-AND mask and original image
        res = cv.bitwise_and(bgRemoved,bgRemoved, mask= mask)
        
        cv.imshow('frame',bgRemoved)
        cv.imshow('mask',mask)
        cv.imshow('res',res)
        k = cv.waitKey(5) & 0xFF
        if k == 27:
            break


except Exception as e:
    print(f"An error occurred: {e}")

finally:
    camera.pipeline.stop()
    cv.destroyAllWindows()