import cv2 as cv
import argparse
import CameraObject
import numpy as np



max_value = 255
max_value_H = 360//2

low_H = 111
high_H = 159

low_S = 55
high_S = 194

low_V = 124
high_V = 255

window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'


import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import CameraObject


title_window = "Filter Image"

def doNothing(nothing):
    pass

def main():
    camera = CameraObject.RealSense2()
    camera.setupCamera()
    camera.setRemoveBackgroundThreshold(1.2)
    align = camera.getAlign()

    cv.namedWindow(title_window, cv.WINDOW_NORMAL)

    cv.createTrackbar("H-Max", title_window, 158, 179, lambda x : None)
    cv.createTrackbar("S-Max", title_window, 255, 255, lambda x : None)  
    cv.createTrackbar("V-Max", title_window, 177, 255, lambda x : None)  

    cv.createTrackbar("H-Min", title_window, 119, 179, lambda x : None) 
    cv.createTrackbar("S-Min", title_window, 79, 255, lambda x : None)  
    cv.createTrackbar("V-Min", title_window, 105, 255, lambda x : None)  

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
            hsvImage = cv.cvtColor(colorImage, cv.COLOR_BGR2HSV)

            # get current positions of the HSV trackbars
            maxH = cv.getTrackbarPos("H-Max", title_window)
            maxS = cv.getTrackbarPos("S-Max", title_window)
            maxV = cv.getTrackbarPos("V-Max", title_window)

            minH = cv.getTrackbarPos("H-Min", title_window)
            minS = cv.getTrackbarPos("S-Min", title_window)
            minV = cv.getTrackbarPos("V-Min", title_window)

            # create a mask based on the HSV trackbar values
            lowerHSV = np.array([minH, minS, minV])
            upperHSV = np.array([maxH, maxS, maxV])  

            mask = cv.inRange(hsvImage, lowerHSV, upperHSV)
            filteredImage = cv.bitwise_and(colorImage, colorImage, mask=mask)

            frame_threshold = cv.inRange(hsvImage, lowerHSV, upperHSV)
            # display the filtered image
            cv.imshow(title_window, filteredImage)

            cv.imshow(window_capture_name, bgRemoved)
            cv.imshow(window_detection_name, frame_threshold)
            
            key = cv.waitKey(30)
            if key == ord('q') or key == 27:
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        camera.pipeline.stop()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()


    