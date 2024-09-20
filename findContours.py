import cv2 as cv
import argparse
import CameraObject
import numpy as np
import pyrealsense2 as rs

window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'

filtered_image_name = "Filter Image"

def doNothing(nothing):
    pass

def main():
    camera = CameraObject.RealSense2()
    camera.setupCamera()
    camera.setRemoveBackgroundThreshold(1.2)
    align = camera.getAlign()

    cv.namedWindow(filtered_image_name, cv.WINDOW_NORMAL)

    cv.createTrackbar("H-Max", filtered_image_name, 158, 179, lambda x : None)
    cv.createTrackbar("S-Max", filtered_image_name, 255, 255, lambda x : None)  
    cv.createTrackbar("V-Max", filtered_image_name, 177, 255, lambda x : None)  

    cv.createTrackbar("H-Min", filtered_image_name, 119, 179, lambda x : None) 
    cv.createTrackbar("S-Min", filtered_image_name, 79, 255, lambda x : None)  
    cv.createTrackbar("V-Min", filtered_image_name, 105, 255, lambda x : None)  

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
            maxH = cv.getTrackbarPos("H-Max", filtered_image_name)
            maxS = cv.getTrackbarPos("S-Max", filtered_image_name)
            maxV = cv.getTrackbarPos("V-Max", filtered_image_name)

            minH = cv.getTrackbarPos("H-Min", filtered_image_name)
            minS = cv.getTrackbarPos("S-Min", filtered_image_name)
            minV = cv.getTrackbarPos("V-Min", filtered_image_name)

            # create a mask based on the HSV trackbar values
            lowerHSV = np.array([minH, minS, minV])
            upperHSV = np.array([maxH, maxS, maxV])  

            mask = cv.inRange(hsvImage, lowerHSV, upperHSV)
            filteredImage = cv.bitwise_and(colorImage, colorImage, mask=mask)

            frame_threshold = cv.inRange(hsvImage, lowerHSV, upperHSV)
            contours, hierarchy = cv.findContours(frame_threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(bgRemoved, contours, -1, (0, 255, 0), 3)
            # display the filtered image
            cv.imshow(filtered_image_name, filteredImage)

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


    