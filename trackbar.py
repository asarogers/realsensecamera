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

    cv.createTrackbar("H-Max", title_window, 138, 179, lambda x : None)
    cv.createTrackbar("S-Max", title_window, 167, 255, lambda x : None)  
    cv.createTrackbar("V-Max", title_window, 188, 255, lambda x : None)  

    cv.createTrackbar("H-Min", title_window, 123, 179, lambda x : None) 
    cv.createTrackbar("S-Min", title_window, 61, 255, lambda x : None)  
    cv.createTrackbar("V-Min", title_window, 34, 255, lambda x : None)  

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
            filteredImage = cv.bitwise_and(bgRemoved, bgRemoved, mask=mask)

            # display the filtered image
            cv.imshow(title_window, filteredImage)

            key = cv.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        camera.pipeline.stop()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
