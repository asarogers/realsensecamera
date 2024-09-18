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

h_name = "H"
s_name = "S"
v_name = "V"

h = 127
s = 255
v = 0

def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H - 1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)

def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H + 1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)

def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S - 1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)

def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S + 1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)

def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V - 1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)

def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V + 1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)

def on_H_thresh_trackbar(val):
    global h
    h = val
    cv.setTrackbarPos(h_name, window_detection_name, h)

def on_S_thresh_trackbar(val):
    global s
    s = val
    cv.setTrackbarPos(s_name, window_detection_name, s)

def on_V_thresh_trackbar(val):
    global v
    v = val
    cv.setTrackbarPos(v_name, window_detection_name, v)

parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera device number.', default=0, type=int)
args = parser.parse_args()
cap = cv.VideoCapture(args.camera)
cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)

cv.createTrackbar(low_H_name, window_detection_name, low_H, max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_detection_name, high_H, max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, window_detection_name, low_S, max_value, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, window_detection_name, high_S, max_value, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, window_detection_name, low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_detection_name, high_V, max_value, on_high_V_thresh_trackbar)

cv.createTrackbar(h_name, window_capture_name, h, max_value, on_H_thresh_trackbar)
cv.createTrackbar(s_name, window_capture_name, s, max_value, on_S_thresh_trackbar)
cv.createTrackbar(v_name, window_capture_name, v, max_value, on_V_thresh_trackbar)

camera = CameraObject.RealSense2()
camera.setupCamera()
camera.setRemoveBackgroundThreshold(1)
align = camera.getAlign()

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

    # Convert the color image to HSV
    hsvImage = cv.cvtColor(bgRemoved, cv.COLOR_BGR2HSV)

    frame_threshold = cv.inRange(hsvImage, (low_H, low_S, low_V), (high_H, high_S, high_V))
    
    ret, thresh = cv.threshold(frame_threshold, h, s, v)
    center = []
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = contours[0]

        m = cv.moments(cnt)
        if m["m00"] > 0:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            area = cv.contourArea(cnt)
            print(cx, cy, area)
            epsilon = 0.1*cv.arcLength(cnt,True)
            approx = cv.approxPolyDP(cnt,epsilon,True)
            box = cv.minAreaRect(approx)
            end = approx[1][0]
            start = approx[0][0]
            if area > 300:
                center = [cx, cy]

            # print("\n")
            # print(box)
            # print(f"old  = {approx}")
            # print(f"new  = {approx[0][0]}")
            # print(f"center  = {(end[0] + start[0])}, {(end[1] + start[1])/2}")
    cv.drawContours(bgRemoved, contours, -1, (0, 255, 0), 3)

    if center:
        newImage = cv.circle(bgRemoved, (center[0], center[1]), 2, color=(0, 0, 255), thickness=-1)
        cv.imshow(window_capture_name, newImage)
    else:
        cv.imshow(window_capture_name, bgRemoved)
    
    
    cv.imshow(window_detection_name, frame_threshold)
    
    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break
