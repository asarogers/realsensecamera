import pyrealsense2 as rs
import numpy as np

class RealSense2:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.pipelineWrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipelineProfile = self.config.resolve(self.pipelineWrapper)
        self.device = self.pipelineProfile.get_device()
        self.deviceProductLine = str(self.device.get_info(rs.camera_info.product_line))
        self.profile = None
        self.depthSensor = None
        self.depthScale = None
        self.removeBackgroundThresholdInMeters = 1
        self.removeBackgroundThreshold = None

    def setupCamera(self):
        foundRGB = False
        for s in self.device.sensors:
            if s.get_info(rs.camera_info.name) == "RGB Camera":
                foundRGB = True
                break
        if not foundRGB:
            print("The demo requires a Depth camera with a Color sensor")
            exit(0)
        
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        self.setDepthScale()
        self.pipeline.start(self.config)

    def setDepthScale(self):
        self.depthSensor = self.device.first_depth_sensor()
        self.depthScale = self.depthSensor.get_depth_scale()

    def setRemoveBackgroundThreshold(self, removeBackgroundThresholdInMeters):
        self.removeBackgroundThreshold = removeBackgroundThresholdInMeters / self.depthScale

    def getAlign(self):
        alignTo = rs.stream.color
        align = rs.align(alignTo)
        return align

    def getDepthFrame(self, alignedFrames):
        return alignedFrames.get_depth_frame()
    
    def getColorFrame(self, alignedFrames):
        return alignedFrames.get_color_frame()
    
    def getDataFromFrame(self, frame):
        return np.asanyarray(frame.get_data())
    
    def removeObject(self, depth_image_3d, color_image):
        grey_color = 153
        bg_removed = np.where((depth_image_3d > self.removeBackgroundThreshold) | (depth_image_3d <= -1), grey_color, color_image)
        return bg_removed