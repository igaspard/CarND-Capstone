from styx_msgs.msg import TrafficLight
import numpy as np
import cv2
import rospy

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.RED_MIN1 = np.array([0, 100, 100], np.uint8)
        self.RED_MAX1 = np.array([10, 255, 255], np.uint8)

        self.RED_MIN2 = np.array([160, 100, 100], np.uint8)
        self.RED_MAX2 = np.array([179, 255, 255], np.uint8)

        self.YELLOW_MIN = np.array([40.0/360*255, 100, 100], np.uint8)
        self.YELLOW_MAX = np.array([66.0/360*255, 255, 255], np.uint8)

        self.GREEN_MIN = np.array([90.0/360*255, 100, 100], np.uint8)
        self.GREEN_MAX = np.array([140.0/360*255, 255, 255], np.uint8)
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        frame_thresh1 = cv2.inRange(hsv_img, self.RED_MIN1, self.RED_MAX1)
        frame_thresh2 = cv2.inRange(hsv_img, self.RED_MIN2, self.RED_MAX2)
        red_thresh = cv2.countNonZero(frame_thresh1) + cv2.countNonZero(frame_thresh2)
        #rospy.logwarn('Red Thresh %d', red_thresh)
        if red_thresh > 50:
            return TrafficLight.RED

        frame_thresh = cv2.inRange(hsv_img, self.YELLOW_MIN, self.YELLOW_MAX)
        yellow_thresh = cv2.countNonZero(frame_thresh)
        if yellow_thresh > 50:
            return TrafficLight.YELLOW

        frame_thresh = cv2.inRange(hsv_img, self.GREEN_MIN, self.GREEN_MAX)
        green_thresh = cv2.countNonZero(frame_thresh)
        if green_thresh > 50:
            return TrafficLight.GREEN
        
        return TrafficLight.UNKNOWN
