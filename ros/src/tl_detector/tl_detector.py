#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree
import os

STATE_COUNT_THRESHOLD = 3

ENABLE_SAVE_IMG = False 
NUM_OF_SAVE_IMG = 5000
NUM_OF_SAVE_INTERVAL = 1
SAVE_SIM_IMG_PATH = '/Sim_Img/'
SAVE_SITE_IMG_PATH = '/Site_Img/'
FILE_NAME = 'image'

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        # Use for Save IMG
        self.save_cnt = 0
        self.savered_cnt = 0
        self.saveyellow_cnt = 0
        self.savegreen_cnt = 0
        self.savenone_cnt = 0
        self.save_interval = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            #rospy.loginfo('Traffic Light state changed pre: %d, now: %d', self.state, state)
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            #rospy.loginfo('Traffic Light state_count over Threshold Cur State: %d', state)
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            #rospy.loginfo('Traffic Light state_count below Threshold, update the last_wp: %d', self.last_wp)
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # For testing, just return the light state
        if(not self.has_image):
            self.prev_light_loc = None
            return False
       
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        if ENABLE_SAVE_IMG:
        # Save the image for traffic light classifier
            if (self.save_interval == NUM_OF_SAVE_INTERVAL-1) and (self.save_cnt < NUM_OF_SAVE_IMG):
                path = os.path.abspath(os.path.dirname(__file__))
                is_site = self.config["is_site"]
                if is_site:
                    IMG_PATH = path + SAVE_SITE_IMG_PATH
                else:
                    IMG_PATH = path + SAVE_SIM_IMG_PATH

                if light.state == TrafficLight.RED:
                    IMG_PATH = IMG_PATH + 'RED/' + FILE_NAME + str(self.savered_cnt) + '.jpg'
                    self.savered_cnt += 1
                elif light.state == TrafficLight.YELLOW:
                    IMG_PATH = IMG_PATH + 'YELLOW/' + FILE_NAME + str(self.saveyellow_cnt) + '.jpg'
                    self.saveyellow_cnt += 1
                elif light.state == TrafficLight.GREEN:
                    IMG_PATH = IMG_PATH + 'GREEN/' + FILE_NAME + str(self.savegreen_cnt) + '.jpg'
                    self.savegreen_cnt += 1
                else:
                    IMG_PATH = IMG_PATH + 'NONE/' + FILE_NAME + str(self.savenone_cnt) + '.jpg'
                    self.savenone_cnt += 1
                rospy.loginfo('Save Img for classifier %s', IMG_PATH)
                cv2.imwrite(IMG_PATH, cv_image)
                self.save_cnt += 1

            self.save_interval = (self.save_interval+1) % NUM_OF_SAVE_INTERVAL
            #rospy.loginfo('self.save_interval: %d', self.save_interval)
            return light.state
        
        #Get classification
        is_site = self.config["is_site"]
        if is_site:
            return self.light_classifier.get_classification_site(cv_image)
        else:
            return self.light_classifier.get_classification_sim(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = None
        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            #TODO find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                # Get stop line waypoint index
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                # Find the closest stop line waypoint index
                d = temp_wp_idx - car_wp_idx
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx

        if closest_light:
            state = self.get_light_state(closest_light)
            return line_wp_idx, state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
