from styx_msgs.msg import TrafficLight
import numpy as np
import cv2
import rospy

import os
import tensorflow as tf
import urllib
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
# What model to load.
MODEL_NAME = 'traffic_light_detection_inference'

PWD = os.path.abspath(os.path.dirname(__file__))
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = PWD + '/' + MODEL_NAME + '/frozen_inference_graph.pb'

PATH_TO_CLASSIFER_MODEL = PWD + '/model.h5'

AWSLINK_TO_CLASSIFIER_MODEL = 'https://s3-us-west-1.amazonaws.com/carndmodel/model.h5'

ENABLE_SAVE_IMG = False 
SAVE_TL_IMG_PATH = '/TrafficLight/'

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
    
        self.save_cnt = 0
        # Load a frozen Tensorflow model into memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        if not os.path.exists(PATH_TO_CLASSIFER_MODEL):
            rospy.loginfo('Need to download the model first time due to the size limit')
            urllib.urlretrieve(AWSLINK_TO_CLASSIFIER_MODEL, PATH_TO_CLASSIFER_MODEL)
            rospy.loginfo('model downloaded')
        # Load the classifer model
        self.classifier = load_model(PATH_TO_CLASSIFER_MODEL)
        rospy.loginfo('Traffic Light Classifier model loaded')
        #self.classifier.summary()
        self.classifier._make_predict_function()
        self.classifier_graph = tf.get_default_graph()

    def run_inference_for_single_image(self, image, graph):
        with graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict
        
    def load_cv2image_into_numpy_array(self, image):
        return np.asarray(image, dtype="uint8")

    def get_classification_site(self, image):
        image_np = self.load_cv2image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = self.run_inference_for_single_image(image_np, self.detection_graph)
        if output_dict['detection_scores'][0] > 0.5:
            rospy.loginfo('Found the TrafficLight scores: %f', output_dict['detection_scores'][0])
            box = output_dict['detection_boxes'][0]
            height, width, channels = image.shape

            box_pixel = [int(box[1]*width), int(box[0]*height), int(box[3]*width), int(box[2]*height)]
            roi = image[box_pixel[1]:box_pixel[3], box_pixel[0]:box_pixel[2]]
            if ENABLE_SAVE_IMG:
                path = os.path.abspath(os.path.dirname(__file__))
                IMG_PATH = path + SAVE_TL_IMG_PATH + 'image' + str(self.save_cnt) + '.jpg'
                #rospy.loginfo('Save traffic light roi %s', IMG_PATH)
                cv2.imwrite(IMG_PATH, roi)
                self.save_cnt += 1
            
            # send roi to traffic light classifier
            # resize to 120x256 as the image size we training
            roi_resize = cv2.resize(roi, (100, 200), interpolation=cv2.INTER_CUBIC)
            roi_resize = np.asarray(roi_resize) / 255
            roi_test = np.array([roi_resize],)
            with self.classifier_graph.as_default():
                pred = self.classifier.predict(roi_test)
                pred = np.round(pred)

                if pred[0][0] == 1.:
                    rospy.loginfo("Predict Red Light")
                    return TrafficLight.RED
                elif pred[0][1] == 1.:
                    rospy.loginfo("Predict Yellow Light")
                    return TrafficLight.YELLOW
                elif pred[0][2] == 1.:
                    rospy.loginfo("Predict Green Light")
                    return TrafficLight.GREEN
                else:
                    return TrafficLight.UNKNOWN
            return TrafficLight.UNKNOWN
        else:
            return TrafficLight.UNKNOWN

    def get_classification_sim(self, image):
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
