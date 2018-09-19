# CarND Capstone - Gaspard Shen
  In this system integration project which is the final project of the Udacity Self-Driving Car Engineer Nanodegree, I Individually implement the ROS nodes core functionality of the autonomous vehicle system including traffic light detection, control and the waypoint following.

## System Architecture Diagram

![](/imgs/system_architecture.png)

Above diagram shows major ROS nodes we need to implement and how those ROS nodes communicate with each other. There are four steps we need to make the Carla/simulator work correctly.

## Waypoint Update
Write the first version of the waypoint_updater to publish a list of all waypoints for the track. At this moment, we don't consider the traffic lights first.
## DBW Node
The 2nd step is to implement the drive-by-wire node which will use the various controller to provide the appropriate throttle, brake, and steering command. This part I use the PID controller and low pass filter.

## Traffic Light Detection

The traffic light detect is the most major component of this project and take most of the time in this components. Here I use two different approaches for simulator and the real cases.

For simulator cases, it is much easier and actually works pretty well simply by converting the RGB domain to HSV domain which is much sensitive color. Then I define the different threshold for each red, green and yellow to identify the traffic light detection.

![](/imgs/HSV.png)

For the real cases, I use two steps strategies, first detect the traffic light location based on the `Tensorflow object detection API` with transfer learning of the pre-trained model `ssd_mobilenet_v1_coco`. After we can detect the location of the traffic light, use the CNN for traffic light classifier.

By using the Tensorflow object detection API, first I save the real cases image by running the ROS bags and manually label the traffic light location by the tool "labelImg". The major challenge I met here is that at the beginning I didn't realize that the Tensorflow version inside the ROS environment is quite old 1.3.0. And the training process is very time-consuming, it takes almost 1 day on my local machine's CPU. Somehow I meet several errors by using the AWS GPU service, by not observed on the CPU. After the model trained, at that moment I finally found the Tensorflow version compatible issues due to the training environment using the Tensorflow 1.9.0. Luckily, other students on slack meet the similar issue before and suggest that use the same model train by Tensorflow 1.9.0 and export by Tensorflow 1.3.0 again can fix this issue. Many thanks to the support.

After I can successfully detect the traffic light location, following the similar process that manually saves the traffic light picture and labels with Red, green and yellow light for image classifier which implemented by Keras. Here I meet the similar issue again that Keras version is old in ROS environment. Better this time, the training time is much less than the object detection, so I create another conda environment match all the python, Tensorflow and Keras version to train the model. Finally, it can work and detect the traffic light successfully. Below was the CNN architecture i used in this projects.

The detail how these two object detection and the image classifier will summarize in [other git repo](https://github.com/igaspard/Traffic-Light-Classifier) later on.

```
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 254, 118, 20)      560       
_________________________________________________________________
activation_1 (Activation)    (None, 254, 118, 20)      0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 252, 116, 44)      7964      
_________________________________________________________________
activation_2 (Activation)    (None, 252, 116, 44)      0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 250, 114, 68)      26996     
_________________________________________________________________
activation_3 (Activation)    (None, 250, 114, 68)      0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 248, 112, 92)      56396     
_________________________________________________________________
activation_4 (Activation)    (None, 248, 112, 92)      0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 124, 56, 92)       0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 638848)            0         
_________________________________________________________________
dense_1 (Dense)              (None, 120)               76661880  
_________________________________________________________________
activation_5 (Activation)    (None, 120)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 3)                 363       
_________________________________________________________________
activation_6 (Activation)    (None, 3)                 0         
=================================================================
Total params: 76,754,159
Trainable params: 76,754,159
Non-trainable params: 0
```

## Full Waypoint Update
After we can detect the traffic light, our waypoint_updater can adjust based on the info from the detection.

## Summary
This final project is very challenging and needs all the knowledge that we learn from term1 to term3 of the self-driving car nanodegree. In the past, the training data and labels usually were well prepared, but this time we need to do it from the ground up. Save the images from the simulator, label it and fight with the training model architecture, solving the environmental problem. It was very tough and I will say a very unforgettable experience.
