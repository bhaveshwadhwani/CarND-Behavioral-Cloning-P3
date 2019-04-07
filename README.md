# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/432/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Files Submitted & Code Quality
---

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md/writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py,model.h5 files, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works and what is the code written for .

### Model Architecture and Training Strategy
---

#### 1. An appropriate model architecture has been employed

I had started using basic neural network with single layer network and went upto working with Nvidia end to end training network but found Letnet to give best  results in minimum computations/time as compared others

My model consists of a two convolution layer with 5x5 filter size and depths 6 , pooling layers and fully connected layers .

The model includes RELU layers to introduce nonlinearity, and the data is normalized and in the model using a Keras lambda layer.
Cropped using Cropping2D

Basic architecture was inspired by Lenet Architecture as it has been used widely in recognising objects in images 

##Lenet Architecture
 ![alt text](/examples/lenet.png "Lenet Architecture")

#### 2. Attempts to reduce overfitting in the model

To prevent overfitting, I used several data augmentation techniques like flipping images horizontally as well as using left and right images to help the model generalize. The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Udacity sample data was used for training.  I used a combination of center lane driving, recovering from the left and right sides of the road by adding 0.2 and -0.2 to the steering angle of the that particular image respectively. More data was generated by augmentation techniques like flipping images horizontally

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to drive a car through any terrain and any conditions.

My first step was to use a convolution neural network model similar to the NVIDIA I thought this model might be appropriate because it was designed for autonomous vehicles to drive through all kinds of terrains. 

But as compared to realtime scenarios the conditions in the simulation are simpler. So I decided to reduce the model complexity further by reducing convolutional layers and dense layers. This I have done by keeping in mind that the features required by car to drive through road are extracted at initial levels of convolutional layer and by following this approach I am able to achieve satisfactory results by using 2 convolution layer followed by Pooling and  Dense layers which is inspired from Lenet Architecture.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that model was performing well on the test data on simulator track one and low `mse loss` o trainig and validation set .

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track to improve the driving behavior in these cases, I have added left and right side images with offeseted driving angle, I have cropped the image to include only required details so that noise can be reduced, I flipped images so that additional details are available while training the vehicle. 

At the end of the process, the vehicle is able to drive autonomously around  the track one without leaving the road lane.

#### 2. Final Model Architecture

The final model architecture consisted of :
**Lambda layer for normalizing data and cropping image .
**Two convolution layers with 5x5 filter having depth 6 .
**2 Max Pooling layers .
**Flatten layer to flatten data .
**Fully connected layers
**Output layer


Here is a deatiled description of the architecture layer wise:
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 61, 316, 6)        456       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 30, 158, 6)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 26, 154, 6)        906       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 13, 77, 6)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 6006)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 120)               720840    
_________________________________________________________________
dense_2 (Dense)              (None, 84)                10164     
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 85        
_________________________________________________________________
						Output
=================================================================

#### 3. Creation of the Training Set & Training Process

To train my model I used dataset provided by Udacity. Some of the samples from original dataset are

![alt text](/examples/left_center_right_orignal.png "Original Images")

Converted to RGB :

| Camera | Image 1 | Image 2 |
| ---- | ---- | ---- |
| **Left Camera Images** | ![alt text](/examples/left1.jpg "Left Images") | ![alt text](/examples/left2.jpg "Left Images") |
| **Center Camera Images** | ![alt text](/examples/center1.jpg "Center Images") | ![alt text](/examples/center2.jpg "Center Images") |
| **Right Camera Images** | ![alt text](/examples/right1.jpg "Right Images") | ![alt text](/examples/right2.jpg "Right Images") |

First of all I divided data into 80% training data and 20% validation data.

To remove redundant details I cropped image's 70px from top and 25px from bottom for all the images that is center camera images, left camera images and right camera images (right and left camera images are added by adding respective steering angle offset), images after cropping are :

Cropped Original Images:

![alt text](/examples/Cropped_original.png " Cropped Original Images")

RGB Cropped Images:

![alt text](/examples/Cropped_rgb.png " RGB Cropped Images")

To augment the data set, I flipped center camera images to get model generalization and more input data. For example, here is an image that has then been flipped:

Flipped Cropped Images:

![alt text](/examples/flipped.png " Flipped Cropped  Images")

After cropping image I applied normalization on images and these normalized images are passed to the network as input

Normalized Cropped  Images:

![alt text](/examples/Normalized.png " Normalized Cropped  Images")

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.Augmenting data helped model to reduce overfitting and introduced additional data to generalize better . I used an adam optimizer so that manually training the learning rate wasn't necessary.

### Evaluation Video

| Track 1  |
| ---- |
| [Output Video](./output_video.mp4) | 

Thank you for reading !!!