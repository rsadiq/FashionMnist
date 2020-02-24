# FashionMnist Test Attempts for Advertima 
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction
1.  This repo contains python3 scripts for training and inference of different CNN based 
 models for fashion objects classification using ```FashionMnist``` Dataset  
 2. I have tried various architectures and finalized few of them to be added to this repo.
 3. Hyper parameters optimization for all the models is not possible due to limitation of time and resource.
 But i tried to get as good results as possible.
 4. I believe some of them can be further to get better results.
 
###Requirements
1. All models were trained and tested in python 3.6      
 ```   Python3
    keras
    cv2
    imutils
    tensorflow
    efficientnet
    matplotlib
    numpy
```

 
 ### Problem Statement and provided solutions:
 Given the ```FashionMnist``` Dataset, Build CNN based architectures to classify images
 into 10 different classes with high efficiency.
 * Although Its a classification problem, I approached it with two method
    1. Multiclass Classification
    2. Object detection and localization 
 * For ***Multiclass Classification*** task, I have 3 different models
    1. Trained a BaseLine CNN model with a signle convolution layer, followed by 
    a fully connected layer. Since the dataset is quite simple and even with a simpler model
    I was able to achieve ***90%*** accuracy.
    2. Trained a little deeper network with increased the number of layers, and filters, used learning rate scheduling, used 
    some data augmentations and i was able to get ***94.10 test accuracy***
    3. Trained EfficientNet (B0, B3, B5) with various settings. Maximum trained accuracy 
    was around ***92%***. But i believe that EffcientNet can be further improved by playing with 
    Hyperparnameters, especially with learning rate. Right now i am using drop learning rate by 20% after every 20 epochs.
    ***Note:*** You can install EfficientNet binded with keras via pip install.
 * For ***Object Detection*** task, i used TinyYoloV3. I used most of the default settings for training 
 yolo except for anchors which i modified based on the images. 
    1. I generated the Jpeg images from original 28x28 FashionMnist gray scale images by creating an empty green canvas,
    placed resized fashionMnist image in its center and used the edges as Bounding Box coordinates of
    the objects.
    2. After data prepration, (train.txt) in yolo formate, i used default values to train 
    the model. but due to limitations on computation power and time constraints. I could only
    train the network for 6 EPOCHS. On the training set, i was able to achive the loss value lesserr than
    ***0.1***. 
    3 For packages and other requirements, kindly refer to README.md of keras-yolo3-master directory. 
 
###DataPreperation for Realtime Inference
Since the FashionMnist data set is quite small (28x28) single channels and according to its archive paper
the objects were trimmed from the images, placed on canvas with white background, inverted colors, converted to 
grayscale and resized to 28x28.
* I also tried to follow the same procedure, 
But for that i need to do image segmentation in order to crop out the main object and make a black background. I used 
opencv contours methods followed by morphological operators along with some smoothing to rule out very small objects.
and at the end conversion to grayscale and resizing to 28x28.

### Reason to select The above models for this problem
1. All of the models are light weight and can run in real time 
    * 15fps with webcam and 28fps from video for efficient net
    * 19fps for webcam and 39fps for CNN model with CNN with multilayers
2. As it is mentioned on FashionMnist dataset github page, Current SOTA is around 96%.
and all the leading models are based on WRN(Wide residual nets). But i believe they can ***not*** be 
***efficiently*** used for realtime with limited resources as they are quite heavy models and only their training will require
much time and inference also suffers from low speed. 
 
###Usage
* For training kindly run:
  ```shell script
    python3 fashion_mnist.py
    python3 fashion_mnist_aug.py (with data augmentations)
    python3 efffciecnt_fashion_mnist.py

   ```  
* For Realtime Inference from trained models kindly run:
```shell script
python3 realtime_inference_fashion.py -v /path/to/input/video/ -o /path/to/save/output/viideo
python3 realtime_inference_fashion.py -v /path/to/input/video/ -o /path/to/save/output/viideoEFN
```
   *  If no valid input video path is given, it will capture the video from webcam


### Limitations
Althoguh i was able to get above ***90%*** test accuracy for all models, i was unable to get
decent results on videos and webcam feeds. The main reason is the representation of training data.
Since it is built only for benchmarking purpose, it can still not represent the real world images with whole lot of objects
with various artifects and outliers.
* Right now i have seperate CLI for each code.
* Todo:
    * Code cleaning
    * Merging all modules under a single CLI using arguemts.
    * Further tunning of efficientnets
 