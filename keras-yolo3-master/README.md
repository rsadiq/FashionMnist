# keras-yolo3

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

A Keras implementation of TinyYOLOv3 (Tensorflow backend) inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K).


---

### Usage -Inference
Use --help to see usage of yolo_video.py:
```
usage: yolo_video.py [-h] [--model MODEL] [--anchors ANCHORS]
                     [--classes CLASSES] [--gpu_num GPU_NUM] [--image]
                     [--input] [--output]

positional arguments:
  --input        Video input path
  --output       Video output path

optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      path to model weight file, default model_data/yolo.h5
  --anchors ANCHORS  path to anchor definitions, default
                     model_data/tiny_yolo_anchors_fashion.txt
  --classes CLASSES  path to class definitions, default
                     model_data/fashion_mnist_classes.txt
  --gpu_num GPU_NUM  Number of GPU to use, default 1
  --image            Image detection mode, will ignore all positional arguments
```
---

## Training
0. Run ```fashion_mnist_to_jpg_and_label.py``` to generate the training file and images for
   ```FashionMnist``` Dataset.
            It should save all images to disk, and create an annotation file for training.

1. 
    One row for one image;  
    Row format: `image_file_path box1 box2 ... boxN`;  
    Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  
    Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```
   
2. Make sure you have run `python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5`  
    The file model_data/yolo_weights.h5 is used to load pretrained weights.

3. Modify train.py and start training.  
    `python train.py`  
    Use your trained weights or checkpoint weights with command line option `--model model_file` when using yolo_video.py
    Remember to modify class path or anchor path, with `--classes class_file` and `--anchors anchor_file`.

If you want to use original pretrained weights for YOLOv3:  
    1. `wget https://pjreddie.com/media/files/darknet53.conv.74`  
    2. rename it as darknet53.weights  
    3. `python convert.py -w darknet53.cfg darknet53.weights model_data/darknet53_weights.h5`  
    4. use model_data/darknet53_weights.h5 in train.py (I used TinyYoloV3)
    
---

## Some issues to know

1. The test environment is
    - Python 3.5.2
    - Keras 2.1.5
    - tensorflow 1.6.0

2. Modified anchors are used.

3. The inference result is not able to detect objects right now due to very low amount of training, only for couple of EPOCHS
