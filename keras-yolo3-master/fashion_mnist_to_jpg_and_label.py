'''
Run this script to generate save Fashion Mnist images as JPEG images with Bounding box labels
A text file "yolov3_train.txt" will generated to train Yolo
Images will be saved in  Fashion_Mnist_JPEGImages

'''

import keras
from keras.datasets import fashion_mnist
import numpy as np
from PIL import Image, ImageOps
import os
import random

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def save_image(filename, data_array):

    bgcolor = (0x00, 0xff, 0x00) # Green background
    screen = (640, 480)          # Create Empty Banner for Pasting rescaled Fashion Mnist image

    img = Image.new('RGB', screen, bgcolor)

    mnist_img = Image.fromarray(data_array.astype('uint8'))

    # mnist_img_invert = ImageOps.invert(mnist_img)

    #w = int(round(mnist_img.width * random.uniform(8.0, 10.0)))
    # Sclae up the 28x28 Image to 280x280
    w = int(mnist_img.width*10)
    # mnist_img_invert = mnist_img_invert.resize((w,w))
    mnist_img_invert = mnist_img.resize((w,w))

    #x = random.randint(0, img.width-w)
    #y = random.randint(0, img.height-w)
    x = int((img.width-w)/2)
    y = int((img.height-w)/2)
    img.paste(mnist_img_invert, (x, y))
    img.save(filename)
    print()
    # return convert((img.width,img.height), (float(x), float(x+w), float(y), float(y+w)))
    return int(x), int(x+w), int(y), int(y+w)

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
LABEL_DIR_NAME = os.getcwd()
DIR_NAME = "Fashion_Mnist_JPEGImages"
if os.path.exists(DIR_NAME) == False:
    os.mkdir(DIR_NAME)

j = 0
no = 0
wd = os.getcwd()
f = open('yolov3_train.txt', 'w')

for li in [x_train]:
    j += 1
    i = 0
    print("[---------------------------------------------------------------]")
    for x in li:
        # Write Image file
        filename = "{0}/{1:05d}.jpg".format(DIR_NAME,no)
        print(filename)
        ret = save_image(filename, x)

        # Write label file
        label_filename = "{0}/{1:05d}.txt".format(LABEL_DIR_NAME,no)
        # f = open(label_filename, 'w')

        y = 0
        if j == 1:
            y = y_train[i]
        else:
            y = y_test[i]

        #labelfile for yolo training : label followed by Bbox x1y1x2y2
        str = "{0:s}/{1:s} {2:d},{3:d},{4:d},{5:d},{6:d}".format(wd, filename,ret[0], ret[1], ret[2], ret[3],y)
        f.write(str)
        f.write('\n')
        # f.close()

        i += 1
        no += 1
f.close()

f = open('yolov3_test.txt', 'w')

for li in [x_test]:
    j += 1
    i = 0
    print("[---------------------------------------------------------------]")
    for x in li:
        # Write Image file
        filename = "{0}/{1:05d}.jpg".format(DIR_NAME,no)
        print(filename)
        ret = save_image(filename, x)

        # Write label file
        label_filename = "{0}/{1:05d}.txt".format(LABEL_DIR_NAME,no)
        # f = open(label_filename, 'w')

        y = 0
        if j == 1:
            y = y_train[i]
        else:
            y = y_test[i]

        #labelfile for yolo training : label followed by Bbox x1y1x2y2
        str = "{0:s}/{1:s} {2:d},{3:d},{4:d},{5:d},{6:d}".format(wd, filename,ret[0], ret[1], ret[2], ret[3],y)
        f.write(str)
        f.write('\n')
        # f.close()

        i += 1
        no += 1
f.close()