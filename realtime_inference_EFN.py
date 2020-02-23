__author__ = "Rizwan Sadiq - Real time implementation of Fashion object detection "
__copyright__ = "Copyright (C) 2020 Rizwan Sadiq"
'''
This is a realtime implementation of Fashion Object detection using Efficient NEt
requirements:
        python3.6
        cv2 (tested with opencv 4)
        keras (tested with 2.2.4)
        numpy
        imutils
        argparse
        efficientnet

Usage:
        python3 realtime_inference_fashion.py -v /path/to/input/video -o /path/to/output/video/
If there is no video path is provided, it will switch to camera feed
Model file "BN_model1.h5" should be placed in the same directory

-- OutPut video will be written at 20fps with no audio
-- for audio merging we can use ffmpeg or pyav 
'''
import os
import cv2
import efficientnet.keras
from keras.preprocessing import image as Image
import numpy as np
from imutils.video import FPS
import argparse

from keras import models
import tensorflow as tf

from tensorflow.compat.v1 import InteractiveSession

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to input video file")
ap.add_argument("-o", "--out_video", default='outvideoEFN',
                help="path to output video file")
args = vars(ap.parse_args())

# Load Model
model = models.load_model('EFNB0Weight.h5')
out_directory = args["out_video"]
os.makedirs(out_directory, exist_ok=True)

# Check if a valid video Argument is pass
try:
    video = cv2.VideoCapture(args["video"])
except:
    print('exception')

else:
    # If video is not valid, Switch to Camera
    if video.read()[0] == False:
        video = cv2.VideoCapture(0)

# video = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(os.path.join(out_directory, 'output.avi'), fourcc, 20.0, (640, 480), False)

count = 0
labelNames = ["T-shirt/top", "trouser", "pullover", "dress", "coat",
              "sandal", "shirt", "sneaker", "bag", "ankle boot"]

ftext_font = cv2.FONT_HERSHEY_SIMPLEX
text_Location = (50, 50)
fontScale = 1
text_color = (255, 0, 0)
text_thickness = 2
fps = FPS().start()

while True:
    # try:
    count += 1
    _, frame = video.read()
    frame = cv2.resize(frame, (640, 480))
    # im =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(frame, (5, 5), 1)
    # gray = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    gray = 255 - cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    edges = cv2.dilate(gray, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    edges = cv2.GaussianBlur(edges, (1, 1), 0)

    # -- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    # -- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))

    # -- Smooth mask, then blur it --------------------------------------------------------
    # mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.GaussianBlur(mask, (19, 19), 0)
    # mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)

    # -- Blend masked img into MASK_COLOR background --------------------------------------
    frame = frame.astype('float32') / 255.0  # for easy blending
    # cv2.imwrite('mask.png', mask)

    c_red, c_green, c_blue = cv2.split(frame)
    mask = mask.astype(np.uint8)

    cv2.addWeighted(mask, 0.8, gray, 0.8, 0.2, gray)

    im = cv2.resize(gray, (28, 28))
    im = np.full((28, 28, 3), 0.0)

    test_img = Image.img_to_array(im)
    test_img = test_img[:, :].reshape(1, 28, 28, 3)

    test_img = test_img / 255

    preds = model.predict(test_img)
    print(np.max(preds))
    if np.max(preds) > 0.5:
        label = labelNames[int(preds.argmax(axis=1))]
        cv2.putText(gray, str(label), text_Location, ftext_font,
                    fontScale, text_color, text_thickness, cv2.LINE_AA)
    out.write(gray)

    cv2.imshow("Press q to Exit", gray)
    key = cv2.waitKey(1)
    if key == ord('q'):
        out.release()
        break
    fps.update()
# except:
#         break
video.release()
out.release()
cv2.destroyAllWindows()
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))