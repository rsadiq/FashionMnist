# USAGE
# python fashion_mnist.py

# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from sklearn.metrics import classification_report
from keras.optimizers import SGD, Adam
from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras import backend as K
# import the necessary packages
from keras.engine import Input, Model
from keras.layers import merge, Add
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D,Convolution2D
from keras.layers.convolutional import MaxPooling2D,AveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten,Lambda
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from imutils import build_montages
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



NUM_EPOCHS = 100
INIT_LR = 1e-1
BS = 32
classes=10

def zero_pad_channels(x, pad=0):
    """
    Function for Lambda layer
    """
    pattern = [[0, 0], [0, 0], [0, 0], [pad - pad // 2, pad // 2]]
    return tf.pad(x, pattern)


def residual_block(x, nb_filters=16, subsample_factor=1):
    prev_nb_channels = K.int_shape(x)[3]

    if subsample_factor > 1:
        subsample = (subsample_factor, subsample_factor)
        # shortcut: subsample + zero-pad channel dim
        shortcut = AveragePooling2D(pool_size=subsample, dim_ordering='tf')(x)
    else:
        subsample = (1, 1)
        # shortcut: identity
        shortcut = x

    if nb_filters > prev_nb_channels:
        shortcut = Lambda(zero_pad_channels,
                          arguments={'pad': nb_filters - prev_nb_channels})(shortcut)

    y = BatchNormalization(axis=3)(x)
    y = Activation('relu')(y)
    y = Convolution2D(nb_filters, 3, 3, subsample=subsample,
                      init='he_normal', border_mode='same', dim_ordering='tf')(y)
    y = BatchNormalization(axis=3)(y)
    y = Activation('relu')(y)
    y = Dropout(0.5)(y)
    y = Convolution2D(nb_filters, 3, 3, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(y)

    out = Add()([y, shortcut])

    return out

img_rows, img_cols = 28, 28
img_channels = 1

blocks_per_group = 4
widening_factor = 10

inputs = Input(shape=(img_rows, img_cols, img_channels))

x = Convolution2D(16, 3, 3,
                  init='he_normal', border_mode='same', dim_ordering='tf')(inputs)

for i in range(0, blocks_per_group):
    nb_filters = 16 * widening_factor
    x = residual_block(x, nb_filters=nb_filters, subsample_factor=1)

for i in range(0, blocks_per_group):
    nb_filters = 32 * widening_factor
    if i == 0:
        subsample_factor = 2
    else:
        subsample_factor = 1
    x = residual_block(x, nb_filters=nb_filters, subsample_factor=subsample_factor)

for i in range(0, blocks_per_group):
    nb_filters = 64 * widening_factor
    if i == 0:
        subsample_factor = 2
    else:
        subsample_factor = 1
    x = residual_block(x, nb_filters=nb_filters, subsample_factor=subsample_factor)

x = BatchNormalization(axis=3)(x)
x = Activation('relu')(x)
x = AveragePooling2D(pool_size=(4, 4), strides=None, border_mode='valid', dim_ordering='tf')(x)
x = Flatten()(x)

predictions = Dense(classes, activation='softmax')(x)

wide_model = Model(input=inputs, output=predictions)
sgd = SGD(lr=0.1, decay=5e-4, momentum=0.9, nesterov=True)

wide_model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# grab the Fashion MNIST dataset (if this is your first time running
# this the dataset will be automatically downloaded)
print("[INFO] loading Fashion MNIST...")
((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()

# if we are using "channels first" ordering, then reshape the design
# matrix such that the matrix is:
# 	num_samples x depth x rows x columns
if K.image_data_format() == "channels_first":
    trainX = trainX.reshape((trainX.shape[0], 1, 28, 28))
    testX = testX.reshape((testX.shape[0], 1, 28, 28))

# otherwise, we are using "channels last" ordering, so the design
# matrix shape should be: num_samples x rows x columns x depth
else:
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))


# one-hot encode the training and testing labels
trainY = np_utils.to_categorical(trainY, 10)
testY = np_utils.to_categorical(testY, 10)

# initialize the label names
labelNames = ["top", "trouser", "pullover", "dress", "coat",
              "sandal", "shirt", "sneaker", "bag", "ankle boot"]

# Data Augmentation
train_generator = ImageDataGenerator(rescale=1 / 255,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     rotation_range=20,width_shift_range=0.2,
                                     height_shift_range=0.2, shear_range=0.15,
                                     fill_mode="nearest"
                                     )

test_generator = ImageDataGenerator(rescale=1 / 255)

train_generator = train_generator.flow(np.array(trainX),
                                       trainY,
                                       batch_size=BS,
                                       shuffle=False)

test_generator = test_generator.flow(np.array(testX),
                                     testY,
                                     batch_size=BS,
                                     shuffle=False)
testX = testX.astype("float32") / 255.0

# initialize the optimizer and model
opt = Adam(lr=INIT_LR)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=3, min_lr=0.000001, verbose=1, cooldown=1)


# train the network
print("[INFO] training model...")
# print(len(trainX)/BS)
# H = model.fit_generator(train_generator.flow(trainX, trainY, batch_size=BS),
# 	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
# 	epochs=NUM_EPOCHS,callbacks=[reduce_lr],verbose=1)
H_Aug = wide_model.fit_generator(train_generator,steps_per_epoch = int(len(trainX)/BS),epochs = NUM_EPOCHS,
                            shuffle=True,validation_data=(testX,testY),callbacks=[reduce_lr],verbose=2)
# make predictions on the test set
# preds = model.predict(testX)
# print("Test_Accuracy(after augmentation): {:.2f}%".format(model.evaluate_generator(test_generator, steps = len(testX), verbose = 2)[1]*100))
# # show a nicely formatted classification report
# print("[INFO] evaluating network...")
# print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1),
#                             target_names=labelNames))
# model.save("Aug_BN_model1.h5")
