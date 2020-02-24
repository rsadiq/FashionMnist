from efficientnet.keras import EfficientNetB0

# from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Input
from keras.models import Model
from keras.models import load_model

# from keras import optimizers
from keras.datasets import fashion_mnist
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras.optimizers import SGD, Adam,rmsprop
from keras import backend as K
import cv2
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

# import cv2
import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from pathlib import Path
# %matplotlib inline
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--test_train",choices=['train','test'],default='test',
                help="Train or Evaluate the pretrained model")
ap.add_argument("-w", "--model", default='deep',choices=['base','deep'],
                help="Which model to use for training and testing")
args = vars(ap.parse_args())

##############################################################################################


NUM_EPOCHS = 1
INIT_LR = 1e-3
BS = 128



print("[INFO] loading Fashion MNIST...")
((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
# Normilize data
trainX = trainX.astype('float32')
testX = testX.astype('float32')
trainX3 = np.full((trainX.shape[0], 28, 28, 3), 0.0)
testX3 = np.full((testX.shape[0], 28, 28, 3), 0.0)

for i, s in enumerate(trainX):
    trainX3[i] = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB)
for i, s in enumerate(testX):
    testX3[i] = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB)

print(np.shape(trainX3))
# if we are using "channels first" ordering, then reshape the design
# matrix such that the matrix is:
# 	num_samples x depth x rows x columns
if K.image_data_format() == "channels_first":
    trainX3 = trainX3.reshape((trainX3.shape[0], 3, 28, 28))
    testX3 = testX3.reshape((testX3.shape[0], 3, 28, 28))

# otherwise, we are using "channels last" ordering, so the design
# matrix shape should be: num_samples x rows x columns x depth
else:
    trainX3 = trainX3.reshape((trainX3.shape[0], 28, 28, 3))
    testX3 = testX3.reshape((testX3.shape[0], 28, 28, 3))


# one-hot encode the training and testing labels
trainY = np_utils.to_categorical(trainY, 10)
testY = np_utils.to_categorical(testY, 10)

# initialize the label names
labelNames = ["top", "trouser", "pullover", "dress", "coat",
              "sandal", "shirt", "sneaker", "bag", "ankle boot"]

# Data Augmentation
train_generator = ImageDataGenerator(rescale=1 / 255,
                                     horizontal_flip=True,
                                     )

test_generator = ImageDataGenerator(rescale=1 / 255)

train_generator = train_generator.flow(np.array(trainX3),
                                       trainY,
                                       batch_size=BS,
                                       shuffle=False)

test_generator = test_generator.flow(np.array(testX3),
                                     testY,
                                     batch_size=BS,
                                     shuffle=False)
testX3 = testX3.astype("float32") / 255.0
if args['test_train'] == 'train':

    opt = Adam(lr=INIT_LR)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=3, min_lr=1e-10, verbose=1, cooldown=2)

    model_checkpoint = ModelCheckpoint('EFNB3Weight.h5',monitor='val_loss',
                                save_best_only=True,period=3)
    input_tensor = Input(shape=(28, 28, 3))

    model = EfficientNetB0(weights=None,
                            include_top=False,
                            input_tensor = input_tensor)

    x = model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(units = 10, activation="softmax")(x)
    model_f = Model(input = model.input, output = predictions)
    model_f.compile(Adam(lr=INIT_LR),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    print(model.summary())

    # train the network
    print("[INFO] training model...")
    # print(len(trainX)/BS)
    # H = model.fit_generator(train_generator.flow(trainX, trainY, batch_size=BS),
    # 	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
    # 	epochs=NUM_EPOCHS,callbacks=[reduce_lr],verbose=1)
    H_Aug = model_f.fit_generator(train_generator,steps_per_epoch = int(len(trainX)/BS),epochs = NUM_EPOCHS,
                                shuffle=True,validation_data=(testX3,testY),callbacks=[reduce_lr,model_checkpoint],verbose=2)
    # make predictions on the test set
    preds = model_f.predict(testX3)
    model.save("EFNetB0.h5")

elif args['test_train'] == 'test':
    print('Loading Model')
    model = load_model('EFNB0Weight.h5')
    print('Predicting')
    preds = model.predict(testX3)
    # print("Test_Accuracy(after augmentation): {:.2f}%".format(model.evaluate_generator(test_generator, steps = len(testX3), verbose = 2)[1]*100))
    # show a nicely formatted classification report
    print("[INFO] evaluating network...")
    print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1),
                                target_names=labelNames))
