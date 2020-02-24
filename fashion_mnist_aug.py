
from sklearn.metrics import classification_report
from keras.optimizers import SGD, Adam
from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
# import the necessary packages
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
import math
# from imutils import build_montages
import matplotlib.pyplot as plt
import numpy as np
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--test_train",choices=['train','test'],default='test',
                help="Train or Evaluate the pretrained model")
ap.add_argument("-w", "--model", default='deep',choices=['base','deep'],
                help="Which model to use for training and testing")
args = vars(ap.parse_args())

import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


from My_Fash_Models import Models as CNNmodels

def step_decay(epoch):
   initial_lrate = 1e-3
   drop = 0.2
   epochs_drop = 20.0
   lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
   return lrate

# initialize the number of epochs to train for, base learning rate,
# and batch size

NUM_EPOCHS = 50
INIT_LR = 1e-3
BS = 128

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
                                    rotation_range = 10,
                                    zoom_range = 0.1,
                                    width_shift_range = 0.1,
                                    height_shift_range = 0.1,
                                    horizontal_flip=True,
                                     )

test_generator = ImageDataGenerator(rescale=1 / 255)

train_generator = train_generator.flow(np.array(trainX),
                                       trainY,
                                       batch_size=BS,
                                       shuffle=True)

test_generator = test_generator.flow(np.array(testX),
                                     testY,
                                     batch_size=BS,
                                     shuffle=False)
testX = testX.astype("float32") / 255.0

# initialize the optimizer and model
# opt = Adam(lr=INIT_LR)
if args['test_train'] == 'train':

    opt = SGD(lr = INIT_LR,momentum=0.9,decay=0)
    lrate_decay = LearningRateScheduler(step_decay)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=3, min_lr=1e-10, verbose=1, cooldown=2)

    # opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)
    model = CNNmodels.Deep_with_BN(width=28, height=28, depth=1, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    print(model.summary())

    # train the network
    print("[INFO] training model...")
    # print(len(trainX)/BS)
    # H = model.fit_generator(train_generator.flow(trainX, trainY, batch_size=BS),
    # 	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
    # 	epochs=NUM_EPOCHS,callbacks=[reduce_lr],verbose=1)
    H_Aug = model.fit_generator(train_generator,steps_per_epoch = int(len(trainX)/BS),epochs = NUM_EPOCHS,
                                shuffle=True,validation_data=(testX,testY),callbacks=[lrate_decay],verbose=2)
    # make predictions on the test set
    model.save("Aug_BN_model1.h5")
     # plot the training loss and accuracy
    # N = NUM_EPOCHS
    # plt.style.use("ggplot")
    # plt.figure()
    # plt.plot(np.arange(0, N), H_Aug.history["loss"], label="train_loss")
    # plt.plot(np.arange(0, N), H_Aug.history["val_loss"], label="val_loss")
    # plt.plot(np.arange(0, N), H_Aug.history["acc"], label="train_acc")
    # plt.plot(np.arange(0, N), H_Aug.history["val_acc"], label="val_acc")
    # plt.title("Training Loss and Accuracy on Dataset")
    # plt.xlabel("Epoch #")
    # plt.ylabel("Loss/Accuracy")
    # plt.legend(loc="lower left")
    # plt.savefig("loss_Aug_plot2.png")
    # plt.show()
elif args['test_train'] == 'test':
    print('Loading Model')
    model = load_model('BN_model1.h5')

    preds = model.predict(testX)
    print("Test_Accuracy(after augmentation): {:.2f}%".format(model.evaluate_generator(test_generator, steps = len(testX), verbose = 2)[1]*100))
    # show a nicely formatted classification report
    print("[INFO] evaluating network...")
    print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1),
                                target_names=labelNames))

