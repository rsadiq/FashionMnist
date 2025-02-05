# USAGE
# python fashion_mnist.py

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from sklearn.metrics import classification_report
from keras.optimizers import SGD,Adam
from keras.datasets import fashion_mnist
from keras.models import load_model

import argparse
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau,LearningRateScheduler
import  math
import tensorflow as tf

from tensorflow.compat.v1 import InteractiveSession
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from My_Fash_Models import Models as CNNmodels

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--test_train",choices=['train','test'],default='test',
                help="Train or Evaluate the pretrained model")
ap.add_argument("-c", "--model", default='deep',choices=['base','deep'],
                help="Which model to use for training and testing")
args = vars(ap.parse_args())

def step_decay(epoch):
   initial_lrate = 1e-3
   drop = 0.1
   epochs_drop = 20.0
   lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
   return lrate


NUM_EPOCHS = 100
INIT_LR = 1e-3
BS = 256

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
 
# scale data to the range of [0, 1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# one-hot encode the training and testing labels
trainY = np_utils.to_categorical(trainY, 10)
testY = np_utils.to_categorical(testY, 10)

# initialize the label names
labelNames = ["top", "trouser", "pullover", "dress", "coat",
	"sandal", "shirt", "sneaker", "bag", "ankle boot"]

# initialize the optimizer and model

# opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)
if args['test_train'] == 'train':
		opt = Adam(lr=INIT_LR)
		# opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)
		reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
									  patience=3, min_lr=0.000001, verbose=1, cooldown=2)
		lrate_decay = LearningRateScheduler(step_decay)

		model = CNNmodels.baseline_one_layer(width=28, height=28, depth=1, classes=10)
		model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

		# train the network
		print("[INFO] training model...")
		H = model.fit(trainX, trainY, verbose=2,
			validation_data=(testX, testY),
			batch_size=BS, epochs=NUM_EPOCHS,callbacks=[reduce_lr])

		model.save("Baseline1.h5")
		# plot the training loss and accuracy
		# N = NUM_EPOCHS
		# plt.style.use("ggplot")
		# plt.figure()
		# plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
		# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
		# plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
		# plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
		# plt.title("Training Loss and Accuracy on Dataset")
		# plt.xlabel("Epoch #")
		# plt.ylabel("Loss/Accuracy")
		# plt.legend(loc="lower left")
		# plt.savefig("plot.png")

elif args['test_train'] == 'test':
	print('Loading Model')
	if args["model"] == 'deep':
		model = load_model('Deep_model.h5')
	elif args["model"] == 'base':
		model = load_model('Baseline1.h5')

		# make predictions on the test set
	preds = model.predict(testX)

	# show a nicely formatted classification report
	print("[INFO] evaluating network...")
	print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1),
		target_names=labelNames))


#
# # initialize our list of output images
# images = []
#
# # randomly select a few testing fashion items
# for i in np.random.choice(np.arange(0, len(testY)), size=(16,)):
# 	# classify the clothing
# 	probs = model.predict(testX[np.newaxis, i])
# 	prediction = probs.argmax(axis=1)
# 	label = labelNames[prediction[0]]
#
# 	# extract the image from the testData if using "channels_first"
# 	# ordering
# 	if K.image_data_format() == "channels_first":
# 		image = (testX[i][0] * 255).astype("uint8")
#
# 	# otherwise we are using "channels_last" ordering
# 	else:
# 		image = (testX[i] * 255).astype("uint8")
#
# 	# initialize the text label color as green (correct)
# 	color = (0, 255, 0)
#
# 	# otherwise, the class label prediction is incorrect
# 	if prediction[0] != np.argmax(testY[i]):
# 		color = (0, 0, 255)
#
# 	# merge the channels into one image and resize the image from
# 	# 28x28 to 96x96 so we can better see it and then draw the
# 	# predicted label on the image
# 	image = cv2.merge([image] * 3)
# 	image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
# 	cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
# 		color, 2)
#
# 	# add the image to our list of output images
# 	images.append(image)
#
# # construct the montage for the images
# montage = build_montages(images, (96, 96), (4, 4))[0]
#
# # show the output montage
# cv2.imshow("Fashion MNIST", montage)
# cv2.waitKey(0)