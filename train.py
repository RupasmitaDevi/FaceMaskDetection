'''
Authors:
1. Rupasmita Devi 
2. Salil kulkarni
'''

import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.linear_model import LogisticRegression
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from PIL import Image
import requests
import cv2
import imutils
import numpy as np
import random
import os

def _ImageDataGenerator(directory_name, batch_size, image_height, image_width):

    X = []
    y = []
    image_list = os.listdir(directory_name)
    images = dict(zip(map(lambda x: x.split('.')[0], image_list), image_list))
    images_already_read = []
    size = (200, 200)

    i = 0
    while(True):
        if len(images)==0:
            break
        image_label = random.choice(list(images.keys()))
        image_file = images[image_label]

        # remove the image that has been once read
        images_already_read.append(images.pop(image_label))

        # read the image from the directory
        im = cv2.imread(os.path.join(directory_name, image_file))
        try:
            # rescale it to (200, 200)
            im = cv2.resize(im, size, interpolation = cv2.INTER_AREA)
        except Exception as e:
            continue
        # print("................HERE.................\n")
        # print("shape: ",im.shape)
        
        rgb_image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        rgb_image = rgb_image.reshape([image_height, image_width, 3])
        processed_data = np.array(rgb_image) / 255.0
        
        # Augmentation ----- Flip
        # im1 = cv2.flip(im, 0)
        # rgb_image_1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
        # rgb_image_1 = rgb_image_1.reshape([image_height, image_width, 3])
        # processed_data_1 = np.array(rgb_image_1) / 255.0
  
        X.append(processed_data)
        # X.append(processed_data_1)
        if image_label.lower().startswith("no"):
            y.append(0)
            # y.append(0)
        else:
            y.append(1)
            # y.append(1)
        i = i + 1
    X = (np.array(X)).reshape(i, 200, 200, 3)
    y = (np.array(y)).reshape(i,1)


    return X, y

def logistic(x_train, y_train, x_test, y_test):
	
	x_train = x_train[:, :, 0, 0]

	# logistic regression
	logistic_model = LogisticRegression()
	logistic_model = logistic_model.fit(x_train, y_train)

	preds = logistic_model.predict(x_train)
	print(classification_report(preds, y_train))
	print(confusion_matrix(preds, y_train))

	x_test = x_test[:, :, 0, 0]
	preds = logistic_model.predict(x_test)
	print(classification_report(preds, y_test))
	print(confusion_matrix(preds, y_test))

def createmodel(X, y, x_test, y_test, batch_size = 10, image_height=200, image_width=200):

    model = keras.Sequential()
    model.add(Conv2D(100, (3,3), activation='relu', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D(2,2))
    #model.add(Dropout(0.20))
    model.add(Conv2D(100, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    # model.add(Dropout(0.15))
    model.add(Flatten())
    model.add(Dropout(0.45))
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(2, activation ='softmax'))
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    batch_size = 10
    epochs = 5
    history = model.fit(X, y, batch_size=batch_size,
                        epochs=epochs, validation_split=0.1)
    model.save("model2.h5")

    # tf.keras.utils.plot_model(model, to_file="img1.png")

    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss'); plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    preds = model.predict(X)
    y_pred = np.argmax(preds, axis=1)
    y_train1 = np.argmax(y, axis=1)
    print(classification_report(y_train1, y_pred))
    print(confusion_matrix(y_train1, y_pred))

    preds = model.predict(x_test)
    y_pred = np.argmax(preds, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    print(classification_report(y_test1, y_pred))
    print(confusion_matrix(y_test1, y_pred))

def create_baseline_cnn_model(X, y, x_test, y_test, batch_size = 10, image_height=200, image_width=200):

    model = keras.Sequential()
    model.add(Conv2D(100, (3,3), activation='relu', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D(2,2))
    model.add(Flatten())
    model.add(Dense(2, activation ='softmax'))
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    batch_size = 10
    epochs = 5
    history = model.fit(X, y, batch_size=batch_size,
                        epochs=epochs, validation_split=0.1)
    model.save("model_baseline_cnn_2.h5")

    # tf.keras.utils.plot_model(model, to_file="img1.png")

    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('baseline cnn model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('baseline cnn model loss')
    plt.ylabel('loss'); plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    preds = model.predict(X)
    y_pred = np.argmax(preds, axis=1)
    y_train1 = np.argmax(y, axis=1)
    print(classification_report(y_train1, y_pred))
    print(confusion_matrix(y_train1, y_pred))

    preds = model.predict(x_test)
    y_pred = np.argmax(preds, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    print(classification_report(y_test1, y_pred))
    print(confusion_matrix(y_test1, y_pred))


def main():
    training_data = "./train_images/train_set"

    X, y = _ImageDataGenerator(training_data, batch_size = 10, image_height=200, image_width=200)

    validation_data = "./train_images/train_set"

    _X, _y = _ImageDataGenerator(validation_data, batch_size=10, image_height = 200, image_width=200)

    # baseline model --- Logistic regression
    logistic(X, y, _X, _y)

    create_baseline_cnn_model(X, y, _X, _y, 10, 200, 200)

    # keras model
    createmodel(X, y, _X, _y, 10, 200, 200)

if __name__ == "__main__":
    main()          