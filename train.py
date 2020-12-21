from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense,Dropout
from keras.models import Model, load_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import imutils
import cv2
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
        im1 = cv2.flip(im, 0)
        rgb_image_1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
        rgb_image_1 = rgb_image_1.reshape([image_height, image_width, 3])
        processed_data_1 = np.array(rgb_image_1) / 255.0
  
        X.append(processed_data)
        X.append(processed_data_1)
        if image_label.lower().startswith("no"):
            y.append(0)
            y.append(0)
        else:
            y.append(1)
            y.append(1)
        i = i + 1
    X = (np.array(X)).reshape(2*i, 150, 150, 3)
    y = (np.array(y)).reshape(2*i,1)
    # print("___________________Features__________________\n")
    # print(X)
    # print("No. of Features: ", len(X))
    # print("___________________Labels__________________\n")
    # print("No. of Labels: ", len(y))
    # print(y)

    return X, y

training_data = "./train_images"

X, y = _ImageDataGenerator(training_data, batch_size = 10, image_height=200, image_width=200)

validation_data = "./train_images"

X, _y = _ImageDataGenerator(validation_data, batch_size=10, image_height = 200, image_width=200)

model = keras.Sequential()
model.add(Conv2D(100, (3,3), activation='relu', input_shape=(200, 200, 3)))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.15))
model.add(Conv2D(100, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.20))
model.add(Flatten())
model.add(Dense(50, activation = 'relu'))
model.add(Dense(2, activation ='softmax'))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

batch_size = 10
epochs = 5
history = model.fit(X, y, batch_size=batch_size,
                    epochs=epochs, validation_split=0.1)
model.save("apna.h5")

tf.keras.utils.plot_model(model, to_file="img1.png")
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

                            