from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense,Dropout
from keras.models import Model, load_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import imutils
import cv2
import numpy as np
import random
import os

def _ImageDataGenerator(directory_name, batch_size, image_height, image_width):

    file_list = os.listdir(directory_name)
    files = dict(zip(map(lambda x: x.split('.')[0], file_list), file_list))
    used_files = []
    count = len(file_list)

    iterations = int(np.floor(count / batch_size))

    X = np.zeros((batch_size, image_height, image_width, 3), dtype=np.float32)
    y = np.zeros((batch_size, image_height, image_width, 3), dtype=np.float32)

    for itr in range(iterations):
        for i in range(batch_size):
            if len(files)==0:
                break
            random_image_label = random.choice(list(files.keys()))
            random_image_file = files[random_image_label]

            # We've used this image now, so we can't repeat it in this iteration
            used_files.append(files.pop(random_image_label))

            # We have to scale the input pixel values to the range [0, 1] for
            # Keras so we divide by 255 since the image is 8-bit RGB
            raw_data = cv2.imread(os.path.join(directory_name, random_image_file))
            rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
            rgb_data = rgb_data.reshape([-1, image_height, image_width, 3])
            processed_data = np.array(rgb_data) / 255.0
            
            X[i] = processed_data

            if random_image_label.lower().startswith("no"):
                y[i] = 0
            else:
                y[i] = 1

    return X, y


model =Sequential([
    Conv2D(100, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),
    
    Conv2D(100, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dropout(0.5),
    Dense(50, activation='relu'),
    Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

TRAINING_DIR = "./train"
# train_datagen = ImageDataGenerator(rescale=1.0/255,
#                                    rotation_range=40,
#                                    width_shift_range=0.2,
#                                    height_shift_range=0.2,
#                                    shear_range=0.2,
#                                    zoom_range=0.2,
#                                    horizontal_flip=True,
#                                    fill_mode='nearest')
train_datagen = _ImageDataGenerator(TRAINING_DIR, batch_size = 10, image_height=150, image_width=150)
# train_generator = train_datagen.flow_from_directory(TRAINING_DIR, 
#                                                     batch_size=10, 
#                                                     target_size=(150, 150))
VALIDATION_DIR = "./test"
# validation_datagen = ImageDataGenerator(rescale=1.0/255)

validation_datagen = _ImageDataGenerator(VALIDATION_DIR, batch_size=10, image_height = 150, image_width=150)

# validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, 
#                                                          batch_size=10, 
#                                                          target_size=(150, 150))
checkpoint = ModelCheckpoint('model2.h5',monitor='val_loss',verbose=0,save_best_only=False,mode='auto')


# history = model.fit_generator(train_generator,
#                               epochs=10,
#                               validation_data=validation_generator,
#                               callbacks=[checkpoint])

history = model.fit_generator(train_datagen,
                              epochs=10,
                              validation_data=validation_datagen,
                              callbacks=[checkpoint],
                              use_multiprocessing=True)