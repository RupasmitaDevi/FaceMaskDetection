import cv2
import sys
import os
import random

directory_name = sys.argv[1]
image_list = os.listdir(directory_name)
images_to_read = dict(zip(map(lambda x: x.split('.')[0], image_list), image_list))
images_already_read = []

while len(images_already_read) != len(images_to_read):
    image_label = random.choice(list(images_to_read.keys()))
    image_file = images_to_read[image_label]
    images_already_read.append(images_to_read.pop(image_label)) 
    try:
        raw_data = cv2.imread(os.path.join(directory_name, image_file))
    except Exception as e:
        if os.path.exists(image_label + '.jpg'):
            os.remove(image_label + '.jpg')

print("Data Cleaning Successful")
