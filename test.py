'''
Authors:
1. Rupasmita Devi 
2. Salil kulkarni
'''

import cv2
import numpy as np
from keras.models import load_model

IMPORT_XML = 'haarcascade_frontalface_default.xml'
MODEL = "./apna3.h5"
WAIT_TIME = 10
webcam = cv2.VideoCapture(0)
TEST_IMAGE = 'selfno.jpg'
def plot_box(image, x, y, width, height, color, text):
    font_style = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(image,(x,y),(x+width,y+height),color,3)
    cv2.putText(image, text, (x, y), font_style, 1, color, 2, cv2.LINE_AA)

def start_capturing(model, object_detection, webcam):

    while True:
        '''
        Un - Coment the block below (Line 27-29) to use Webcame
        '''
        # if not (webcam.isOpened()):
        #     print("Could not open webcam on this device!")
        # (_, image) = webcam.read()

        image = cv2.imread(TEST_IMAGE) # This line predicts on a static image
        image = cv2.flip(image, 1)
        heads = object_detection.detectMultiScale(image)
        # print(heads)

        for h in heads:
            (x, y, width, height) = h
            head = image[y:y+height, x:x+width]
            
            rescaled_image=cv2.resize(head,(200,200))

            # cv2.imshow('preview',rescaled_image)
            processed_image=rescaled_image/255.0
            processed_image = processed_image.reshape([1,200,200,3])
            
            pred=model.predict(processed_image)

            # print(pred)
            
            label=np.argmax(pred,axis=1)[0]
            labels={0:'No Mask',1:'Mask'}
            text = labels[label]
            colors={0:(0,0,255),1:(0,255,0)}
            color = colors[label]
            plot_box(image, x, y, width, height, color, text)
        
        cv2.imshow('HELLO', image)
        key = cv2.waitKey(WAIT_TIME)


def main():
    model=load_model(MODEL)
    object_detection = cv2.CascadeClassifier(IMPORT_XML)
    start_capturing(model, object_detection, webcam)
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 