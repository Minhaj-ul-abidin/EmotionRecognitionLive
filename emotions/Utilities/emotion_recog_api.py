import json
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import SGD
import os
from emotionsApi import model , graph, face_cascade
IMG_SIZE = 48
classes = {
     0:'Angry',
     1:'Disgust',
     2:'Fear',
     3:'Happy',
     4:'Sad',
     5:'Surprise',
     6:'Neutral'
}

def load_img_into_numpy(input_img, show=False):
    imgs = []
    # print(input_img)

    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    input_img_resize = cv2.resize(input_img, (IMG_SIZE, IMG_SIZE))

    imgs.append(input_img_resize)
    img_tensor = np.array(imgs, dtype='float32')
    print(img_tensor.shape)
    img_tensor /= 255  # (height, width, channels)
    img_tensor = np.expand_dims(
        img_tensor, axis=1
    )  # (1, channels,height, width,), # imshow expects values in the range [0, 1]
    print(img_tensor.shape, "_______________________-----")
    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor


class Object(object):
    def __init__(self , **kwargs):
        self.__dict__.update(kwargs)

    def toJSON(self):
        return json.dumps(self.__dict__)




def get_objects(image, threshold=0.5):
    print("HELLLO")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if(len(faces)):
        print("_+++++++++++++++++",*faces[0])
        x,y,w,h = faces[0]
        
        image = image[y:y+h, x:x+w]
        np_arr = load_img_into_numpy(image)
        lr = 0.01
        sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
        # model = load_model(os.path.join( os.path.abspath(os.path.dirname( __file__)), 'model.h5'))
        with graph.as_default():
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            pred = model.predict(np_arr)
        emotion_dict = {}
        for c in range(0,len(classes.values())):
            emotion = classes[c]
            emotion_dict[emotion] = str(pred[0][c])
        print(pred)
        emotions = Object(**emotion_dict)
        item = [emotions]

        return json.dumps([em.__dict__ for em in item])
    else:
        return json.dumps({'msg': "No Face Found"})