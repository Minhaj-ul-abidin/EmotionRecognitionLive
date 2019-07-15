from keras.models import load_model
from .settings import BASE_DIR

import tensorflow as tf
import os
import cv2

face_cascade = cv2.CascadeClassifier(os.path.join(BASE_DIR, 'emotions/Utilities/haarcascade_frontalface_default.xml'))

model = load_model(os.path.join(BASE_DIR, 'emotions/Utilities/model.h5'))
graph = tf.get_default_graph()