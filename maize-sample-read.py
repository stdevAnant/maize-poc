import keras.models
import pickle
from keras.models import Sequential
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import numpy as np
from cv2 import *
from flask import Flask,request
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
model_str = ""
pklfile= "C:/Users/hp/Downloads/WinPython/settings/.spyder-py3/modelweights-gcp.pkl"
WIDTH = 256
HEIGHT = 256
DEPTH = 3
DEFAULT_IMAGE_SIZE = tuple((256, 256))

filename = 'plant_disease_label_transform-gcp.pkl'
image_labels = pickle.load(open(filename, 'rb'))
n_classes = len(image_labels.classes_)
print(n_classes)
def mymodel():

    inputShape = (HEIGHT, WIDTH, DEPTH)
    chanDim = -1
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(n_classes))
    model.add(Activation("softmax"))
    return model

f= open(pklfile, 'rb')     #Python 3                 
weigh= pickle.load(f);                
f.close();
restoredmodel= mymodel()
restoredmodel.set_weights(weigh)

# model = keras.models.load_model(pklfile)

restoredmodel.summary()
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, DEFAULT_IMAGE_SIZE)   
            return img_to_array(image)
        else:
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None
    


def predict_disease(image_path):
    image_array = convert_image_to_array(image_path)
    np_image = np.array(image_array, dtype=np.float16) / 225.0
    np_image = np.expand_dims(np_image,0)
    result = restoredmodel.predict_classes(np_image)
    # print((image_labels.classes_[result][0]))
    return image_labels.classes_[result][0]


@app.route('/')
def hello_world():
    baseImage = request.args.get('imgSrc')

    return predict_disease(baseImage)

if __name__ == "__main__":
    app.run()
# predict_disease('C:/Users/hp/Downloads/maize/test/maize-rust.jpg')