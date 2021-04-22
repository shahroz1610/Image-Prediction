import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import pickle
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Dropout,Activation,Dense
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time
from tensorflow import keras

NAME = "Cats-vs-Dogs-CNN{}".format(int(time.time()))

DATADIR = "C:/Users/mshah/Desktop/Project/Machine Learning/DogsVsCats/PetImages"
Categories = ["Dog" , "Cat"]

for category in Categories:
    path = os.path.join(DATADIR,category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        #plt.imshow(img_array,cmap="gray")
        plt.show()
        break
    break
IMG_SIZE = 50

new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))

#plt.imshow(new_array,cmap='gray')
#plt.show()

training_data = []
create_data = False
def create_training_data():
    for category in Categories:
        path = os.path.join(DATADIR,category)
        class_num = Categories.index(category)

        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path,img) , cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass

if create_data == True:
    create_training_data()
    X,y = [] , []
    for features,label in training_data:
        X.append(features)
        y.append(label)


    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(y)

    pickle_out = open('X.pickle','wb')
    pickle.dump(X,pickle_out)
    pickle_out.close()

    pickle_out = open('y.pickle','wb')
    pickle.dump(y,pickle_out)
    pickle_out.close()

pickle_in = open('X.pickle','rb')
X = pickle.load(pickle_in)


pickle_in = open('y.pickle','rb')
y = pickle.load(pickle_in)

X = X/255.0
pre_trained = True
if pre_trained == False:
    model = Sequential()

    model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

    model.compile(loss='binary_crossentropy' , optimizer='adam' , metrics=['accuracy'])

    model.fit(X,y,batch_size=32,epochs=20,validation_split=0.3)

    model.save('64x3-CNN.model')
else:
    model = keras.models.load_model('64x3-CNN.model')

test_dir = 'C:/Users/mshah/Desktop/Project/Machine Learning/DogsVsCats/Test data/'

test_img = 'chotu_leo.jpeg'
img_dir = os.path.join(test_dir,test_img)

test_img_array = cv2.imread('C:/Users/mshah/Desktop/Project/Machine Learning/DogsVsCats/Test data/jasper.jpeg',cv2.IMREAD_GRAYSCALE)
# plt.imshow(test_img_array)
test_img_array_resized = cv2.resize(test_img_array,(IMG_SIZE,IMG_SIZE))
test = test_img_array_resized.reshape(-1,50,50,1)
# plt.imshow(test_img_array_resized)
# print(test_img_array_resized.shape)
prediction = model.predict(test)
print(prediction)
if prediction[0][0] == 1:
    print("It is a picture of cat.")
else:
    print("It is a picture of dog.")
