from __future__ import print_function

import matplotlib.image as mpimg
import os.path
import numpy as np
import os
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from glob import glob
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam , SGD
from keras.applications.vgg16 import VGG16
from PIL import Image

size_image = 64
# cutoff = 1000
# numbers = 39

def getSum(path):
    sum = 0
    for d in os.listdir(path):
        num = len(os.listdir(os.path.join(path, d)))
        sum += num
    return sum
#lay tong so file train :
sum = getSum("/mnt/volume/ggcluster/spark-2.1.1-bin-hadoop2.7/thangbk2209/food-101/images")
# print(sum)
#
Y_all = np.zeros(sum)
X_all = np.zeros((sum, size_image, size_image, 3), dtype='float64')
# Y_all = np.zeros(cutoff * numbers)
# X_all = np.zeros((cutoff * numbers, size_image, size_image, 3), dtype='float64')

def listdirs(path):
    count_X = 0
    label = 0

    for d in os.listdir(path):
        if os.path.isdir(os.path.join(path, d)):
            local_count = 0
            print(d)
            for img in os.listdir(os.path.join(path, d)):
                if img.endswith("jpg"):
                    # np.resize()
                    # image = mpimg.imread(os.path.join(os.path.join(path, d), img))
                    # image.resize(size_image, size_image, 3)


                    image = Image.open(os.path.join(os.path.join(path, d), img))
                    image = image.resize((size_image, size_image), Image.ANTIALIAS)
                    image = np.array(image)

                    print('%d ...' % count_X, end='\r')
                    # print image.shape
                    # print os.path.join(os.path.join(path, d), img)
                    X_all[count_X]= image
                    Y_all[count_X] = label
                    count_X +=1
                    local_count += 1
                    # if local_count >= cutoff:
                    #     break
        label +=1
        # if(label >= numbers ):
        #     break



#
if __name__ == '__main__':
    listdirs("/mnt/volume/ggcluster/spark-2.1.1-bin-hadoop2.7/thangbk2209/food-101/images")
    sum_laber = len(os.listdir("/mnt/volume/ggcluster/spark-2.1.1-bin-hadoop2.7/thangbk2209/food-101/images"))

    Y_all = np_utils.to_categorical(Y_all, sum_laber)
    X_all /= 255.0
    # #print(Y_all)

    X_train,X_test,Y_train,Y_test = train_test_split(X_all, Y_all, test_size=0.1, random_state=42)
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)
    print(X_all)
    #



    # model CNN
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, activation='relu', input_shape=(size_image, size_image,3)))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64,(3,3)))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    #
    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, (3, 3)))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, (3, 3)))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, (3, 3)))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(sum_laber, activation='softmax'))

    # model = VGG16(include_top=True, input_shape=(size_image, size_image, 3),
    #               classes=sum_laber, weights=None, pooling='max')

    model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=32, nb_epoch=1, verbose=1, shuffle=True,
              validation_data=(X_test, Y_test))

    score = model.evaluate(X_test, Y_test, verbose=1)
    # print score








