import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from PIL import Image
from glob import glob
import csv
import seaborn as sn
import pandas as pd

#from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from helper import *

from keras.preprocessing.image import ImageDataGenerator





from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization

def createMyModel():
    
    model = Sequential()

    n_filters = 16
    # this applies n_filters convolution filters of size 5x5 resp. 3x3 each in the 2 layers below

    # Layer 1
    model.add(Convolution2D(n_filters, 3, 3, border_mode='valid', input_shape=input_shape))
    # input shape: 100x100 images with 3 channels -> input_shape should be (3, 100, 100) 
    model.add(BatchNormalization())
    model.add(Activation('relu'))  # ReLu activation
    model.add(MaxPooling2D(pool_size=(2, 2))) # reducing image resolution by half
    model.add(Dropout(0.3))  # random "deletion" of %-portion of units in each batch

    # Layer 2
    model.add(Convolution2D(n_filters, 3, 3))  # input_shape is only needed in 1st layer
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten()) # Note: Keras does automatic shape inference.
    
    # Full Layer
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(1,activation='sigmoid'))
    
    return model


car_data = CarDataSet()
car_data.loadImages("/home/lazafi/labor/ml-2019/male2019/3/data/CarData/TrainImages")
car_data.loadImages("/home/lazafi/labor/ml-2019/male2019/3/data/CarData/TestImages", "neg")
(df_train, df_test) = car_data.getDataFrames()

datagen = ImageDataGenerator(rescale=1.0/255.0, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

train_it = datagen.flow_from_dataframe(df_train, batch_size=64, target_size=(224, 224))
test_it = datagen.flow_from_dataframe(df_test, batch_size=64, target_size=(224, 224))

model = createMyModel()

# fit model
history = model.fit_generator(train_it, steps_per_epoch=len(train_it), validation_data=test_it, validation_steps=len(test_it), epochs=50, verbose=0)
# evaluate model
_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
print('> %.3f' % (acc * 100.0))







fids30_data = FIDS30DataSet("/home/lazafi/labor/ml-2019/male2019/3/data/FIDS30", limit=None)
fids30_data.resetFeatures()
fids30_data.addFeatures(BOV(10, orb=True, debug=True))

exp1 = Experiment(fids30_data,  MLPClassifier())
exp1.train()
exp1.evaluate(figure=True)



car_data = CarDataSet()
car_data.loadImages("/home/lazafi/labor/ml-2019/male2019/3/data/CarData/TrainImages")
car_data.loadImages("/home/lazafi/labor/ml-2019/male2019/3/data/CarData/TestImages", "neg")
(df_train, df_test) = car_data.getDataFrames()


fruit_data = FIDS30DataSet("/home/lazafi/labor/ml-2019/male2019/3/data/FIDS30", 5)
print(fruit_data.count)
# TODO: standartize=True is very slow!
#fruit_data.addFeatures(Pixel(standartize = False, debug=False))
(df, dfs) = fruit_data.getDataFrames()

datagen = ImageDataGenerator(rescale=1.0/255.0)

train_it = datagen.flow_from_dataframe(df, batch_size=64)



from keras.datasets import cifar10
from keras.datasets import fashion_mnist

## keras test
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)

# here's a more "manual" example
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
        model.fit(x_batch, y_batch)
        batches += 1
        if batches >= len(x_train) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break



# prepare the car dataset with histogram features
#datapath = "/home/lazafi/labor/ml-2019/male2019/3/data/CarData"
car_data = CarDataSet()
car_data.loadImages("/home/lazafi/labor/ml-2019/male2019/3/data/CarData/TrainImages")
car_data.loadImages("/home/lazafi/labor/ml-2019/male2019/3/data/CarData/TestImages", "neg")
print(car_data.count)

# TODO: standartize=True is very slow!
car_data.addFeatures(Pixel(standartize = False, debug=True))

(x_train, x_test, y_train, y_test) = car_data.getData(ratio=0.3)

#datagen = ImageDataGenerator(
#    featurewise_center=True,
#    featurewise_std_normalization=True,
#    rotation_range=20,
#    width_shift_range=0.2,
#    height_shift_range=0.2,
#    horizontal_flip=True)

# fits the model on batches with real-time data augmentation:
#model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),steps_per_epoch=len(x_train) / 32, epochs=epochs)


##
#car_data.addFeatures(Pixel)



car_data.addFeatures(BOV(20))


# knn
exp1 = Experiment(car_data, KNeighborsClassifier(n_neighbors=3, weights = 'distance'),  "BOV with Knn k=3")
exp1.train()
exp1.evaluate(figure=True, text=True)

# BOV
# TODO: BOV does not work well with cardata (bug) becouse it cannot create features for small resolution images. 
# the trainset images are 40x100 pixels
# maybe we leave this experiment out
#car_data.addFeatures(BOV(50, debug=False))
## svc
#exp1 = Experiment(car_data,  svm.SVC())
#exp1.train()
#exp1.evaluate()


        
# prepare the fids30 dataset with histogram features
datapath = "/home/lazafi/labor/ml-2019/male2019/3/data/FIDS30"
fids30_data = FIDS30DataSet("/home/lazafi/labor/ml-2019/male2019/3/data/FIDS30", 10)
fids30_data.addFeatures(BOV(50))
print(fids30_data.count)

# svc

exp1 = Experiment(fids30_data, svm.SVC(), "Histogram with SVC")
exp1.train()
exp1.evaluate(figure=True, text=True)

# knn
exp1 = Experiment(fids30_data, KNeighborsClassifier(n_neighbors=3, weights = 'distance'),  "Histogram with Knn k=3")
exp1.train()
exp1.evaluate(figure=True, text=True)

score = []
for k in range(1, 10):
    print(k)
    classifier = KNeighborsClassifier(n_neighbors=k, weights = 'distance')
    classifier.fit(x_train, y_train)
    predicted = classifier.predict(x_test)
    score.append(average_precision_score(y_test, predicted))
plt.figure(figsize=(10,10))
plt.plot(score)
plt.show()


# mlp
exp1 = Experiment(fids30_data,  MLPClassifier())
exp1.train()
exp1.evaluate(figure=True, text=True)


# prepare dataset with bov features
fids30_data.resetFeatures()
fids30_data.addFeatures(BOV(50))

# svc
exp1 = Experiment(fids30_data,  svm.SVC())
exp1.train()
exp1.evaluate(figure=True, text=True)

# knn
exp1 = Experiment(fids30_data, KNeighborsClassifier(n_neighbors=3, weights = 'distance'))
exp1.train()
exp1.evaluate(figure=True, text=True)



x_train, x_test, y_train, y_test =  fids30_pixel_dataset(DATA_DIR, labels)

#mnist = tf.keras.datasets.fashion_mnist

#(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
    print(y_train[i])
plt.show()

model = tf.keras.models.Sequential([
tf.keras.layers.Flatten(input_shape=(28, 28)),
tf.keras.layers.Dense(512, activation=tf.nn.relu),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1)
model.evaluate(x_test, y_test)


x = np.linspace(0, 20, 100)  # Create a list of evenly-spaced numbers over the range
plt.plot(x, np.sin(x))       # Plot the sine of each x point
plt.show()

