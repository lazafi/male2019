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
from bov.Bag import BOV


# prepare the car dataset with histogram features
#datapath = "/home/lazafi/labor/ml-2019/male2019/3/data/CarData"
car_data = CarDataSet()
car_data.loadImages("/home/lazafi/labor/ml-2019/male2019/3/data/CarData/TrainImages")
car_data.loadImages("/home/lazafi/labor/ml-2019/male2019/3/data/CarData/TestImages", "neg")
print(car_data.count)

##
#car_data.addFeatures(Pixel)



car_data.addFeatures(Histogram(10))

# knn
exp1 = Experiment(car_data, KNeighborsClassifier(n_neighbors=3, weights = 'distance'),  "Histogram with Knn k=3")
exp1.train()
exp1.evaluate()

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
fids30_data.addFeatures(Histogram(50))
print(fids30_data.count)

# svc

exp1 = Experiment(fids30_data, svm.SVC(), "Histogram with SVC")
exp1.train()
exp1.evaluate()

# knn
exp1 = Experiment(fids30_data, KNeighborsClassifier(n_neighbors=3, weights = 'distance'),  "Histogram with Knn k=3")
exp1.train()
exp1.evaluate()

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
exp1.evaluate()


# prepare dataset with bov features
fids30_data.resetFeatures()
fids30_data.addFeatures(BOV(50))

# svc
exp1 = Experiment(fids30_data,  svm.SVC())
exp1.train()
exp1.evaluate()

# knn
exp1 = Experiment(fids30_data, KNeighborsClassifier(n_neighbors=3, weights = 'distance'))
exp1.train()
exp1.evaluate()



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

