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
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score
from sklearn.neighbors import KNeighborsClassifier

import progressbar

from Bag import BOV

def getPixelData(src):
    print(src)
    image = Image.open(src,'r')
    return image.getdata()

def getHistogramData(img, bins):
    if isinstance(img, Image.Image):
        image = img
    else:
        image = Image.fromarray(img)
    image = image.convert("RGB")
    rawhist = image.histogram()
    if (len(rawhist) > 768):
        raise Exception('color histogram to large: {}'.format(len(rawhist)))
    # split raw histogram to colors
    r = rawhist[:256]
    g = rawhist[257:512]
    b = rawhist[513:]
    colhist = []
    # summ up histogram bins
    for chanel in (r,g,b):
        for idx in range(bins):
            lower = int(idx*256/bins)
            upper = int((idx+1)*256/bins)
            colhist.append(sum(chanel[lower : upper]))
    return colhist 

def fids30_histogram_dataset(picdir, labels, bins=8, test_size=0.33, randomseed=123, debug=True):
    plt.ioff()
    x_data = []
    y_data = []
    for label in labels:
        print(label)

        files = glob(picdir+'/'+label+'/*')
        if debug:
            f, axarr = plt.subplots(len(files),2,figsize = (10,7))
            f.canvas.set_window_title(label)
        bar = progressbar.ProgressBar(maxval=len(files), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Counter()])
        bar.start()
        for i, img_p in enumerate(files):
            #print("%d: %s"%(i,img_p))
            imagePIL = Image.open(img_p,'r')
            # extract histogram
            hist = getHistogramData(imagePIL, bins)
            #standartize data
            #max = max(hist)
            #min = min(hist)
            hist = [(x - min(hist)) / (max(hist) - min(hist)) for x in hist] 
            if debug:
                fig = axarr[i,0].imshow(imagePIL)   
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
                axarr[i,1].bar(list(range(bins*3)), hist, width=1.0, color=['red'  for x in range(bins)] + ['green'  for x in range(bins)] + ['blue'  for x in range(bins)])
                axarr[i,1].xaxis.set_visible(False)
                axarr[i,1].yaxis.set_visible(False)
            x_data.append(hist)
            y_data.append(labels.index(label))
            bar.update(i)
        bar.finish()
        if debug:
            plt.savefig(label+ ".histograms.png")
            plt.close(f)
            print("saved: " + label+ ".histograms.png")
    return train_test_split(x_data, y_data, test_size=test_size, random_state=randomseed)

def main():
    DATA_DIR = '/home/lazafi/labor/ml-2019/male2019/3/data/FIDS30'
    limit = 5

    with open(DATA_DIR+'/fruits.txt', 'r') as f:
        reader = csv.reader(f, delimiter=';')
        labels = list(reader)[0]
        if limit is not None:
            labels = labels[:limit]

    x_train, x_test, y_train, y_test =  fids30_histogram_dataset(DATA_DIR, labels, debug=False, bins=16)

    for i, x in enumerate(x_test):
        print("%d %d %d %s " % (i, len(x), y_test[i], labels[y_test[i]]) )


    # random forest

    #classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)

    # svm
    #classifier = svm.SVCL(gamma='scale', decision_function_shape='ovo')
    #classifier = svm.LinearSVC(C=1.0, max_iter=100)

    #knn
    classifier = KNeighborsClassifier(n_neighbors=2, weights = 'distance')
    #A = classifier.kneighbors_graph()
    #print(A)


    classifier.fit(x_train, y_train)

    #distances, indices = classifier.kneighbors(x_test)

    #print(indices)
    #print(distances)

    #print(classifier.kneighbors_graph(x_test).toarray())

    predicted = classifier.predict(x_test)

    confusion = confusion_matrix(y_test, predicted)

    df_cm = pd.DataFrame(confusion, index = [i for i in labels[:limit]],
                    columns = [i for i in labels[:limit]])
    plt.figure(figsize = (10,7))
    cm = sn.heatmap(df_cm, annot=True)
    plt.show()
    report = classification_report(y_test, predicted)
    print("classifier %s:\n%s\n" % (classifier, report))
    print("Confusion matrix:\n%s" % confusion)

    score = []
    for k in range(1, 10):
        print(k)
        classifier = KNeighborsClassifier(n_neighbors=k, weights = 'distance')
        classifier.fit(x_train, y_train)
        predicted = classifier.predict(x_test)
        score.append(average_precision_score(y_test, predicted))
        plt.figure(figsize=(10,10))
        plt.plot(k, score)
        plt.show()






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

bov = BOV(no_clusters=100)

# set training paths
#bov.train_path = args['train_path'] 
# set testing paths
#bov.test_path = args['test_path'] 
#--train_path ../../3/data/CarData/TrainImages --test_path ../../3/data/CarData/TestImages
#bov.train_path="/home/lazafi/labor/ml-2019/male2019/3/data/CarData/TrainImages"
#bov.test_pat="/home/lazafi/labor/ml-2019/male2019/3/data/CarData/TestImages"

bov.train_path="/home/lazafi/labor/ml-2019/male2019/3/data/FIDS30"
bov.test_path="/home/lazafi/labor/ml-2019/male2019/3/data/FIDS30"
# train the model
bov.trainModel()
# test model
bov.testModel()