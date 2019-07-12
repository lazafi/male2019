#files
from glob import glob
from os.path import isdir
#images
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sn
#ml
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
#data
import pandas as pd
import numpy as np



class Experiment:
        def __init__(self, data, classifier, label=None):
            #self.featuresExtractor = featuresExtr
            self.classifier = classifier
            self.dataset = data
            if label == None:
                self.label = "%s %s" % (type(self.dataset).__name__, type(self.classifier).__name__)
            else:
                self.label = label
            pass

        #TODO: specify classifier here
        def train(self):
            #if features == None:
            #    x_data, y_data = self.featuresExtractor.features(self.dataset.images)
            #else
            self.x_train, self.x_test, self.y_train, self.y_test =  train_test_split(self.dataset.x_data, self.dataset.y_data, test_size=0.33, random_state=123)
            self.classifier.fit(self.x_train, self.y_train)

        def evaluate(self):
            predicted = self.classifier.predict(self.x_test)

            for i, x in enumerate(self.x_test):
                print("%d %d %d %s %s" % (i, self.y_test[i], predicted[i], self.dataset.labels[int(self.y_test[i])], self.dataset.labels[int(predicted[i])]) )
            confusion = confusion_matrix(self.y_test, predicted)

            df_cm = pd.DataFrame(confusion, index = [i for i in self.dataset.labels],
                            columns = [i for i in self.dataset.labels])
            ax = plt.axes()
            cm = sn.heatmap(df_cm, annot=True, ax=ax) 
            ax.set_title("ConfusionMatrix %s" % self.label)
            plt.show()
            report = classification_report(self.y_test, predicted)
            print("classifier %s:\n%s\n" % (self.classifier, report))
            print("Confusion matrix:\n%s" % confusion)

class File:
    
    def __init__(self):
        pass

    def getFiles_FIDS30(self, path, limit=None):
        """
        - returns  a dictionary of all files 
        having key => value as  objectname => image path

        - returns total number of files.

        """
        imlist = {}
        count = 0
        labels = []
        for each in glob(path + "/*")[:limit]:
            if isdir(each):
                word = each.split("/")[-1]
                print (" #### Reading image category " + word + " ##### ")
                labels.append(word)
                imlist[word] = []
                for imagefile in glob(path + "/" + word+"/*")[:limit]:
                    print ("Reading file " + imagefile)
                    im = cv2.imread(imagefile, 0)
                    imlist[word].append(im)
                    count +=1 

        return [imlist, labels, count]

    # TODO
    def getFiles_CarData(self, path, clazz=None, limit=None):
        imlist = {}
        count = 0 
        for each in glob(path + "/*"):
            im = cv2.imread(each, 0)
            count +=1 
            if clazz == None:
                filename = each.split("/")[-1]
                word = filename.split("-")[0]
            else:
                word = clazz
            print (word + ": " + each + " " + str(im.shape))
            #cv2.imshow("im", im)
            #cv2.waitKey(10)
                
            if word not in imlist:
                imlist[word] = []
            imlist[word].append(im)
        return [imlist, list(imlist.keys()), count]

class ImageDataSet:
        def __init__(self):
            self.images = {}
            self.labels = []
            self.x_data = []
            self.y_data = []
            self.count = 0
            pass
        
        def addFeatures(self, featureExtractor):
            self.x_data, self.y_data = featureExtractor.features(self.images)

        def resetFeatures(self):
            self.x_data = []
            self.y_data = []

class FIDS30DataSet(ImageDataSet):
        def __init__(self, path, limit=None):
            super(FIDS30DataSet, self).__init__()
            self.images, self.labels, self.count = File().getFiles_FIDS30(path, limit)
            #self.labels
            pass

class CarDataSet(ImageDataSet):
        def __init__(self, path=None, limit=None):
            super(CarDataSet, self).__init__()
            if path != None:
                self.images, self.labels, self.count = File().getFiles_CarData(path, limit)
            #self.labels
            pass

        def loadImages(self, path, clazz=None, limit=None):
            images, labels, count = File().getFiles_CarData(path, clazz, limit)
            # merge read images with existing ones
            self.images = {**self.images, **images}
            #self.labels = self.labels.union(labels)
            labelslist = list(self.labels)
            labelslist.extend(x for x in labels if x not in labelslist)
            self.labels = labelslist
            #self.labels = set(self.labels)
            self.count += count


class Histogram:
        def __init__(self, bins, debug = False):
            self.images = []
            self.bins = bins
            self.debug = debug
            pass
        
        def getHistogramData(self, img, bins):
            image = Image.fromarray(img)
            #TODO: work with grayscale images
            # image.getbands() -> returns collor chanels used
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

        def features(self, images):
            x_data = []
            y_data = []
            labels = []
            for word, imlist in images.items():
                labels.append(word)
                for i, img in enumerate(imlist):
                    hist = self.getHistogramData(img, self.bins)
                    #standartize
                    hist = [(x - min(hist)) / (max(hist) - min(hist)) for x in hist]
                    x_data.append(hist)
                    y_data.append(labels.index(word))
                if self.debug:
                    f, axarr = plt.subplots(20,2,figsize = (10,7))
                    f.canvas.set_window_title(word)
                    for i, img in enumerate(imlist[:20]):
                        fig = axarr[i,0].imshow(img)   
                        fig.axes.get_xaxis().set_visible(False)
                        fig.axes.get_yaxis().set_visible(False)
                        axarr[i,1].bar(list(range(self.bins*3)), x_data[i], width=1.0, color=['red'  for x in range(self.bins)] + ['green'  for x in range(self.bins)] + ['blue'  for x in range(self.bins)])
                        axarr[i,1].xaxis.set_visible(False)
                        axarr[i,1].yaxis.set_visible(False)

                    plt.savefig(word+ ".histograms.png")
                    plt.close(f)
                    print("saved: " + word+ ".histograms.png")

            return (x_data, y_data)
