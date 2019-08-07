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
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score, precision_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.cluster import MiniBatchKMeans

#data
import pandas as pd
import numpy as np

def merge_dols(dol1, dol2):
    keys = set(dol1).union(dol2)
    no = []
    return dict((k, dol1.get(k, no) + dol2.get(k, no)) for k in keys)


# class representing experiments.
class Experiment:
        def __init__(self, data, classifier=None, label=None):
            #self.featuresExtractor = featuresExtr
            if classifier == None:
                self.classifier = None
            else:
                self.classifier = classifier
            self.dataset = data
            if label == None:
                self.label = "%s %s" % (type(self.dataset).__name__, type(self.classifier).__name__)
            else:
                self.label = label
            pass

        def train(self, classifier=None):
            #if features == None:
            #    x_data, y_data = self.featuresExtractor.features(self.dataset.images)
            #else
            self.x_train, self.x_test, self.y_train, self.y_test =  train_test_split(self.dataset.x_data, self.dataset.y_data, test_size=0.33, random_state=123)
            if classifier == None:
                if self.classifier == None:
                    raise Exception("no classifier specified")
            else:
                self.classifier = classifier
            


            #for i, t in enumerate(self.x_train):
            #    for j, v in enumerate(t):
            #        self.x_train[i][j] = float(self.x_train[i][j])
            #        if self.x_train[i][j] < 0.01:
            #            self.x_train[i][j] = float(0.01)
            #        if isinstance(v, str):
            #            print('x is a str!')

            #for i, t in enumerate(self.y_train):
            #    self.y_train[i] = int(self.y_train[i])


            #np.savetxt('test_exp2.out', self.x_train, delimiter=',')
            
            self.classifier.fit(self.x_train, self.y_train)
        def precision(self):
            predicted = self.classifier.predict(self.x_test)
            return precision_score(self.y_test, predicted, average='macro')

        def evaluate(self, text=False, figure=False, precision=False, debug=False):
            predicted = self.classifier.predict(self.x_test)
            confusion = confusion_matrix(self.y_test, predicted)

            if debug:
                for i, x in enumerate(self.x_test):
                    print("%d %d %d %s %s" % (i, self.y_test[i], predicted[i], self.dataset.labels[int(self.y_test[i])], self.dataset.labels[int(predicted[i])]) )
            if figure:
                df_cm = pd.DataFrame(confusion, index = [i for i in self.dataset.labels], columns = [i for i in self.dataset.labels])
                plt.figure(figsize=(10,10))
                ax = plt.axes()
                cm = sn.heatmap(df_cm, annot=True, ax=ax) 
                ax.set_title("ConfusionMatrix %s" % self.label)

                plt.show()
            if text:
                report = classification_report(self.y_test, predicted)
                print("classifier %s:\n%s\n" % (self.classifier, report))
                #print("Confusion matrix:\n%s" % confusion)
            if precision:
                print(self.predision())    

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
                dcount = 0
                word = each.split("/")[-1]
                #print (" #### Reading image category " + word + " ##### ")
                labels.append(word)
                imlist[word] = []
                for imagefile in glob(path + "/" + word+"/*")[:limit]:
                    #print ("Reading file " + imagefile)
                    im = cv2.imread(imagefile, cv2.COLOR_BGR2RGB)
                    rgb = np.array(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                    imlist[word].append(rgb)
                    #plt.imshow(rgb)
                    #plt.show()
                    count += 1
                    dcount += 1 
                print (" category %s %d" % (word, dcount) )

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
            self.name_dict = {}
            self.dataframe = pd.DataFrame([], columns = ['filename', 'class'])
            pass
        
        def addFeatures(self, featureExtractor):
            """
            use the featureExtractor to add features from image data
            overwrites old features
            """
            self.x_data, self.y_data = featureExtractor.features(self.images)

        def resetFeatures(self):
            """
            reset feature vector
            """
            self.x_data = []
            self.y_data = []
        
        def getData(self, ratio=0.33):
            return train_test_split(self.x_data, self.y_data, test_size=ratio, random_state=123)
        def getDataFrame(self):
            return self.dataframe
        def getDataFrames(self, ratio=0.33):
            return train_test_split(self.dataframe, test_size=ratio)
    
class FIDS30DataSet(ImageDataSet):
        def __init__(self, path=None, images=True, limit=None):
            super(FIDS30DataSet, self).__init__()
            if path != None:
                if images:
                    self.images, self.labels, self.count = File().getFiles_FIDS30(path, limit)
                # dataframe
                data = []
                for each in glob(path + "/*")[:limit]:
                    if isdir(each):
                        word = each.split("/")[-1]
                        for imagefile in glob(path + "/" + word+"/*")[:limit]:
                            #print(imagefile)
                            #print(word)
                            line = list([imagefile, word])
                            data.append(line)
                self.dataframe = pd.DataFrame(data, columns = ['filename', 'class']) 
            #self.labels
            
            pass

        def loadImages(self, path, clazz=None, limit=None):
            """
            load images from directory
            all images in clazz if specified
            if no clazz, the image class is guessed from name (neg* or pos*)
            NOTE: DO NOT USE
            """
            images, labels, count = File().getFiles_FIDS30(path, limit)
            # merge read images with existing ones
            self.images = {**self.images, **images}
            #self.labels = self.labels.union(labels)
            labelslist = list(self.labels)
            labelslist.extend(x for x in labels if x not in labelslist)
            self.labels = labelslist
            #self.labels = set(self.labels)
            self.count += count

class CarDataSet(ImageDataSet):
        def __init__(self, path=None, limit=None):
            super(CarDataSet, self).__init__()
            if path != None:
                self.images, self.labels, self.count = File().getFiles_CarData(path, limit)
            #self.labels
            pass

        def loadImages(self, path, clazz=None, images=True, limit=None):
            """
            load images from directory
            all images in clazz if specified
            if no clazz, the image class is guessed from name (neg* or pos*)
            """
            if images:
                images, labels, count = File().getFiles_CarData(path, clazz, limit)
                # merge read images with existing ones
                #for k, v in d.items():
                #    v = 
                #self.images = {**self.images, **images}
                #self.images.update(images)
                self.images = merge_dols(self.images, images)
                #self.labels = self.labels.union(labels)
                labelslist = list(self.labels)
                labelslist.extend(x for x in labels if x not in labelslist)
                self.labels = labelslist
                #self.labels = set(self.labels)
                self.count += count
            # dataframe
            data = []
            for each in glob(path + "/*"):
                print(each)
                count +=1 
                if clazz == None:
                    filename = each.split("/")[-1]
                    word = filename.split("-")[0]
                else:
                    word = clazz
                print (word)
                line = list([each, word])
                data.append(line)
            df = pd.DataFrame(data, columns = ['filename', 'class']) 
            self.dataframe = self.dataframe.append(df) 

# classes for extracting features from image datasets

class BOV:
    def __init__(self, no_clusters, orb = False, debug = False):
        self.no_clusters = no_clusters
        if orb:
            self.sift = cv2.ORB_create()
        else:
            self.sift = cv2.xfeatures2d.SIFT_create()

    def features(self, images):

        #
        dico = []
        labels = []
        label_count = 0
        name_dict = {}
        imcount = 0
        imglist = []
        y_data = []


        for word, imlist in images.items():
            labels.append(word)
            name_dict[str(label_count)] = word
            for i, img in enumerate(imlist):
                kp, des = self.sift.detectAndCompute(img, None)
                if des is None:
                    print("skipping img size %s" % str(img.shape))
                    #cv2.imshow("im", img)
                    #cv2.waitKey(100)
                else:
                    for d in des:
                        dico.append(d)
                    imcount += 1
                    imglist.append(img)
                    y_data.append(label_count)
            label_count += 1

#k = np.size(species) * 10

        batch_size = imcount * 3
        kmeans = MiniBatchKMeans(n_clusters=self.no_clusters, batch_size=batch_size, verbose=0).fit(dico)
        
        kmeans.verbose = False
        histo_list = []
        for img in imglist:
            kp, des = self.sift.detectAndCompute(img, None)

            histo = np.zeros(self.no_clusters)
            nkp = np.size(kp)

            for d in des:
                idx = kmeans.predict([d])
                histo[idx] += 1/nkp # Because we need normalized histograms, I prefere to add 1/nkp directly

            histo_list.append(histo)

        return (histo_list, y_data)


class Histogram:
        """
        extracts Histogram data from images
        """
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
            label_count = 0
            name_dict = {}
            for word, imlist in images.items():
                labels.append(word)
                name_dict[str(label_count)] = word
                for i, img in enumerate(imlist):
                    hist = self.getHistogramData(img, self.bins)
                    #standartize
                    hist = [(x - min(hist)) / (max(hist) - min(hist)) for x in hist]
                    x_data.append(hist)
                    y_data.append(labels.index(word))
                label_count += 1
                if self.debug:
                    f, axarr = plt.subplots(len(imlist),2,figsize = (10,7))
                    f.canvas.set_window_title(word)
                    for i, img in enumerate(imlist):
                        fig = axarr[i,0].imshow(img)   
                        fig.axes.get_xaxis().set_visible(False)
                        fig.axes.get_yaxis().set_visible(False)
                        axarr[i,1].bar(list(range(self.bins*3)), x_data[i], width=1.0, color=['red'  for x in range(self.bins)] + ['green'  for x in range(self.bins)] + ['blue'  for x in range(self.bins)])
                        axarr[i,1].xaxis.set_visible(False)
                        axarr[i,1].yaxis.set_visible(False)

                    #plt.savefig(word+ ".histograms.png")
                    #plt.close(f)
                    #print("saved: " + word+ ".histograms.png")
                    plt.show()
            return (x_data, y_data)

class Pixel:
        """
        extract pixel data
        """
        def __init__(self, flatten = False, standartize = False, debug = False):
            self.debug = debug
            self.standartize = standartize
            self.flatten = flatten
            pass

        def features(self, images):
            imcount = 0
            for word, imlist in images.items():
                imcount += 1
            x_data = np.empty((imcount))
            y_data = []
            labels = []
            for word, imlist in images.items():
                labels.append(word)
                for i, img in enumerate(imlist):
                    pixels = img
                    if self.flatten:
                        pixels = img.flatten()
                    #standartize
                    if self.standartize:
                        pixels = [(x - min(pixels)) / (max(pixels) - min(pixels)) for x in pixels]
                    x_data = np.append(x_data, pixels, axis = 0)
                    y_data.append(labels.index(word))
                    if self.debug:
                        print(word)
                        print(pixels)
            return (x_data, y_data)
            
