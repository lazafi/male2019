import cv2
import numpy as np
import os
import pandas as pd
import csv

from sklearn.cluster import MiniBatchKMeans
from sklearn.neural_network import MLPClassifier


class BOVKG:
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
        kmeans = MiniBatchKMeans(n_clusters=self.no_clusters, batch_size=batch_size, verbose=1).fit(dico)
        
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
        
