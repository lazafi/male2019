import argparse
from helper import *
from bov.Bag import BOV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier


parser = argparse.ArgumentParser(description='Run Experiments in ML.')
parser.add_argument('--dataset', nargs=1, help='dataset name either "CarData" or "FIDS30"')
parser.add_argument('--classifier', nargs=1, help='"svm", "knn", "mlp"')
parser.add_argument('--classifier-par', nargs="+", help='parameter to pass to classifier')
parser.add_argument('--features', nargs="+", help='"histogram", "bov", "pixel"')
parser.add_argument('--histogram-bins', nargs="+", help='bins per chanel for histogram features (default 10)')
parser.add_argument('--features-par', nargs="+", help='parameter to pass to feature extractor')
parser.add_argument('--dir', nargs='+', help='dataset root directory')
parser.add_argument('--pos', nargs='+', help='positive samples data directory')
parser.add_argument('--neg', nargs='+', help='negative samples data directory')

args = parser.parse_args()
print (args)

#dataset


# data dir
datapaths = args.dir

dataset = {}
if args.dataset[0] == "CarData":
    dataset = CarDataSet()
    # load samples from dirs
    for d in args.dir:
        dataset.loadImages(d)
    # load neg samples
    if args.neg != None:
        for d in args.neg:
            dataset.loadImages(d, "neg")
    # load pos samples
    if args.pos != None:
        for d in args.pos:
            dataset.loadImages(d, "pos")
elif args.dataset[0] == "FIDS30":
    dataset = FIDS30DataSet()
    for d in args.dir:
        dataset.loadImages(d)
else:
    raise Exception("unknown dataset")

print(dataset.count)


## extract features
##

for f in args.features:
    if f == "histogram":
        k = int(args.histogram_bins[0]) if args.histogram_bins != None else 10
        extractor = Histogram(k)
#    , "bov", "pixel"
    if f == "bov":
        p = 50
        extractor = BOV(p)
    if f == "pixel":
        extractor = Pixel()
    dataset.addFeatures(extractor)

## train dataset
models = {}
classpar = args.classifier_par[0] if args.classifier_par != None else None
classifier = args.classifier[0]
print('training %s with parameter %s' % (classifier, classpar))

models['knn'] = KNeighborsClassifier(n_neighbors=classpar, weights = 'distance')
models['svm'] = svm.SVC()


exp = Experiment(dataset, models[classifier], "exp")
exp.train()
exp.evaluate()


