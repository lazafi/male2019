import argparse
from helper import *

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
parser.add_argument('--bov-descriptors', nargs="+", help='amount of descriptors for BOV (default 20)')
parser.add_argument('--features-par', nargs="+", help='parameter to pass to feature extractor')
parser.add_argument('--dir', nargs='+', help='dataset root directory')
parser.add_argument('--pos', nargs='+', help='positive samples data directory')
parser.add_argument('--neg', nargs='+', help='negative samples data directory')
parser.add_argument('--output', nargs=1, action='append', help='output: "confusionmatrix", "report", "precision" ')
parser.add_argument('--verbose',  help='verbose output')

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
    dataset = FIDS30DataSet(args.dir[0])
    #for d in args.dir:
    #    dataset.loadImages(d)
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
        p = int(args.bov_descriptors[0]) if args.bov_descriptors != None else 20
        extractor = BOV(p)
    if f == "pixel":
        extractor = Pixel()
    dataset.addFeatures(extractor)

## train dataset
models = {}
classpar = args.classifier_par[0] if args.classifier_par != None else None
classifier = args.classifier[0]
print('training %s with parameter %s' % (classifier, classpar))

# default to mlp
m = classpar if classpar != None else 3200
exp = Experiment(dataset, MLPClassifier(max_iter=m, verbose=False), "exp")
# TODO: max_iter or other param
if classifier == 'svm':
    exp = Experiment(dataset, svm.SVC(gamma='auto'), "exp")
if classifier == 'knn':
    k = classpar if classpar != None else 3
    exp = Experiment(dataset, KNeighborsClassifier(n_neighbors=k, weights = 'distance'), "exp")

exp.train()

# output
for o in args.output:
    o = o[0]
    if o == 'confusionmatrix':
        exp.evaluate(figure=True)
    if o == 'report':
        exp.evaluate(text=True)
    if o == 'precision':
        exp.evaluate(precision=True)
    






