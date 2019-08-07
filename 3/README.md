# Machine Learning Excercise 3

## Usage

```
> python src/fruits/cli.py  -h
usage: cli.py [-h] [--dataset DATASET] [--classifier CLASSIFIER]
              [--classifier-par CLASSIFIER_PAR [CLASSIFIER_PAR ...]]
              [--features FEATURES [FEATURES ...]]
              [--histogram-bins HISTOGRAM_BINS [HISTOGRAM_BINS ...]]
              [--bov-descriptors BOV_DESCRIPTORS [BOV_DESCRIPTORS ...]]
              [--features-par FEATURES_PAR [FEATURES_PAR ...]]
              [--dir DIR [DIR ...]] [--pos POS [POS ...]]
              [--neg NEG [NEG ...]] [--output OUTPUT] [--verbose VERBOSE]

Run Experiments in ML.

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     dataset name either "CarData" or "FIDS30"
  --classifier CLASSIFIER
                        "svm", "knn", "mlp"
  --classifier-par CLASSIFIER_PAR [CLASSIFIER_PAR ...]
                        parameter to pass to classifier
  --features FEATURES [FEATURES ...]
                        "histogram", "bov", "pixel"
  --histogram-bins HISTOGRAM_BINS [HISTOGRAM_BINS ...]
                        bins per chanel for histogram features (default 10)
  --bov-descriptors BOV_DESCRIPTORS [BOV_DESCRIPTORS ...]
                        amount of descriptors for BOV (default 20)
  --features-par FEATURES_PAR [FEATURES_PAR ...]
                        parameter to pass to feature extractor
  --dir DIR [DIR ...]   dataset root directory
  --pos POS [POS ...]   positive samples data directory
  --neg NEG [NEG ...]   negative samples data directory
  --output OUTPUT       output: "confusionmatrix", "report", "precision"
  --verbose VERBOSE     verbose output

  ```

### examples

```
> python src/fruits/cli.py --dataset=CarData --dir=/home/lazafi/labor/ml-2019/male2019/3/data/CarData/TrainImages --neg=/home/lazafi/labor/ml-2019/male2019/3/data/CarData/TestImages --classifier=mlp --classifier-par=1000 --features=bov --bov-descriptors=30 --output=confusionmatrix --output=report

```