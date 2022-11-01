# Decision Trees Coursework
## Usage guide for running on DoC lab machines

In order to run the main entrypoint of the coursework solution invoke
```
python main.py PATH_TO_DATASET
```
and substitute PATH_TO_DATASET with the relative path to the file of the dataset being used.

----
 
### Flags
When running 
```
python main.py -h
```

You will see these available flags:
```
Decision Tree and Evaluation Metrics

positional arguments:
  PATH_TO_DATASET       The relative path to the dataset

optional arguments:
  -h, --help            show this help message and exit
  --visualise, -v       Use this flag to produce images visualising the decision trees
  --k_folds K_FOLDS, -k K_FOLDS
                        The number of folds to use to split the dataset up into
  --pruning, -p         Use this flag to turn on tree pruning utilising a validation dataset fold
  --debug, -d           Use this flag to turn on debugging prints
  -s SEED, --seed SEED  Provide a seed for shuffling the dataset, leave empty for random seed
```

The `-h` and `--help` flags will bring up the menu above to see what flags are available.

The `-v` and `--visualise` flags will turn on the visualiser and produce images that represent the resulting decision trees.

The `-k` and `--k_folds` arguments will allow you to specify the number of folds to split the dataset into.

The `-p` and `--pruning` flags will enable the validation set being used in pruning the trees to improve accuracy.

The `-s` and `--seed` arguments will allow you to select the seed being used for shuffling of the dataset

----

### Example
Running:
```
python main.py ./intro2ML-coursework1/wifi_db/clean_dataset.txt -k 8 --visualise --pruning -s 123
```
Will:
- Set the folds to 8.
- Turn on visualisation and will create images for the trees.
- Turn on pruning using a validation fold from the initial folds.
- Set the shuffling seed to be 123