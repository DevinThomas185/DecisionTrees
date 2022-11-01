# Decision Trees Coursework
## Usage guide for running on DOC lab machines

In order to run the main entrypoint of the coursework solution invoke
```
python main.py
```

To swap out the dataset for the secret dataset for assessing our solution, 
the following line needs to be updated: 

```
dataset, unique_classes = file_utils.read_dataset(
  "./intro2ML-coursework1/wifi_db/noisy_dataset.txt"
)
```

Located in the main entrypoint function in main.py (at the bottom).
