# Data

The project uses `sklearn.datasets.load_digits()` directly at runtime.
The Digits dataset is downloaded/provided by scikit-learn; no separate dataset
file is required in this repository. The experiment creates the stratified
train/test split and five label-skewed client partitions in memory.
