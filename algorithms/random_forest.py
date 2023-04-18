import numpy as np
import pandas as pd
import os
import glob
from sklearn.ensemble import RandomForestClassifier


# Load data from csv
assert os.path.isfile('data_processed/test_data_NEW.csv'), 'file does not exist' # sanity check
assert os.path.isfile('data_processed/training_data_NEW.csv'), 'file does not exist' # sanity check
test_data_file = glob.glob('data_processed/test_data_NEW.csv')
train_data_file = glob.glob('data_processed/training_data_NEW.csv')

import pdb; pdb.set_trace()

train_data = pd.read_csv(train_data_file[0])
test_data = pd.read_csv(test_data_file[0])

