import numpy as np
import pandas as pd
import os
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# Load data from csv
assert os.path.isfile('data_processed/Iteration_2/test_data_Final.csv'), 'file does not exist' # sanity check
assert os.path.isfile('data_processed/Iteration_2/training_data_Final.csv'), 'file does not exist' # sanity check
test_data_file = glob.glob('data_processed/Iteration_2/test_data_Final.csv')
train_data_file = glob.glob('data_processed/Iteration_2/training_data_Final.csv')

# Get train and test set
train_set = pd.read_csv(train_data_file[0])
test_set = pd.read_csv(test_data_file[0])

# Split train and test set into features and response variables
X_train = []
y_train = np.array(train_set['Genre']) 

X_test = []
y_test = np.array(test_set['Genre'])

for column_name in train_set.columns:
    if "Segment" in column_name:
        # print(column_name)
        X_train.append(train_set[column_name].values.tolist())
        X_test.append(test_set[column_name].values.tolist())

X_train = np.array(X_train).T
X_test = np.array(X_test).T

# Random Forest (Without Optimized hyperparameters)
rnd_clf = RandomForestClassifier(n_estimators=200, max_leaf_nodes=15, n_jobs=-1, criterion='entropy', random_state=42)
rnd_clf.fit(X_train, y_train)

y_pred = rnd_clf.predict(X_test)


# import pdb; pdb.set_trace()
# param_grid = [{'max_leaf_nodes': [range(1, np.shape(X_train)[0])]}]
# tree_clf = RandomForestClassifier(random_state=42)

# grid_search = GridSearchCV(tree_clf, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
# grid_search.fit(X_train, y_train)

# print("Optimal value:", grid_search.best_params_) # This prints out 14 for max_leaf_nodes


# Accuracy score
print(rnd_clf.__class__.__name__, accuracy_score(y_test, y_pred))
