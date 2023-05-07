import numpy as np
import pandas as pd
import os
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
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

# # Random Forest (Without Optimized hyperparameters -- Initial Guess)
# rnd_clf = RandomForestClassifier(n_estimators=100, max_leaf_nodes=10, n_jobs=-1, criterion='entropy', random_state=42)
# rnd_clf.fit(X_train, y_train)
# y_pred = rnd_clf.predict(X_test)

# # Accuracy score (Without optimized hyperparameters -- Initial Guess)
# print("Random Forest Classifier (Without Optimized Hyperparameters):", accuracy_score(y_test, y_pred))


# Use GridSearchCV to try and find optimum hyperparameters for random trees classifier
# Define the parameter grid to search
param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'max_features': [20, 30, 50],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [100, 500, 1000],
    'min_samples_leaf': [100, 200, 500],
    'max_leaf_nodes': [10, 15, 20],
}
tree_clf = RandomForestClassifier()

# Using negative mean squared error score for classification
grid_search = GridSearchCV(tree_clf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print the best combination of hyperparameters
print("Best hyperparameters for Random Forest:", grid_search.best_params_)

# # New random forest classifier (with optimized hyperparameters -- using results from GridSearchCV)
# rnd_clf_optimized = RandomForestClassifier(n_estimators=200, min_samples_leaf=100, min_samples_split=100, max_features=50, max_depth=20, criterion='entropy', random_state=42)
# rnd_clf_optimized.fit(X_train, y_train)
# y_pred = rnd_clf_optimized.predict(X_test)

# # Accuracy score (With optimized hyperparameters)
# print("Random Forest Classifier (Optimized Hyperparameters):", accuracy_score(y_test, y_pred))

# # Now, let's try performing bagging on the random forest classifier (with the optimized hyperparameters)
# rnd_clf_bagging = BaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=200, min_samples_leaf=100, min_samples_split=100, max_features=50, max_depth=20, criterion='entropy', random_state=42), n_estimators=10, random_state=42)
# rnd_clf_bagging.fit(X_train, y_train)
# y_pred = rnd_clf_bagging.predict(X_test)

# # Accuracy score (With optimized hyperparameters and bagging)
# print("Random Forest Classifier (Optimized Hyperparameters and Bagging):", accuracy_score(y_test, y_pred))

# Use GridSearchCV to try and find optimum hyperparameters for random trees classifier
# Define the parameter grid to search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': [20, 30, 50],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [100, 500, 1000],
    'min_samples_leaf': [100, 200, 500],
    'max_leaf_nodes': [10, 15, 20],
}
extra_trees_clf = ExtraTreesClassifier()

# Using negative mean squared error score for classification
grid_search = GridSearchCV(extra_trees_clf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print the best combination of hyperparameters
print("Best hyperparameters for Extra Trees:", grid_search.best_params_)
