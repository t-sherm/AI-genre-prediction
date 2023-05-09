import numpy as np
import pandas as pd
import os
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
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

# 1) Random Forest (Without Optimized hyperparameters -- Initial Guess)
rnd_clf = RandomForestClassifier(n_estimators=100, max_leaf_nodes=10, n_jobs=-1, criterion='gini', random_state=42)
rnd_clf.fit(X_train, y_train)

# Accuracy score (Without optimized hyperparameters -- Initial Guess)
print("Random Forest Classifier (Without Optimized Hyperparameters) -- Train Set:", rnd_clf.score(X_train, y_train)) # 0.2436168317083726
print("Random Forest Classifier (Without Optimized Hyperparameters) -- Test Set:", rnd_clf.score(X_test, y_test)) # 0.23755218409530598


# # Use RandomSearchCV to try and find optimum hyperparameters for random trees classifier
# # Define the parameter grid to search
# param_grid = {
#     'n_estimators': [50, 100, 200, 500, 1000],
#     'max_features': [10, 20, 30, 50],
#     'max_depth': [5, 10, 20, 30, None],
#     'min_samples_split': [50, 100, 500, 1000],
#     'min_samples_leaf': [50, 100, 200, 500],
# }
# tree_clf = RandomForestClassifier()

# # Using negative mean squared error score for classification
# grid_search = RandomizedSearchCV(tree_clf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
# grid_search.fit(X_train, y_train)

# # Print the best combination of hyperparameters
# print("Best hyperparameters for Random Forest:", grid_search.best_params_)

# # Use RandomSearchCV to try and find optimum hyperparameters for extra trees classifier
# extra_trees_clf = ExtraTreesClassifier()

# # Using negative mean squared error score for classification
# grid_search = RandomizedSearchCV(extra_trees_clf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
# grid_search.fit(X_train, y_train)

# # Print the best combination of hyperparameters
# print("Best hyperparameters for Extra Trees:", grid_search.best_params_)

# 2) New random forest classifier (with optimized hyperparameters -- using results from RandomSearchCV)
rnd_clf_optimized = RandomForestClassifier(n_estimators=100, min_samples_split=100, min_samples_leaf=100, max_features=30, max_depth=None, criterion='gini', random_state=42)
rnd_clf_optimized.fit(X_train, y_train)

# Accuracy score (With optimized hyperparameters)
print("Random Forest Classifier (Optimized Hyperparameters) -- Train Set:", rnd_clf_optimized.score(X_train, y_train)) # 0.34618028154672503
print("Random Forest Classifier (Optimized Hyperparameters) -- Test Set:", rnd_clf_optimized.score(X_test, y_test)) # 0.3037368903370329

# 3) Now, let's try performing bagging on the random forest classifier (with the optimized hyperparameters)
rnd_clf_bagging = BaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=100, min_samples_split=100, min_samples_leaf=100, max_features=30, max_depth=None, criterion='gini', random_state=42), n_estimators=10, random_state=42)
rnd_clf_bagging.fit(X_train, y_train)

# Accuracy score (With optimized hyperparameters and bagging)
print("Random Forest Classifier (Optimized Hyperparameters and Bagging) -- Train Set:", rnd_clf_bagging.score(X_train, y_train)) # 0.32894636356693735
print("Random Forest Classifier (Optimized Hyperparameters and Bagging) -- Test Set:", rnd_clf_bagging.score(X_test, y_test)) # 0.30017309846247836

# 4) Extra Trees Classifier using optimized parameters
rnd_clf_extra_trees = ExtraTreesClassifier(n_estimators=50, min_samples_split=100, min_samples_leaf=500, max_features=50, max_depth=30, criterion='gini', random_state=42)
rnd_clf_extra_trees.fit(X_train, y_train)

# Accuracy score (Extra trees classifier)
print("Extra Trees Classifier (Optimized Hyperparameters) -- Train Set:", rnd_clf_extra_trees.score(X_train, y_train)) # 0.2623017590306239
print("Extra Trees Classifier (Optimized Hyperparameters) -- Test Set:", rnd_clf_extra_trees.score(X_test, y_test)) # 0.2559820792180022624 
