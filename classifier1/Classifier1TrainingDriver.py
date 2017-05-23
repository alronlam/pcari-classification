
# Load CSV file
import os
from collections import Counter

# Load input CSV file
import numpy
from imblearn.under_sampling import RandomUnderSampler
from sklearn.externals import joblib

import Settings
import Utils
from classifier1 import DataLoading, ModelTraining, DataPreprocessing, DataParsing


# Load input CSV file


def print_dataset(x_y_tuples):
    print()
    for category, (X, Y) in x_y_tuples.items():
        print("{}: {} instances - {}".format(category, len(X), Counter(Y)))
        # print("Sample X: {}".format(X[0]))


pk_tweet_data_tuples , tweet_categories = DataLoading.load_raw_data()

x_y_tuples = DataParsing.transform_raw_data_to_binary_data(pk_tweet_data_tuples, tweet_categories)

# Filter data with less than 100 instances
x_y_tuples = DataPreprocessing.remove_categories_with_less_than_n(x_y_tuples, 100)

# Undersample
rus = RandomUnderSampler(return_indices=False)
for category, (X, Y) in x_y_tuples.items():
    X_undersampled, Y_undersampled = rus.fit_sample(numpy.reshape(X, (len(X), 1)),Y)
    X_undersampled = [x[0] for x in X_undersampled]
    x_y_tuples[category] = (X_undersampled, Y_undersampled)


# Pre-process
x_y_tuples = DataParsing.standard_preprocess_x_y_tuples(x_y_tuples)


print_dataset(x_y_tuples)

# Train models on entire dataset
print_file = open(Utils.construct_path_from_project_root("models/models_performance.txt"), "w")
category_model_tuples = ModelTraining.train_model_for_each_category(x_y_tuples, verbose=True, print_file=print_file)

for category, model in category_model_tuples:
    joblib.dump(model, Utils.construct_path_from_project_root("models/{}_model.pickle".format(category)))

joblib.dump(category_model_tuples, Utils.construct_path_from_project_root("models/category_model_tuples.pickle"))
