import os
from collections import Counter

# Load input CSV file
import numpy
from imblearn.under_sampling import RandomUnderSampler
from sklearn.externals import joblib

import Settings
import Utils
from classifier1 import DataLoading, ModelTraining, DataPreprocessing, DataParsing

def print_dataset(x_y_tuples):
    print()
    for category, (X, Y) in x_y_tuples.items():
        print("{}: {} instances - {}".format(category, len(X), Counter(Y)))
        # print("Sample X: {}".format(X[0]))


def merge_two_tuples(x_y_tuples1, x_y_tuples2):
    merged_x_y_tuples = {}
    keys = [key for key in x_y_tuples1]

    for key in keys:
        x1, y1 = x_y_tuples1[key]
        x2, y2 = x_y_tuples2[key]

        #remove duplicates
        x_y_list_of_tuples1 = [(x1[index], y1[index])for index in range(len(x1))]
        x_y_list_of_tuples2 = [(x2[index], y2[index])for index in range(len(x2))]

        merged_x_y_list_of_tuples = list(set(x_y_list_of_tuples1 + x_y_list_of_tuples2))

        merged_x_list = [tuple[0] for tuple in merged_x_y_list_of_tuples]
        merged_y_list = [tuple[1] for tuple in merged_x_y_list_of_tuples]

        merged_x_y_tuples[key] = (merged_x_list, merged_y_list)

    return merged_x_y_tuples


pk_tweet_data_tuples , tweet_categories = DataLoading.load_raw_data()
strong_x_y_tuples = DataParsing.transform_raw_data_to_binary_data(pk_tweet_data_tuples, tweet_categories)
strong_x_y_tuples = DataPreprocessing.remove_categories_with_less_than_n(strong_x_y_tuples, 100)

weak_x_y_tuples = DataLoading.load_weak_annotations_binary_data()


# merge strong and weak x y tuples
x_y_tuples = merge_two_tuples(strong_x_y_tuples , weak_x_y_tuples)

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
print_file = open(Utils.construct_path_from_project_root("models2/models_performance.txt"), "w")
category_model_tuples = ModelTraining.train_model_for_each_category(x_y_tuples, verbose=True, print_file=print_file)

for category, model in category_model_tuples:
    joblib.dump(model, Utils.construct_path_from_project_root("models2/{}_model.pickle".format(category)))

joblib.dump(category_model_tuples, Utils.construct_path_from_project_root("models2/category_model_tuples.pickle"))
