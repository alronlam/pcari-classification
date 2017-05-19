
# Load CSV file
from collections import Counter

# Load input CSV file
import numpy
from imblearn.under_sampling import RandomUnderSampler
from sklearn.externals import joblib

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
print_dataset(x_y_tuples)

# Undersample
rus = RandomUnderSampler(return_indices=False)
for category, (X, Y) in x_y_tuples.items():
    X_undersampled, Y_undersampled = rus.fit_sample(numpy.reshape(X, (len(X), 1)),Y)
    X_undersampled = [x[0] for x in X_undersampled]
    x_y_tuples[category] = (X_undersampled, Y_undersampled)


print_dataset(x_y_tuples)

# Train models on entire dataset
category_model_tuples = ModelTraining.train_model_for_each_category(x_y_tuples, verbose=True)

for category, model in category_model_tuples:
    print(category)

joblib.dump(category_model_tuples, "category_model_tuples.pickle")
