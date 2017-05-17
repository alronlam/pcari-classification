
# Load CSV file
from collections import Counter

# Load input CSV file
import DataLoading

# Load input CSV file
import DataParsing
import DataPreprocessing

pk_tweet_data_tuples , tweet_categories = DataLoading.load_raw_data()

# Filter data with less than 100 instances
print("\nAll:\n")
x_y_tuples = DataParsing.transform_raw_data_to_binary_data(pk_tweet_data_tuples, tweet_categories)
for category, (X, Y) in x_y_tuples.items():
    print("{}: {} instances - {}".format(category, len(X), Counter(Y)))

x_y_tuples = DataPreprocessing.remove_categories_with_less_than_n(x_y_tuples, 100)
print("\nFiltered:\n")
for category, (X, Y) in x_y_tuples.items():
    print("{}: {} instances - {}".format(category, len(X), Counter(Y)))

# Filter









