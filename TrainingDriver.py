
# Load CSV file
from collections import Counter

# Load input CSV file
import DataLoading

# Load input CSV file
import DataParsing

pk_tweet_data_tuples , tweet_categories = DataLoading.load_raw_data()

# Filter data with less than 100 instances
x_y_tuples = DataParsing.transform_raw_data_to_binary_data(pk_tweet_data_tuples, tweet_categories)

for category, (X, Y) in x_y_tuples.items():
    print("{}: {} instances - {}".format(category, len(X), Counter(Y)))

# Filter









