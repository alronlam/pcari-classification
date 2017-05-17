
# Load CSV file
from collections import Counter

# Load input CSV file
import numpy
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import _num_samples

import DataLoading

# Load input CSV file
import DataParsing
import DataPreprocessing


def print_dataset(x_y_tuples):
    print()
    for category, (X, Y) in x_y_tuples.items():
        print("{}: {} instances - {}".format(category, len(X), Counter(Y)))
        print("Sample X: {}".format(X[0]))



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


classification_pipeline = Pipeline(
    [
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB())
    ]
)

for category, (X,Y) in x_y_tuples.items():
    X_train = X
    Y_train = Y
    X_test = X
    Y_test = Y

    # X_train_counts = CountVectorizer().fit_transform(X_train)

    model = classification_pipeline.fit(X_train, Y_train)
    predicted = model.predict(X_test)
    print("\n\n")
    print(category)

    try:
        print(metrics.classification_report(Y_train, predicted))
        print(metrics.confusion_matrix(Y_train, predicted))
    except Exception as e:
        # print(e)
        pass