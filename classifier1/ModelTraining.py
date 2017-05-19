from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def train_one_model(X, Y):
    classification_pipeline = Pipeline(
        [
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB())
        ]
    )

    return classification_pipeline.fit(X,Y)

def train_model_for_each_category(x_y_tuples, verbose=False):
    category_model_tuples = []

    for category, (X, Y) in x_y_tuples.items():
        model = train_one_model(X,Y)
        category_model_tuples.append((category,model))
        if verbose:
            print("\n")
            print(category)
            print_model_training_accuracy(model, X, Y)

    return category_model_tuples


def print_model_training_accuracy(model, X_test, Y_test):
    predicted = model.predict(X_test)
    try:
        print(metrics.classification_report(Y_test, predicted))
        print(metrics.confusion_matrix(Y_test, predicted))
    except Exception as e:
        # print(e)
        pass