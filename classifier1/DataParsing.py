from preprocessing import PreProcessing


def transform_raw_data_to_binary_data(pk_tweet_data_tuples, tweet_categories):
    # construct binary list (1/0) for each category
    labels = {}
    for index, tweet_category in enumerate(tweet_categories):
        labels[tweet_category] = [ 1 if int(data[index]) > 0 else 0 for pk, tweet, data in pk_tweet_data_tuples]

    # collect all tweets
    tweet_list = [row[1] for row in pk_tweet_data_tuples]

    # construct final x,y tuples
    x_y_tuples = {}
    for category, data, in labels.items():
        x_y_tuples[category] = (tweet_list.copy(),data)

    return x_y_tuples

STANDARD_PREPROCESSORS = [
    PreProcessing.SplitWordByWhitespace(),
    PreProcessing.WordToLowercase(),
    PreProcessing.RemoveRT(),
    PreProcessing.ReplaceURL(),
    PreProcessing.ReplaceUsernameMention(),
    PreProcessing.RemovePunctuationFromWords(),
    PreProcessing.ConcatWordArray()
]

def standard_preprocess_x_y_tuples(x_y_tuples):

    preprocessed_x_y_tuples = {}

    for category, (X, Y) in x_y_tuples.items():
        preprocessed_X = PreProcessing.preprocess_strings(X, STANDARD_PREPROCESSORS)

        preprocessed_x_y_tuples[category] = (preprocessed_X, Y)

    return preprocessed_x_y_tuples

def standard_preprocess_tweet_strings(strings):
    return PreProcessing.preprocess_strings(strings, STANDARD_PREPROCESSORS)
