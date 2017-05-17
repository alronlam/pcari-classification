def transform_raw_data_to_binary_data(pk_tweet_data_tuples, tweet_categories):
    # construct binary list (1/0) for each category
    labels = {}
    for index, tweet_category in enumerate(tweet_categories):
        print("Constructing data for {}".format(tweet_category))
        labels[tweet_category] = [ 1 if int(data[index]) > 0 else 0 for pk, tweet, data in pk_tweet_data_tuples]

    # collect all tweets
    tweet_list = [row[1] for row in pk_tweet_data_tuples]

    # construct final x,y tuples
    x_y_tuples = {}
    for category, data, in labels.items():
        x_y_tuples[category] = (tweet_list.copy(),data)

    return x_y_tuples