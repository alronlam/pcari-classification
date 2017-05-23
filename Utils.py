import os

import Settings


def construct_path_from_project_root(path):
    return os.path.join(Settings.PROJECT_ROOT, path)


from collections import OrderedDict


def count_themes_per_month(pk_tweet_data_month_tuples, tweet_categories):

    themes_per_month = OrderedDict()

    for tweet_category in tweet_categories:
        themes_per_month[tweet_category] = [0 for x in range(5)]

    for pk, tweet, data, month in pk_tweet_data_month_tuples:
        for index, value in enumerate(data):
            category = tweet_categories[index]
            if int(value) > 0:
                themes_per_month[category][int(month)-1] += 1

    # print(themes_per_month)

    for category, frequency_count in themes_per_month.items():
        print(category)
        print(frequency_count)

