import csv
import os

from sklearn.externals import joblib

import Settings
from classifier1 import DataParsing


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


def classify_on_dataset(model_file_names, months_to_classify_per_model, data_per_month):

    # pre-process data
    preprocessed_data_per_month = []
    for data in data_per_month:
        preprocessed_data_per_month.append(DataParsing.standard_preprocess_tweet_strings(data))

    all_results = {}
    for index, model_file_name in enumerate(model_file_names):
        model = joblib.load(construct_path_from_project_root("models/"+model_file_name))
        months = months_to_classify_per_model[index]
        model_results = {}

        for month in months:
            curr_data = data_per_month[month]
            model_results[month] = model.predict(curr_data)
        all_results[model_file_name] = model_results

    return all_results

def generate_csv(classification_results, data_per_month, output_dir=construct_path_from_project_root('data/weak_annotations')):
    header_data = []
    for file_name, month_values in classification_results.items():

        csv_file_name = os.path.join(output_dir, file_name+".csv")
        with open(csv_file_name, 'w', encoding='utf-8', newline='') as csv_file:

            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['tweet', 'month', 'classification'])

            for month, values in month_values.items():
                for index, value in enumerate(values):
                    tweet = data_per_month[month][index]
                    csv_writer.writerow([tweet, month, value])


