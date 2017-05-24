##################################
###          Functions         ###
##################################
import csv
import os

from sklearn.externals import joblib

import Utils
from classifier1 import DataParsing, DataLoading
from parsing.folders import FolderIO




def classify_on_dataset(model_file_names, months_to_classify_per_model, data_per_month):

    # pre-process data
    preprocessed_data_per_month = []
    for data in data_per_month:
        preprocessed_data_per_month.append(DataParsing.standard_preprocess_tweet_strings(data))

    all_results = {}
    for index, model_file_name in enumerate(model_file_names):
        model = joblib.load(Utils.construct_path_from_project_root("models/"+model_file_name))
        months = months_to_classify_per_model[index]
        model_results = {}

        for month in months:
            curr_data = data_per_month[month]
            model_results[month] = model.predict(curr_data)
        all_results[model_file_name] = model_results

    return all_results

def generate_csv(classification_results, data_per_month, output_dir=Utils.construct_path_from_project_root('data/weak_annotations')):
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


##################################
###          Constants         ###
##################################


model_file_names = ['Agonism or Engagement in Debate_model.pickle',
          'Celebrification_model.pickle',
          'Solidaristic_model.pickle',
          'Tweeting about a charity event (run, cookfest, walk, etc)_model.pickle']

months = [
    [0],
    [1],
    [1],
    [0]
]

##################################
###         Driver Code        ###
##################################

data_per_month = DataLoading.load_full_dataset_per_month()

classifications = classify_on_dataset(model_file_names, months, data_per_month)

generate_csv(classifications, data_per_month)