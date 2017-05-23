##################################
###          Functions         ###
##################################
from sklearn.externals import joblib

import Utils
from classifier1 import DataParsing
from parsing.folders import FolderIO


def load_full_dataset_per_month():
    files = FolderIO.get_files(Utils.construct_path_from_project_root('data/yolanda_tweets_nov2013_feb2014'), False, '.txt')

    data_per_month = []

    # Files are assumed to be ordered sequentially
    for index, file in enumerate(files):
        with open(file.absolute().__str__(), encoding='utf-8') as txt_file:
            file_lines = txt_file.readlines()
            file_lines = [line.strip() for line in file_lines]
            data_per_month.append(file_lines)

    return data_per_month

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

data_per_month = load_full_dataset_per_month()

classifications = classify_on_dataset(model_file_names, months, data_per_month)
