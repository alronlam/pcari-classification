##################################
###          Functions         ###
##################################
import Utils
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

def classify_on_dataset(models, months, data_per_month):
    pass


##################################
###          Constants         ###
##################################


models = ['Agonism or Engagement in Debate_model.pickle',
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

classifications = classify_on_dataset(models, months, data_per_month)
