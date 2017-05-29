##################################
###          Functions         ###
##################################
import csv
import os

from sklearn.externals import joblib

import Utils
from classifier1 import DataParsing, DataLoading
from parsing.folders import FolderIO

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

classifications = Utils.classify_on_yolanda_dataset_per_month(model_file_names, months, data_per_month)

Utils.generate_csv_per_month(classifications, data_per_month)