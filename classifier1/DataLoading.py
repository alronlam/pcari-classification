import csv

import Utils
from parsing.csv_parser import CSVParser
from parsing.folders import FolderIO

def load_raw_data(folder_path="../data"):
    csv_files = FolderIO.get_files(folder_path, False, '.csv')
    csv_rows = CSVParser.parse_files_into_csv_row_generator(csv_files, False)

    # Parse input into lists/tuples
    pk_tweet_data_tuples = []
    tweet_categories = []

    for index, csv_row in enumerate(csv_rows):
        if index != 0:
            pk = csv_row[0]
            tweet = csv_row[1]
            data = csv_row[3:]

            pk_tweet_data_tuples.append((pk, tweet, data))
        else:
            tweet_categories = csv_row[3:]

    return pk_tweet_data_tuples, tweet_categories

def load_raw_data_with_months(folder_path="../data"):
    csv_files = FolderIO.get_files(folder_path, False, '.csv')
    csv_rows = CSVParser.parse_files_into_csv_row_generator(csv_files, False)

    # Parse input into lists/tuples
    pk_tweet_data_month_tuples = []
    tweet_categories = []

    for index, csv_row in enumerate(csv_rows):
        if index != 0:
            pk = csv_row[0]
            tweet = csv_row[1]
            month = csv_row[2]
            data = csv_row[3:]

            pk_tweet_data_month_tuples.append((pk, tweet, data, month))
        else:
            tweet_categories = csv_row[3:]

    return pk_tweet_data_month_tuples, tweet_categories

def load_weak_annotations_binary_data(folder_path=Utils.construct_path_from_project_root('data/weak_annotations')):
    csv_file_paths = FolderIO.get_files(folder_path, False, '.csv')

    x_y_tuples = {}

    for csv_file_path in csv_file_paths:

        csv_row_generator = CSVParser.parse_file_into_csv_row_generator(csv_file_path, True, encoding='utf-8')
        curr_X = [row[0] for row in csv_row_generator]

        csv_row_generator = CSVParser.parse_file_into_csv_row_generator(csv_file_path, True, encoding='utf-8')
        curr_Y = [int(row[2]) for row in csv_row_generator]

        category = csv_file_path.stem

        x_y_tuples[category] = (curr_X, curr_Y)

    return x_y_tuples

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


