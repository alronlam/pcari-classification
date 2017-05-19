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