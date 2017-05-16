
# Load CSV file
from parsing.csv_parser import CSVParser
from parsing.folders import FolderIO

csv_files = FolderIO.get_files('data/', False, '.csv')
csv_rows = CSVParser.parse_files_into_csv_row_generator(csv_files, False)

pk_tweet_data_tuples = []
tweet_categories = []

for index, csv_row in enumerate(csv_rows):
    if index != 0:
        pk = csv_row[0]
        tweet = csv_row[1]
        data = csv_row[2:]

        pk_tweet_data_tuples.append((pk, tweet, data))
    else:
        tweet_categories = csv_row[2:]


labels = {}
for index, tweet_category in enumerate(tweet_categories):
    print("Constructing data for {}".format(tweet_category))
    labels[tweet_category] = [ 1 if int(data[index]) > 0 else 0 for pk, tweet, data in pk_tweet_data_tuples]

for category, data in labels.items():
    print("{} - {} instances - {}".format(category, len(data)))








