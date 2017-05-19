from classifier1 import DataLoading
from classifier1.classification import Utils

pk_tweet_data_month_tuples, tweet_categories = DataLoading.load_raw_data_with_months()
Utils.count_themes_per_month(pk_tweet_data_month_tuples, tweet_categories)

