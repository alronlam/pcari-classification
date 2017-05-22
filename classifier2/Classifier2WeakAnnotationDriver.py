def load_full_dataset_per_month():
    pass


data_per_month = load_full_dataset_per_month()

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

def classify_on_dataset(models, months, data_per_month):
    pass

classifications = classify_on_dataset(models, months, data_per_month)