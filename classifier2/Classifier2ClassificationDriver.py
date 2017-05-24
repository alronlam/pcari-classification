import Utils
from classifier1 import DataLoading

model_file_names = ['Agonism or Engagement in Debate_model.pickle',
          'Celebrification_model.pickle',
          'Solidaristic_model.pickle',
          'Tweeting about a charity event (run, cookfest, walk, etc)_model.pickle']

months = [
    [0,1,2,3],
    [0,1,2,3],
    [0,1,2,3],
    [0,1,2,3]
]

data_per_month = DataLoading.load_full_dataset_per_month()

classifications = Utils.classify_on_dataset(model_file_names, months, data_per_month)

Utils.generate_csv(classifications, data_per_month, output_dir=Utils.construct_path_from_project_root('data/final_classifications'))
