import os

import pickle

import Utils
from parsing.csv_parser import CSVParser
from parsing.folders import FolderIO


# CONSTANTS
from preprocessing import PreProcessing
from topic_modelling.LDATopicModeller import LDATopicModeller

YOLANDA_CLASSIFICATIONS_DIR = Utils.construct_path_from_project_root('data/final_classifications')
YOLANDA_TOPIC_MODELS_DIR = os.path.join(YOLANDA_CLASSIFICATIONS_DIR, 'topic_models')

lda_topic_modeller = LDATopicModeller()

# load classifications
files = FolderIO.get_files(YOLANDA_CLASSIFICATIONS_DIR, False, '.csv')

for file in files:

    print("Processing {}".format(file.stem))

    # run topic modelling
    category = file.stem
    csv_rows = CSVParser.parse_file_into_csv_row_generator(file, True, encoding='utf-8')
    positive_texts = [csv_row[0] for csv_row in csv_rows if str(csv_row[2]) == '1']

    preprocessors = [
        PreProcessing.SplitWordByWhitespace(),
        PreProcessing.WordToLowercase(),
        PreProcessing.RemoveRT(),
        PreProcessing.ReplaceURL(replacement_token=""),
        PreProcessing.ReplaceUsernameMention(replacement_token=""),
        PreProcessing.RemovePunctuationFromWords(),
        PreProcessing.RemoveLetterRepetitions(),
        # PreProcessing.RemoveExactTerms(['…', '”', '“', 'haiyan', 'yolanda', 'quot', 'typhoon', 'yolandaph', 'amp', '’', 'apos', 'haha', 'hahaha']),
        PreProcessing.RemoveExactTerms(Utils.load_function_words(Utils.construct_path_from_project_root('preprocessing/other-function-words.txt'), encoding='utf-8')),
        PreProcessing.RemoveExactTerms(Utils.load_function_words(Utils.construct_path_from_project_root('preprocessing/fil-function-words.txt'))),
        PreProcessing.RemoveDigits(),
        PreProcessing.WordLengthFilter(3),
        PreProcessing.RemoveEmptyStrings(),
        PreProcessing.ConcatWordArray()
    ]
    positive_texts = PreProcessing.preprocess_strings(positive_texts, preprocessors)
    print("{} tweets".format(len(positive_texts)))
    topic_model, topic_model_string = lda_topic_modeller.generate_topic_models_and_string(positive_texts)


    # save output (pickle + readable txt format)
    topic_model_file = open(os.path.join(YOLANDA_TOPIC_MODELS_DIR, category+".pickle"), "wb")
    pickle.dump(topic_model, topic_model_file)
    topic_model_file.close()

    topic_model_string_file = open(os.path.join(YOLANDA_TOPIC_MODELS_DIR, category+".txt"), "w", encoding='utf-8')
    topic_model_string_file.write(topic_model_string)
    topic_model_string_file.close()



