import abc


class TopicModeller(object):

    @abc.abstractmethod
    def generate_topic_models_and_string(self, document_list):
        """
        :param document_list: list of text documents for topic modelling
        :return:
        """
