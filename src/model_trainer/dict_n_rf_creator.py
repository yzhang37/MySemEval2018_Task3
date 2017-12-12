# encoding: utf-8
import sys
import json
sys.path.append("../..")
from src import config
from src.model_trainer import dict_creator
from src.model_trainer import rf_calculate


def load_traindata():
    data = json.load(open(config.PROCESSED_TRAIN, "r"), encoding="utf-8")
    return data


def create_nltk_unigram(creator):
    dict_creator.create_nltk_unigram_dict(creator)
    rf_calculate.create_nltk_unigram_rf(creator.texts)


def create_hashtag(creator):
    dict_creator.create_hashtag_dict(creator)
    rf_calculate.create_hashtag_rf(creator.texts)


def create_nltk_bigram(creator):
    dict_creator.create_nltk_bigram_dict(creator)
    rf_calculate.create_nltk_bigram_rf(creator.texts)


def create_nltk_trigram(creator):
    dict_creator.create_nltk_trigram_dict(creator)
    rf_calculate.create_nltk_trigram_rf(creator.texts)


if __name__ == '__main__':
    train_data = load_traindata() # load training data
    d_creator = dict_creator.Dict_creator()
    d_creator.texts = train_data

    # create_hashtag(d_creator)
    # create_nltk_bigram(d_creator)
    # create_nltk_unigram(d_creator)
    create_nltk_trigram(d_creator)

    # dict_creator.create_dict(dict_util.get_hashtag_unigram, config.DICT_HASHTAG_UNIGRAM_T1, threshold=1)
    # dict_creator.create_dict(dict_util.get_stem_unigram, config.DICT_UNIGRAM_STEM_T2, threshold=2)
    # dict_creator.create_dict(dict_util.get_bigram, config.DICT_BIGRAM_T3, threshold=3)
    # dict_creator.create_dict(dict_util.get_trigram, config.DICT_TRIGRAM_T5, threshold=5)

    from src.model_trainer import feature_functions
