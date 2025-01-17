# encoding: utf-8
import sys
import json
sys.path.append("../..")
from src import config
from src.model_trainer import dict_creator
from src.model_trainer import rf_calculate


def load_traindata():
    data = json.load(open(config.PROCESSED_TRAIN), encoding="utf-8")
    return data


def load_testdata():
    data = json.load(open(config.PROCESSED_TEST), encoding="utf-8")
    return data


def create_nltk_unigram(creator, freq):
    dict_creator.create_nltk_unigram_dict(creator, freq)
    rf_calculate.create_nltk_unigram_rf(creator.texts, freq)


def create_hashtag(creator, freq):
    dict_creator.create_hashtag_dict(creator, freq)
    rf_calculate.create_hashtag_rf(creator.texts, freq)


def create_nltk_bigram(creator, freq):
    dict_creator.create_nltk_bigram_dict(creator, freq)
    rf_calculate.create_nltk_bigram_rf(creator.texts, freq)


def create_nltk_trigram(creator, freq):
    dict_creator.create_nltk_trigram_dict(creator, freq)
    rf_calculate.create_nltk_trigram_rf(creator.texts, freq)


def create_hashtag_unigram(creator, freq):
    dict_creator.create_hashtag_unigram_dict(creator, freq)
    rf_calculate.create_hashtag_unigram_rf(creator.texts, freq)


def create_nltk_unigram_for_test(creator, freq):
    dict_creator.create_nltk_unigram_dict_for_test(creator, freq)


def create_url_unigram(creator, freq):
    dict_creator.create_url_unigram(creator, freq)


if __name__ == '__main__':
    task = "train"

    if task == "train":
        print("Make train dictionary...")
        # load training data
        train_data = load_traindata()
        d_creator = dict_creator.Dict_creator()
        d_creator.texts = train_data

        for f in range(1, 6):
            # create_hashtag(d_creator, f)
            # create_nltk_bigram(d_creator, f)
            # create_nltk_unigram(d_creator, f)
            # create_nltk_trigram(d_creator, f)
            # create_hashtag_unigram(d_creator, f)
            create_url_unigram(d_creator, f)

    elif task == "test":
        print("Make test dictionary...")
        test_data = load_testdata()
        d_creator = dict_creator.Dict_creator()
        d_creator.texts = test_data

        for f in range(1, 6):
            create_nltk_unigram_for_test(d_creator, f)

    # dict_creator.create_dict(dict_util.get_hashtag_unigram, config.DICT_HASHTAG_UNIGRAM_T1, threshold=1)
    # dict_creator.create_dict(dict_util.get_stem_unigram, config.DICT_UNIGRAM_STEM_T2, threshold=2)
    # dict_creator.create_dict(dict_util.get_bigram, config.DICT_BIGRAM_T3, threshold=3)
    # dict_creator.create_dict(dict_util.get_trigram, config.DICT_TRIGRAM_T5, threshold=5)

    from src.model_trainer import feature_functions
