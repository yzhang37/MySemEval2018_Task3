#coding:utf-8
import sys
sys.path.append("../..")
from src import config
from src import util
from src.model_trainer import dict_util


class Dict_creator(object):
    def __init__(self):
        self.texts = None

    def create_dict(self, dict_function, dict_path, threshold=1):
        print("=="*40)
        print("Create dict for %s ..." % (dict_function.__name__.replace("get_", "")))
        dictionary = {}
        self._create_dict(self.texts, dict_function, dictionary)
        # 删除频率小于threshold的键
        util.removeItemsInDict(dictionary, threshold)
        # 字典keys写入文件
        util.write_dict_keys_to_file(dictionary, dict_path)

    def _create_dict(self, texts, dict_function, dictionary):
        for text in texts:
            result = dict_function(text)
            if type(result) == list:
                for item in result:
                    # 计算key频率
                    util.set_dict_key_value(dictionary, item)
            else:
                util.set_dict_key_value(dictionary, result)


def create_nltk_unigram_dict(dict_creator, freq):
    dict_creator.create_dict(dict_util.get_nltk_unigram, config.DICT_NLTK_UNIGRAM_TU % freq, threshold=freq)


def create_hashtag_dict(dict_creator, freq):
    dict_creator.create_dict(dict_util.get_hashtag, config.DICT_HASHTAG_TU % freq, threshold=freq)


def create_nltk_bigram_dict(dict_creator, freq):
    dict_creator.create_dict(dict_util.get_nltk_bigram, config.DICT_NLTK_BIGRAM_TU % freq, threshold=freq)


def create_nltk_trigram_dict(dict_creator, freq):
    dict_creator.create_dict(dict_util.get_nltk_trigram, config.DICT_NLTK_TRIGRAM_TU % freq, threshold=freq)


def create_hashtag_unigram_dict(dict_creator, freq):
    dict_creator.create_dict(dict_util.get_hashtag_unigram, config.DICT_HASHTAG_UNIGRAM_TU % freq, threshold=freq)