#coding:utf-8
import sys
sys.path.append("../..")
import json
from src import config
from src import util
from src.model_trainer import dict_util
from src.model_trainer import feature_functions
from src.model_trainer import rf_calculate


class Dict_creator(object):
    def __init__(self):
        self.texts = None

    def create_dict(self, dict_function, dict_path, threshold=1):
        print("=="*40)
        print("Create dict for %s ..." % (dict_function.__name__.replace("get_", "")))
        print("=="*40)
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


def load_traindata():
    data = json.load(open(config.PROCESSED_TRAIN, "r"), encoding="utf-8")
    return data


def calc_rc(data, feature_function, output_path):
    rf = rf_calculate.Rf_Calculator(data, config.get_label_map)
    rf.calc(feature_function, output_path)


if __name__ == '__main__':
    train_data = load_traindata() # load training data
    dict_creator = Dict_creator()
    dict_creator.texts = train_data

    dict_creator.create_dict(dict_util.get_nltk_unigram, config.DICT_NLTK_UNIGRAM_T2, threshold=2)
    calc_rc(train_data, feature_functions.nltk_unigram, config.RF_DATA_NLTK_UNIGRAM_PATH)

    # dict_creator.create_dict(dict_util.get_unigram, config.DICT_UNIGRAM_T1, threshold=1)
    # dict_creator.create_dict(dict_util.get_unigram, config.DICT_UNIGRAM_T2, threshold=2)
    # dict_creator.create_dict(dict_util.get_hashtag_unigram, config.DICT_HASHTAG_UNIGRAM_T1, threshold=1)
    # dict_creator.create_dict(dict_util.get_stem_unigram, config.DICT_UNIGRAM_STEM_T2, threshold=2)
    # dict_creator.create_dict(dict_util.get_bigram, config.DICT_BIGRAM_T3, threshold=3)
    # dict_creator.create_dict(dict_util.get_trigram, config.DICT_TRIGRAM_T5, threshold=5)


