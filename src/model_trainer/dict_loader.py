#coding:utf8
import sys
import numpy
import json
sys.path.append("../..")
from src.model_trainer.LazyLoader import LazyLoader
from src.util import singleton
from src.util import load_dict_from_file
from src import config
from src.word2vec import Word2Vec
from src.word2vec import Word2VecTag
from src.word2vec import GloVe
from src.model_trainer.dict_cacher import DictCache


@singleton
class DictLoader(LazyLoader):
    def __init__(self):
        super().__init__()
        self._map_name_to_handler = {
            "sent_BL": lambda: self.__dict_Senti_Lexi_0(config.LEXI_BL),
            "sent_GI": lambda: self.__dict_Senti_Lexi_0(config.LEXI_GI),
            "sent_IMDB": lambda: self.__dict_Senti_Lexi_0(config.LEXI_IMDB),
            "sent_MPQA": lambda: self.__dict_Senti_Lexi_0(config.LEXI_MPQA),
            "sent_NRCE": lambda: self.__dict_Senti_Lexi_0(config.LEXI_NRCEMOTION),
            "sent_AF": lambda: self.__dict_Senti_Lexi_1(config.LEXI_AFINN),
            "sent_NRC140_U": lambda: self.__dict_Senti_Lexi_1(config.LEXI_NRC140_U),
            "sent_NRCH_U": lambda: self.__dict_Senti_Lexi_1(config.LEXI_NRCHASHTAG_U),
            "sent_NRC140_B": lambda: self.__dict_Senti_Lexi_2(config.LEXI_NRC140_B),
            "sent_NRCH_B": lambda: self.__dict_Senti_Lexi_2(config.LEXI_NRCHASHTAG_B),
            "embed_Word2Vec": lambda: Word2Vec(config.WORD2VEC_GOOGLE),
            "embed_GloVe": self.__get_glove_handler,
            "url_crawled_data": lambda: self.__get_url_creeper_data(config.URL_CACHE_PATH),
        }
        # GloVe is too large, make cache for it.

        for freq in range(1, 6):
            self._map_name_to_handler["nltk_unigram_t%d" % freq] = lambda freq=freq: load_dict_from_file(
                config.DICT_NLTK_UNIGRAM_TU % freq)
            self._map_name_to_handler["nltk_bigram_t%d" % freq] = lambda freq=freq: load_dict_from_file(
                config.DICT_NLTK_BIGRAM_TU % freq)
            self._map_name_to_handler["nltk_trigram_t%d" % freq] = lambda freq=freq: load_dict_from_file(
                config.DICT_NLTK_TRIGRAM_TU % freq)
            self._map_name_to_handler["hashtag_t%d" % freq] = lambda freq=freq: load_dict_from_file(
                config.DICT_HASHTAG_TU % freq)
            self._map_name_to_handler["hashtag_unigram_t%d" % freq] = lambda freq=freq: \
                load_dict_from_file(config.DICT_HASHTAG_UNIGRAM_TU % freq)
            self._map_name_to_handler["url_unigram_t%d" % freq] = lambda freq=freq: load_dict_from_file(
                config.DICT_URL_UNIGRAM_TU % freq)


            self._map_name_to_handler["nltk_unigram_for_test_t%d" % freq] = lambda freq=freq: \
                load_dict_from_file(config.DICT_NLTK_UNIGRAM_TU_TEST % freq)

        for k, v in self._map_name_to_handler.items():
            try:
                v.__name__ = "%s_handler" % k
            except Exception:
                pass


    def __dict_Senti_Lexi_0(self, fLexi):
        """Bing Liu & General Inquirer & imdb & MPQA & NRCEmotion"""
        # format: word \t positive_score \t negative_score
        dict_ = {}
        f = open(fLexi, encoding="ISO-8859-1")
        for line in f:
            line = line.strip().split("\t")
            score = float(line[1]) - float(line[-1])
            dict_[line[0]] = score
        return dict_

    def __dict_Senti_Lexi_1(slef, fLexi):
        """AFINN & NRC140_U & NRCHash_U"""
        # format: word \t score
        dict_ = {}
        for line in open(fLexi,encoding="ISO-8859-1"):
            line = line.strip().split("\t")
            score = float(line[-1])
            dict_[line[0]] = score
        return dict_

    def __dict_Senti_Lexi_2(slef, fLexi):
        """NRC140_B & NRCHash_B"""
        dict_ = {}
        for line in open(fLexi,encoding="ISO-8859-1"):
            line = line.strip().split("\t")
            score = float(line[-1])
            dict_[tuple(line[0].split(" "))] = score
        return dict_

    def __get_glove_handler(self):

        def get_train_n_test_small_dict():
            train_small_dict = self.get("nltk_unigram_t2").keys()
            test_small_dict = self.get("nltk_unigram_for_test_t2").keys()
            return train_small_dict | test_small_dict

        glove_cache = DictCache(config.GLOVE_CACHE_PATH,
                                lambda: GloVe(config.GLOVE_840B_300_PATH, get_train_n_test_small_dict()).word2vec)
        glove_vec = {k: numpy.asarray(v) for k, v in glove_cache.load_dict().items()}
        return Word2VecTag(glove_vec)

    def __get_url_creeper_data(self, json_path):
        try:
            creep_data = json.load(open(json_path))
        except Exception as ex:
            print("Error: ", ex)
            creep_data = dict()
        return creep_data


@singleton
class UrlCrawledLoader(LazyLoader):
    def __init__(self):
        super().__init__()
        self._map_name_to_handler = {
            "url_cache": lambda :json.load(open(config.PROCESSED_URL_DATA))
        }