#coding:utf8
import sys
import numpy
sys.path.append("../..")
from src.util import singleton
from src.util import load_dict_from_file
from src import config
from src.word2vec import Word2Vec
from src.word2vec import Word2VecTag
from src.word2vec import GloVe
from src.model_trainer.dict_cacher import DictCache


@singleton
class Dict_loader(object):
    def __init__(self):
        self.dict_unigram = load_dict_from_file(config.DICT_UNIGRAM_T2)

        self.dict_nltk_unigram_t = {}
        self.dict_nltk_bigram_t = {}
        self.dict_nltk_trigram_t = {}
        self.dict_hashtag_t = {}
        for freq in range(1, 6):
            self.dict_nltk_unigram_t[freq] = load_dict_from_file(config.DICT_NLTK_UNIGRAM_TU % freq)
            self.dict_nltk_bigram_t[freq] = load_dict_from_file(config.DICT_NLTK_BIGRAM_TU % freq)
            self.dict_nltk_trigram_t[freq] = load_dict_from_file(config.DICT_NLTK_TRIGRAM_TU % freq)
            self.dict_hashtag_t[freq] = load_dict_from_file(config.DICT_HASHTAG_TU % freq)
            pass

        # self.dict_bigram = load_dict_from_file(config.DICT_BIGRAM_T3)
        self.google_vec = Word2Vec(config.WORD2VEC_GOOGLE)
        # Sentiment_Lexicon
        self.dict_BL = self._dict_Senti_Lexi_0(config.LEXI_BL)
        self.dict_GI = self._dict_Senti_Lexi_0(config.LEXI_GI)
        self.dict_IMDB = self._dict_Senti_Lexi_0(config.LEXI_IMDB)
        self.dict_MPQA = self._dict_Senti_Lexi_0(config.LEXI_MPQA)
        self.dict_NRCE = self._dict_Senti_Lexi_0(config.LEXI_NRCEMOTION)
        self.dict_AF = self._dict_Senti_Lexi_1(config.LEXI_AFINN)
        self.dict_NRC140_U = self._dict_Senti_Lexi_1(config.LEXI_NRC140_U)
        self.dict_NRCH_U = self._dict_Senti_Lexi_1(config.LEXI_NRCHASHTAG_U)
        self.dict_NRC140_B = self._dict_Senti_Lexi_2(config.LEXI_NRC140_B)
        self.dict_NRCH_B = self._dict_Senti_Lexi_2(config.LEXI_NRCHASHTAG_B)

        # GloVe is too large, make cache for it.
        glove_cache = DictCache(config.GLOVE_CACHE_PATH, self._load_glove)
        glove_vec = {k: numpy.asarray(v) for k, v in glove_cache.load_dict().items()}
        self.dict_glove_vec = Word2VecTag(glove_vec)

    def _load_glove(self):
        glove = GloVe(config.GLOVE_840B_300_PATH, self.dict_nltk_unigram_t[2].keys())
        return glove.word2vec

    def _dict_Senti_Lexi_0(slef, fLexi):
        """Bing Liu & General Inquirer & imdb & MPQA & NRCEmotion"""
        # format: word \t positive_score \t negative_score
        dict_ = {}

        f = open(fLexi, encoding="ISO-8859-1")
        for line in f:
            line = line.strip().split("\t")
            score = float(line[1]) - float(line[-1])
            dict_[line[0]] = score

        return dict_

    def _dict_Senti_Lexi_1(slef, fLexi):
        """AFINN & NRC140_U & NRCHash_U"""
        # format: word \t score
        dict_ = {}

        for line in open(fLexi,encoding="ISO-8859-1"):
            line = line.strip().split("\t")
            score = float(line[-1])
            dict_[line[0]] = score

        return dict_

    def _dict_Senti_Lexi_2(slef, fLexi):
        """NRC140_B & NRCHash_B"""
        dict_ = {}

        for line in open(fLexi,encoding="ISO-8859-1"):
            line = line.strip().split("\t")
            score = float(line[-1])
            dict_[tuple(line[0].split(" "))] = score

        return dict_

