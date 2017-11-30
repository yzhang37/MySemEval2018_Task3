#coding:utf8
import sys
sys.path.append("../..")
from src.util import singleton
from src.util import load_dict_from_file
from src import config
from src.word2vec import Word2Vec

@singleton
class Dict_loader(object):
    def __init__(self):
        self.dict_unigram = load_dict_from_file(config.DICT_UNIGRAM_T2)
        self.dict_nltk_unigram = load_dict_from_file(config.DICT_NLTK_UNIGRAM_T2)
        # self.dict_bigram = load_dict_from_file(config.DICT_BIGRAM_T3)

        self.google_vec = Word2Vec(config.WORD2VEC_GOOGLE)

        # # Sentiment_Lexicon
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
