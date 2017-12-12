# coding: utf-8
import socket
import os
import pwd

__CLASS = "B"


def get_label_map(x):
    if __CLASS == "A":
        if x == "0":
            return "0"
        else:
            return "1"
    else:
        return x


def get_label_list():
    if __CLASS == "A":
        return [0, 1]
    else:
        return [0, 1, 2, 3]

hostname = socket.gethostname()
cur_user = pwd.getpwuid(os.getuid())[0]
if hostname == "precision":
    if cur_user.lower() == "feixiang":
        CWD = "/home/feixiang/pyCharmSpace/SemEval2018_T3"
        PCCMD = "/home/feixiang/pyCharmSpace"
        YXPCCMD = "/home/yunxiao/workspace/SemEval2017_T4"
        LIB_LINEAR_PATH = "/home/feixiang/tools/liblinear-multicore-2.11-1"
        GLOVE_TWITTER_PATH = "/home/zhenghang/dict/GloVe"
        GLOVE_PATH = "/home/junfeng/GloVe"
    elif cur_user.lower() == "zhenghang":
        CWD = "/home/zhenghang/SemEval2018_T3"
        PCCMD = "/home/feixiang/pyCharmSpace"
        YXPCCMD = "/home/yunxiao/workspace/SemEval2017_T4"
        LIB_LINEAR_PATH = "/home/feixiang/tools/liblinear-multicore-2.11-1"
        GLOVE_TWITTER_PATH = "/home/zhenghang/dict/GloVe"
        GLOVE_PATH = "/home/junfeng/GloVe"
elif hostname.lower().startswith("l-mbookpro") or hostname.startswith("192.168"):
    CWD = "/Users/l/Projects/Python/MySemEval2018_Task3"
    PCCMD = "/Users/l/Projects/External/feixiang/pyCharmSpace"
    YXPCCMD = "/Users/l/Projects/External/yunxiao/SemEval2017_T4"
    LIB_LINEAR_PATH = "/Users/l/.tools/liblinear-multicore-2.11-1"
    GLOVE_TWITTER_PATH = "/Users/l/Projects/External/zhenghang/dict/GloVe"
    GLOVE_PATH = "/Users/l/Projects/External/junfeng/GloVe"
else:
    assert False, "没有设置启动路径。"

SLANGS_PATH = PCCMD + "/data/slangs"
NORMAL_WORDS_PATH = PCCMD + "/data/normal_word.pkl"
EMOTICON = PCCMD + "/data/Emoticon.txt"

DATA_PATH = CWD + "/data"
DICT_PATH = CWD + "/dict"
DICT_CACHE_PATH = os.path.join(CWD, "dict_cache")
FEATURE_PATH = CWD + "/feature"
MODEL_PATH = CWD + "/model/binary_clf.model"
RESULT_PATH = CWD + "/result/predict.txt"
RESULT_MYDIR = os.path.join(CWD, "result")

RAW_TRAIN = DATA_PATH + "/train/SemEval2018-T4-train-task%s.txt" % __CLASS.upper()

PROCESSED_TRAIN = DATA_PATH + "/train/processed_train_%s.json" % __CLASS.lower()

DICT_UNIGRAM_T2 = os.path.join(DICT_PATH, "unigram_t2.txt")
DICT_UNIGRAM_T1 = os.path.join(DICT_PATH, "unigram_t1.txt")
DICT_HASHTAG_UNIGRAM_T1 = os.path.join(DICT_PATH, "hashtag_unigram_t1.txt")
DICT_HASHTAG_T1 = os.path.join(DICT_PATH, "hashtag_t1.txt")
DICT_HASHTAG_T2 = os.path.join(DICT_PATH, "hashtag_t2.txt")
DICT_UNIGRAM_STEM_T2 = DICT_PATH + "/unigram_stem_t2.txt"
DICT_BIGRAM_T3 = DICT_PATH + "/bigram_t3.txt"
DICT_TRIGRAM_T5 = DICT_PATH + "/trigram_t5.txt"

DICT_NLTK_UNIGRAM_T1 = os.path.join(DICT_PATH, "nltk_unigram_t1.txt")
DICT_NLTK_UNIGRAM_T2 = os.path.join(DICT_PATH, "nltk_unigram_t2.txt")
DICT_NLTK_BIGRAM_T1 = os.path.join(DICT_PATH, "nltk_bigram_t1.txt")
DICT_NLTK_BIGRAM_T2 = os.path.join(DICT_PATH, "nltk_bigram_t2.txt")
DICT_NLTK_BIGRAM_T3 = os.path.join(DICT_PATH, "nltk_bigram_t3.txt")
DICT_NLTK_BIGRAM_T4 = os.path.join(DICT_PATH, "nltk_bigram_t4.txt")
DICT_NLTK_BIGRAM_T5 = os.path.join(DICT_PATH, "nltk_bigram_t5.txt")
DICT_NLTK_TRIGRAM_T1 = os.path.join(DICT_PATH, "nltk_trigram_t1.txt")
DICT_NLTK_TRIGRAM_T2 = os.path.join(DICT_PATH, "nltk_trigram_t2.txt")
DICT_NLTK_TRIGRAM_T3 = os.path.join(DICT_PATH, "nltk_trigram_t3.txt")
DICT_NLTK_TRIGRAM_T4 = os.path.join(DICT_PATH, "nltk_trigram_t4.txt")
DICT_NLTK_TRIGRAM_T5 = os.path.join(DICT_PATH, "nltk_trigram_t5.txt")

WORD2VEC_GOOGLE = os.path.join(PCCMD, "SemEval2017_T8/data_new/Google.txt")
VOCABULARY_PATH = os.path.join(YXPCCMD, "vocabulary")
NEGATION_PATH = VOCABULARY_PATH + "/negation terms.txt"

LEXI_SOURCE = os.path.join(YXPCCMD, "data/Senti_Lexi")
LEXI_BL = LEXI_SOURCE + "/Bing Liu/BL.lexi"
LEXI_AFINN = LEXI_SOURCE + "/AFINN/AFINN-111.lexi"
LEXI_GI = LEXI_SOURCE + "/General Inquirer/GI.lexi"
LEXI_IMDB = LEXI_SOURCE + "/imdb/IMDB.lexi"
LEXI_MPQA = LEXI_SOURCE + "/MPQA/MPQA.lexi"
LEXI_NRC140_U = LEXI_SOURCE + "/NRC140/Nunigram.lexi"
LEXI_NRC140_B = LEXI_SOURCE + "/NRC140/Nbigram.lexi"
LEXI_NRCEMOTION = LEXI_SOURCE + "/NRCEmotion/NRCEmoti.lexi"
LEXI_NRCHASHTAG_U = LEXI_SOURCE + "/NRCHashtag/Hunigram.lexi"
LEXI_NRCHASHTAG_B = LEXI_SOURCE + "/NRCHashtag/Hbigram.lexi"
LEXI_SENTIWORDNET = LEXI_SOURCE + "/SentiWordNet_3.0.0/SentiWordNet.lexi"

GLOVE_TWITTER_27B_25_PATH = os.path.join(GLOVE_TWITTER_PATH, "glove.twitter.27B.25d.txt")

GLOVE_840B_300_PATH = os.path.join(GLOVE_PATH, "glove.840B.300d.txt")

TRAIN_FEATURE_PATH = FEATURE_PATH + "/train.fea.txt"
DEV_FEATURE_PATH = FEATURE_PATH + "/dev.fea.txt"

GLOVE_CACHE_PATH = os.path.join(DICT_CACHE_PATH, "glove.small.300d.txt")



# the following is RF files
RF_DATA_NLTK_UNIGRAM_PATH = os.path.join(RESULT_MYDIR, "rf_nltk_unigram_%s.txt" % __CLASS.lower())
RF_DATA_HASHTAG_PATH = os.path.join(RESULT_MYDIR, "rf_hashtag_%s.txt" % __CLASS.lower())
RF_DATA_NLTK_BIGRAM_PATH = os.path.join(RESULT_MYDIR, "rf_nltk_bigram_%s.txt" % __CLASS.lower())
RF_DATA_NLTK_TRIGRAM_PATH = os.path.join(RESULT_MYDIR, "rf_nltk_trigram_%s.txt" % __CLASS.lower())