# coding: utf-8
import socket
import os
import pwd
import time
import uuid


__CLASS = "B"
__MULTIBIN = True


if __CLASS == "B" and __MULTIBIN:
    def get_binary_label_map(cur_label, x):
        return "1" if str(cur_label) == str(x) else "0"

    def get_label_map(x):
        return x

    def get_label_list():
        return [0, 1]

    def get_all_label_list():
        return [0, 1, 2, 3]

else:
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


def get_class():
    return __CLASS.upper()


def if_multi_binary():
    return __CLASS == "B" and __MULTIBIN


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
elif hostname == "tembusu":
    if cur_user.lower() == "feixiang":
        assert False, "没有设置启动路径。"
    elif cur_user.lower() == "zhenghang":
        CWD = "/home/zhenghang/projects/python/SemEval2018_T3"
        PCCMD = "/home/zhenghang/External/feixiang"
        YXPCCMD = "/home/zhenghang/External/yunxiao/SemEval2017_T4"
        LIB_LINEAR_PATH = "/home/feixiang/tools/liblinear-multicore-2.11-1"
        GLOVE_TWITTER_PATH = "/home/zhenghang/dict/GloVe"
        GLOVE_PATH = "/home/zhenghang/dict/GloVe"
elif hostname.lower().startswith("l-mbookpro") or hostname.startswith("192.168"):
    CWD = "/Users/l/Projects/Python/MySemEval2018_Task3"
    PCCMD = "/Users/l/Projects/External/feixiang/pyCharmSpace"
    YXPCCMD = "/Users/l/Projects/External/yunxiao/SemEval2017_T4"
    LIB_LINEAR_PATH = "/Users/l/.tools/liblinear-multicore-2.11-1"
    GLOVE_TWITTER_PATH = "/Users/l/Projects/External/zhenghang/dict/GloVe"
    GLOVE_PATH = "/Users/l/Projects/External/junfeng/GloVe"
else:
    raise NotImplementedError("Required path not implemented.")


def __chkfold(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


if if_multi_binary():
    MULTI_BINARY_ROOT = __chkfold(os.path.join(CWD, "multi_binary_1000"))

SLANGS_PATH = PCCMD + "/data/slangs"
NORMAL_WORDS_PATH = PCCMD + "/data/normal_word.pkl"
EMOTICON = PCCMD + "/data/Emoticon.txt"

DATA_PATH = os.path.join(CWD, "data")
DICT_PATH = CWD + "/dict"
DICT_CACHE_PATH = os.path.join(CWD, "dict_cache")
MODEL_PATH = CWD + "/model/binary_clf.model"

if if_multi_binary():
    FEATURE_PATH = __chkfold(os.path.join(MULTI_BINARY_ROOT, "feature"))
    RESULT_MYDIR = __chkfold(os.path.join(MULTI_BINARY_ROOT, "result"))
    HILLCLIMB_ROOT_PATH = __chkfold(os.path.join(MULTI_BINARY_ROOT, "hill_climb"))
    ENSEMBLE_SCORE_PATH = __chkfold(os.path.join(MULTI_BINARY_ROOT, "ensemble_score"))
    ENSEMBLE_PATH = __chkfold(os.path.join(MULTI_BINARY_ROOT, "ensemble"))
    MODEL_MYDIR = __chkfold(os.path.join(MULTI_BINARY_ROOT, "model"))
else:
    FEATURE_PATH = __chkfold(os.path.join(CWD, "feature"))
    RESULT_MYDIR = __chkfold(os.path.join(CWD, "result"))
    HILLCLIMB_ROOT_PATH = __chkfold(os.path.join(CWD, "hill_climb"))
    ENSEMBLE_SCORE_PATH = __chkfold(os.path.join(CWD, "ensemble_score"))
    ENSEMBLE_PATH = __chkfold(os.path.join(CWD, "ensemble"))
    MODEL_MYDIR = __chkfold(os.path.join(CWD, "model"))

RESULT_EXCEL_PATH = __chkfold(os.path.join(CWD, "result_excel"))

RELATION_FREQ_PATH = __chkfold(os.path.join(CWD, "RelFreq"))

RAW_TRAIN = os.path.join(DATA_PATH, "train", "SemEval2018-T4-train-task%s.txt" % __CLASS.upper())
RAW_TEST = os.path.join(DATA_PATH, "test", "SemEval2018-T3_input_test_task%s.txt" % __CLASS.upper())
RAW_GOLDEN_TEST = os.path.join(DATA_PATH, "test", "SemEval2018-T3_gold_test_task%s_emoji.txt" % __CLASS.upper())

# here, we can use config.get_class_map to convert class b to class a.
PROCESSED_TRAIN = os.path.join(DATA_PATH, "train", "processed_train_%s.json" % "b")
# due to the unknown of class label, a and b processed file is identical.
PROCESSED_TEST = os.path.join(DATA_PATH, "test", "processed_test_%s.json" % "b")

PROCESSED_URL_DATA = os.path.join(DATA_PATH, "processed_url_%s.json" % "b")

GOLDEN_TRAIN_LABEL_FILE = os.path.join(DATA_PATH, "train", "golden_label_%s.txt" % __CLASS.lower())
GOLDEN_TEST_LABEL_FILE = os.path.join(DATA_PATH, "test", "golden_label_%s.txt" % __CLASS.lower())
if if_multi_binary():
    GOLDEN_TRAIN_LABEL_BINARY_FILE = os.path.join(DATA_PATH, "train", "golden_label_binary%%d_%s.txt" % __CLASS.lower())
ENSEMBLE_RESULT_PATH = os.path.join(RESULT_MYDIR, "ensemble_result.txt")

DICT_UNIGRAM_T2 = os.path.join(DICT_PATH, "unigram_t2.txt")
DICT_UNIGRAM_T1 = os.path.join(DICT_PATH, "unigram_t1.txt")
# DICT_HASHTAG_UNIGRAM_T1 = os.path.join(DICT_PATH, "hashtag_unigram_t1.txt")
# DICT_HASHTAG_T1 = os.path.join(DICT_PATH, "hashtag_t1.txt")
# DICT_HASHTAG_T2 = os.path.join(DICT_PATH, "hashtag_t2.txt")
DICT_UNIGRAM_STEM_T2 = DICT_PATH + "/unigram_stem_t2.txt"
DICT_BIGRAM_T3 = DICT_PATH + "/bigram_t3.txt"
DICT_TRIGRAM_T5 = DICT_PATH + "/trigram_t5.txt"

# UNITAG FILENAME
DICT_HASHTAG_UNIGRAM_TU = os.path.join(DICT_PATH, "hashtag_unigram_t%d.txt")
DICT_HASHTAG_TU = os.path.join(DICT_PATH, "hashtag_t%d.txt")
DICT_NLTK_UNIGRAM_TU = os.path.join(DICT_PATH, "nltk_unigram_t%d.txt")
DICT_NLTK_UNIGRAM_TU_TEST = os.path.join(DICT_PATH, "nltk_unigram_for_test_t%d.txt")
DICT_NLTK_BIGRAM_TU = os.path.join(DICT_PATH, "nltk_bigram_t%d.txt")
DICT_NLTK_TRIGRAM_TU = os.path.join(DICT_PATH, "nltk_trigram_t%d.txt")
DICT_URL_UNIGRAM_TU = os.path.join(DICT_PATH, "url_unigram_t%d.txt")

WORD2VEC_GOOGLE = os.path.join(PCCMD, "SemEval2017_T8/data_new/Google.txt")
VOCABULARY_PATH = os.path.join(YXPCCMD, "vocabulary")
NEGATION_PATH = VOCABULARY_PATH + "/negation terms.txt"

LEXI_SOURCE = os.path.join(YXPCCMD, "data", "Senti_Lexi")
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


def __make_unique_string(pattern: str):
    return pattern.replace("<uni>", str(uuid.uuid1()))


def make_feature_path(dev=False, dspr="", unique=True):
    path = "dev" if dev else "train"
    path = path + ".fea."
    if len(dspr.strip()) > 0:
        path += dspr + "."
    else:
        path += __make_time_string() + "."
    if unique:
        path += "<uni>."
    path += "feature"
    return __make_unique_string(os.path.join(FEATURE_PATH, path))


def make_model_path(dspr="", unique=True):
    pattern = ""
    if len(dspr.strip()) > 0:
        pattern += dspr + "."
    else:
        pattern += __make_time_string() + "."
    if unique:
        pattern += "<uni>."
    pattern += "model"
    return __make_unique_string(os.path.join(MODEL_MYDIR, pattern))


def make_result_path(dspr="", unique=True):
    pattern = "predict."
    if len(dspr.strip()) > 0:
        pattern += dspr + "."
    else:
        pattern += __make_time_string() + "."
    if unique:
        pattern += "<uni>."
    pattern += "txt"
    return __make_unique_string(os.path.join(RESULT_MYDIR, pattern))


def __make_time_string():
    sTime = time.localtime(time.time())
    return "%04d-%02d-%02d" % (sTime.tm_year, sTime.tm_mon, sTime.tm_mday)

GLOVE_CACHE_PATH = os.path.join(DICT_CACHE_PATH, "glove.small.300d.txt")


# the following is RF files
RF_DATA_NLTK_UNIGRAM_TU_PATH = os.path.join(RELATION_FREQ_PATH, "nltk_unigram_t%%d_%s.txt" % __CLASS.lower())
RF_DATA_HASHTAG_TU_PATH = os.path.join(RELATION_FREQ_PATH, "hashtag_t%%d_%s.txt" % __CLASS.lower())
RF_DATA_HASHTAG_UNIGRAM_TU_PATH = os.path.join(RELATION_FREQ_PATH, "hashtag_unigram_t%%d_%s.txt" % __CLASS.lower())
RF_DATA_NLTK_BIGRAM_TU_PATH = os.path.join(RELATION_FREQ_PATH, "nltk_bigram_t%%d_%s.txt" % __CLASS.lower())
RF_DATA_NLTK_TRIGRAM_TU_PATH = os.path.join(RELATION_FREQ_PATH, "nltk_trigram_t%%d_%s.txt" % __CLASS.lower())

RESULT_HC_DICT = os.path.join(RESULT_MYDIR, "feature_hc_dict_<uni>_%s.txt" % __CLASS.lower())


def make_result_hc_output(dspr="", unique=True):
    filename = "hc"

    if len(dspr) > 0:
        filename += ("_" + dspr)

    filename += "_" + __make_time_string()

    if unique:
        filename += "_<uni>"

    filename += "_%s.log" % __CLASS.lower()
    filename = __make_unique_string(filename)

    return os.path.join(HILLCLIMB_ROOT_PATH, filename)


def make_result_hc_dict(classifier_name="", dspr="", digit=3, unique=False):
    filename = "hcdict_"

    if len(classifier_name) > 0:
        filename += classifier_name
    else:
        filename += "<algo>"

    if len(dspr) > 0:
        filename += ("_" + dspr)

    filename += "_" + __make_time_string()

    filename += "_%%0%dd" % digit

    if unique:
        filename += "_<uni>"

    filename += "_%s.dict" % __CLASS.lower()
    filename = __make_unique_string(filename)

    return os.path.join(HILLCLIMB_ROOT_PATH, filename)


# URL dumping file list
URL_CACHE_PATH = os.path.join(DICT_CACHE_PATH, "url_cache.json")


# ensemble directory path:


def make_ensemble_path(dspr= "", unique=True):
    name = "ensemble"
    name += "." + __make_time_string()
    if len(dspr) > 0:
        name += "." + dspr
    else:
        name += ".<algo>" + dspr

    if unique:
        name += ".<uni>"
    name += ".%s.json" % __CLASS.lower()
    name = __make_unique_string(name)
    return os.path.join(ENSEMBLE_PATH, name)


def make_ensemble_score_path(dspr="", unique=True):
    name = "score"
    name += "." + __make_time_string()
    if len(dspr) > 0:
        name += "." + dspr
    if unique:
        name += ".<uni>"
    name += ".%s.json" % __CLASS.lower()
    name = __make_unique_string(name)
    return os.path.join(ENSEMBLE_SCORE_PATH, name)
