# coding: utf-8
import socket
hostname = socket.gethostname()

if hostname == "precision":
    CWD = "/home/zhenghang/SemEval2018_T3"
    PCCMD = "/home/feixiang/pyCharmSpace"
    LIB_LINEAR_PATH = "/home/feixiang/tools/liblinear-multicore-2.11-1"
# elif hostname.lower().startswith("l-mbookpro"):
#
else:
    CWD = "D:/pyCharmSpace/SemEval18_T3"
    LIB_LINEAR_PATH = ""

SLANGS_PATH = PCCMD + "/data/slangs"
NORMAL_WORDS_PATH = PCCMD + "/data/normal_word.pkl"
EMOTICON = PCCMD + "/data/Emoticon.txt"

DATA_PATH = CWD + "/data"
DICT_PATH = CWD + "/dict"
FEATURE_PATH = CWD + "/feature"
MODLE_PATH = CWD + "/model/binary_clf.model"

RAW_TRAIN_A = DATA_PATH + "/train/SemEval2018-T4-train-taskA.txt"
RAW_TRAIN_B = DATA_PATH + "/train/SemEval2018-T4-train-taskB.txt"

PROCESSED_TRAIN_A = DATA_PATH + "/train/processed_train_a.json"
PROCESSED_TRAIN_B = DATA_PATH + "/train/processed_train_b.json"

DICT_UNIGRAM_T2 = DICT_PATH + "/unigram_t2.txt"
DICT_UNIGRAM_STEM_T2 = DICT_PATH + "/unigram_stem_t2.txt"
DICT_BIGRAM_T3 = DICT_PATH + "/bigram_t3.txt"
DICT_TRIGRAM_T5 = DICT_PATH + "/trigram_t5.txt"


