# encoding: utf-8
import sys
import os
import json
import re
import numpy as np
sys.path.append("../..")
from src import config
from src import util
from src.model_trainer import feature_functions


class Rf_Calculator(object):
    def __init__(self, tweets=None, map_func=lambda x:x, Rf=None):
        if map_func is None:
            map_func = {}
        self.tweets = tweets
        self.class_map_function = map_func
        if Rf is None:
            self.Rf = self._inner_Rf
        else:
            self.Rf = Rf

    def _inner_Rf(self, c, s):
        return np.log2(2+c/max(1, s-c))

    def calc(self, feature_function, out=""):
        rc = re.compile("(\d+):(\d+)")

        rc_data = {}
        existed_label = []
        for tw in self.tweets:
            feature = feature_function(tw)
            label = self.class_map_function(tw["label"])
            if label not in existed_label:
                existed_label.append(label)

            if feature == None:
                feature = feature

            for sValue, sFreq in rc.findall(feature.feat_string):
                iValue = int(sValue)
                iFreq = int(sFreq)
                rc_data.setdefault(iValue, {})
                rc_data[iValue].setdefault(label, 0)
                ''' todo += 1 or += iFreq?
                # ask feixiang for detail'''
                rc_data[iValue][label] += 1

        # calc the sum
        rc_sum = {}
        for value, dat in rc_data.items():
            rc_sum[value] = sum(dat.values())

        existed_label.sort()

        # calc all the label
        rf_dict = {}
        rf_sorted = {}
        rf_valueDict = {}
        for class_id in existed_label:
            rf_dict[class_id] = {}

            for value, dat in rc_data.items():
                c = dat.get(class_id, 0)
                s = rc_sum[value]
                iRf = self.Rf(c, s)
                rf_dict[class_id][value] = iRf
                # find the biggest rf for each word
                rf_valueDict.setdefault(value, {})
                rf_valueDict[value].setdefault("cls", [class_id])
                rf_valueDict[value].setdefault("max_rf", iRf)
                if iRf > rf_valueDict[value]["max_rf"]:
                    rf_valueDict[value]["max_rf"] = iRf
                    rf_valueDict[value]["cls"] = [class_id]
                elif iRf == rf_valueDict[value]["max_rf"] and class_id not in rf_valueDict[value]["cls"]:
                    rf_valueDict[value]["cls"].append(class_id)

            rf_sorted[class_id] = sorted(list(rf_dict[class_id].items()), key=lambda x:-x[1])

        if len(out) > 0:
            print("==" * 40)
            print("Creating Rf files..., dumped to\n%s" % out)
            json.dump({"rf_sorted": rf_sorted, "rf_value": rf_valueDict ,"rf_dict": rf_dict}, open(out, "w"))
            print("==" * 40)


def load_data():
    tweets = json.load(open(config.PROCESSED_TRAIN, "r"), encoding="utf-8")
    return tweets


def calc_rc(data, feature_function, output_path):
    rf = Rf_Calculator(data, config.get_label_map)
    rf.calc(feature_function, output_path)


def create_nltk_unigram_rf(train_data, freq):
    calc_rc(train_data, feature_functions.nltk_unigram_t[freq], config.RF_DATA_NLTK_UNIGRAM_TU_PATH % freq)


def create_hashtag_rf(train_data, freq):
    calc_rc(train_data, feature_functions.hashtag_t[freq], config.RF_DATA_HASHTAG_TU_PATH % freq)


def create_nltk_bigram_rf(train_data, freq):
    calc_rc(train_data, feature_functions.nltk_bigram_t[freq], config.RF_DATA_NLTK_BIGRAM_TU_PATH % freq)


def create_nltk_trigram_rf(train_data, freq):
    calc_rc(train_data, feature_functions.nltk_trigram_t[freq], config.RF_DATA_NLTK_TRIGRAM_TU_PATH % freq)