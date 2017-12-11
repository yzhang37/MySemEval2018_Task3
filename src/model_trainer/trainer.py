#coding:utf-8
import sys
import time
import json
import random
import numpy as np
sys.path.append("../..")
from src import config
from src.evaluation import *
from src.model_trainer.make_feature_file import make_feature
from src import util
from src.classifier import *
from src.model_trainer.feature_functions import *

class Trainer(object):
    def __init__(self,
                 train_tweets,
                 dev_tweets,
                 feature_functions,
                 train_feature_path,
                 dev_feature_path,
                 classifier,
                 model_path,
                 result_file_path,

    ):
        self.train_tweets = train_tweets
        self.dev_tweets = dev_tweets
        self.feature_functions = feature_functions
        self.train_feature_path = train_feature_path
        self.dev_feature_path = dev_feature_path
        self.classifier = classifier
        self.model_path = model_path
        self.result_file_path = result_file_path

    def make_feature(self):
        make_feature(self.train_tweets, self.feature_functions, self.train_feature_path)
        make_feature(self.dev_tweets, self.feature_functions, self.dev_feature_path)
        util.handle_train_test_dim(self.train_feature_path, self.dev_feature_path)

    def train_model(self):
        print("training!!!!!")
        self.classifier.train_model(self.train_feature_path, self.model_path)

    def test_model(self):
        print("testing!!!!!")
        self.classifier.test_model(self.dev_feature_path, self.model_path, self.result_file_path)

    def evaluation_for_several_label(self, label):
        print("evaluating!!!!!")
        cm = Evaluation(self.dev_feature_path, self.result_file_path, label)
        # cm.print_out()
        return cm


def classification(train_feature_path, dev_feature_path, model_path, result_file_path, feature_functions, classifier, train_tweets, dev_tweets):

    trainer = Trainer(train_tweets, dev_tweets, feature_functions, train_feature_path, dev_feature_path, classifier,
                       model_path, result_file_path)
    trainer.make_feature()
    trainer.train_model()
    trainer.test_model()

    cm = trainer.evaluation_for_several_label(config.get_label_list())

    cm.print_out()
    return cm


def write_to_file(tuple, file_path):
    with open(file_path, "a") as file_out:
        file_out.write("\n-----------------------------------------------\n")
        file_out.write("\n".join(["s:%s c:%s :%s" % (str(t[0][0]), str(t[0][1]), str(t[1])) for t in tuple]))


def load_data():
    tweets = json.load(open(config.PROCESSED_TRAIN, "r"), encoding="utf-8")
    return tweets


def algorithm_liblinear(train_tweets, dev_tweets, feature_list):
    '''classifier'''
    classifier = Classifier(LibLinear(0, 1))
    cm = classification(config.TRAIN_FEATURE_PATH, config.DEV_FEATURE_PATH,
                           config.MODEL_PATH, config.RESULT_PATH,
                           feature_list, classifier, train_tweets, dev_tweets)
    return cm


def build_cv(tweets, map_function, fold=4):
    data_dict = {}

    for tw in tweets:
        new_label = map_function(tw["label"])
        data_dict.setdefault(new_label, [])
        data_dict[new_label].append(tw)

    data_len_dict = {}
    for label in data_dict.keys():
        data_len_dict[label] = len(data_dict[label])
        random.shuffle(data_dict[label])

    cv = fold
    index_cv = []
    for i in range(cv):
        curCV = []
        for label, data in data_dict.items():
            curCV += data[i * data_len_dict[label] // cv: (i + 1) * data_len_dict[label] // cv]
        index_cv.append(curCV)

    return index_cv


def main():
    '''load data'''
    tweets = load_data()

    '''build_cv'''
    index_cv = build_cv(tweets, config.get_label_map, 10)

    '''feature_function'''
    features = [
        # nltk_unigram,
        nltk_unigram_with_rf,
        # nltk_bigram,
        hashtag_with_rf,
        ners_existed,
        # bigram,
        wv_google,
        wv_GloVe,
        sentilexi,
        emoticon,
        punction,
        elongated
    ]

    print("Using following features:")
    print("=" * 30)
    for fe_func in features:
        print(fe_func.__name__)
    print("=" * 30)
    print()

    prec_score = []
    recall_score = []
    f1_score = []
    for i, list_item in enumerate(index_cv):
        dev = list_item
        train = []
        for j, list_item in enumerate(index_cv):
            if i == j:
                continue
            else:
                train += list_item
        cm = algorithm_liblinear(train, dev, features)
        p, r, f1 = cm.get_average_prf()
        print("p:{},r:{},f1:{}".format(p, r, f1))
        prec_score.append(p)
        recall_score.append(r)
        f1_score.append(f1)

    average_score = sum(f1_score) / len(f1_score)
    print(average_score)
    util.print_dedicated_mean(prec_score, recall_score, f1_score)
    util.print_markdown_mean(prec_score, recall_score, f1_score)


if __name__ == '__main__':
    print("Trainer started at", time.asctime(time.localtime(time.time())))
    print("==" * 30)
    main()
