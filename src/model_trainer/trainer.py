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


def classification_hc(train_feature_path, dev_feature_path, model_path,
                      result_file_path, feature_functions, get_classifier_function, tweet_cv):
    num_iter = (len(feature_functions) + 1) * len(feature_functions) // 2
    curr_iter = 0

    foutput = open(config.RESULT_HC_OUTPUT, "w+")
    print("", file=foutput)
    print("Trainer started at", time.asctime(time.localtime(time.time())), file=foutput)
    current_best_features = set([])
    dict_all = {}
    feature_functions = set(feature_functions)
    best_score = -1
    best_features = set([])
    while len(feature_functions) > 0:
        dict_pending = {}
        for feature_function in feature_functions:
            curr_iter += 1
            pending_feature_functions = current_best_features | { feature_function }

            prec_score = []
            recall_score = []
            f1_score = []
            for i, list_item in enumerate(tweet_cv):
                dev_tweets = list_item
                train_tweets = []
                for j, list_item in enumerate(tweet_cv):
                    if i == j:
                        continue
                    else:
                        train_tweets += list_item

                classifier = get_classifier_function()
                trainer = Trainer(train_tweets, dev_tweets, pending_feature_functions, train_feature_path,
                                  dev_feature_path, classifier, model_path, result_file_path)
                trainer.make_feature()
                trainer.train_model()
                trainer.test_model()
                cm = trainer.evaluation_for_several_label(config.get_label_list())

                p, r, f1 = cm.get_average_prf()
                prec_score.append(p)
                recall_score.append(r)
                f1_score.append(f1)

            score = np.mean(f1_score)
            dict_pending[score] = pending_feature_functions
            cur_funcs = " | ".join([func.__name__ for func in pending_feature_functions])
            dict_all[score] = cur_funcs

            if score > best_score:
                best_score = score
                best_features = pending_feature_functions

            print()
            print("--> %d/%d" % (curr_iter, num_iter), file=foutput)
            print("##" * 45, file=foutput)
            print("Best score: %.4f" % best_score, file=foutput)
            print("Best feature set: ", " | ".join([func.__name__ for func in best_features]), file=foutput)
            print("-" * 45, file=foutput)
            print("Current functions: %s" % cur_funcs, file=foutput);
            util.print_markdown_mean_file(prec_score, recall_score, f1_score, foutput)
            print("##" * 45, file=foutput)
            foutput.flush()

        current_best_score = -1
        for key in dict_pending:
            if key > current_best_score:
                current_best_score = key
                current_best_features = dict_pending[key]

        feature_functions -= current_best_features
    util.write_dict_to_file(dict_all, os.path.join(config.RESULT_PATH, "hc.txt"))


def write_to_file(tuple, file_path):
    with open(file_path, "a") as file_out:
        file_out.write("\n-----------------------------------------------\n")
        file_out.write("\n".join(["s:%s c:%s :%s" % (str(t[0][0]), str(t[0][1]), str(t[1])) for t in tuple]))


def load_data():
    tweets = json.load(open(config.PROCESSED_TRAIN, "r"), encoding="utf-8")
    return tweets


def get_classifier():
    return Classifier(LibLinear(0, 1))

def algorithm_liblinear(train_tweets, dev_tweets, feature_list):
    '''classifier'''
    classifier = get_classifier()
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


def main(mode="default"):
    '''load data'''
    tweets = load_data()

    '''build_cv'''
    index_cv = build_cv(tweets, config.get_label_map, 10)

    '''feature_function'''
    features = [
        ners_existed,
        wv_google,
        wv_GloVe,
        sentilexi,
        emoticon,
        punction,
        elongated
    ]
    for __freq in range(1, 6):
        features.append(nltk_unigram_t[__freq])
        features.append(nltk_bigram_t[__freq])
        features.append(nltk_trigram_t[__freq])
        features.append(hashtag_t[__freq])
        features.append(nltk_unigram_t_with_rf[__freq])
        features.append(nltk_bigram_t_with_rf[__freq])
        features.append(nltk_trigram_with_t_rf[__freq])
        features.append(hashtag_t_with_rf[__freq])

    print("Using following features:")
    print("=" * 30)
    for fe_func in features:
        print(fe_func.__name__)
    print("=" * 30)
    print()

    if mode.lower() == "default":
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
    elif mode.lower() == "hc":
        classification_hc(config.TRAIN_FEATURE_PATH, config.DEV_FEATURE_PATH,
                       config.MODEL_PATH, config.RESULT_PATH, features,
                       get_classifier, index_cv)


if __name__ == '__main__':
    print("Trainer started at", time.asctime(time.localtime(time.time())))
    print("==" * 30)
    main("hc")

