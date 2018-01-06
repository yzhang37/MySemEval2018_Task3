# coding:utf-8
import sys
import os
import time
import json
import random
import numpy as np

sys.path.append("../..")
from src import config
from src.evaluation import *
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
        self.classifier.make_feature(self.train_tweets, self.feature_functions, self.train_feature_path)
        self.classifier.make_feature(self.dev_tweets, self.feature_functions, self.dev_feature_path)
        util.handle_train_test_dim(self.train_feature_path, self.dev_feature_path)

    def train_model(self):
        # print("training!!!!!")
        self.classifier.train_model(self.train_feature_path, self.model_path)

    def test_model(self):
        # print("testing!!!!!")
        self.classifier.test_model(self.dev_feature_path, self.model_path, self.result_file_path)

    def evaluation_for_several_label(self, label):
        print("Evaluating...")
        cm = Evaluation(self.dev_feature_path, self.result_file_path, label)
        # cm.print_out()
        return cm


def classification_hc(train_feature_path, dev_feature_path, model_path,
                      result_file_path, feature_functions, get_classifier_function, tweet_cv, output_file_name):
    num_iter = (len(feature_functions) + 1) * len(feature_functions) // 2
    curr_iter = 0

    filePath = config.make_result_hc_output()
    foutput = open(filePath, "w+")
    print("Hill Climbing file dumping to %s." % filePath)
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
            pending_feature_functions = current_best_features | {feature_function}

            cm = run(tweet_cv, pending_feature_functions)
            p, r, f1 = cm.get_average_prf()

            score = f1
            dict_pending[score] = pending_feature_functions
            cur_funcs = " | ".join([func.__name__ for func in pending_feature_functions])
            dict_all[score] = cur_funcs

            if score > best_score:
                best_score = score
                best_features = pending_feature_functions

            print()
            print("--> %d/%d" % (curr_iter, num_iter), file=foutput)
            print("##" * 45, file=foutput)
            print("Best score: %.2f%%" % best_score, file=foutput)
            print("Best feature set: ", " | ".join([func.__name__ for func in best_features]), file=foutput)
            print("-" * 45, file=foutput)
            print("Current functions: %s" % cur_funcs, file=foutput)
            print("Current score: %.2f%%" % score, file=foutput)
            print("##" * 45, file=foutput)
            foutput.flush()

        current_best_score = -1
        for key in dict_pending:
            if key > current_best_score:
                current_best_score = key
                current_best_features = dict_pending[key]

        feature_functions -= current_best_features
    print("Hill Climbing file dumped to %s." % filePath)
    util.write_dict_to_file(dict_all, os.path.join(config.RESULT_MYDIR, output_file_name))


def write_to_file(tuple, file_path):
    with open(file_path, "a") as file_out:
        file_out.write("\n-----------------------------------------------\n")
        file_out.write("\n".join(["s:%s c:%s :%s" % (str(t[0][0]), str(t[0][1]), str(t[1])) for t in tuple]))


def load_data():
    tweets = json.load(open(config.PROCESSED_TRAIN, "r"), encoding="utf-8")
    return tweets


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


def get_features_on_liblinear(feature: list):
    # feature += [
    #     ners_existed,
    #     wv_google,
    #     wv_GloVe,
    #     sentilexi,
    #     emoticon,
    #     punction,
    #     elongated
    # ]

    # feature.append(nltk_unigram_t_with_rf[2])
    # feature.append(nltk_bigram_t_with_rf[3])
    # feature.append(nltk_trigram_t[2])
    # feature.append(hashtag_t_with_rf[3])
    for __freq in range(1, 6):
        feature.append(nltk_unigram_t[__freq])
        feature.append(nltk_bigram_t[__freq])
        feature.append(nltk_trigram_t[__freq])
        feature.append(hashtag_t[__freq])
        feature.append(nltk_unigram_withrf_t[__freq])
        feature.append(nltk_bigram_withrf_t[__freq])
        feature.append(nltk_trigram_withrf_t[__freq])
        feature.append(hashtag_t_withrf_t[__freq])
    return feature


def get_features_on_AdaBoost(features: list):
    print("!!!!! uncleared !!!!!")
    features.extend([
        ners_existed,
        wv_google,
        wv_GloVe,
        sentilexi,
        emoticon,
        punction,
        elongated
    ])

    features.append(nltk_unigram_t[2])

    # features.append(nltk_bigram_t[__freq])

    features.append(nltk_trigram_t[2])

    features.append(hashtag_t[2])

    features.append(nltk_unigram_withrf_t[2])

    features.append(nltk_bigram_withrf_t[2])

    features.append(nltk_trigram_withrf_t[1])
    features.append(nltk_trigram_withrf_t[4])

    features.append(hashtag_t_withrf_t[4])
    features.append(hashtag_t_withrf_t[5])


def get_features_on_DecisionTree(features: list):
    print("!!!!! uncleared !!!!!")
    for __freq in range(1, 6):
        # features.append(nltk_unigram_t[__freq])
        # features.append(nltk_bigram_t[__freq])
        # features.append(nltk_trigram_t[__freq])
        # features.append(hashtag_t[__freq])
        # features.append(nltk_unigram_t_with_rf[__freq])
        # features.append(nltk_bigram_t_with_rf[__freq])
        # features.append(nltk_trigram_with_t_rf[__freq])
        features.append(hashtag_t_withrf_t[__freq])


def get_features_on_NaiveBayes(features: list):
    print("!!!!! uncleared !!!!!")
    for __freq in range(1, 6):
        # features.append(nltk_unigram_t[__freq])
        # features.append(nltk_bigram_t[__freq])
        # features.append(nltk_trigram_t[__freq])
        features.append(hashtag_t[__freq])
        # features.append(nltk_unigram_t_with_rf[__freq])
        # features.append(nltk_bigram_t_with_rf[__freq])
        # features.append(nltk_trigram_with_t_rf[__freq])
        # features.append(hashtag_t_with_rf[__freq])


def run(index_cv, feature_list, keep_train=False, keep_pw=False):

    # define the power file
    power_dev_fea_path = config.make_feature_path(dev=True, dspr="power")
    fpower = open(power_dev_fea_path, "w+")
    power_result_path = config.make_result_path(dspr="power")
    fresPower = open(power_result_path, "w+")

    for i, __item in enumerate(index_cv):
        dev_tweets = __item
        train_tweets = []
        for j, __item in enumerate(index_cv):
            if i != j:
                train_tweets += __item

        print("Fold %d / %d" % (i+1, len(index_cv)))
        # get classifier
        classifier = get_classifier()

        # get paths
        train_fea_path = config.make_feature_path(dev=False)
        dev_fea_path = config.make_feature_path(dev=True)
        model_path = config.make_model_path()
        result_path = config.make_result_path()

        trainer = Trainer(train_tweets, dev_tweets, feature_list, train_fea_path, dev_fea_path, classifier,
                          model_path, result_path)

        trainer.make_feature()
        trainer.train_model()
        trainer.test_model()

        with open(dev_fea_path) as fdev_in:
            for line in fdev_in:
                fpower.write(line)
                if line[-1] != '\n':
                    fpower.write('\n')

        fpower.flush()
        with open(result_path) as fres_in:
            for line in fres_in:
                fresPower.write(line)
                if line[-1] != '\n':
                    fpower.write('\n')
        fresPower.flush()

        if keep_train:
            print("--" * 30)
            print("current train file:")
            print("train_feature: %s" % (train_fea_path))
            print("dev_feature: %s" % (dev_fea_path))
            print("model: %s" % (model_path))
            print("result: %s" % (result_path))
        else:
            for path in [train_fea_path, dev_fea_path, model_path, result_path]:
                if os.path.exists(path):
                    os.remove(path)
        print()

    fpower.close()
    fresPower.close()
    cm = Evaluation(power_dev_fea_path, power_result_path, config.get_label_list())
    cm.print_out()
    if keep_pw:
        print("==" * 30)
        print("Power dev_feature and result file path is:")
        print(power_dev_fea_path)
        print(power_result_path)
        print()
    else:
        for path in [power_dev_fea_path, power_result_path]:
            if os.path.exists(path):
                os.remove(path)
    return cm


def main(mode="default", hc_output_filename="%05d.txt"):
    '''load data'''
    tweets = load_data()

    '''build_cv'''
    index_cv = build_cv(tweets, config.get_label_map, 10)

    '''feature_function'''
    features = []
    # feature_func = get_features_on_NaiveBayes
    feature_func = get_features_on_liblinear

    print(feature_func.__name__.replace("_", " ").replace("get", "Using"))
    feature_func(features)

    # features = [
    #     ners_existed,
    #     wv_google,
    #     wv_GloVe,
    #     sentilexi,
    #     emoticon,
    #     punction,
    #     elongated
    # ]

    # for __freq in range(1, 6):
        # features.append(nltk_unigram_t[__freq])
        # features.append(nltk_bigram_t[__freq])
        # features.append(nltk_trigram_t[__freq])
        # features.append(hashtag_t[__freq])
        # features.append(nltk_unigram_t_with_rf[__freq])
        # features.append(nltk_bigram_t_with_rf[__freq])
        # features.append(nltk_trigram_with_t_rf[__freq])
        # features.append(hashtag_t_with_rf[__freq])

    print("Using following features:")
    print("=" * 30)
    for fe_func in features:
        print(fe_func.__name__)
    print("=" * 30)
    print()

    if mode.lower() == "default":
        cm = run(index_cv, features)
        p, r, f1 = cm.get_average_prf()

        # prec_score = []
        # recall_score = []
        # f1_score = []
        # for i, list_item in enumerate(index_cv):
        #     dev = list_item
        #     train = []
        #     for j, list_item in enumerate(index_cv):
        #         if i == j:
        #             continue
        #         else:
        #             train += list_item
        #     cm = single_train_algorithm(train, dev, features)
        #     p, r, f1 = cm.get_average_prf()
        #     # print("p:{},r:{},f1:{}".format(p, r, f1))
        #     prec_score.append(p)
        #     recall_score.append(r)
        #     f1_score.append(f1)
        #
        # average_score = sum(f1_score) / len(f1_score)
        # print(average_score)
        # util.print_dedicated_mean(prec_score, recall_score, f1_score)
        # util.print_markdown_mean_file(prec_score, recall_score, f1_score)
    elif mode.lower() == "hc":
        # execute the hc procedure several times

        for exec_id in range(1):
            print()
            print("Running hc time %d" % (exec_id + 1))

            train_fea_path = config.make_feature_path(dev=False)
            dev_fea_path = config.make_feature_path(dev=True)
            model_path = config.make_model_path()
            result_path = config.make_result_path()

            classification_hc(train_fea_path, dev_fea_path,
                              model_path, result_path, features,
                              get_classifier, index_cv, hc_output_filename % (exec_id + 1))

            for path in [train_fea_path, dev_fea_path, model_path, result_path]:
                if os.path.exists(path):
                    os.remove(path)

        util.standard_hc_info_output(os.path.join(config.RESULT_MYDIR, hc_output_filename), range(1), 2)


def get_classifier():
    return Classifier(LibLinear(0, 1))
    # return Classifier(skLearn_AdaBoostClassifier())
    # return Classifier(skLearn_DecisionTree())
    # return Classifier(skLearn_NaiveBayes())


if __name__ == '__main__':
    print("Trainer started at", time.asctime(time.localtime(time.time())))
    print("==" * 30)
    # output_format = "hc_hashtag_NaiveBayes_%05d.txt"
    # main("hc", "liblinear_licorice_masterrun_%05d.txt")
    main("default")


