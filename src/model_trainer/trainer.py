# coding:utf-8
import sys
import os
import time
import json
import random
import numpy as np

sys.path.append("../..")
from src import config
from src import evaluation
from src import util
from src.classifier import *
from src.model_trainer.feature_functions import *
from src import ensemble


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
        cm = evaluation.Evaluation(self.dev_feature_path, self.result_file_path, label)
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

            cm_list = run(tweet_cv, pending_feature_functions)
            p, r, f1 = evaluation.get_cm_eval(cm_list[0])

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
            print("Best score: %.2f%%" % (best_score * 100), file=foutput)
            print("Best feature set: ", " | ".join([func.__name__ for func in best_features]), file=foutput)
            print("-" * 45, file=foutput)
            print("Current functions: %s" % cur_funcs, file=foutput)
            print("Current score: %.2f%%" % (score * 100), file=foutput)
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


def load_data(is_test=False):
    tweets = json.load(open(config.PROCESSED_TRAIN, "r"), encoding="utf-8")
    if is_test:
        test_tweets = json.load(open(config.PROCESSED_TEST, 'r'), encoding="utf-8")
        return tweets, test_tweets
    else:
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
    feature += [
        ners_existed,
        wv_google,
        wv_GloVe,
        sentilexi,
        emoticon,
        punction,
        elongated
    ]
    #
    # for __freq in range(1, 6):
    #     feature.append(nltk_unigram_t[__freq])
    #     feature.append(nltk_bigram_t[__freq])
    #     feature.append(nltk_trigram_t[__freq])
    #     feature.append(hashtag_t[__freq])
    #     feature.append(hashtag_unigram_t[__freq])
    #     feature.append(nltk_unigram_withrf_t[__freq])
    #     feature.append(nltk_bigram_withrf_t[__freq])
    #     feature.append(nltk_trigram_withrf_t[__freq])
    #     feature.append(hashtag_t_withrf_t[__freq])
    #     feature.append(hashtag_unigram_withrf_t[__freq])
    feature += [
        # ners_existed,
        nltk_trigram_withrf_t[4],
        nltk_bigram_withrf_t[2],
        hashtag_withrf_t[2],
        hashtag_unigram_t[1],
        hashtag_withrf_t[1],
        nltk_unigram_withrf_t[2],
        nltk_trigram_withrf_t[2],
        url_unigram_t[2]
    ]
    return feature


def run(index_cv, feature_list, keep_train=False, keep_pw=False, use_ensemble=False, ensemble_get_classifier_list=None,
        is_test=False):
    """
    运行评估
    :param index_cv: 交叉验证集。如果是测试集，则所有内容放在一个 [] 中使用。
    :param feature_list:
    :param keep_train:
    :param keep_pw:
    :param use_ensemble:
    :param ensemble_get_classifier_list:
    :param is_test: 为测试评估提供优化
    :return:
    """
    # define the power file
    power_dev_fea_path = config.make_feature_path(dev=True, dspr="power")
    fpower = open(power_dev_fea_path, "w+")

    # currently, we need a lot of result file for each classifier algorithm

    fresPowerList = []
    power_result_path_List = []
    classifier_name_list = []
    ensemble_file_name_list = []
    train_feature_path_list = []
    dev_feature_path_list = []

    if not is_test:
        train_feature_path_list = [""] * len(index_cv)
        dev_feature_path_list = [""] * len(index_cv)
    else:
        train_feature_path_list.append("")
        dev_feature_path_list.append("")

    current_run_ensemble_path = config.make_ensemble_path()

    for i, __item in enumerate(index_cv):

        if is_test and i > 0:
            break

        train_tweets = []
        if not is_test:
            dev_tweets = __item

            for j, __item in enumerate(index_cv):
                if i != j:
                    train_tweets += __item

            print("Fold %d / %d" % (i + 1, len(index_cv)))
            train_feature_path_list[i] = config.make_feature_path(dspr="fold%d_of%d" % (i+1, len(index_cv)), dev=False)
            dev_feature_path_list[i] = config.make_feature_path(dspr="fold%d_of%d" % (i+1, len(index_cv)), dev=True)
        else:
            dev_tweets = index_cv[1]

            train_tweets = index_cv[0]

            print("Test Fold")
            train_feature_path_list[i] = config.make_feature_path(dspr="testfold", dev=False)
            dev_feature_path_list[i] = config.make_feature_path(dspr="testfold", dev=True)

        # here, we use many kind of classifier

        if ensemble_get_classifier_list is None:
            ensemble_get_classifier_list = [get_classifier]

        if len(power_result_path_List) == 0:
            power_result_path_List = [None] * len(ensemble_get_classifier_list)
            fresPowerList = [None] * len(ensemble_get_classifier_list)
            classifier_name_list = [""] * len(ensemble_get_classifier_list)
            ensemble_file_name_list = [""] * len(ensemble_get_classifier_list)

        # get classifier for each ensemble
        for handler_idx, get_classifier_handler in enumerate(ensemble_get_classifier_list):
            # first we get the classifier
            classifier = get_classifier_handler()
            classifier.is_test = is_test

            classifier_file_name = classifier.idname()

            if is_test:
                classifier_file_name = "test_" + classifier_file_name

            if power_result_path_List[handler_idx] is None:
                filename = config.make_result_path(dspr="power.%s" % classifier_file_name)
                power_result_path_List[handler_idx] = filename
                fresPowerList[handler_idx] = open(filename, "w+")
                classifier_name_list[handler_idx] = classifier.strategy.trainer
                ensemble_file_name_list[handler_idx] = \
                    (current_run_ensemble_path.replace("<algo>", "%s")) % classifier_file_name

            # then make path for the current selected algorithm
            model_path = config.make_model_path(dspr=classifier_file_name)
            result_path = config.make_result_path()

            # get the trainer
            trainer = Trainer(train_tweets, dev_tweets, feature_list, train_feature_path_list[i],
                              dev_feature_path_list[i], classifier, model_path, result_path)

            # train the modeler
            if handler_idx == 0:
                trainer.make_feature()

            trainer.train_model()
            trainer.test_model()

            # copy the dev file to power
            if handler_idx == 0:
                # only in the first run, we need to copy the power predict
                with open(dev_feature_path_list[i]) as fdev_in:
                    for line in fdev_in:
                        fpower.write(line)
                        if line[-1] != '\n':
                            fpower.write('\n')
                fpower.flush()

            dev_cls = []
            with open(result_path) as fres_in:
                for line in fres_in:
                    dev_cls.append(line.strip())
                    fresPowerList[handler_idx].write(line)
                    if line[-1] != '\n':
                        fresPowerList[handler_idx].write('\n')
            fresPowerList[handler_idx].flush()

            ensem_list = []
            for idx, tweet in enumerate(dev_tweets):
                ensem_list.append((tweet["id"], dev_cls[idx]))

            if keep_train:
                print("--" * 30)
                print("current train file:")
                print("train_feature: %s" % (train_feature_path_list[i]))
                print("dev_feature: %s" % (dev_feature_path_list[i]))
                print("model: %s" % (model_path))
                print("result: %s" % (result_path))
            else:
                for path in [model_path, result_path]:
                    if os.path.exists(path):
                        os.remove(path)

            if use_ensemble:
                try:
                    ensemble.make_ensemble(ensem_list, ensemble_file_name_list[handler_idx])
                except Exception as e:
                    print(e)

            print()

        if not keep_train:
            for path in [train_feature_path_list[i], dev_feature_path_list[i]]:
                if os.path.exists(path):
                    os.remove(path)

    fpower.close()
    for fresPower in fresPowerList:
        if fresPower is not None:
            fresPower.close()

    cm_list = []

    if not is_test:
        # for test file, evaluation is not available.

        for idx, handler in enumerate(ensemble_get_classifier_list):
            print("Evaluation on %s" % classifier_name_list[idx])
            print("--" * 30)
            cm = evaluation.Evaluation(power_dev_fea_path, power_result_path_List[idx], config.get_label_list())
            cm_list.append(cm)
            cm.print_out()
            print("--" * 30)
            print()
            if keep_pw:
                print("==" * 30)
                print("Power dev_feature and result file path is:")
                print(power_dev_fea_path)
                print(power_result_path_List[idx])
                print()
            else:
                del_list = [power_result_path_List[idx]]
                if idx == len(ensemble_get_classifier_list) - 1:
                    del_list.append(power_dev_fea_path)
                for path in del_list:
                    if os.path.exists(path):
                        os.remove(path)

        # make the ensemble score files
        if use_ensemble:
            ensemble_score_out_path = config.make_ensemble_score_path(dspr="train", unique=False)
            ensemble_scores = []

            for idx, cm in enumerate(cm_list):
                ensemble_scores.append(dict())
                cur_score = ensemble_scores[-1]
                cur_score["name"] = classifier_name_list[idx]
                cur_score["accuracy"] = cm.get_accuracy()
                cur_score["ensemble_path"] = ensemble_file_name_list[idx]

                p, r, f1 = cm.get_average_prf()
                cur_score["avrg_score"] = {
                    "precision": p,
                    "recall": r,
                    "f1": f1
                }

                cur_score["score"] = dict()
                for cls in config.get_label_list():
                    class_label = str(cls)
                    p, r, f1 = cm.get_prf(class_label)
                    cur_score["score"][class_label] = {
                        "precision": p,
                        "recall": r,
                        "f1": f1
                    }

            json.dump(ensemble_scores, open(ensemble_score_out_path, "w"), indent=4)

        # if use_ensemble:
        #     file_reg = current_run_ensemble_path.replace("<algo>", "*")
        #     ret = os.popen("ls %s" % file_reg).read()
        #     ensemble.make_ensemble_from_file(ret.strip().split('\n'),
        #                                      current_run_ensemble_path.replace("<algo>", "total"))
    else:
        if use_ensemble:
            ensemble_score_in_path = config.make_ensemble_score_path(dspr="train", unique=False)
            ensemble_score_out_path = config.make_ensemble_score_path(dspr="test", unique=False)
            ensemble_scores = json.load(open(ensemble_score_in_path))

            print("Copying ensemble scores from Training files: %s" % ensemble_score_in_path)
            print("To Test files: %s" % ensemble_score_out_path)

            handled_id = set()
            for idx in range(len(ensemble_scores)):
                algo_name = ensemble_scores[idx]['name']

                if algo_name not in classifier_name_list:
                    print("Algorithm %s not found." % algo_name)
                else:
                    _id = classifier_name_list.index(algo_name)
                    ensemble_scores[idx]["ensemble_path"] = ensemble_file_name_list[_id]
                    handled_id.add(_id)
            print("%d algorithm score copied." % len(handled_id))

            json.dump(ensemble_scores, open(ensemble_score_out_path, "w"), indent=4)

    print("=="*30)

    return cm_list


def main(mode="default", hc_output_filename="%05d.txt", is_test=False):
    # load data and build cv
    if not is_test:
        tweets = load_data(is_test=is_test)
        index_cv = build_cv(tweets, config.get_label_map, 10)
    else:
        tweets, test_tweets = load_data(is_test=is_test)
        index_cv = [tweets, test_tweets]

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
        classifier_list = [
            lambda: Classifier(LibLinearSVM(0, 1)),
            # lambda: Classifier(SkLearnAdaBoostClassifier()),
            # lambda: Classifier(SkLearnDecisionTree()),
            # lambda: Classifier(SkLearnKNN()),
            # lambda: Classifier(SkLearnLogisticRegression()),
            # lambda: Classifier(SkLearnNaiveBayes()),
            # lambda: Classifier(SkLearnRandomForestClassifier()),
            # lambda: Classifier(SkLearnSGD()),
            # lambda: Classifier(SkLearnSVM()),
            # lambda: Classifier(SkLearnVotingClassifier()),
            # lambda: Classifier(SkLearnXGBoostClassifier()),
        ]
        cm_list = run(index_cv, features, use_ensemble=False, ensemble_get_classifier_list=classifier_list,
                      is_test=is_test)

        for cm in cm_list:
            p, r, f1 = evaluation.get_cm_eval(cm)

    elif mode.lower() == "hc":
        # execute the hc procedure several times

        for exec_id in range(5):
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

        util.standard_hc_info_output(os.path.join(config.RESULT_MYDIR, hc_output_filename), range(5), 2)


def get_classifier():
    return Classifier(LibLinearSVM(0, 1))


if __name__ == '__main__':
    print("Trainer started at", time.asctime(time.localtime(time.time())))
    print("==" * 30)
    # output_format = "hc_hashtag_NaiveBayes_%05d.txt"
    # main("hc", "liblinear_licorice_masterrun_%05d.txt")
    main("default", is_test=False)
