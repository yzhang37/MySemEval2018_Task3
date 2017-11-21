#coding:utf-8
import sys
sys.path.append("../..")
import json
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

    cm = trainer.evaluation_for_several_label([0,1,2,3]) #四分类
    # cm = trainer.evaluation_for_several_label([0,1]) #二分类

    cm.print_out()
    return cm

def write_to_file(tuple, file_path):
    with open(file_path, "a") as file_out:
        file_out.write("\n-----------------------------------------------\n")
        file_out.write("\n".join(["s:%s c:%s :%s" % (str(t[0][0]), str(t[0][1]), str(t[1])) for t in tuple]))


def load_data():
    tweets = json.load(open(config.PROCESSED_TRAIN_B, "r"), encoding="utf-8")
    return tweets

def algorithm_liblinear(train_tweets, dev_tweets):

    '''feature_function'''
    feature_list = [
        unigram,
        # bigram,
        wv_google,
        sentilexi,
        emoticon,
        punction,
        elongated
    ]

    '''classifier'''
    classifier = Classifier(LibLinear(0, 1))
    cm = classification(config.TRAIN_FEATURE_PATH, config.DEV_FEATURE_PATH,
                           config.MODEL_PATH, config.RESULT_PATH,
                           feature_list, classifier, train_tweets, dev_tweets)
    return cm

def build_cv(tweets):
    zero_list = []
    one_list = []
    two_list = []
    three_list = []
    for tw in tweets:
        if tw["label"] == "0":
            zero_list.append(tw)
        elif tw["label"] == "1": # 二分类
        # elif tw["label"] == "1" or tw["label"] == "2" or tw["label"] == "3" : # 四分类
            one_list.append(tw)
        elif tw["label"] == "2":
            two_list.append(tw)
        elif tw["label"] == "3":
            three_list.append(tw)

    len_0 = len(zero_list)
    len_1 = len(one_list)
    len_2 = len(two_list)
    len_3 = len(three_list)

    cv = 4
    index_cv = []
    for i in range(cv):
        curCV = zero_list[i * len_0 // cv: (i + 1) * len_0 // cv] + \
                one_list[i * len_1 // cv: (i + 1) * len_1 // cv] + \
                two_list[i * len_2 // cv: (i + 1) * len_2 // cv] + \
                three_list[i * len_3 // cv: (i + 1) * len_3 // cv]

        index_cv.append(curCV)

    return index_cv



def main():
    '''load data'''
    tweets = load_data()

    '''build_cv'''
    index_cv = build_cv(tweets)

    score = []
    for i, list_item in enumerate(index_cv):
        dev = list_item
        train = []
        for j, list_item in enumerate(index_cv):
            if i == j:
                continue
            else:
                train += list_item
        cm = algorithm_liblinear(train, dev)
        p, r, f1 = cm.get_average_prf()
        print("p:{},r:{},f1:{}".format(p, r, f1))
        score.append(f1)



    average_score = sum(score) / len(score)
    print(average_score)



if __name__ == '__main__':
    main()