#coding:utf8
import os
import pickle

from sklearn.datasets import load_svmlight_file
from sklearn import svm, tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

from src import config

class Strategy(object):
    def train_model(self, train_feature_path, model_path):
        return None
    def test_model(self, test_feature_path, model_path, result_file_path):
        return None

class Classifier(object):
    def __init__(self, strategy):
        self.strategy = strategy
    def train_model(self, train_feature_path, model_path):
        self.strategy.train_model(train_feature_path, model_path)
    def test_model(self, test_feature_path, model_path, result_file_path):
        self.strategy.test_model(test_feature_path, model_path, result_file_path)

'''liblinear'''
class LibLinear(Strategy):

    def __init__(self,s,c):
        self.s = s
        self.c = c
        self.trainer = "Liblinear"
        print ("Using %s Classfier" % self.trainer)

    def train_model(self, train_feature_path, model_path):
        #   options:
        #         -s type : set type of solver (default 1)
        #   for multi-class classification
        #      0 -- L2-regularized logistic regression (primal)
        #      1 -- L2-regularized L2-loss support vector classification (dual)
        #      2 -- L2-regularized L2-loss support vector classification (primal)
        #      3 -- L2-regularized L1-loss support vector classification (dual)
        #      4 -- support vector classification by Crammer and Singer
        #      5 -- L1-regularized L2-loss support vector classification
        #      6 -- L1-regularized logistic regression
        #      7 -- L2-regularized logistic regression (dual)
        #   for regression
        #     11 -- L2-regularized L2-loss support vector regression (primal)
        #     12 -- L2-regularized L2-loss support vector regression (dual)
        #     13 -- L2-regularized L1-loss support vector regression (dual)
        #   -c cost : set the parameter C (default 1)

        # cmd = config.LIB_LINEAR_PATH + "/train -s 0 -n 8 -c 0.9 " + train_feature_path + " " + model_path + " 1> " + config.DATA_PATH + "/tmp1.txt" + " 2> " + config.DATA_PATH + "/tmp2.txt"
        cmd = config.LIB_LINEAR_PATH + "/train -s " + str(self.s) + " -n 8 -c " + str(self.c) + " " + train_feature_path + " " + model_path + " 1> " + config.DATA_PATH + "/tmp1.txt" + " 2> " + config.DATA_PATH + "/tmp2.txt"
        # cmd = config.LIB_LINEAR_PATH + "/train -s 0 -n 8 -c 1 -w0 9 -w1 1" + train_feature_path + " " + model_path + " 1> " + config.DATA_PATH + "/tmp1.txt" + " 2> " + config.DATA_PATH + "/tmp2.txt"
        # print(cmd)
        os.system(cmd)

    def test_model(self, test_feature_path, model_path, result_file_path):
        cmd = config.LIB_LINEAR_PATH + "/predict " + test_feature_path + " " + model_path + " " + result_file_path
        # print(cmd)
        os.system(cmd)

    # def test_model(self, test_feature_path, model_path, result_file_path):
    #     cmd = config.LIB_LINEAR_PATH + "/predict -b 1 " + test_feature_path + " " + model_path + " " + result_file_path
    #     # print(cmd)
    #     os.system(cmd)