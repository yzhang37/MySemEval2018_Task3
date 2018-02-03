#coding:utf8
import os
import pickle

from sklearn.datasets import load_svmlight_file
from sklearn import svm, tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
import xgboost as xgb

from src import config
from src.model_trainer import make_feature_file
from src import util

DEBUG = False


class Strategy(object):
    def __init__(self):
        self.make_feature_handler = make_feature_file.make_feature_for_liblinear
        self.idname = "noname"
        self.use_proba = False

    def train_model(self, train_feature_path, model_path):
        return None

    def test_model(self, test_feature_path, model_path, result_file_path):
        return None


class Classifier(object):
    def __init__(self, strategy):
        self.strategy = strategy
        self.use_proba = False
        self.is_test = False

    def idname(self):
        return self.strategy.idname

    def train_model(self, train_feature_path, model_path):
        if self.use_proba:
            self.strategy.use_proba = True
        self.strategy.train_model(train_feature_path, model_path)

    def test_model(self, test_feature_path, model_path, result_file_path):
        if self.use_proba:
            self.strategy.use_proba = True
        self.strategy.test_model(test_feature_path, model_path, result_file_path)

    def make_feature(self, tweets, feature_function_list, to_file):
        if self.use_proba:
            self.strategy.use_proba = True
        self.strategy.make_feature_handler(tweets, feature_function_list, to_file, is_test=self.is_test)


''' skLearn '''
# 决策树算法
# 朴素贝叶斯
# 支持向量机


# 决策树算法
class SkLearnDecisionTree(Strategy):
    def __init__(self):
        super().__init__()
        self.trainer = "Scikit-Learn DecisionTree"
        self.idname = "sklearn_dcstree"
        self.clf = tree.DecisionTreeClassifier()
        print("Using %s Classifier" % (self.trainer))

    def train_model(self, train_file_path, model_path):
        train_X, train_y = load_svmlight_file(train_file_path)

        # print("==> Train the model ...")
        self.clf.fit(train_X, train_y)

    def test_model(self, test_file_path, model_path, result_file_path):

        # print("==> Test the model ...")
        test_X, test_y = load_svmlight_file(test_file_path)

        if self.use_proba:
            pred_y = self.clf.predict_proba(test_X)
        else:
            pred_y = self.clf.predict(test_X)

        # write prediction to file
        if not self.use_proba:
            with open(result_file_path, 'w') as fout:
                fout.write("\n".join(map(str, map(int, pred_y))))
        else:
            util.write_result_with_proba(pred_y, result_file_path)


# 朴素贝叶斯算法
class SkLearnNaiveBayes(Strategy):
    def __init__(self):
        super().__init__()
        self.trainer = "Scikit-Learn NaïveBayes"
        self.idname = "sklearn_navbayes"
        self.clf = GaussianNB()
        print("Using %s Classifier" % (self.trainer))

    def train_model(self, train_file_path, model_path):
        train_X, train_y = load_svmlight_file(train_file_path)

        train_X = train_X.toarray()
        # print("==> Train the model ...")
        self.clf.fit(train_X, train_y)

    def test_model(self, test_file_path, model_path, result_file_path):

        # print("==> Test the model ...")
        test_X, test_y = load_svmlight_file(test_file_path)
        test_X = test_X.toarray()

        if self.use_proba:
            pred_y = self.clf.predict_proba(test_X)
        else:
            pred_y = self.clf.predict(test_X)

        # write prediction to file
        if not self.use_proba:
            with open(result_file_path, 'w') as fout:
                fout.write("\n".join(map(str, map(int, pred_y))))
        else:
            util.write_result_with_proba(pred_y, result_file_path)


# Sklearn 支持向量机
class SkLearnSVM(Strategy):
    def __init__(self):
        super().__init__()
        self.trainer = "Scikit-Learn Support Vector Machine"
        self.idname = "sklearn_svm"
        self.clf = svm.LinearSVC()
        print("Using %s Classifier" % (self.trainer))

    def train_model(self, train_file_path, model_path):
        train_X, train_y = load_svmlight_file(train_file_path)

        # print("==> Train the model ...")
        self.clf.fit(train_X, train_y)

    def test_model(self, test_file_path, model_path, result_file_path):
        # print("==> Test the model ...")
        test_X, test_y = load_svmlight_file(test_file_path)

        if self.use_proba:
            raise NotImplementedError("Sklearn SVM didn't support probability output.")
        else:
            pred_y = self.clf.predict(test_X)

        # write prediction to file
        with open(result_file_path, 'w') as fout:
            fout.write("\n".join(map(str, map(int, pred_y))))


# 随机梯度下降
class SkLearnSGD(Strategy):
    def __init__(self):
        super().__init__()
        self.trainer = "Scikit-Learn Stochastic Gradient Descent (log)"
        self.idname = "sklearn_sgd_log"
        # only log and modified_huber support
        self.clf = SGDClassifier(loss='log')
        print("Using %s Classifier" % self.trainer)

    def train_model(self, train_file_path, model_path):
        train_X, train_y = load_svmlight_file(train_file_path)
        # print("==> Train the model ...")
        self.clf.fit(train_X, train_y)

    def test_model(self, test_file_path, model_path, result_file_path):
        # print("==> Test the model ...")
        test_X, test_y = load_svmlight_file(test_file_path)

        if self.use_proba:
            pred_y = self.clf.predict_proba(test_X)
        else:
            pred_y = self.clf.predict(test_X)

        # write prediction to file
        if not self.use_proba:
            with open(result_file_path, 'w') as fout:
                fout.write("\n".join(map(str, map(int, pred_y))))
        else:
            util.write_result_with_proba(pred_y, result_file_path)


# 逻辑回归
class SkLearnLogisticRegression(Strategy):
    def __init__(self):
        super().__init__()
        self.trainer = "Scikit-Learn LogisticRegression"
        self.idname = "sklearn_logreg"
        self.clf = LogisticRegression()
        print("Using %s Classifier" % self.trainer)

    def train_model(self, train_file_path, model_path):
        train_X, train_y = load_svmlight_file(train_file_path)
        # print("==> Train the model ...")
        self.clf.fit(train_X, train_y)

    def test_model(self, test_file_path, model_path, result_file_path):
        # print("==> Test the model ...")
        test_X, test_y = load_svmlight_file(test_file_path)

        if self.use_proba:
            pred_y = self.clf.predict_proba(test_X)
        else:
            pred_y = self.clf.predict(test_X)

        # write prediction to file
        if not self.use_proba:
            with open(result_file_path, 'w') as fout:
                fout.write("\n".join(map(str, map(int, pred_y))))
        else:
            util.write_result_with_proba(pred_y, result_file_path)


# KNN
class SkLearnKNN(Strategy):
    def __init__(self):
        super().__init__()
        self.trainer = "Scikit-Learn KNN"
        self.idname = "sklearn_knn"
        self.clf = KNeighborsClassifier(n_neighbors=3)
        print("Using %s Classifier" % (self.trainer))

    def train_model(self, train_file_path, model_path):
        train_X, train_y = load_svmlight_file(train_file_path)
        # print("==> Train the model ...")
        self.clf.fit(train_X, train_y)

    def test_model(self, test_file_path, model_path, result_file_path):
        # print("==> Test the model ...")
        test_X, test_y = load_svmlight_file(test_file_path)

        if self.use_proba:
            pred_y = self.clf.predict_proba(test_X)
        else:
            pred_y = self.clf.predict(test_X)

        # write prediction to file
        if not self.use_proba:
            with open(result_file_path, 'w') as fout:
                fout.write("\n".join(map(str, map(int, pred_y))))
        else:
            util.write_result_with_proba(pred_y, result_file_path)


# AdaBoost
class SkLearnAdaBoostClassifier(Strategy):
    def __init__(self):
        super().__init__()
        self.trainer = "Scikit-Learn AdaBoostClassifier"
        self.idname = "sklearn_adaboost"
        self.clf = AdaBoostClassifier()
        print("Using %s Classifier" % (self.trainer))

    def train_model(self, train_file_path, model_path):
        train_X, train_y = load_svmlight_file(train_file_path)
        # print("==> Train the model ...")
        self.clf.fit(train_X.toarray(), train_y)
        pickle.dump(self.clf, open(model_path, 'wb'))

    def test_model(self, test_file_path, model_path, result_file_path):
        # print("==> Test the model ...")
        test_X, test_y = load_svmlight_file(test_file_path)
        self.clf = pickle.load(open(model_path, 'rb'))

        test_X = test_X.toarray()

        if self.use_proba:
            pred_y = self.clf.predict_proba(test_X)
        else:
            pred_y = self.clf.predict(test_X)

        # write prediction to file
        if not self.use_proba:
            with open(result_file_path, 'w') as fout:
                fout.write("\n".join(map(str, map(int, pred_y))))
        else:
            util.write_result_with_proba(pred_y, result_file_path)


# 随机森林
class SkLearnRandomForestClassifier(Strategy):
    def __init__(self):
        super().__init__()
        self.trainer = "Scikit-Learn RandomForestClassifier"
        self.idname = "sklearn_rand_forest"
        self.clf = RandomForestClassifier()
        print("Using %s Classifier" % (self.trainer))

    def train_model(self, train_file_path, model_path):
        train_X, train_y = load_svmlight_file(train_file_path)
        # print("==> Train the model ...")
        self.clf.fit(train_X, train_y)

    def test_model(self, test_file_path, model_path, result_file_path):
        # print("==> Test the model ...")
        test_X, test_y = load_svmlight_file(test_file_path)

        if self.use_proba:
            pred_y = self.clf.predict_proba(test_X)
        else:
            pred_y = self.clf.predict(test_X)

        # write prediction to file
        if not self.use_proba:
            with open(result_file_path, 'w') as fout:
                fout.write("\n".join(map(str, map(int, pred_y))))
        else:
            util.write_result_with_proba(pred_y, result_file_path)


# Voting 分类器
class SkLearnVotingClassifier(Strategy):
    def __init__(self):
        super().__init__()
        self.trainer = "Scikit-Learn VotingClassifier"
        self.idname = "sklearn_voting"

        clf1 = LogisticRegression()
        clf2 = svm.LinearSVC()
        clf3 = AdaBoostClassifier()

        self.clf = VotingClassifier(estimators=[('lr', clf1), ('svm', clf2), ('ada', clf3)], voting='hard')
        print("Using %s Classifier" % (self.trainer))

    def train_model(self, train_file_path, model_path):
        train_X, train_y = load_svmlight_file(train_file_path)

        # print("==> Train the model ...")
        self.clf.fit(train_X, train_y)

    def test_model(self, test_file_path, model_path, result_file_path):
        # print("==> Test the model ...")
        test_X, test_y = load_svmlight_file(test_file_path)

        if self.use_proba:
            pred_y = self.clf.predict_proba(test_X)
        else:
            pred_y = self.clf.predict(test_X)

        # write prediction to file
        if not self.use_proba:
            with open(result_file_path, 'w') as fout:
                fout.write("\n".join(map(str, map(int, pred_y))))
        else:
            util.write_result_with_proba(pred_y, result_file_path)


class XGBoost(Strategy):
    def __init__(self):
        super().__init__()
        self.trainer = "XGBoost"
        self.idname = "xgboost"
        print("Using %s Classifier" % self.trainer)

    def train_model(self, train_file_path, model_path, current_relation=None):
        # print("==> Train the model ...")
        # read in data
        # print (train_file_path)

        # 去掉comment 部分
        with open(train_file_path) as fin, open(train_file_path + ".tmp", "w") as fout:
            new_lines = []
            for line in fin:
                line = line.strip().split(" #")[0]
                new_lines.append(line)
            fout.write("\n".join(new_lines))

        dtrain = xgb.DMatrix(train_file_path + ".tmp")
        # specify parameters via map
        # param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'reg:linear'}
        # param = { 'silent': 1, 'eta': 0.1, 'max_depth': 10, "booster" : "gbtree",}

        # param = {'silent': 1, "booster":"gbtree",  "subsample": 0.9, "colsample_bytree": 0.7, "seed":1301 }
        # param = {'silent':1, 'objective':'multi:softmax', 'num_class':5}

        param = {'objective': 'multi:softmax',  # 'booster':'gbtree',
                  'num_class': 4,
                 }

        num_round = 50

        bst = xgb.train(param, dtrain, num_round)
        # make prediction
        # print("==> Save the model ...")
        pickle.dump(bst, open(model_path, 'wb'))
        return bst

    def test_model(self, test_file_path, model_path, result_file_path):
        # print("==> Load the model ...")
        bst = pickle.load(open(model_path, 'rb'))
        # print("==> Test the model ...")

        # 去掉comment 部分
        with open(test_file_path) as fin, open(test_file_path + ".tmp", "w") as fout:
            new_lines = []
            for line in fin:
                line = line.strip().split(" #")[0]
                new_lines.append(line)
            fout.write("\n".join(new_lines))

        dtest = xgb.DMatrix(test_file_path + ".tmp")
        pred_y = bst.predict(dtest)
        # print("==> Save the result ...")

        with open(result_file_path, 'w') as fout:
            fout.write("\n".join(map(str, map(int, pred_y))))


'''liblinear'''

class LibLinearSVM(Strategy):
    def __init__(self, s, c):
        super().__init__()
        self.s = s
        self.c = c
        self.trainer = "LibLinear SVM"
        self.idname = "liblinear_svm"
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
        cmd = config.LIB_LINEAR_PATH + "/train -s " + str(self.s) + " -n 8 -c " + str(self.c) + " " + train_feature_path + " " + model_path + " 1> " + config.DATA_PATH + "/lib_out.txt" + " 2> " + config.DATA_PATH + "/lib_last_error.txt"
        # cmd = config.LIB_LINEAR_PATH + "/train -s 0 -n 8 -c 1 -w0 9 -w1 1" + train_feature_path + " " + model_path + " 1> " + config.DATA_PATH + "/tmp1.txt" + " 2> " + config.DATA_PATH + "/tmp2.txt"
        if DEBUG:
            print(cmd)
        os.system(cmd)

    def test_model(self, test_feature_path, model_path, result_file_path):
        cmd = config.LIB_LINEAR_PATH + "/predict "
        if self.use_proba:
            cmd += "-b 1 "
        cmd += test_feature_path + " " + model_path + " " + result_file_path
        if DEBUG:
            print(cmd)
        os.system(cmd)

    # def test_model(self, test_feature_path, model_path, result_file_path):
    #     cmd = config.LIB_LINEAR_PATH + "/predict -b 1 " + test_feature_path + " " + model_path + " " + result_file_path
    #     # print(cmd)
    #     os.system(cmd)


