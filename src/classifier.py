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


class Strategy(object):
    def __init__(self):
        self.make_feature_handler = make_feature_file.make_feature_for_liblinear
        self.idname = "noname"
    def train_model(self, train_feature_path, model_path):
        return None
    def test_model(self, test_feature_path, model_path, result_file_path):
        return None

    # We no longer need this static method, because scikit contains a function
    # to convert liblinear format to its own data format.
    #
    # @staticmethod
    # def make_feature_handler(tweets, feature_function_list, to_file):
    #     assert False, "Make feature function not yet implemented."


class Classifier(object):
    def __init__(self, strategy):
        self.strategy = strategy
    def idname(self):
        return self.strategy.idname
    def train_model(self, train_feature_path, model_path):
        self.strategy.train_model(train_feature_path, model_path)
    def test_model(self, test_feature_path, model_path, result_file_path):
        self.strategy.test_model(test_feature_path, model_path, result_file_path)
    def make_feature(self, tweets, feature_function_list, to_file):
        self.strategy.make_feature_handler(tweets, feature_function_list, to_file)


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
        pred_y = self.clf.predict(test_X)

        # write prediction to file
        with open(result_file_path, 'w') as fout:
            fout.write("\n".join(map(str, map(int, pred_y))))


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
        pred_y = self.clf.predict(test_X)

        # write prediction to file
        with open(result_file_path, 'w') as fout:
            fout.write("\n".join(map(str, map(int, pred_y))))


# Sklearn 支持向量机
class SkLearnSVM(Strategy):
    def __init__(self):
        super().__init__()
        self.trainer = "Scikit-Learn Support Vector Machine"
        self.idname = "sklearn_svm"
        self.clf = svm.LinearSVC()
        self.make_feature_handler = make_feature_file.make_feature_for_sklearn
        print("Using %s Classifier" % (self.trainer))

    def train_model(self, train_file_path, model_path):
        train_X, train_y = load_svmlight_file(train_file_path)

        # print("==> Train the model ...")
        self.clf.fit(train_X, train_y)

    def test_model(self, test_file_path, model_path, result_file_path):
        # print("==> Test the model ...")
        test_X, test_y = load_svmlight_file(test_file_path)
        pred_y = self.clf.predict(test_X)

        # write prediction to file
        with open(result_file_path, 'w') as fout:
            fout.write("\n".join(map(str, map(int, pred_y))))


# 随机梯度下降
class SkLearnSGD(Strategy):
    def __init__(self):
        super().__init__()
        self.trainer = "Scikit-Learn Stochastic Gradient Descent (hinge)"
        self.idname = "sklearn_sgd_hinge"
        self.clf = SGDClassifier(loss='hinge')
        print("Using %s Classifier" % self.trainer)

    def train_model(self, train_file_path, model_path):
        train_X, train_y = load_svmlight_file(train_file_path)
        # print("==> Train the model ...")
        self.clf.fit(train_X, train_y)

    def test_model(self, test_file_path, model_path, result_file_path):
        # print("==> Test the model ...")
        test_X, test_y = load_svmlight_file(test_file_path)
        pred_y = self.clf.predict(test_X)

        # write prediction to file
        with open(result_file_path, 'w') as fout:
            fout.write("\n".join(map(str, map(int, pred_y))))


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
        pred_y = self.clf.predict(test_X)

        # write prediction to file
        with open(result_file_path, 'w') as fout:
            fout.write("\n".join(map(str, map(int, pred_y))))

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
        pred_y = self.clf.predict(test_X)

        # write prediction to file
        with open(result_file_path, 'w') as fout:
            fout.write("\n".join(map(str, map(int, pred_y))))


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
        pred_y = self.clf.predict(test_X.toarray())

        # write prediction to file
        with open(result_file_path, 'w') as fout:
            fout.write("\n".join(map(str, map(int, pred_y))))


class SkLearnXGBoostClassifier(Strategy):
    def __init__(self):
        super().__init__()
        self.trainer = "Scikit-Learn XGBoost"
        self.idname = "sklearn_xgboost"
        self.clf = KNeighborsClassifier(n_neighbors=3)
        print("Using %s Classifier" % (self.trainer))

    def train_model(self, train_file_path, model_path):
        train_X, train_y = load_svmlight_file(train_file_path)
        # print("==> Train the model ...")
        self.clf.fit(train_X, train_y)

    def test_model(self, test_file_path, model_path, result_file_path):
        # print("==> Test the model ...")
        test_X, test_y = load_svmlight_file(test_file_path)
        pred_y = self.clf.predict(test_X)

        # write prediction to file
        with open(result_file_path, 'w') as fout:
            fout.write("\n".join(map(str, map(int, pred_y))))


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
        pred_y = self.clf.predict(test_X)

        # write prediction to file
        with open(result_file_path, 'w') as fout:
            fout.write("\n".join(map(str, map(int, pred_y))))


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
        pred_y = self.clf.predict(test_X)

        # write prediction to file
        with open(result_file_path, 'w') as fout:
            fout.write("\n".join(map(str, map(int, pred_y))))


'''XgBoost'''
class XGBOOST(Strategy):
    def __init__(self):
        super().__init__()
        self.trainer = "XGBOOST"
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
        y_pred = bst.predict(dtest)
        # print("==> Save the result ...")


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


