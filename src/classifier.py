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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier

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

class skLearn_DecisionTree(Strategy):
    def __init__(self):
        super().__init__()
        self.trainer = "skLearn decisionTree"
        self.idname = "sklearn_dcstree"
        self.clf = tree.DecisionTreeClassifier()
        print("Using %s Classifier" % (self.trainer))

    def train_model(self, train_file_path, model_path):
        train_X, train_y = load_svmlight_file(train_file_path)

        print("==> Train the model ...")
        self.clf.fit(train_X, train_y)


    def test_model(self, test_file_path, model_path, result_file_path):

        print("==> Test the model ...")
        test_X, test_y = load_svmlight_file(test_file_path)
        pred_y = self.clf.predict(test_X)

        # write prediction to file
        with open(result_file_path, 'w') as fout:
            fout.write("\n".join(map(str, map(int, pred_y))))




class skLearn_NaiveBayes(Strategy):
    def __init__(self):
        super().__init__()
        self.trainer = "skLearn NaiveBayes"
        self.idname = "sklearn_navbayes"
        self.clf = GaussianNB()
        print("Using %s Classifier" % (self.trainer))

    def train_model(self, train_file_path, model_path):
        train_X, train_y = load_svmlight_file(train_file_path)

        train_X = train_X.toarray()
        print("==> Train the model ...")
        self.clf.fit(train_X, train_y)


    def test_model(self, test_file_path, model_path, result_file_path):

        print("==> Test the model ...")
        test_X, test_y = load_svmlight_file(test_file_path)
        test_X = test_X.toarray()
        pred_y = self.clf.predict(test_X)

        # write prediction to file
        with open(result_file_path, 'w') as fout:
            fout.write("\n".join(map(str, map(int, pred_y))))

class skLearn_svm(Strategy):
    def __init__(self):
        super().__init__()
        self.trainer = "skLearn Support Vector Machine"
        self.idname = "sklearn_svm"
        self.clf = svm.LinearSVC()
        self.make_feature_handler = make_feature_file.make_feature_for_sklearn
        print("Using %s Classifier" % (self.trainer))

    def train_model(self, train_file_path, model_path):
        train_X, train_y = load_svmlight_file(train_file_path)

        print("==> Train the model ...")
        self.clf.fit(train_X, train_y)


    def test_model(self, test_file_path, model_path, result_file_path):

        print("==> Test the model ...")
        test_X, test_y = load_svmlight_file(test_file_path)
        pred_y = self.clf.predict(test_X)

        # write prediction to file
        with open(result_file_path, 'w') as fout:
            fout.write("\n".join(map(str, map(int, pred_y))))


class skLearn_lr(Strategy):
    def __init__(self):
        super().__init__()
        self.trainer = "skLearn LogisticRegression"
        self.idname = "sklearn_logreg"
        self.clf = LogisticRegression()
        print("Using %s Classifier" % (self.trainer))

    def train_model(self, train_file_path, model_path):
        train_X, train_y = load_svmlight_file(train_file_path)

        print("==> Train the model ...")
        self.clf.fit(train_X, train_y)


    def test_model(self, test_file_path, model_path, result_file_path):

        print("==> Test the model ...")
        test_X, test_y = load_svmlight_file(test_file_path)
        pred_y = self.clf.predict(test_X)

        # write prediction to file
        with open(result_file_path, 'w') as fout:
            fout.write("\n".join(map(str, map(int, pred_y))))


class skLearn_KNN(Strategy):
    def __init__(self):
        super().__init__()
        self.trainer = "skLearn KNN"
        self.idname = "sklearn_knn"
        self.clf = KNeighborsClassifier(n_neighbors=3)
        print("Using %s Classifier" % (self.trainer))

    def train_model(self, train_file_path, model_path):
        train_X, train_y = load_svmlight_file(train_file_path)

        print("==> Train the model ...")
        self.clf.fit(train_X, train_y)


    def test_model(self, test_file_path, model_path, result_file_path):

        print("==> Test the model ...")
        test_X, test_y = load_svmlight_file(test_file_path)
        pred_y = self.clf.predict(test_X)

        # write prediction to file
        with open(result_file_path, 'w') as fout:
            fout.write("\n".join(map(str, map(int, pred_y))))



class skLearn_AdaBoostClassifier(Strategy):
    def __init__(self):
        super().__init__()
        self.trainer = "skLearn AdaBoostClassifier"
        self.idname = "sklearn_adaboost"
        self.clf = AdaBoostClassifier()
        print("Using %s Classifier" % (self.trainer))

    def train_model(self, train_file_path, model_path):
        train_X, train_y = load_svmlight_file(train_file_path)
        print("==> Train the model ...")
        self.clf.fit(train_X.toarray(), train_y)
        pickle.dump(self.clf, open(model_path, 'wb'))

    def test_model(self, test_file_path, model_path, result_file_path):
        print("==> Test the model ...")
        test_X, test_y = load_svmlight_file(test_file_path)
        self.clf = pickle.load(open(model_path, 'rb'))
        pred_y = self.clf.predict(test_X.toarray())

        # write prediction to file
        with open(result_file_path, 'w') as fout:
            fout.write("\n".join(map(str, map(int, pred_y))))


class sklearn_RandomForestClassifier(Strategy):
    def __init__(self):
        super().__init__()
        self.trainer = "skLearn RandomForestClassifier"
        self.idname = "sklearn_rand_forest"
        self.clf = RandomForestClassifier()
        print("Using %s Classifier" % (self.trainer))

    def train_model(self, train_file_path, model_path):
        train_X, train_y = load_svmlight_file(train_file_path)

        print("==> Train the model ...")
        self.clf.fit(train_X, train_y)


    def test_model(self, test_file_path, model_path, result_file_path):

        print("==> Test the model ...")
        test_X, test_y = load_svmlight_file(test_file_path)
        pred_y = self.clf.predict(test_X)

        # write prediction to file
        with open(result_file_path, 'w') as fout:
            fout.write("\n".join(map(str, map(int, pred_y))))


class sklearn_VotingClassifier(Strategy):
    def __init__(self):
        super().__init__()
        self.trainer = "skLearn VotingClassifier"
        self.idname = "sklearn_voting"

        clf1 = LogisticRegression()
        clf2 = svm.LinearSVC()
        clf3 = AdaBoostClassifier()

        self.clf = VotingClassifier(estimators=[('lr', clf1), ('svm', clf2), ('ada', clf3)], voting='hard')

        print("Using %s Classifier" % (self.trainer))

    def train_model(self, train_file_path, model_path):
        train_X, train_y = load_svmlight_file(train_file_path)

        print("==> Train the model ...")
        self.clf.fit(train_X, train_y)


    def test_model(self, test_file_path, model_path, result_file_path):

        print("==> Test the model ...")
        test_X, test_y = load_svmlight_file(test_file_path)
        pred_y = self.clf.predict(test_X)

        # write prediction to file
        with open(result_file_path, 'w') as fout:
            fout.write("\n".join(map(str, map(int, pred_y))))


'''liblinear'''


class LibLinear(Strategy):

    def __init__(self, s, c):
        super().__init__()
        self.s = s
        self.c = c
        self.trainer = "Liblinear"
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


