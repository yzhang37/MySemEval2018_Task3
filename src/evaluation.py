# encoding:utf8
import sys
import sklearn
from sklearn import metrics
sys.path.append("..")
from src.confusion_matrix import Alphabet, ConfusionMatrix
from src import config


def get_cm_eval(matrix: ConfusionMatrix):
    if config.get_class() == "A":
        # as for the task A, because it only evaluate on class "1",
        # so instead of using average f1, we use f1 of class "1".
        return matrix.get_prf("1")
    else:
        return matrix.get_average_prf()


def Evaluation(gold_file_path, predict_file_path, label):

    with open(gold_file_path) as gold_file, open(predict_file_path) as predict_file:
        golds = [line.strip().split(" ")[0] for line in gold_file]
        predictions = map(lambda x: str(int(float(x.strip()))), predict_file)
        # predictions = [line.strip() for line in predict_file]
        alphabet = Alphabet()

        for i in label:
            alphabet.add(str(i))
        cm = ConfusionMatrix(alphabet)
        cm.add_list(predictions, golds)

        # cm.print_out()
        return cm

def Evaluation_b(gold_file_path, predict_file_path, label):

    with open(gold_file_path) as gold_file, open(predict_file_path) as predict_file:
        golds = [line.strip().split(" ")[0] for line in gold_file]
        predict_lines = [line.strip() for line in predict_file][1:]
        predictions = [str(int(float(line.split(" ")[0]))) for line in predict_lines]

        # predictions = map(lambda x: str(int(float(x.strip()))), predict_file)
        # predictions = [line.strip() for line in predict_file]
        alphabet = Alphabet()

        for i in label:
            alphabet.add(str(i))
        cm = ConfusionMatrix(alphabet)
        cm.add_list(predictions, golds)

        # cm.print_out()
        return cm

def Evaluation2(golds, predictions, label):

    alphabet = Alphabet()
    for i in range(label):
        alphabet.add(str(i))

    cm = ConfusionMatrix(alphabet)
    cm.add_list(predictions, golds)

    # cm.print_out()
    return cm

def Evaluation3(golds, predictions, label):

    alphabet = Alphabet()
    for i in label:
        alphabet.add(str(i))

    cm = ConfusionMatrix(alphabet)
    predictions = map(str, predictions)
    golds = map(str, golds)
    cm.add_list(predictions, golds)

    return cm


    # F1score = metrics.f1_score(golds, predicts)
    # precision = metrics.precision_score(golds, predicts)
    # recall = metrics.recall_score(golds, predicts)




    # return F1score, precision, recall
if __name__ == '__main__':
    print('a')
    cm = Evaluation2([0,1,2,3,4],[0, 1,1,1,1],5)
    cm.print_out()
    p,r,f = cm.get_average_prf()
    print(p,r,f)