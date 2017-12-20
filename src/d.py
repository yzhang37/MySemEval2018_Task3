import sys
sys.path.append("..")
from src import config
import json
from nltk.tokenize import TweetTokenizer
import os

data = json.load(open(config.PROCESSED_TRAIN, "r"))

list_a = []
list_b = []


for item in data:
    txt = item["clean_tweet"].strip().lower()
    token_txt = TweetTokenizer().tokenize(txt)

    label_b = int(item["label"])
    if label_b == 0:
        label_a = 0
    else:
        label_a = 1
    txt_a = str(label_a) + " " + " ".join(token_txt)
    list_a.append(txt_a)
    txt_b = str(label_b) + " " + " ".join(token_txt)
    list_b.append(txt_b)

data_path = "/home/feixiang/pyCharmSpace/SemEval2018_T3/data/train/forwv"
wv_path = "/home/feixiang/pyCharmSpace/SemEval2018_T3/data/train/wvforab"

file_a = data_path + "/forwv_a"
file_b = data_path + "/forwv_b"

fin_a = open(file_a, "w")
for sen in list_a:
    fin_a.write(sen + "\n")
fin_a.close()

fin_b = open(file_b, "w")
for sen in list_b:
    fin_b.write(sen + "\n")
fin_b.close()

# print("training data ok~~~~~~~~~~~~~")

#  "./run.sh word2vec_C_AB /home/feixiang/pyCharmSpace/ABSA_W2V/data/Sem15-16/absa15_rest_forwv_ep.txt  ./vector/15_rest_ep_c.vec",
# p1 = "/home/feixiang/pyCharmSpace/ABSA_W2V"
# run = os.path.join(p1, "run.sh")
# method = os.path.join(p1, "word2vec_C_AB")
# train_a = os.path.join(data_path, file_a)
# train_b = os.path.join(data_path, file_b)
# towv_a = os.path.join(wv_path, "wv_a.vec")
# towv_b = os.path.join(wv_path, "wv_b.vec")
#
# cmd_a = "{} {} {} {}".format(run, method, train_a, towv_a)
# cmd_b = "{} {} {} {}".format(run, method, train_b, towv_b)
# print(cmd_a)
# # os.system(cmd_a)
# print(cmd_b)
# os.system(cmd_b)





