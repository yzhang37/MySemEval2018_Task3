#coding:utf-8
import sys
import re
import numpy as np
sys.path.append("..")
from src.feature import Feature


class ProgressBar:
    def __init__(self, count=0, total=0, width=50):
        self.count = count
        self.total = total
        self.width = width

    def move(self):
        self.count += 1

    def log(self, s=""):
        if len(s) > 1:
            sys.stdout.write(' ' * (self.width + 9) + '\r')
            sys.stdout.flush()
            print(s)
        progress = int(self.width * self.count / self.total)
        sys.stdout.write('{0:3}/{1:3}: '.format(self.count, self.total))
        sys.stdout.write('#' * progress + '-' * (self.width - progress) + '\r')
        if progress == self.width:
            sys.stdout.write('\n')
        sys.stdout.flush()


# 计算字典中词的频率
def set_dict_key_value(dict, key):
    if key not in dict:
        dict[key] = 0
    dict[key] += 1

#在字典中删除 value < threshold 的item
def removeItemsInDict(dict, threshold=1):
    if threshold > 1:
        for key in list(dict.keys()):
            if dict[key] < threshold:
                dict.pop(key)

def write_dict_keys_to_file(dict, file_path):
    with open(file_path, "w", encoding="utf-8") as file_out:
        file_out.write("\n".join([str(key) for key in sorted(dict.keys())]))

# key : index; index 从1开始
def load_dict_from_file(dict_file_path):
    with open(dict_file_path) as dict_file:
        d = {}
        lines = [line.strip() for line in dict_file]
        # ngram_list = clean_unigram(lines)
        for index, line in enumerate(lines):
            if line == "":
                continue
            d[line] = index+1
        return d


# 专门用于读取 GloVe 词典的函数
def load_dict_from_GloVe_file(dict_file_path):
    d = dict()
    rc = re.compile("^(.+?)(?=(?:\s|$))")
    with open(dict_file_path) as dict_file:
        idx = 0
        for line in dict_file:
            w = rc.findall(line)
            if len(w) > 0:
                d[w[0]] = idx
                idx += 1
    return d


def dict_overlap(dict1=None, dict2=None):
    assert isinstance(dict1, dict), "dict1 should be an instance of Dictionary."
    assert isinstance(dict2, dict), "dict2 should be an instance of Dictionary."

    s1 = set(dict1.keys())
    s2 = set(dict2.keys())
    s3 = s1.intersection(s2)

    return {"intersection": len(s3),
            "overlap_1": len(s3) / len(s1),
            "overlap_2": len(s3) / len(s2),
            "len_1": len(s1),
            "len_2": len(s2)}


def singleton(cls):
    instances = {}
    def _singleton(*args, **kw):
        if (cls, args) not in instances:
            instances[(cls, args)] = cls(*args, **kw)
        return instances[(cls, args)]
    return _singleton

# [0, 1, 0, 1]
def get_feature_by_list(list):
    feat_dict = {}
    for index, item in enumerate(list):
        if item != 0:
            feat_dict[index+1] = item
    return Feature("", len(list), feat_dict)


def get_feature_by_feat_list(dict, token_list):
    feat_dict = {}
    for token in token_list:
        if token in dict:
            feat_dict[dict[token]] = 1
    # print (len(dict))
    return Feature("", len(dict), feat_dict)


def get_feature_by_feature_list_with_rf(dict, token_list, rf_object):
    feat_dict = {}
    for token in token_list:
        if token in dict:
            cls, rf_v = rf_object.get_word_id_rf(dict[token])
            feat_dict[dict[token]] = rf_v
    # print (len(dict))
    return Feature("", len(dict), feat_dict)


''' 合并 feature_list中的所有feature '''
def mergeFeatures(feature_list, name = ""):
    # print "-"*80
    # print "\n".join([feature_file.feat_string+feature_file.name for feature_file in feature_list])
    dimension = 0
    feat_string = ""
    for feature in feature_list:
        if dimension == 0:#第一个
            feat_string = feature.feat_string
        else:
            if feature.feat_string != "":
                #修改当前feature的index
                temp = ""
                for item in feature.feat_string.split(" "):
                    index, value = item.split(":")
                    temp += " %d:%s" % (int(index)+dimension, value)
                feat_string += temp
        dimension += feature.dimension

    merged_feature = Feature(name, dimension, {})
    merged_feature.feat_string = feat_string.strip()
    return merged_feature


def split_reg_tokens(data_list, split_reg_list):
    assert(isinstance(data_list, list))
    assert(len(split_reg_list) < 1 or (isinstance(split_reg_list[0], tuple) and len(split_reg_list[0]) == 2))

    for re_pat, rsub in split_reg_list:
        rc = re.compile(re_pat)

        i = 0
        while i < len(data_list):
            token = data_list[i]
            if rc.match(token):
                del data_list[i]
                j = 0
                for sub_str in rsub:
                    data_list.insert(i + j, rc.sub(sub_str, token))
                    j += 1
            else:
                i += 1


def rewrite_reg_tokens(data_list, rewrite_reg_list):
    split_list = []
    for item in rewrite_reg_list:
        item = (item[0], [item[1]])
        split_list.append(item)
    split_reg_tokens(data_list, split_list)


def delete_reg_tokens(data_list, delete_reg_list):
    split_list = []
    for item in delete_reg_list:
        split_list.append((item, []))
    split_reg_tokens(data_list, split_list)


def write_example_list_to_file(example_list, to_file):
    with open(to_file, "w") as fout:
        fout.write("\n".join([example.content + " # " + example.comment for example in example_list]))


def _get_max_dim(in_path):
    with open(in_path) as fin:
        max_dim = 0
        for line in fin:
            feature = line.strip().split(" ", 1)[-1].split(" #")[0]
            dim = 0
            if feature != "":
                dim = int(feature.split(" ")[-1].split(":")[0])
            if dim > max_dim:
                max_dim = dim

        return max_dim

def _add_max_dim_for_file(in_file, max_dim):
    # 在dev中给某个维度加 train_max_dim:0
    with open(in_file) as fin:
        new_lines = []
        flag = 0
        for line in fin:
            if flag == 1:
                new_lines.append(line.strip())
            else:
                target, feature_comment = line.strip().split(" ", 1)
                feature, comment = feature_comment.split("#")
                feature = feature.strip()
                comment = comment.strip()

                dim = 0
                if feature != "":
                    dim = int(feature.split(" ")[-1].split(":")[0])

                if dim < max_dim:
                    feature += " %d:0" % (max_dim)
                    new_line = "%s %s # %s" % (target, feature, comment)
                    new_lines.append(new_line)
                    flag = 1
                else:
                    new_lines.append(line.strip())

def handle_train_test_dim(train_path, dev_path):
    train_max_dim = _get_max_dim(train_path)
    dev_max_dim = _get_max_dim(dev_path)

    if train_max_dim > dev_max_dim:
        # 在dev中给某个维度加 train_max_dim:0
        _add_max_dim_for_file(dev_path, train_max_dim)

    if dev_max_dim > train_max_dim:
        # 在train中给某个维度加 train_max_dim:0
        _add_max_dim_for_file(train_path, dev_max_dim)


def print_dedicated_mean(prec, recl, f1l):
    print("=" * 40)
    line_count = len(prec)
    for id in range(0, line_count):
        print("Fold %d\t: %.2f%%\t%.2f%%\t%.2f%%" % (id + 1, prec[id] * 100, recl[id] * 100, f1l[id] * 100))
    print("=" * 40)
    print("Mean\t: %.2f%%\t%.2f%%\t%.2f%%" % (np.mean(prec) * 100, np.mean(recl) * 100, np.mean(f1l) * 100))


def print_markdown_mean(prec, recl, f1l):
    print("#### Train Result Table: ")
    print("| Fold | Precision | Recall | F-1 |")
    print("| ---- | --------- | ------ | --- |")
    line_count = len(prec)
    for id in range(0, line_count):
        print("| Fold %d | %.2f%% | %.2f%% | %.2f%% |" % (id + 1, prec[id] * 100, recl[id] * 100, f1l[id] * 100))
    print("| **Mean** | **%.2f%%** | **%.2f%%** | **%.2f%%** |" % (np.mean(prec) * 100, np.mean(recl) * 100, np.mean(f1l) * 100))