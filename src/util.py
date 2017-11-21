#coding:utf-8
import sys
sys.path.append("..")
from src.feature import Feature

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