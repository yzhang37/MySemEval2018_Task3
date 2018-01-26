#coding:utf-8
import sys
import random
sys.path.append("../..")
from src import config
from src import util
from src.example import Example


def make_feature_for_liblinear(tweets, feature_function_list, to_file, is_test=False):

    example_list = []
    for tw in tweets:
        features = [feature_function(tw) for feature_function in feature_function_list]
        feature = util.mergeFeatures(features)

        if not is_test:
            target = int(config.get_label_map(tw["label"]))
        else:
            # 测试文件，label 是无效的。
            target = random.choice(config.get_label_list())

        comment = str(tw["id"])
        example = Example(target, feature, comment)
        example_list.append(example)
    util.write_example_list_to_file(example_list, to_file)
