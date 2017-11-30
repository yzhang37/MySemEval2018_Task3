#coding:utf-8
import sys
sys.path.append("../..")
from src import config
from src import util
from src.example import Example


def make_feature(tweets, feature_function_list, to_file):

    example_list = []
    for tw in tweets:
        features = [feature_function(tw) for feature_function in feature_function_list]
        feature = util.mergeFeatures(features)

        target = int(config.get_label_map(tw["label"]))

        comment = str(tw["id"])
        example = Example(target, feature, comment)
        example_list.append(example)
    util.write_example_list_to_file(example_list, to_file)