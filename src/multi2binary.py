# encoding: utf-8
import sys
import os
import copy
import json
sys.path.append("..")
from src import config


if config.if_multi_binary():
    def build_binary_tweets_from_multi(original_tweets):
        print("Building Binary Classifiers from Multi-class Classifiers")
        label_count = len(config.get_all_label_list())
        ret = [copy.deepcopy(original_tweets) for i in range(label_count)]

        try:
            for cur_label_idx in range(len(ret)):
                cur_tweets = ret[cur_label_idx]
                for idx in range(len(cur_tweets)):
                    if 'label' in cur_tweets[idx]:
                        old_label = cur_tweets[idx]['label']
                        new_label = config.get_binary_label_map(cur_label_idx, old_label)
                        cur_tweets[idx]['label'] = new_label

            return ret

        except Exception as e:
            print(e)
            raise e


    def duplicate_class_data(tweets_data, label=None, num=None):
        """
        Duplicate the train data.
        :param [in/out] tweets_data: A list contain tweet data.
        :param [in] label: Either a string or a list of string.
        :param [in] num: Either an int or a list of int. If a list of int is placed, then label param should
                    also be a list of string.
        :return: Nothing.
        """
        if num is None:
            num = 1
        if not isinstance(num, int) and any([not isinstance(item, int) for item in num]):
            raise TypeError("num should be an integer or a list of integer.")

        if label is None or (isinstance(num, int) and num <= 0) or \
                (isinstance(num, list) and all([item <= 0 for item in num])):
            print("Nothing needed to process.")
            return

        if isinstance(num, list) and not isinstance(label, list):
            raise TypeError("Num and label should have the same type.")

        if isinstance(label, list) and isinstance(num, list) and len(label) != len(num):
            raise TypeError("Num and label should have the same size.")

        print("=="*30)
        if isinstance(label, str):
            print("Duplicate data of class \"%s\" %d times." % (label, num))
            print("Original Data Count: %d" % len(tweets_data))
            calc_str = "%d" % len(tweets_data)
            label_data = []
            for item in tweets_data:
                if 'label' in item and item['label'] == label:
                    label_data.append(item)
            print("Label \"%s\" Count: %d, after duplicated: %d x %d = %d" % (label, len(label_data), len(label_data), num, len(label_data) * (num + 1)))
            calc_str += " + %d" % (len(label_data) * (num + 1))
            for i in range(num):
                tweets_data += label_data
            print()
            print("After duplicate, " + calc_str + " =")
            print("Count: %d" % len(tweets_data))
        elif isinstance(label, list) and all([isinstance(item, str) for item in label]):
            if isinstance(num, int):
                print("Duplicate data of class [" + ", ".join(["\"" + item + "\"" for item in label]) + "] %d times" % num)
            else:
                print("Duplicate data of class [" + ", ".join(["\"" + item + "\"" for item in label]) + "],")
                print("respectively [" + ", ".join([str(item) for item in num]) + "] times.")
            print("Original Data Count: %d" % len(tweets_data))
            calc_str = "%d" % len(tweets_data)

            if isinstance(num, int):
                for __label in label:
                    label_data = []
                    for item in tweets_data:
                        if 'label' in item and item['label'] == __label:
                            label_data.append(item)
                    print("Label \"%s\" Count: %d, after duplicated: %d x %d = %d" % (__label, len(label_data), len(label_data), num, len(label_data) * (num + 1)))
                    calc_str += " + %d" % (len(label_data) * (num + 1))
                    for i in range(num):
                        tweets_data += label_data
            else:
                for __label, __num in zip(label, num):
                    label_data = []
                    for item in tweets_data:
                        if 'label' in item and item['label'] == __label:
                            label_data.append(item)
                    print("Label \"%s\" Count: %d, after duplicated: %d x %d = %d" % (__label, len(label_data), len(label_data), __num, len(label_data) * (__num + 1)))
                    calc_str += " + %d" % (len(label_data) * (__num + 1))
                    for i in range(__num):
                        tweets_data += label_data
            print()
            print("After duplicate, " + calc_str + " =")
            print("Count: %d" % len(tweets_data))
        else:
            raise TypeError("label should be an string or a list of string.")


    if __name__ == "__main__":
        a = json.load(open(config.PROCESSED_TRAIN))
        duplicate_class_data(a, ["2", "3"], [5, 9])
