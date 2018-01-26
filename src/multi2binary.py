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


    if __name__ == "__main__":
        tweets = json.load(open(config.PROCESSED_TRAIN))
        binary_tweets = build_binary_tweets_from_multi(tweets)
