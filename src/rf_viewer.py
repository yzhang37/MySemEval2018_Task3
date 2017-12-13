# encoding: utf-8
import sys
import os
import re
import json

sys.path.append("..")

from src import config
from src.model_trainer.dict_loader import Dict_loader


class Rf_Viewer(object):
    def __init__(self, dict=None, rf_file=None):
        self.dict = dict
        self.rf_data = None
        if rf_file is not None:
            self.load_rf_file(rf_file)

    def __check_data(self):
        assert self.dict is not None, "没有加载字典。"
        assert self.rf_data is not None, "RF 数据不能是空的。"

    def load_rf_file(self, rf_file=None):
        if rf_file is not None:
            if not os.path.exists(rf_file):
                fout = open(rf_file, "w")
                fout.close()
            self.rf_data = json.load(open(rf_file, "r"))

    def view_class_top_k(self, dict=None, k=10):
        if dict is not None:
            self.dict = dict
        self.__check_data()
        inv_dict = {v: k for k, v in self.dict.items()}

        rf_sorted = self.rf_data.get("rf_sorted")
        assert rf_sorted, "数据错误"
        for clsid, cls_data in rf_sorted.items():
            print("Class", clsid)
            print("="*40)
            for i in range(0, k):
                print(inv_dict[cls_data[i][0]], cls_data[i][1])
            print()

    def print_list_rf(self):
        self.__check_data()
        inv_dict = {v: k for k, v in self.dict.items()}
        rf_value = self.rf_data.get("rf_value")
        assert rf_value, "数据错误"
        print("word;cls;max_rf")
        for word_id, rf_data in rf_value.items():
            print(inv_dict[int(word_id)], ";", rf_data["cls"], ";", rf_data["max_rf"])

    def print_word_rf(self, word):
        self.__check_data()
        word_id = self.dict.get(word)
        assert word_id, "没有找到单词 \'%s\'."%word
        inv_dict = {v: k for k, v in self.dict.items()}
        rf_value = self.rf_data.get("rf_value")
        assert rf_value, "数据错误"
        if str(word_id) in rf_value:
            print("word;cls;max_rf")
            print(word, ";", rf_value[str(word_id)]["cls"], ";", rf_value[str(word_id)]["max_rf"])

    def get_word_id_rf(self, word_id):
        rf_value = self.rf_data.get("rf_value")
        return rf_value[str(word_id)]["cls"], rf_value[str(word_id)]["max_rf"] if str(word_id) in rf_value else None


if __name__ == "__main__":
    dict_loader = Dict_loader()
    viewer = Rf_Viewer(dict=dict_loader.dict_nltk_unigram)
    viewer.load_rf_file(rf_file=config.RF_DATA_NLTK_UNIGRAM_PATH)
    # viewer.view_class_top_k(k=20)
    # viewer.print_list_rf()
    # viewer.print_word_rf("Oxford")

