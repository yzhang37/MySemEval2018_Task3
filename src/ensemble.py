# encoding: utf-8
import sys
import os
import json
import time
sys.path.append("..")
from src import config


def make_ensemble(data_list, file_path):
    try:
        ensemble_data = json.load(open(file_path))
    except:
        ensemble_data = dict()

    for idx, cls in data_list:
        ensemble_data.setdefault(idx, dict())
        ensemble_data[idx].setdefault(cls, 0)
        ensemble_data[idx][cls] += 1
    json.dump(ensemble_data, open(file_path, "w"))
    print("--" * 30)
    print("Ensemble data dumped to %s" % file_path)
    print("at %s" % time.asctime(time.localtime(time.time())) )


def make_ensemble_from_file(file_path_list: list, output_path):

    print("--" * 30)
    print("Loading ensemble from:")
    total_data = dict()

    try:
        for file_path in file_path_list:

            data = json.load(open(file_path))
            print(file_path)

            for idx, cls_f in data.items():
                total_data.setdefault(idx, dict())
                for cls, freq in cls_f.items():
                    total_data[idx].setdefault(cls, 0)
                    total_data[idx][cls] += freq

        json.dump(total_data, open(output_path, "w"))

        print()
        print("Ensemble data dumped to %s" % output_path)
        print("at %s" % time.asctime(time.localtime(time.time())))
        print()
    except Exception as ex:
        print("Exception: ", ex)

