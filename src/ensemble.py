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

        json.dump(total_data, open(output_path, "w"), indent=4)

        print()
        print("Ensemble data dumped to %s" % output_path)
        print("at %s" % time.asctime(time.localtime(time.time())))
        print()
    except Exception as ex:
        print("Exception: ", ex)


def __print_score_helper(score_dict: dict):
    if "precision" in score_dict and \
       "recall" in score_dict and \
       "f1" in score_dict:
        print("precision: %f, recall: %f, f1: %f" % (score_dict["precision"], score_dict["recall"], score_dict["f1"]))
    else:
        raise KeyError("Cannot find necessary key in", score_dict)


def build_top_ensemble_score_json(old_json_path, new_json_path, threshold=None, top=None):
    print("Loading ensemble data from:")
    print(old_json_path)
    ensemble_data_list = json.load(open(old_json_path))

    if config.get_class() == "A":
        ensemble_data_list.sort(key=lambda x: -x["score"]["1"]["f1"])
    else:
        raise NotImplementedError("排序算法尚未实现。")

    new_ensemble_data_list = []
    for ensemble_data in ensemble_data_list:
        if top is not None and len(new_ensemble_data_list) >= top:
            break
        if config.get_class() == "A":
            if threshold is not None and ensemble_data["score"]["1"]["f1"] < threshold:
                break
            new_ensemble_data_list.append(ensemble_data)
        else:
            raise NotImplementedError("尚未实现。")

    json.dump(new_ensemble_data_list, open(new_json_path, "w"), indent=4)
    print("New ensemble data built at:")
    print(new_json_path)


def get_ensemble_path_list_from_score_json(json_path, threshold=None, top=None):
    ensemble_data_list = json.load(open(json_path))
    ensemble_file_path_list = []
    for ensemble_data in ensemble_data_list:
        if top is not None and len(ensemble_file_path_list) >= top:
            break
        if config.get_class() == "A":
            class_1_score = ensemble_data["score"]["1"]["f1"]
            if threshold is not None and class_1_score < threshold:
                continue
            print("Ensemble on %s" % ensemble_data["name"])
            print("--" * 30)
            print("Class \"1\" score: ", end="")
            __print_score_helper(ensemble_data["score"]["1"])

            print("Ensemble file loaded from:")
            print(ensemble_data["ensemble_path"])
            ensemble_file_path_list.append(ensemble_data["ensemble_path"])
        elif config.get_class() == "B":
            raise NotImplementedError()
        print()
    return ensemble_file_path_list


def make_ensemble_from_score_json(json_path, output_path, threshold=None, top=None):
    path_list = get_ensemble_path_list_from_score_json(json_path, threshold, top)
    if len(path_list) > 0:
        make_ensemble_from_file(path_list, output_path)
    else:
        print("0 files needed to be handled.")


def make_result_from_ensemble(json_path, result_path):
    ensemble_data = json.load(open(json_path))
    line_data = []
    for str_idx, data in sorted(ensemble_data.items(), key=lambda x:int(x[0])):
        dat = sorted(data.items(), key=lambda x: -x[1])
        line_data.append(dat[0][0])
    with open(result_path, "w") as fout:
        for line in line_data:
            fout.write(line)
            fout.write('\n')
    print("Result file saved to:")
    print(result_path)


OUTPUT_LIST_JSON = os.path.join(config.ENSEMBLE_SCORE_PATH, "output.json")
TOP_LIST_JSON = os.path.join(config.ENSEMBLE_SCORE_PATH, "top.json")
NN_PATH = os.path.join(config.ENSEMBLE_PATH, "nn_ali_%s.json" % config.get_class().lower())
FINAL_PATH = os.path.join(config.ENSEMBLE_PATH, "all.json")
RESULT_PATH = os.path.join(config.RESULT_MYDIR, "ensemble_result.txt")


def main():
    path_list = []
    # build_top_ensemble_score_json(OUTPUT_LIST_JSON, TOP_LIST_JSON, top=5)
    # path_list = get_ensemble_path_list_from_score_json(TOP_LIST_JSON, top=5)
    path_list.append(NN_PATH)
    make_ensemble_from_file(path_list, FINAL_PATH)
    make_result_from_ensemble(FINAL_PATH, RESULT_PATH)


if __name__ == "__main__":
    main()
    # build_top_ensemble_score_json(os.path.join(config.ENSEMBLE_SCORE_PATH, "output.json"),
    #                               os.path.join(config.ENSEMBLE_SCORE_PATH, "top.json"),
    #                               top=4)

