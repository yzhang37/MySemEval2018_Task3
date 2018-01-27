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
    print("at %s" % time.asctime(time.localtime(time.time())))


def make_ensemble_with_proba(data_list, file_path):
    try:
        ensemble_data = json.load(open(file_path))
    except:
        ensemble_data = dict()
    for idx, cls_proba in data_list:
        ensemble_data.setdefault(idx, dict())
        for cls, proba in enumerate(cls_proba):
            cls = str(cls)
            ensemble_data[idx].setdefault(cls, 0)
            ensemble_data[idx][cls] += proba
    json.dump(ensemble_data, open(file_path, "w"))
    print("--" * 30)
    print("Ensemble data with probability dumped to %s" % file_path)
    print("at %s" % time.asctime(time.localtime(time.time())))


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
        ensemble_data_list.sort(key=lambda x: -x["avrg_score"]["f1"])

    new_ensemble_data_list = []
    for ensemble_data in ensemble_data_list:
        if top is not None and len(new_ensemble_data_list) >= top:
            break
        if config.get_class() == "A":
            if threshold is not None and ensemble_data["score"]["1"]["f1"] < threshold:
                break
            new_ensemble_data_list.append(ensemble_data)
        else:
            if threshold is not None and ensemble_data["avrg_score"]["f1"] < threshold:
                break
            new_ensemble_data_list.append(ensemble_data)

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
            f1_macro_score = ensemble_data["avrg_score"]["f1"]
            if threshold is not None and f1_macro_score < threshold:
                continue
            print("Ensemble on %s" % ensemble_data["name"])
            print("--" * 30)
            print("Average score: ", end="")
            __print_score_helper(ensemble_data["avrg_score"])

            print("Ensemble file loaded from:")
            print(ensemble_data["ensemble_path"])
            ensemble_file_path_list.append(ensemble_data["ensemble_path"])
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


def make_partly_result_from_ensemble(json_path, original_golden_path, result_path, target_golden_path):
    with open(original_golden_path) as fopen:
        golden_file_data = fopen.readlines()

    golden_data = [int(line.strip()) for line in golden_file_data]
    new_golden_data = []
    ensemble_data = json.load(open(json_path))
    line_data = []
    for str_idx, data in sorted(ensemble_data.items(), key=lambda x: int(x[0])):
        dat = sorted(data.items(), key=lambda x: -x[1])
        line_data.append(dat[0][0])
        new_golden_data.append(golden_data[int(str_idx)-1])
    with open(result_path, "w") as fout:
        for line in line_data:
            fout.write(line)
            fout.write('\n')
    print("Result file saved to:")
    print(result_path)

    with open(target_golden_path, "w") as fout:
        for line in new_golden_data:
            fout.write(str(line))
            fout.write('\n')
    print("Part golden file saved to:")
    print(target_golden_path)


EXCEL_NAME = "6841_no_drop"
OUTPUT_LIST_JSON = config.make_ensemble_score_path(dspr="test", unique=False)
TOP_LIST_JSON = config.make_ensemble_score_path(dspr="test_top", unique=False)
NN_PATH = os.path.join(config.ENSEMBLE_PATH, "nn_ali_%s_%s.json" % (EXCEL_NAME, config.get_class().lower()))
NN_VALID_PATH = os.path.join(config.ENSEMBLE_PATH, "nn_ali_valid_%s_%s.json" % (EXCEL_NAME, config.get_class().lower()))
NN_TEST_PATH = os.path.join(config.ENSEMBLE_PATH, "nn_ali_test_%s_%s.json" % (EXCEL_NAME, config.get_class().lower()))
PARTLY_GOLDEN = os.path.join(config.CWD, "golden.txt")
FINAL_PATH = os.path.join(config.ENSEMBLE_PATH, "all.json")
RESULT_PATH = os.path.join(config.RESULT_MYDIR, "ensemble_result.txt")


def main(task, is_test=False):
    if task == "0": # 取 top_5 的结果，然后 ensemble (所有的数据)
        build_top_ensemble_score_json(OUTPUT_LIST_JSON, TOP_LIST_JSON, top=5)
        path_list = get_ensemble_path_list_from_score_json(TOP_LIST_JSON, top=5)
        make_ensemble_from_file(path_list, FINAL_PATH)
        make_result_from_ensemble(FINAL_PATH, RESULT_PATH)

        if not is_test:
            from src import evaluation
            cm = evaluation.Evaluation(config.GOLDEN_TRAIN_LABEL_FILE, RESULT_PATH, config.get_label_list())
            cm.print_out()
    elif task == "1": # 仅生成阿里的训练结果的数据，然后进行验证 （只能使用阿里的valid，test是无效的）
        make_partly_result_from_ensemble(NN_VALID_PATH, config.GOLDEN_TRAIN_LABEL_FILE, RESULT_PATH, PARTLY_GOLDEN)

        if not is_test:
            from src import evaluation
            cm = evaluation.Evaluation(PARTLY_GOLDEN, RESULT_PATH, config.get_label_list())
            cm.print_out()
    elif task == "2": # 飞翔的建议：因为阿里的效果较好，则1分类全部使用来自阿里的效果，阿里分类为0 的，则使用剩下来的
        if config.get_class() != 'A':
            raise Exception("不允许使用 B 分类")
        final_result = dict()

        ali_result = json.load(open(NN_TEST_PATH))
        for key, value in ali_result.items():
            if len(value) == 1:
                clsid = str(list(value.keys())[0])
                if clsid == "1":
                    final_result[key] = {clsid: 1}

        print(len(final_result))

        build_top_ensemble_score_json(OUTPUT_LIST_JSON, TOP_LIST_JSON, top=5)
        path_list = get_ensemble_path_list_from_score_json(TOP_LIST_JSON, top=5)
        make_ensemble_from_file(path_list, FINAL_PATH)

        top_five_result = json.load(open(FINAL_PATH))

        for key, value in top_five_result.items():
            if key not in final_result:
                final_result[key] = value

        json.dump(final_result, open(FINAL_PATH, "w"))
        make_result_from_ensemble(FINAL_PATH, RESULT_PATH)

        if not is_test:
            from src import evaluation
            cm = evaluation.Evaluation(config.GOLDEN_TRAIN_LABEL_FILE, RESULT_PATH, config.get_label_list())
            cm.print_out()
    elif task == "3": #飞翔的建议2
        if config.get_class() != 'A':
            raise Exception("不允许使用 B 分类")
        final_result = dict()

        ali_result = json.load(open(NN_TEST_PATH))
        for key, value in ali_result.items():
            if len(value) == 1:
                clsid = str(list(value.keys())[0])
                if clsid == "1":
                    final_result[key] = {clsid: 1}

        print(len(final_result))

        build_top_ensemble_score_json(OUTPUT_LIST_JSON, TOP_LIST_JSON, top=4)
        path_list = get_ensemble_path_list_from_score_json(TOP_LIST_JSON, top=4)
        make_ensemble_from_file(path_list, FINAL_PATH)

        top_four_result = json.load(open(FINAL_PATH))

        for key, value in top_four_result.items():
            if key not in final_result:
                value.setdefault("0", 0)
                value["0"] += 1
                final_result[key] = value

        json.dump(final_result, open(FINAL_PATH, "w"))
        make_result_from_ensemble(FINAL_PATH, RESULT_PATH)

        if not is_test:
            from src import evaluation
            cm = evaluation.Evaluation(config.GOLDEN_TRAIN_LABEL_FILE, RESULT_PATH, config.get_label_list())
            cm.print_out()
    elif task == "4":
        # 使用 TOP4 和 NN_ALI 的结果

        build_top_ensemble_score_json(OUTPUT_LIST_JSON, TOP_LIST_JSON, top=4)
        path_list = get_ensemble_path_list_from_score_json(TOP_LIST_JSON, top=4)

        path_list += [NN_TEST_PATH]
        make_ensemble_from_file(path_list, FINAL_PATH)
        make_result_from_ensemble(FINAL_PATH, RESULT_PATH)

    elif task == "5":
        # for Task B, method 1
        print(OUTPUT_LIST_JSON)
        build_top_ensemble_score_json(OUTPUT_LIST_JSON, TOP_LIST_JSON, top=3)
        path_list = get_ensemble_path_list_from_score_json(TOP_LIST_JSON)
        print(path_list)
        _ = input("请按 Enter或Return 继续")
        make_ensemble_from_file(path_list, FINAL_PATH)

        make_result_from_ensemble(FINAL_PATH, RESULT_PATH)

        if not is_test:
            from src import evaluation
            cm = evaluation.Evaluation(config.GOLDEN_TRAIN_LABEL_FILE, RESULT_PATH, config.get_label_list())
            cm.print_out()


if __name__ == "__main__":
    main("5", is_test=True)
    # build_top_ensemble_score_json(os.path.join(config.ENSEMBLE_SCORE_PATH, "output.json"),
    #                               os.path.join(config.ENSEMBLE_SCORE_PATH, "top.json"),
    #                               top=4)

