# encoding: utf-8
import xlrd
import sys
import os
import json
sys.path.append("../..")
from src import config


def get_excel_data(path, train_out_path, test_out_path=""):
    book = xlrd.open_workbook(path)
    train_output_dict = dict()
    test_output_dict = dict()
    for idx, table in enumerate(book.sheets()):
        if idx < 2:
            output_dict = train_output_dict
        else:
            output_dict = test_output_dict
        for idx in range(1, table.nrows):
            line_data = table.row_values(idx)

            clsid = str(int(line_data[0]))

            if config.get_class() == "A":
                prob_0 = line_data[1]
                prob_1 = line_data[2]

                if prob_0 > prob_1:
                    output_dict[clsid] = {"0": 1}
                else:
                    output_dict[clsid] = {"1": 1}
            else:
                raise NotImplementedError("分类器尚未实现。")

    json.dump(train_output_dict, open(train_out_path, "w"))
    print("Train excel data converted to:")
    print(train_out_path)

    if len(test_out_path) > 0:
        json.dump(test_output_dict, open(test_out_path, "w"))
        print("Test excel data converted to:")
        print(test_out_path)


def get_valid_and_test_data(path, train_out_path, test_out_path=""):
    book = xlrd.open_workbook(path)
    train_output_dict = dict()
    test_output_dict = dict()
    for idx, table in enumerate(book.sheets()):
        sheet_name = table.name.lower().strip()
        if sheet_name == "valid":
            output_dict = train_output_dict
        elif sheet_name == "test":
            output_dict = test_output_dict
        else:
            continue
        for idx in range(1, table.nrows):
            line_data = table.row_values(idx)

            clsid = str(int(line_data[0]))

            if config.get_class() == "A":
                prob_0 = line_data[1]
                prob_1 = line_data[2]

                if prob_0 > prob_1:
                    output_dict[clsid] = {"0": 1}
                else:
                    output_dict[clsid] = {"1": 1}
            else:
                raise NotImplementedError("分类器尚未实现。")

    json.dump(train_output_dict, open(train_out_path, "w"))
    print("Train excel data converted to:")
    print(train_out_path)

    if len(test_out_path) > 0:
        json.dump(test_output_dict, open(test_out_path, "w"))
        print("Test excel data converted to:")
        print(test_out_path)



def convert(excel_name, task="0"):
    if task == "0":
        excel_path = os.path.join(config.RESULT_EXCEL_PATH, "%s.xlsx" % excel_name)
        train_path = os.path.join(config.ENSEMBLE_PATH, "nn_ali_%s_%s.json" % (excel_name, config.get_class().lower()))
        test_path = os.path.join(config.ENSEMBLE_PATH,
                                 "nn_ali_test_%s_%s.json"% (excel_name, config.get_class().lower()))
        get_excel_data(excel_path, train_path, test_path)
    else:
        excel_path = os.path.join(config.RESULT_EXCEL_PATH, "%s.xlsx" % excel_name)
        valid_path = os.path.join(config.ENSEMBLE_PATH, "nn_ali_valid_%s_%s.json" % (excel_name, config.get_class().lower()))
        test_path = os.path.join(config.ENSEMBLE_PATH,
                                 "nn_ali_test_%s_%s.json" % (excel_name, config.get_class().lower()))
        get_valid_and_test_data(excel_path, valid_path, test_path)


if __name__ == "__main__":
    convert("ali_old", "1")
