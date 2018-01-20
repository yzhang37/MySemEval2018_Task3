# encoding: utf-8
import xlrd
import sys
import os
import json
sys.path.append("../..")
from src import config


def get_excel_data(path, out_path):
    book = xlrd.open_workbook(path)
    output_dict = dict()
    for table in book.sheets():
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

    json.dump(output_dict, open(out_path, "w"))
    print("excel data converted to:")
    print(out_path)


if __name__ == "__main__":
    get_excel_data(os.path.join(config.CWD, "ali.xlsx"),
                   os.path.join(config.ENSEMBLE_PATH, "nn_ali_a.json") )