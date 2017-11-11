#coding:utf-8
import sys
sys.path.append("..")

# 计算字典中词的频率
def set_dict_key_value(dict, key):
    if key not in dict:
        dict[key] = 0
    dict[key] += 1

#在字典中删除 value < threshold 的item
def removeItemsInDict(dict, threshold=1):
    if threshold > 1:
        for key in list(dict.keys()):
            if dict[key] < threshold:
                dict.pop(key)

def write_dict_keys_to_file(dict, file_path):
    with open(file_path, "w", encoding="utf-8") as file_out:
        file_out.write("\n".join([str(key) for key in sorted(dict.keys())]))