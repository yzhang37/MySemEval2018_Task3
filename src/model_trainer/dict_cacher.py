# encoding: utf-8
import sys
import os
import json
sys.path.append("../..")


class DictCache(object):
    def __init__(self, dict_path, dict_handle_function):
        self.dict_path = dict_path
        self.auto_recalculate = False
        self.dict_handle_function = dict_handle_function

    def invalidate(self):
        if os.path.exists(self.dict_path):
            os.remove(self.dict_path)
        d = self.dict_handle_function()
        json.dump(d, open(self.dict_path, "w"))
        return d

    def load_dict(self):
        if not os.path.exists(self.dict_path) or self.auto_recalculate:
            print()
            if not os.path.exists(self.dict_path):
                print("Loading cached dict_vec from %s failed." % (self.dict_path))
            else:
                print("Auto recalculating on each train.")
            print("Recalculating...")
            return self.invalidate()
        else:
            try:
                data = json.load(open(self.dict_path, "r"))
                return data
            except:
                print()
                print("Loading cached dict_vec from %s failed." % (self.dict_path))
                print("Recalculating...")
                return self.invalidate()
