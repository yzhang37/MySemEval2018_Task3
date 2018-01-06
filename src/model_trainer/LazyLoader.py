# encoding: utf-8


class LazyLoader(object):
    def __init__(self):
        self._dict_manager = {}
        self._map_name_to_handler = {}

    def get(self, dict_name, *args, **kwargs):
        """
        Get the instance of the dictionary.
        :param dict_name: The name of the dictionary.
        :return: Dictionary data.
        """
        if dict_name not in self._dict_manager:
            self._dict_manager[dict_name] = self._map_name_to_handler[dict_name]()
        return self._dict_manager[dict_name]

    def get_dict_list(self):
        """
        List all available dictionary alias.
        :return: A list of name (str).
        """
        return list(self._map_name_to_handler.keys())