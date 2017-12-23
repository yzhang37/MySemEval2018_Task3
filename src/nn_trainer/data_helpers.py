import numpy as np
import re
import json
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(training_data_file):
    """
    Loads data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    data_dict = {}
    tweets = json.load(open(training_data_file, "r"), encoding="utf=8")
    for tw in tweets:
        from src import config
        new_label = config.get_label_map(tw["label"])
        data_dict.setdefault(new_label, [])
        data_dict[new_label].append(" ".join(tw["nltk_tokens"]))

    # positive_examples = list(open(positive_data_file, "r").readlines())
    # positive_examples = [s.strip() for s in positive_examples]
    # negative_examples = list(open(negative_data_file, "r").readlines())
    # negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    label_list = config.get_label_list()
    x_text = []
    y = []
    for label in label_list:
        y_tmp = [0] * len(label_list)
        x_text += data_dict[str(label)]
        y_tmp[label] = 1
        y += [y_tmp] * len(data_dict[str(label)])

    x_text = [clean_str(sent) for sent in x_text]
    y = np.asarray(y)

    # x_text = positive_examples + negative_examples
    # x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    # positive_labels = [[0, 1] for _ in positive_examples]
    # negative_labels = [[1, 0] for _ in negative_examples]
    # y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]