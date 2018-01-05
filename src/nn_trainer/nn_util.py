# encoding: utf-8
import sys
import numpy as np
import os
import tensorflow as tf
sys.path.append("../..")
from src import config
from src.word2vec import GloVe
from src.model_trainer.dict_loader import Dict_loader


def load_word2vec_for_vocab(dict_word_to_index, from_origin=True):

    # print "\n".join(dict_word_to_index.keys())
    train_dir = config.DATA_PATH
    embedding_file = os.path.join(train_dir, "vocab.google_w2v_fortask")
    # print(embedding_file)
    # print()
    # print()
    if from_origin:
        _load_vocab_vec(config.GLOVE_840B_300_PATH, dict_word_to_index, embedding_file)
        # _load_vocab_vec(config.BLLIP_WORD2VEC_PATH, dict_word_to_index, embedding_file)
        # _load_vec_from_corpus(config.DATA_PATH + "/cqa/qatar_corpus_100.wordvec", dict_word_to_index, embedding_file, 100)
        # _load_vec_from_corpus(config.ZH_WORD2VEC_PATH, dict_word_to_index, embedding_file, 300)

    # load embedding matrix
    return _load_wordvec(embedding_file)


# dict_vocab: token -> index
def _load_vocab_vec(fname, dict_word_to_index, to_file):
    """
    Loads word vecs from Google (Mikolov) word2vec
    """
    # dict_word_to_vector, vocab_size, layer1_size = GloVe(fname, smalldict=Dict_loader().dict_nltk_unigram)
    dict_word_to_vector = Dict_loader().dict_glove_vec.word2vec
    vocab_size = Dict_loader().dict_glove_vec.length
    layer1_size = Dict_loader().dict_glove_vec.size
    # dict_word_to_vector = {}
    # with open(fname, "rb") as f:
    #     # header = f.readline()
    #     # vocab_size, layer1_size = map(int, header.split())
    #     print ("==> word embedding size", layer1_size)
    #     binary_len = np.dtype('float32').itemsize * layer1_size
    #     for line in range(vocab_size):
    #         word = []
    #         while True:
    #             ch = f.read(1)
    #             if ch == ' ':
    #                 word = ''.join(word)
    #                 break
    #             if ch != '\n':
    #                 word.append(ch)
    #
    #         if word in dict_word_to_index:
    #             dict_word_to_vector[word] = np.fromstring(f.read(binary_len), dtype='float32')
    #         else:
    #             f.read(binary_len)

    dict_index_to_word = {}
    vocab_embeddings = [np.array([0] * layer1_size)] * len(dict_word_to_index)
    print("The number of word in vec: %d" % len(dict_word_to_vector))
    for word in dict_word_to_index:
        index = dict_word_to_index[word]
        dict_index_to_word[index] = word
        if index == 0: # unk or padding --> 0
            continue
        if word in dict_word_to_vector:
            vocab_embeddings[index] = dict_word_to_vector[word]
        else:
            vocab_embeddings[index] = np.random.uniform(-0.25, 0.25, layer1_size)

    # with open(to_file, "w") as fw:
    #     for i in range(len(dict_word_to_index)):
    #         fw.write(vocab_words[i] + " " + " ".join(map(str, vocab_embeddings[i])) + "\n")

    with open(to_file, "w") as fw:
        for i in range(len(dict_word_to_index)):
            fw.write(dict_index_to_word[i] + " " + " ".join(list(map(str, vocab_embeddings[i]))) + "\n")


def _load_wordvec(filename):
    vocab_embeddings = []
    with open(filename) as fr:
        for line in fr:
            if len(list(map(float, line.strip().split(" ")[1:]))) != 300:
                print("===>", line)
            vocab_embeddings.append(list(map(float, line.strip().split(" ")[1:])))
    return np.array(vocab_embeddings)


def get_data_length(data):
    used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    length_one = tf.ones(tf.shape(length), dtype=tf.int32)
    length = tf.maximum(length, length_one)
    return length


def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = int(output.get_shape()[1])
    output_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, output_size])
    relevant = tf.gather(flat, index)
    return relevant