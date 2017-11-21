#coding:utf8
import sys
sys.path.append("../..")
import numpy

def get_unigram(tw):
    unigram = tw["tokens"]

    return unigram

def get_stem_unigram(tw):
    unigram = tw["stems_n"]
    return unigram


def get_bigram(tw):
    bigram = []
    tokens = tw["tokens"]
    n = len(tokens)
    i = 1
    while i < n:
        bigram.append("%s|%s" % (tokens[i-1], tokens[i]))
        i += 1
    return bigram

def get_trigram(tw):
    trigram = []
    tokens = tw["tokens"]
    n = len(tokens)
    i = 2
    while i < n:
        trigram.append("%s|%s|%s" % (tokens[i-2], tokens[i-1], tokens[1]))
        i += 1
    return trigram


def get_w2v(tweet, vector):
    vec = []
    for token in tweet["tokens"]:
        if token in vector.word2vec:
            vec.append(vector.word2vec[token])
    # vec 是矩阵
    vec = numpy.array(vec)
    if len(vec) == 0: return [0] * 3 * vector.size
    # sum(vec[:,i])表示固定第一列
    feature = [min(vec[:, i]) for i in range(len(vec[0]))] + \
              [max(vec[:, i]) for i in range(len(vec[0]))] + \
              [sum(vec[:, i])/len(vec) for i in range(len(vec[0]))]

    return feature