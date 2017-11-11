#coding:utf8
import sys
sys.path.append("../..")

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