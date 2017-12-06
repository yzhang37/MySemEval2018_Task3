#coding:utf8
import sys
import numpy
import re
sys.path.append("../..")
from src import util


'''
unigram 降维操作 (.md)
已经完成的修建工作：

-[x] !!!, !!!! -> !! …. -> .. ??? -> ??
-[x] ", "@someuser -> @someuser, "Big -> Big (split)
-[x] remove \# (tags)
-[x] \$, \$1.5, \$BIDU?  
-[ ] &\#8211; 这个什么东西？
-[x] 'The, 's -> s (split)
-[x] (Level, (in, (the, (via -> (split)
-[x] .@someuser -> @someuser (split)
-[x] 数字, 数字. 数字! 数字: -> (split)
-[x] 数字单词 ->L (split)
-[ ] 数字是否要处理掉？
-[x] @someuser!, @someuser', @someuser? （结尾的) split
-[x] 单词末尾有标点 (D: 一个字母加单词例外，是表情)
-[ ] 所有单词统一小写化处理，除了缩写单词 
-[x] |#…, |单词 -> (split)
-[x] 单词*标点符号+\| -> (split)

'''

def get_unigram(tw):
    unigram = tw["tokens"]

    # splitting...

    split_reg_list = [
        ("^([.(|)\"\'\$])([@#]?\w+.*)$", ["\g<1>", "\g<2>"]),                   # 分割前导多余的单个标点符号
        ("^(\d+)([.,:;!?\'\"\(\)<>]+)$", ["\g<1>", "\g<2>"]),                   # 数字, 数字. 数字! 数字: -> (split)
        ("^(\d+)([A-Za-z]+)$", ["\g<1>", "\g<2>"]),                             # 数字单词(split)
        ("^(\w+.*)\|(\w+.*)$", ["\g<1>", "|", "\g<2>"]),                        # 两个字符串被|分割的情况
        ("^([@#]?[A-Za-z]{2,})([.,:;!?\'\"\(\)<>]+)$", ["\g<1>", "\g<2>"]),     # 单词末尾有标点(D: 一个字母加单词例外，是表情)
        ("^(.+)(\")$", ["\g<1>", "\g<2>"]),                                     # 右边多了"
        ("^(\")(.+)$", ["\g<1>", "\g<2>"]),                                     # 左边多了"
    ]
    util.split_reg_tokens(unigram, split_reg_list)

    # pruning...

    prune_reg_list = [
        "^#.*$"                                                                 # 删除所有的tag
    ]
    util.delete_reg_tokens(unigram, prune_reg_list)

    # rewriting...
    rewrite_reg_list = [
        (r"^([!.?])\1{2,}$", "\g<1>\g<1>"),                                     # !!!, !!!! -> !! …. -> .. ??? -> ??
        # (r"^(.+)[\x00-\x08\x0E-\x1B]+$", "\g<1>")
    ]
    util.rewrite_reg_tokens(unigram, rewrite_reg_list)
    return unigram


def get_nltk_unigram(tw):
    nltk_unigram = tw["nltk_tokens"]

    # splitting...
    split_reg_list = [
        (r"^(.+) +(.+)$", ["\g<1>", "\g<2>"]),
    ]
    util.split_reg_tokens(nltk_unigram, split_reg_list)

    # pruning...
    prune_reg_list = [
        "^#.*$",
    ]
    util.delete_reg_tokens(nltk_unigram, prune_reg_list)

    # rewriting...
    rewrite_reg_list = [
        #("^[+-]?\d+.*$", "<SomeNumber>"),
    ]
    util.rewrite_reg_tokens(nltk_unigram, rewrite_reg_list)

    return nltk_unigram


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


def get_hashtag(tw):
    all_tokens = tw["tokens"]
    # filtering all the hashtag like '#somewords'
    rc = re.compile("#(\w+)")

    hashtag_list = []
    for token in all_tokens:
        hashtag_list.extend(rc.findall(token))
    return hashtag_list


def get_hashtag_unigram(tw):
    hashtag_list = get_hashtag(tw)
    hashtag_unigram_list = []

    for hashtag in hashtag_list:
        hashtag_unigram_list.append(handle(hashtag))

    # todo: how to split the word in hashtag?

    return hashtag_unigram_list


def get_w2v(tweet, vector):
    vec = []
    for token in tweet["nltk_tokens"]:
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


def get_ners_exist(tweet):
    ners_tag = ["DURATION", "SET", "NUMBER", "LOCATION", "PERSON", "ORGANIZATION", "PERCENT", "MISC", "ORDINAL", "TIME",
                "DATE", "MONEY"]
    ners_c_set = set(tweet["ners"])
    feature = [1 if tag in ners_c_set else 0 for tag in ners_tag]
    return feature