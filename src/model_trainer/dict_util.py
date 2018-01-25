#coding:utf8
import sys
import numpy
import re
import wordsegment as wordseg
sys.path.append("../..")
from src import util

# initialize the wordseg
wordseg.load()


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
'''

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


def get_nltk_bigram(tw):
    nltk_unigram = get_nltk_unigram(tw)
    nltk_bigram = []
    n = len(nltk_unigram)
    i = 1
    while i < n:
        nltk_bigram.append("%s|%s" % (nltk_unigram[i - 1], nltk_unigram[i]))
        i += 1
    return nltk_bigram


def get_nltk_trigram(tw):
    nltk_unigram = get_nltk_unigram(tw)
    nltk_trigram = []
    n = len(nltk_unigram)
    i = 2
    while i < n:
        nltk_trigram.append("%s|%s|%s" % (nltk_unigram[i-2], nltk_unigram[i - 1], nltk_unigram[i]))
        i += 1
    return nltk_trigram


def get_stem_unigram(tw):
    unigram = tw["stems_n"]
    return unigram


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
        hashtag_unigram_list.extend(wordseg.segment(hashtag))

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


def get_url_unigram(tweet, url_data_cache):
    related_urls = tweet["twitter_url"]

    url_unigram = []

    for t_co_url in related_urls:
        if t_co_url in url_data_cache:
            url_data_list = url_data_cache[t_co_url]

            for current_url_data in url_data_list:
                current_tokens = current_url_data["token_pure_sentence"]

                # splitting...
                split_reg_list = [
                    (r"^(.+) +(.+)$", ["\g<1>", "\g<2>"]),
                    (r"(-?)(\d+)([-,/])(\d+)(-?)", ["\g<1>", "\g<2>", "\g<3>", "\g<4>", "\g<5>"]),
                    (r"(-)(\d+[.,:]\d+)(-)", ["\g<1>", "\g<2>", "\g<3>"]),
                    (r"(-)(\d+[.,:]\d+)", ["\g<1>", "\g<2>"]),
                    (r"(\d+[.,:]\d+)(-)", ["\g<1>", "\g<2>"]),
                    (r"^(\d+)([A-Za-z]+)$", ["\g<1>", "\g<2>"])
                ]
                util.split_reg_tokens(current_tokens, split_reg_list)

                # pruning...
                prune_reg_list = [
                    "^#.*$",
                ]
                util.delete_reg_tokens(current_tokens, prune_reg_list)

                # rewriting...
                rewrite_reg_list = [
                    (r"\d+:\d+", "<TIME>"),
                    (r"(.)( )(?:\1\2)+(?:\1?)", ".."),
                    ("^[+-]?\d+.*$", "<NUMBER>"),
                ]
                util.rewrite_reg_tokens(current_tokens, rewrite_reg_list)

                util.clear_empty_tokens(current_tokens)

                url_unigram += current_tokens

    return url_unigram


def get_url_hashtag(tweet, url_data_cache):
    related_urls = tweet["twitter_url"]

    url_hashtag = []

    for t_co_url in related_urls:
        if t_co_url in url_data_cache:
            url_data_list = url_data_cache[t_co_url]

            for current_url_data in url_data_list:
                current_hashtags = current_url_data["hashtags"]
                url_hashtag += current_hashtags
    return url_hashtag