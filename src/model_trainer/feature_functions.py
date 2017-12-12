#coding:utf8
import sys
sys.path.append("../..")
from src import util
from src.model_trainer import dict_util
from src.model_trainer.dict_loader import Dict_loader
from src.model_trainer.dict_creator import Dict_creator
from src import rf_viewer
from src.feature import Feature
from src import config
import json
import codecs
import nltk
import itertools
import pickle
from datetime import datetime
from src.util import singleton
import re

punc = {".", ",", "?", "!", "...", ";"}
normal_word = pickle.load(open(config.NORMAL_WORDS_PATH, "rb"), encoding="utf8", errors="ignore")
dict_emoticon = dict(((t.split("\t")[0], int(t.strip().split("\t")[1])) for t in open(config.EMOTICON, encoding="utf-8", errors="ignore")))


def hashtag(tweet):
    # load dict
    dict_hashtag = Dict_loader().dict_hashtag
    # feature
    hashtag = dict_util.get_hashtag(tweet)
    return util.get_feature_by_feat_list(dict_hashtag, hashtag)


'''
def unigram(tweet):
    # load dict
    dict_unigram = Dict_loader().dict_unigram
    # feature
    unigram = dict_util.get_unigram(tweet)
    return util.get_feature_by_feat_list(dict_unigram, unigram)
'''


def nltk_unigram(tweet):
    # load dict
    dict_nltk_unigram = Dict_loader().dict_nltk_unigram
    # feature
    nltk_uni = dict_util.get_nltk_unigram(tweet)
    return util.get_feature_by_feat_list(dict_nltk_unigram, nltk_uni)


def nltk_bigram(tweet):
    # load dict
    dict_nltk_bigram = Dict_loader().dict_nltk_bigram
    # feature
    nltk_bi = dict_util.get_nltk_bigram(tweet)
    return util.get_feature_by_feat_list(dict_nltk_bigram, nltk_bi)


def nltk_trigram(tweet):
    # load dict
    dict_nltk_bigram = Dict_loader().dict_nltk_trigram
    # feature
    nltk_tri = dict_util.get_nltk_trigram(tweet)
    return util.get_feature_by_feat_list(dict_nltk_bigram, nltk_tri)


def ners_existed(tweet):
    ners_list = dict_util.get_ners_exist(tweet)
    return util.get_feature_by_list(ners_list)

# all the rf values
rfdata_nltk_unigram = rf_viewer.Rf_Viewer(None, config.RF_DATA_NLTK_UNIGRAM_PATH)
rfdata_hashtag = rf_viewer.Rf_Viewer(None, config.RF_DATA_HASHTAG_PATH)
rfdata_nltk_bigram = rf_viewer.Rf_Viewer(None, config.RF_DATA_NLTK_BIGRAM_PATH)


def nltk_unigram_with_rf(tweet):
    # load dict
    dict_nltk_unigram = Dict_loader().dict_nltk_unigram
    # feature
    unigram = dict_util.get_nltk_unigram(tweet)
    return util.get_feature_by_feature_list_with_rf(dict_nltk_unigram, unigram, rfdata_nltk_unigram)


def nltk_bigram_with_rf(tweet):
    # load dict
    dict_nltk_bigram = Dict_loader().dict_nltk_bigram
    # feature
    bigram = dict_util.get_nltk_bigram(tweet)
    return util.get_feature_by_feature_list_with_rf(dict_nltk_bigram, bigram, rfdata_nltk_bigram)


def hashtag_with_rf(tweet):
    # load dict
    dict_hashtag = Dict_loader().dict_hashtag
    # feature
    hashtag = dict_util.get_hashtag(tweet)
    return util.get_feature_by_feature_list_with_rf(dict_hashtag, hashtag, rfdata_hashtag)


def bigram(tweet):
    # load dict
    dict_bigram = Dict_loader().dict_bigram
    # feature
    bigram = dict_util.get_bigram(tweet)
    # 得到unigram Feature 对象
    return util.get_feature_by_feat_list(dict_bigram, bigram)

def wv_google(tweet):
    google_vec = Dict_loader().google_vec
    # 返回的是900 300维sum 300维max 300维min
    word2vec = dict_util.get_w2v(tweet, google_vec)
    # print("!!!!!")
    # print(len(word2vec))
    return util.get_feature_by_list(word2vec)

def wv_GloVe(tweet):
    GloVe_vec = Dict_loader().dict_glove_vec

    word2vec = dict_util.get_w2v(tweet, GloVe_vec)

    return util.get_feature_by_list(word2vec)

#将否定词后的4个词加上_NEG后缀
def reverse_neg(tweet):
    set_neg = set([t.strip() for t in open(config.NEGATION_PATH)])
    mtoken = []
    tokens = list(itertools.chain(*tweet["tokens"]))
    sentence = " ".join(tokens)
    length = len(tokens)

    index = 0
    while(index != length):
        cur_token = tokens[index].lower()
        mtoken.append(cur_token)
        if cur_token in set_neg or cur_token.endswith("n't"):
            for i in range(index + 1, min(length, index + 4)):  # 将否定词后的4个词带上"_NEG"
                index = i
                cur_token_1 = tokens[i].lower()
                if tokens[i] in punc:  # 若遇到标点符号则停止加"_NEG"
                    mtoken.append(cur_token_1)
                    break
                mtoken.append(cur_token_1 + "_NEG")
        index += 1

    return mtoken


def sentilexi(tweet):
    '''SentimentLexicon'''

    feature = []
    #dict的value值都是1维score（若字典中本来有pos_score和neg_score，则pos_score-neg_score）
    Lexicon_dict_list = [
        Dict_loader().dict_BL,
        Dict_loader().dict_GI,
        Dict_loader().dict_IMDB,
        Dict_loader().dict_MPQA,
        Dict_loader().dict_NRCE,
        Dict_loader().dict_AF,
        Dict_loader().dict_NRC140_U,
        Dict_loader().dict_NRCH_U
    ]

    # tokens = list(itertools.chain(*
    # 将否定词后的4个词加上_NEG后缀
    tokens = reverse_neg(tweet)

    for Lexicon in Lexicon_dict_list :
        score = []
        for word in tokens:
            flag = -0.8 if word.endswith("_NEG") else 1
            word = word.replace("_NEG", "")
            if word in Lexicon:
                score.append(Lexicon[word] * flag)

        if len(score) == 0:
            feature += [0] * 11
            continue

        countPos, countNeg, countNeu = 0, 0, 0
        length = len(score) * 1.0
        for s in score:
            if s > 0.49:
                countPos += 1
            elif s < -0.49:
                countNeg += 1
            else:
                countNeu += 1

        feature += [countPos, countNeg, countNeu, countPos / length, countNeg / length, countNeu / length, max(score),
                    min(score)]

        finalscore = sum(score)
        # feature.append(finalscore)
        if finalscore > 0:
            feature += [1, 0]
        elif finalscore < 0:
            feature += [0, 1]
        else:
            feature += [0, 0]

        # pos_score = [t for t in score if t > 0]
        # neg_score = [t for t in score if t < 0]
        # feature.append(sum(pos_score))
        # feature.append(sum(neg_score))

        # if pos_score:
        #     feature.append(pos_score[-1])
        # else:
        #     feature.append(0)
        # if neg_score:
        #     feature.append(neg_score[-1])
        # else:
        #     feature.append(0)

        word = tokens[-1]
        flag = -0.8 if word.endswith("_NEG") else 1
        word = word.replace("_NEG", "")
        if word in Lexicon:
            feature.append(Lexicon[word] * flag)
        else:
            feature.append(0)

    #Bigram Lexicons
    for Lexicon in [Dict_loader().dict_NRC140_B, Dict_loader().dict_NRCH_B]:
        score = []

        bigram = list(nltk.ngrams(tokens, 2))
        for index, bi in enumerate(bigram):
            flag = -0.8 if bi[0].endswith("_NEG") and bi[1].endswith("_NEG") else 1
            bi = (bi[0].replace("_NEG", ""), bi[1].replace("_NEG", ""))
            bigram[index] = bi
            if bi in Lexicon:
                score.append(Lexicon[bi] * flag)

        if not score:
            feature += [0] * 11
            continue

        countPos, countNeg, countNeu = 0, 0, 0
        length = len(score) * 1.0
        for s in score:
            if s > 0.49:
                countPos += 1
            elif s < -0.49:
                countNeg += 1
            else:
                countNeu += 1

        feature += [countPos, countNeg, countNeu, countPos / length, countNeg / length, countNeu / length, max(score),
                    min(score)]

        finalscore = sum(score)
        # feature.append(finalscore)
        if finalscore > 0:
            feature += [1, 0]
        elif finalscore < 0:
            feature += [0, 1]
        else:
            feature += [0, 0]

        pos_score = [t for t in score if t > 0]
        neg_score = [t for t in score if t < 0]
        # feature.append(sum(pos_score))
        # feature.append(sum(neg_score))
        # if pos_score:
        #     feature.append(pos_score[-1])
        # else:
        #     feature.append(0)
        # if neg_score:
        #     feature.append(neg_score[-1])
        # else:
        #     feature.append(0)

        bi = bigram[-1]
        flag = -0.8 if bi[0].endswith("_NEG") and bi[1].endswith("_NEG") else 1
        bi = (bi[0].replace("_NEG", ""), bi[1].replace("_NEG", ""))
        if bi in Lexicon:
            feature.append(Lexicon[bi] * flag)
        else:
            feature.append(0)

    return util.get_feature_by_list(feature)

#是否存在大写词，是否存在多个大写词
def allcaps(microblog):
    '''Allcaps'''
    feature = []
    has_allcaps = 0
    has_several_allcaps = 0
    if microblog["tokens"]:
        tokens_list = microblog["tokens"]
        for token_list in tokens_list:
            if has_several_allcaps == 1:
                break
            else:
                for word in token_list:
                    if word.isupper():
                        if has_allcaps == 0:
                            has_allcaps = 1
                        else:
                            has_several_allcaps = 1
                            break
    feature.append(has_allcaps)
    feature.append(has_several_allcaps)

    return util.get_feature_by_list(feature)

#是否存在加长词，是否存在多个加长词
def elongated(microblog):
    '''Elongated'''
    feature = []
    has_elongated = 0
    has_several_elongated = 0
    text = microblog["raw_tweet"]
    elongated_count = 0
    text = " " + text.strip() + " "
    while re.search(r" [\S]*(\w)\1{2,10}[\S]*[ .,?!\"]", text):
        elongated_count += 1
        comp = re.search(r" [\S]*(\w)\1{2,10}[\S]* ", text)
        elongated = comp.group().strip()
        elongated_char = comp.groups()[0]
        elongated_1 = re.sub(elongated_char + "{3,11}", elongated_char, elongated)
        elongated_2 = re.sub(elongated_char + "{3,11}", elongated_char * 2, elongated)
        if normal_word[elongated_1] >= normal_word[elongated_2]:
            text = re.sub(elongated_char + "{3,11}", elongated_char, text)
        else:
            text = re.sub(elongated_char + "{3,11}", elongated_char * 2, text)

    if elongated_count != 0:
        if elongated_count == 1:
            has_elongated = 1
        else:
            has_elongated = 1
            has_several_elongated = 1

    feature.append(has_elongated)
    feature.append(has_several_elongated)

    return util.get_feature_by_list(feature)

#是否包含！，是否包含多个！，是否包含？，是否包含多个？，是否包含？！或！？
#最后一个token中是否包含！，最后一个token中是否包含？
def punction(microblog):
    '''Punctuation'''
    feature = []
    has_exclamation = 0
    has_several_exclamation = 0
    has_question = 0
    has_several_question = 0
    has_exclamation_question = 0
    end_exclamation = 0
    end_question = 0

    # print microblog["parsed_text"]["tokens"]
    if microblog["tokens"]:
        tokens = []  #本句子的所有tokens
        token_lists = microblog["tokens"]
        for token_list in token_lists:
            for word in token_list:
                tokens.append(word)
        sentence = " ".join(tokens)
        # print tokens

        exclamation_list = re.findall("!", sentence)
        if len(exclamation_list) != 0: #无感叹号
            has_exclamation = 1
            if len(exclamation_list) > 2:
                has_several_exclamation = 1

        question_list = re.findall("\?", sentence)
        if len(question_list) != 0:
            has_question = 1
            if len(question_list) > 2:
                has_several_question = 1

        excla_ques_list = re.findall("!\?", sentence)
        ques_excla_list = re.findall("\?!", sentence)
        if not excla_ques_list and not ques_excla_list:
            has_exclamation_question = 1

        end_exclamation = 1 if "!" in tokens[-1] else 0
        end_question = 1 if "?" in tokens[-1] else 0

    feature = [has_exclamation, has_several_exclamation, has_question, has_several_question, has_exclamation_question]

    feature.append(end_exclamation)
    feature.append(end_question)

    return util.get_feature_by_list(feature)

#是否有pos表情，是否有多个pos表情，是否有neg表情，是否有多个neg表情，是否即有pos又有neg
def emoticon(microblog):
    '''Emoticon'''
    feature = []
    has_pos = 0
    has_neg = 0
    has_several_pos = 0
    has_several_neg = 0
    has_pos_neg = 0
    noisy = {
        "-lrb-": "(",
        "-LRB-": "(",
        "-rrb-": ")",
        "-RRB-": ")"
    }
    if microblog["tokens"]:
        tokens = []  #本句子的所有tokens
        token_lists = microblog["tokens"]
        for token_list in token_lists:
            for word in token_list:
                for item in noisy:   #分词后结果把）和（换为了rrb,lrb
                    if item in word:
                        word = word.replace(item, noisy[item])
                tokens.append(word)
        for token in tokens:
            if token in dict_emoticon:
                score = dict_emoticon[token]
                if score == 1 :
                    if has_pos == 0:
                        has_pos = 1
                    else:
                        has_several_pos = 1
                    if has_neg == 1:
                        has_pos_neg = 1
                if score == -1:
                    if has_neg == 0:
                        has_neg = 1
                    else:
                        has_several_neg = 1
                    if has_pos == 1:
                        has_pos_neg = 1
    feature = [has_pos, has_several_pos, has_neg, has_several_neg, has_pos_neg]

    return util.get_feature_by_list(feature)

