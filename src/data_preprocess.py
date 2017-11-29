#coding:utf-8
import sys
sys.path.append("..")

import json
import re
import pickle
from src import config
from src.stanfordCoreNLP import StanfordCoreNLP
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk import PorterStemmer,pos_tag


def load_data(fp):
    '''
        load raw data
        :return:
            tweet_list:[{'id':'1', 'label':'0', 'text':'...'},{...},...]
    '''

    tweet_list = []

    with open(fp, "r", encoding="utf8") as data_in:
        for line in data_in:
            tweet_dict = {}
            if not line.lower().startswith("tweet index"): #discard first line
                line = line.strip()
                id, label, raw_tweet = line.split("\t")
                tweet_dict["id"], tweet_dict["label"], tweet_dict["raw_tweet"] = id, label, raw_tweet
                tweet_list.append(tweet_dict)
                # print("%s %s %s" % (id, label, raw_tweet))
    return tweet_list

def replace_slang(slangs, text):

    set_ = set(["'", ",", ".", " ", "!", "?"])
    text = " " + text.strip() + " "
    text_lower = text.lower()
    # slang 替换
    for slang in slangs:
        slang = " " + slang
        index = text_lower.find(slang)
        if index != -1 and text[index + len(slang)] in set_:
            text = text[:index] + " " + slangs[slang.strip()] + text[index + len(slang):]
            text_lower = text.lower()
    return text.strip()

def elongated_words(normal_word, text): # Elongated words are words which has characters repeated for 3-11 times

    text = " " + text.strip() + " "
    while re.search(r" [\S]*(\w)\1{2,10}[\S]*[ .,?!\"]", text):
        # elongated_count += 1
        comp = re.search(r" [\S]*(\w)\1{2,10}[\S]* ", text)
        elongated = comp.group().strip()
        elongated_char = comp.groups()[0]
        elongated_1 = re.sub(elongated_char + "{3,11}", elongated_char, elongated)
        elongated_2 = re.sub(elongated_char + "{3,11}", elongated_char * 2, elongated)
        if normal_word[elongated_1] >= normal_word[elongated_2]:
            text = re.sub(elongated_char + "{3,11}", elongated_char, text)
        else:
            text = re.sub(elongated_char + "{3,11}", elongated_char * 2, text)
    return text

def changePosTag(pos):

    POStag = pos
    flag=0
    if POStag=="NN" or POStag=="NNS" or POStag=="NNP" or POStag=="NNPS":
        POStag="n"
        flag=1
    if POStag=="VB" or POStag=="VBD" or POStag=="VBG" or POStag=="VBN" or POStag=="VBP" or POStag=="VBZ":
        POStag="v"
        flag=1
    if POStag=="JJ" or POStag=="JJR" or POStag=="JJS":
        POStag="a"
        flag=1
    if POStag=="RB" or POStag=="RBR" or POStag=="RBS":
        POStag="r"
        flag=1

    if flag==0:
        POStag="\\"
    # sent[i]=(word,POStag)
    return POStag

def normalise_tweet(text):
    '''
        normalise hyperlinks and @_replies or mention to http://someurl and @someuser
        extract the description of emoji and the index of emojis
        replace the slangs of tweet
        normalise the elongated words of tweet
        :return:
            text: cleaned tweet
            emoji_list: [(":books:",(57,64)),(":open_book:", (64,75)),...]  a list of emojis of one tweet
    '''

    slangs = {}  # load dict of slangs
    for line in open(config.SLANGS_PATH):
        line = line.strip().split("  -   ")
        slangs[line[0]] = line[-1]

    normal_word = pickle.load(open(config.NORMAL_WORDS_PATH, "rb"), encoding="utf8", errors="ignore") # load normal_word dict

    url_regex = r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'          # URLs
    at_regex = r'(@|@ )([a-zA-Z]|[0-9]|_)+'     # @-replies or mentions
    emoji_regex = r':[a-z|_|-]+:'               # description of emoji
    emoji_list = []

    text = re.sub(url_regex, "http://someurl", text)
    text = re.sub(at_regex, "@someuser", text)

    for i in re.finditer(emoji_regex, text):    # extract the description of emoji and the index of emojis
        emoji_list.append((i.group(), i.span()))
        # print(i.group(), i.span())
    text = re.sub(emoji_regex, "", text)        # replace emoji to empty
    text = replace_slang(slangs, text)          # replace the slangs of tweet
    text = elongated_words(normal_word, text)   # normalise the elongated words of tweet

    # print(text, emoji_list)
    return text, emoji_list


def parse_tweet(nlp_server, tweet_dict):
    clean_text = tweet_dict["clean_tweet"]
    wnl = WordNetLemmatizer()
    portor = PorterStemmer()
    tokens, lemmas, stem, postag, ners, tw_parse, tw_b_dependencies, tw_c_dependencies = [],[],[],[],[],[],[],[]
    lemmas_nltk, stems_nltk, lemma_stems_nltk = [],[],[]

    parse_result = nlp_server.parse_one_line(clean_text)
    if parse_result is not None:
        for item in parse_result["sentences"]:
            for tk in item["tokens"]:
                tokens.append(tk["word"])
                lemmas.append(tk["lemma"])
                postag.append(tk["pos"])
                ners.append(tk["ner"])
            tw_parse.append(item["parse"])
            tw_b_dependencies.append(item["basic-dependencies"])
            tw_c_dependencies.append(item["collapsed-ccprocessed-dependencies"])
    tweet_dict["tokens"] = tokens
    tweet_dict["lemmas"] = lemmas
    tweet_dict["postag"] = postag
    tweet_dict["ners"] = ners
    tweet_dict["tw_parse"] = tw_parse
    tweet_dict["tw_b_dependencies"] = tw_b_dependencies
    tweet_dict["tw_c_dependencies"] = tw_c_dependencies

    # acquire the stem of word using nltk tool
    for index, token in enumerate(tokens):
        token = token.lower()
        if changePosTag(postag[index]) != "\\":
            lemma_nltk = wnl.lemmatize(token, changePosTag(postag[index]))
        else:
            lemma_nltk = wnl.lemmatize(token)

        lemmas_nltk.append(lemma_nltk)
        stems_nltk.append(portor.stem(token))
        lemma_stems_nltk.append(portor.stem(lemma_nltk))

    tweet_dict["stems_n"] = stems_nltk
    tweet_dict["lemmas_n"] = lemmas_nltk
    tweet_dict["lemma_stems_n"] = lemma_stems_nltk

    # return tweet_dict


def preprocess_data(tweet_list):
    server_url = 'http://precision:9000'
    nlp_server = StanfordCoreNLP(server_url)

    # save a contrast file for raw_tw and clean_tw
    contrast_file_in = open(config.DATA_PATH + "/train/contrast_file.txt", "w")
    nltk_tweet_tokenizer = TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles=False)

    for idx, tw_dict in enumerate(tweet_list):
        if idx % 100 == 0:
            print(idx)
        clean_tweet, emoji_list = normalise_tweet(tw_dict["raw_tweet"])
        tw_dict["clean_tweet"], tw_dict["emojis"] = clean_tweet.strip(), emoji_list

        contrast_file_in.write(tw_dict["raw_tweet"] + "\n")
        contrast_file_in.write(tw_dict["clean_tweet"] + "\n")

        parse_tweet(nlp_server, tw_dict)
        # if idx==100: break
        # print("over!!")
        # break
        tw_dict["nltk_tokens"] = nltk_tweet_tokenizer.tokenize(tw_dict["raw_tweet"])

    json.dump(tweet_list, open(config.PROCESSED_TRAIN_B, "w"), indent=2)
    contrast_file_in.close()


if __name__ == '__main__':
    fp = config.RAW_TRAIN_B
    tweet_list = load_data(fp)
    preprocess_data(tweet_list)




