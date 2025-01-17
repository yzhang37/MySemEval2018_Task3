#coding:utf-8
import sys
import json
import re
import pickle
import copy
sys.path.append("..")
from src import config
from src.stanfordCoreNLP import StanfordCoreNLP
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk import PorterStemmer,pos_tag

# URLs
rc_url = re.compile(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+')
rc_twitterUrl = re.compile(r"http[s]?://t\.co/[0-9A-Za-z]{10}")

rc_super_url = re.compile(r"^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)([^#]*)?(#(.*))?")

# @-replies or mentions
rc_at = re.compile(r'(@|@ )([a-zA-Z]|[0-9]|_)+')

# elongated words compiled regex
rc_elongated = re.compile(r'\b\S*(\w)\1{2,10}\S*\b')

# elongated spaces

rc_space = re.compile(r" +([\s])")


def load_data(fp, is_test=False, debug=False):
    """
        load raw data
    :param fp: file_path
    :param is_test: if False, then try to fetch label data in the file. If True, then label would be skipped.
    :param debug: Print debug information if True
    :return: tweet_list:[{'id':'1', 'label':'0', 'text':'...'},{...},...]
    """
    tweet_list = []

    with open(fp, "r", encoding="utf8") as data_in:
        for line in data_in:
            tweet_dict = {}
            if not line.lower().startswith("tweet index"): #discard first line
                line = line.strip()
                if is_test:
                    id, raw_tweet = line.split("\t")
                    tweet_dict["id"], tweet_dict["raw_tweet"] = id, raw_tweet

                    if debug:
                        print("%s %s" % (id, raw_tweet))
                else:
                    id, label, raw_tweet = line.split("\t")
                    tweet_dict["id"], tweet_dict["label"], tweet_dict["raw_tweet"] = id, label, raw_tweet

                    if debug:
                        print("%s %s %s" % (id, label, raw_tweet))
                tweet_list.append(tweet_dict)

    return tweet_list


def replace_slang(slangs, text):

    set_ = {"'", ",", ".", " ", "!", "?"}
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


def elongated_words(normal_word, text):
    """
    Elongated words are words which has characters repeated for 3-11 times
    :param normal_word:
    :param text:
    :return:
    """
    text = " " + text.strip() + " "
    while rc_elongated.search(text):
        # elongated_count += 1
        comp = rc_elongated.search(text)
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

    url_regex = r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'
    at_regex = r'(@|@ )([a-zA-Z]|[0-9]|_)+'
    emoji_regex = r':[a-z|_|-]+:'               # description of emoji
    emoji_list = []

    text = rc_url.sub("http://someurl", text)
    text = rc_at.sub("@someuser", text)

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


def twitterUrls(text):
    return rc_twitterUrl.findall(text)


def preprocess_data(tweet_list, dump_path, is_test=False):
    server_url = 'http://precision:9000'
    nlp_server = StanfordCoreNLP(server_url)

    debug = False

    # save a contrast file for raw_tw and clean_tw
    if debug:
        contrast_file_in = open(config.DATA_PATH + "/contrast_file.txt", "w")
    nltk_tweet_tokenizer = TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles=False)

    for idx, tw_dict in enumerate(tweet_list):
        if idx % 100 == 0:
            print(idx)
        clean_tweet, emoji_list = normalise_tweet(tw_dict["raw_tweet"])
        tw_dict["clean_tweet"], tw_dict["emojis"] = clean_tweet.strip(), emoji_list
        tw_dict["twitter_url"] = twitterUrls(tw_dict["raw_tweet"])

        if debug:
            contrast_file_in.write(tw_dict["raw_tweet"] + "\n")
            contrast_file_in.write(tw_dict["clean_tweet"] + "\n")

        parse_tweet(nlp_server, tw_dict)
        # if idx==100: break
        # print("over!!")
        # break
        tw_dict["nltk_tokens"] = nltk_tweet_tokenizer.tokenize(tw_dict["raw_tweet"])

    json.dump(tweet_list, open(dump_path, "w"), indent=2)
    if debug:
        contrast_file_in.close()


def get_domain_keyword(url):
    ret = re.findall("\w+", url)

    for kw in [
        "www",
        "com",
        "net",
        "top",
        "tech",
        "org",
        "gov",
        "edu",
        "pub",
        "name",
        "me",
        "info",
        "uk",
        "co",
        "jp",
        "cn",
        "xyz"
    ]:
        if kw in ret:
            ret.remove(kw)

    for word in copy.copy(ret):
        if len(word) <= 3:
            ret.remove(word)
    return ret


def get_url_domain_keywords(url):
    domain_url = rc_super_url.findall(url)[0][3]
    domain_keywords = get_domain_keyword(domain_url)
    return domain_keywords


def url_preprocess_data(raw_data):
    import wordsegment as wordseg
    wordseg.load()
    debug = False

    if debug:
        contrast_file_in = open(config.DATA_PATH + "/train/contrast_file_url.txt", "w")

    nltk_tweet_tokenizer = TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles=False)

    dump_data = dict()

    idx = 0
    for raw_t_co_url, current_url_data in raw_data.items():
        idx += 1
        if idx % 100 == 0:
            print(idx)
        # clear data
        if "is_media" in current_url_data:
            # 照片的 title 都是 adaptive media, 没有意义。
            current_url_data["title"] = ""
            del current_url_data["is_media"]

        url = rc_super_url.findall(current_url_data["current_url"])[0][3]
        print(url)
        if url == "www.instagram.com":
            print("OK")
            current_url_data["title"] = ""

        # 复制句子内容
        sentence_list = []
        for key, value in current_url_data.items():
            if key == "current_url":
                continue
            else:
                if isinstance(value, str):
                    sentence_list.append(value)
                elif isinstance(value, (list, set)):
                    for __item in value:
                        if isinstance(__item, str):
                            sentence_list.append(__item)

        # 域名停词。
        domain_stopwords = get_url_domain_keywords(current_url_data["current_url"])
        for word in domain_stopwords:
            rc = re.compile("(?i)(?<!\w)%s(?!\w)" % word)
            for idx in range(len(sentence_list)):
                sentence_list[idx] = rc.sub("", sentence_list[idx])

        rc = re.compile("#(\w+)")

        dump_data[raw_t_co_url] = []
        for idx in range(len(sentence_list)):
            __cur_sent_dict = dict()

            original = rc_space.sub("\g<1>", sentence_list[idx].strip())
            if len(original) == 0:
                continue
            __cur_sent_dict["original_content"] = original
            sentence, emojis = normalise_tweet(original)
            __cur_sent_dict["emojis"] = emojis

            # 查找所有的 hashtag，然后分割
            __cur_all_tags = rc.findall(sentence)

            # pure_sentence: 不包含 hashtag 的句子
            pure_sentence = sentence
            for hashtag in __cur_all_tags:
                pure_sentence = pure_sentence.replace(("#" + hashtag), "")
            __cur_sent_dict["pure_sentence"] = rc_space.sub("\g<1>", pure_sentence.strip())

            # hashtag: 所有的 hashtag
            __cur_split_hashtag_words = []
            for hashtag in __cur_all_tags:
                split_words = wordseg.segment(hashtag)
                __cur_split_hashtag_words += split_words
                sentence = sentence.replace(("#" + hashtag), " ".join(split_words))
            __cur_sent_dict["hashtags"] = ["#" + tag for tag in __cur_all_tags]
            __cur_sent_dict["split_hashtag_words"] = __cur_split_hashtag_words
            __cur_sent_dict["sentence_with_hashtag"] = sentence.strip()

            __cur_sent_dict["token_pure_sentence"] = nltk_tweet_tokenizer.tokenize(__cur_sent_dict["pure_sentence"])
            __cur_sent_dict["token_sentence_with_hashtag"] = nltk_tweet_tokenizer.tokenize(__cur_sent_dict["sentence_with_hashtag"])

            dump_data[raw_t_co_url].append(__cur_sent_dict)

        json.dump(dump_data, open(config.PROCESSED_URL_DATA, "w"), indent=4)

    if debug:
        contrast_file_in.close()


if __name__ == '__main__':
    handle = "tweet-test"

    if handle == "tweet":
        fp = config.RAW_TRAIN
        tweet_list = load_data(fp)
        preprocess_data(tweet_list, config.PROCESSED_TRAIN)

    elif handle == "tweet-test":
        fp = config.RAW_TEST
        tweet_list = load_data(fp, is_test=True)
        preprocess_data(tweet_list, config.PROCESSED_TEST, is_test=True)

    elif handle == "url":
        fp = config.URL_CACHE_PATH
        raw_url_data = json.load(open(fp))
        url_preprocess_data(raw_url_data)
        pass

#
# if __name__ == "__main__":
#     data = json.load(open(config.PROCESSED_TRAIN_B, "r"))
#     nltk_tweet_tokenizer = TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles=False)
#     for tw in data:
#         tw["nltk_tokens"] = nltk_tweet_tokenizer.tokenize(tw["clean_tweet"])
#     json.dump(data, open(config.PROCESSED_TRAIN_B, "w"), indent=2)
#

