import numpy
import os
import re


class Word2Vec():
    def __init__(self, fname=""):
        self.word2vec=None
        self.length=0
        self.size=0
        if len(fname) > 0:
            info = self.load_vec(fname)
            self.word2vec = info[0]
            self.length = info[1]
            self.size = info[2]

    def load_vec(self, fname):
        dict_ = {}
        fvector = open(fname)

        headlines = fvector.readline()
        length, size = map(int, headlines.strip().split(" "))

        for line in fvector:
            line = line.strip().split(" ")
            dict_[line[0]] = numpy.asarray(list(map(float, line[1: ])))

        return dict_, length, size


class Word2VecTag(Word2Vec):
    def __init__(self, word2vec: dict):
        Word2Vec.__init__(self, "")
        self.word2vec = word2vec
        self.length = len(word2vec)
        self.size = 0
        for key, value in word2vec.items():
            self.size = len(value)
            break


class GloVe(object):
    def __init__(self, fname, smalldict = None):
        self.word2vec, self.length, self.size = self._load_vec(fname, smalldict)

    def _load_vec(self, fname, smalldict = None):
        dict_ = {}

        print("Loading GloVe dict from %s..." % (os.path.split(fname)[1]))
        print("==" * 30)

        fVector = open(fname)

        rc = re.compile(r"[ ]+")
        # GloVe don't have headline

        word_count = 0
        size = 0

        if smalldict is None:
            for sLine in fVector:
                line = rc.split(sLine)
                dict_[line[0]] = list(map(float, line[1:]))
                size = max(size, len(line) - 1)
                word_count += 1
                if word_count % 100 == 0:
                    print(word_count)
        else:
            for sLine in fVector:
                line = rc.split(sLine)
                if line[0] in smalldict:
                    dict_[line[0]] = list(map(float, line[1:]))
                    size = max(size, len(line) - 1)
                    word_count += 1
                    if word_count % 100 == 0:
                        print(word_count)

        print("==" * 30)
        print("Completed in loading GloVe dict from %s." % (os.path.split(fname)[1]))
        print("Total data %d items." % (word_count))
        print()
        return dict_, word_count, size