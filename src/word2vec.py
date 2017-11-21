import numpy

class Word2Vec():
    def __init__(self, fname):
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