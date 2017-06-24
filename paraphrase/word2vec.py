from pyword2vec import MapReader
from .ngram import get_words
from .names import named
import numpy as np
from math import sqrt

WORD2VEC_FILEMAP = "./data/filemap.w2v"

print('loading word2vec file')
_word2vec = MapReader(open(WORD2VEC_FILEMAP, "rb"))

def _norm(v):
    return sqrt(np.dot(v, v))

def _get_words_vec(sentence):
    return sum(_word2vec[word] for word in get_words(sentence)\
        if word in _word2vec)

@named("word2vec_dot")
def word2vec_features(data_row):
    u, v = map(_get_words_vec, (data_row.sent_1, data_row.sent_2))
    c, d = map(_norm, (u, v))
    if c == 0 or d == 0:
    	return None
    else:
    	return [np.dot(u, v) / c / d]
