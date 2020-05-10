import numpy as np
from gensim.models import KeyedVectors


class GloveVectorizer:
    def __init__(self, path):
        self.word2vec = {}
        self.word2idx = {}
        self.embedding = []
        self.idx2word = []
        self.V = None
        self.D = None
        self._load_data(path)

    def _load_data(self, path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vec = np.array(values[1:], dtype='float32')
                self.word2vec[word] = vec
                self.embedding.append(vec)
                self.idx2word.append(word)
        self.embedding = np.array(self.embedding)
        self.V, self.D = self.embedding.shape
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}

    def transform(self, data):
        X = np.zeros((len(data), self.D))
        emptycount = 0
        try:
            for i, sentence in enumerate(data):
                feature_vector = [self.word2vec[word] for word in sentence.lower().split() if word in self.word2vec]
                if len(feature_vector) > 0:
                    X[i] = np.array(feature_vector).mean(axis=0)
                    del feature_vector
                else:
                    emptycount += 1
        except KeyError:
            pass
        print(f'not found word sentence vectors {emptycount}/{len(data)}')
        return X


class Word2VecVectorizer:

    def __init__(self, path):
        self._load_embedding(path)

    def _load_embedding(self, path):
        self.word_vectors = KeyedVectors.load_word2vec_format(path, binary=True)
        self.D = self.word_vectors.get_vector('king').shape[0]

    def transform(self, data):
        X = np.zeros((len(data), self.D))
        emptycount = 0
        try:
            for i, sentence in enumerate(data):
                feature_vector = [self.word_vectors.get_vector(word) for word in sentence.split()]
                if len(feature_vector) > 0:
                    X[i] = np.array(feature_vector).mean(axis=0)
                    del feature_vector
                else:
                    emptycount += 1
        except KeyError:
            pass
        print(f'not found word sentence vectors {emptycount}/{len(data)}')
        return X
