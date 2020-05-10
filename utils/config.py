from utils.embeddings import GloveVectorizer, Word2VecVectorizer

embeds = {
    'glove': {
        'path': 'embeddings/glove.6B.50d.txt',
        'model': GloveVectorizer
    },
    'word2vec': {
        'path': 'embeddings/GoogleNews-vectors-negative300.bin',
        'model': Word2VecVectorizer
    }

}
