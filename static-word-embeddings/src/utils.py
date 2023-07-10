from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
import json

def load_embedding(embed_path):
    if embed_path.endswith('wv'):
        return KeyedVectors.load(embed_path)
    elif embed_path.endswith('txt'):
        # print(embed_path)
        try:
            return KeyedVectors.load_word2vec_format(embed_path, binary=False)
        except Exception:

            glove2word2vec(embed_path, embed_path)
            return KeyedVectors.load_word2vec_format(embed_path, binary=False)
    elif embed_path.endswith('bin'):
        return KeyedVectors.load_word2vec_format(embed_path, binary=True)
    # NOTE reddit embedding is saved as model (no ext) + syn1neg + syn0
    else:
        return Word2Vec.load(embed_path).wv


def load_pairs(pair_path):
    pairs = json.load(open(pair_path, "rb"))
    pairs = [(w[0], w[1]) for w in pairs]
    return pairs


def filter_pairs(embedding, pairs):
    for pair in pairs:
        if pair[0] not in embedding.index_to_key or pair[1] not in embedding.index_to_key:
            pairs.remove(pair)
            print("Pair {} removed -- OOV".format(pair))
    return pairs