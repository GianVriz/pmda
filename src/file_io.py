import gensim
import os
import numpy as np
import json
from tqdm import tqdm

def save_embeddings(emb_model, emb_file='embeddings.txt', vocab=[]):
    """
    emb_model : dictionary containing the embeddings
    emb_file  : file to save the word embeddings (.txt)
    vocab     : list of string specifying the words to save in the emb_file
    """
    # Write the embeddings to a file
    model_vocab = list(emb_model.wv.vocab)
    if not vocab:
        vocab = model_vocab
    n = 0
    f = open(emb_file, 'w')
    for v in tqdm(model_vocab):
        if v in vocab:
            vec = list(emb_model.wv[v])  # convert np.array to list
            f.write(v + ' ')
            vec_str = ['%.9f' % val for val in vec]
            vec_str = ' '.join(vec_str)
            f.write(vec_str + '\n')
            n += 1
    f.close()
    print('saved embeddings for ' + str(n) + ' words out of a total of' + str(len(model_vocab)) + '!')

    
def load_vocab(vocab_file):
    """
    vocab_file : txt file containing one word per line
    returns 'word2id' and 'id2word' of the preprocessing
    """
    word2id = dict()
    id2word = dict()
    with open(vocab_file, 'r') as f:
        for idx, w in enumerate(f):
            word = w.strip()
            word2id[word] = idx
            id2word[idx] = word
    return word2id, id2word