#import gensim
import csv
import json
import numpy as np
import os
import pandas as pd
import requests
import string

from tqdm import tqdm

def download_guardian(query, data_path=''):
    """
    Download Guardian articles using its API and save them in a csv file.
    Adaptation of this code: https://gist.github.com/dannguyen/c9cb220093ee4c12b840
    """
    def flatten_json(y):
        """
        Function to flattening a json
        source: https://www.geeksforgeeks.org/flattening-json-objects-in-python/
        """
        out = {}
        def flatten(x, name=''):
            # If the Nested key-value
            # pair is of dict type
            if type(x) is dict:
                for a in x:
                    flatten(x[a], name + a + '_')
            # If the Nested key-value
            # pair is of list type
            elif type(x) is list:
                i = 0
                for a in x:
                    flatten(a, name + str(i) + '_')
                    i += 1
            else:
                out[name[:-1]] = x
        flatten(y)
        return out

    if data_path != '' and os.path.isfile(data_path):
        raise ValueError('"data_path" not valid: the file already exists.')

    print('Downloading articles...')
    api_endpoint = 'https://content.guardianapis.com/search'
    corpus, current_page, total_pages = [], 1, 1
    while current_page <= total_pages:
        try:
            query['page'] = current_page
            r = requests.get(api_endpoint, query)
            r.raise_for_status()
        except:
            SystemExit(err)
        data = r.json()['response']
        corpus.extend(data['results'])
        current_page += 1
        total_pages = data['pages']
    # Flatten dictionaries corresponding to the articles
    for i in range(len(corpus)):
        corpus[i] = flatten_json(corpus[i])
    # Convert list of articles to pandas dataframe
    corpus = pd.json_normalize(corpus)
    # Save corpus in a csv file
    if data_path != '':
        corpus.to_csv(data_path)
        print(len(corpus), '  articles downloaded and stored in', data_path)
    return corpus


def load_un_general_debates(data_path, flag_split_by_paragraph=False):
    # Read raw data (https://www.kaggle.com/datasets/unitednations/un-general-debates)
    print('reading raw data...')
    with open('./data/raw/un-general-debates.csv', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
        line_count = 0
        all_timestamps_ini = []
        all_docs_ini = []
        for row in csv_reader:
            # skip header
            if(line_count>0):
                all_timestamps_ini.append(row[1])
                all_docs_ini.append(row[3].encode("ascii", "ignore").decode())
            line_count += 1
    if flag_split_by_paragraph:
        print('splitting by paragraphs...')
        docs = []
        timestamps = []
        for dd, doc in enumerate(all_docs_ini):
            splitted_doc = doc.split('.\n')
            for ii in splitted_doc:
                docs.append(ii)
                timestamps.append(all_timestamps_ini[dd])
    else:
        docs = all_docs_ini
        timestamps = all_timestamps_ini
    print('  number of documents: {}'.format(len(docs)))
    return docs, timestamps


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

    c_na, c_a = 0, 0
    with open(emb_file, 'w') as f:
        for v in vocab:
            if v not in model_vocab:
                c_na += 1
                print('(' + str(c_na) + ') "' + v + '" is not available.')
            else:
                c_a += 1
                vec = list(emb_model.wv[v])
                f.write(v + ' ')
                vec_str = ['%.9f' % val for val in vec]
                vec_str = ' '.join(vec_str)
                f.write(vec_str + '\n')
    print('saved embeddings for ' + str(c_a) + '/' + len(vocab) + ' words!')


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
