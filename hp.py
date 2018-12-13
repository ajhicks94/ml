import json
import time
import xml
from collections import Counter, OrderedDict

import matplotlib.pyplot as plt

import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from keras.preprocessing.sequence import sequence
from tqdm import tqdm

from hyperpartisannewsparser import HyperpartisanNewsParser

def print_word_from_idx(data, index_from=0):
    with open('data/word_indexes/training.json') as f:
        word_index = {}
        word_index = json.load(f)

        keys = word_index.keys()
        for idx in data:
            for word, count in word_index.items():
                if(count == (idx-index_from)):
                    print(word + " ", end='')
        
        print("\nNumber of words in sequence: ", len(data))

# Reads in data files
def get_data(filename, filetype, mode, data, labelfile="", word_index={}):

    with open(filename) as iFile:
        xml.sax.parse(  iFile, 
                        HyperpartisanNewsParser(
                                mode=mode,
                                word_index=word_index,
                                data=data))

def create_word_index(datafile, labelfile, mode):

    # Create a new file with a blank dictionary
    # training.json
    # test.json
    #idx_file = "data/word_indexes/" + mode + ".json"
    idx_file = datafile + '.json'
    with open(idx_file, 'w') as f:
        data = {}
        json.dump(data,f)
        f.close()
   
    with open(datafile) as inputRunFile:
        xml.sax.parse(inputRunFile, HyperpartisanNewsParser(mode="widx", data=data))

    f = open(idx_file, 'w+')

    # Create a sorted dictionary
    o = OrderedDict(Counter(data).most_common(len(data)))

    # Replaces dict value with its index
    # {'blue': 56, 'brown': 28, 'red': 24} => {'blue': 0, 'brown': 1, 'red': 2}
    # This decreases runtime DRAMATICALLY.
    for w in enumerate(o):
        o[w[1]] = w[0]

    # Write dictionary to file
    json.dump(o, f)
    f.close()

def get_pretrained_embeddings(tr_widx, embedding_file):

    with open(tr_widx, 'r') as w:
        word_index = {}
        word_index = json.load(w)
    
    # First, read in the embedding_index
    # Reference begin: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    EMBEDDING_DIM = 300

    embedding = KeyedVectors.load_word2vec_format(embedding_file, binary=True)
    embedding_matrix = np.zeros((len(embedding.wv.vocab) + 1, EMBEDDING_DIM))
    #embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    with tqdm(total=len(embedding_matrix), unit='it', unit_scale=True, unit_divisor=1024) as pbar:
        i = 0
        unk_words = []
        for word in embedding.wv.vocab:
            #if word in embedding.wv.vocab:
            embedding_vector = embedding.wv[word]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            #else:
            #    unk_words.append(word)
            
            pbar.update()
            i += 1

    print("Num of words in corpus vocab not found in embedding:", len(unk_words))

    return embedding_matrix
    # Reference end

# Loads data from xml files and transforms them for use with keras
def load_data(tr, tr_labels, val, val_labels, te, te_labels, num_words=None, skip_top=0, maxlen=None,
              seed=113, start_char=1, oov_char=2, index_from=3):

    with open(tr + '.json', 'r') as f:
    #with open('data/word_indexes/training.json', 'r') as f:
        training_widx = {}
        training_widx = json.load(f)

    print('len(training_widx)= ', len(training_widx), "\n")

    #with open('data/word_indexes/validation.json', 'r') as f:
    #    validation_widx = {}
    #    validation_widx = json.load(f)

    #with open('data/word_indexes/test.json', 'r') as f:
    #    test_widx = {}
    #    test_widx = json.load(f)

    # Start with python lists, then convert to numpy when finished for better runtime
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []

    # Populate x_train
    start = time.time()
    get_data(filename=tr, labelfile=tr_labels, filetype='xml', mode="x", data=x_train, word_index=training_widx)
    finish = time.time()
    print("get_data(x_train) took:", finish-start)

    #articlelengths = []
    #for article in x_train:
    #    articlelengths.append(len(article))

    #print("maximum length of an article in training: ", max(articlelengths))
    #print("average length of an article in training: ", sum(articlelengths)/len(articlelengths))
    #plt.plot(articlelengths)
    #plt.title('Training article word counts')
    #plt.ylabel('length')
    #plt.xlabel('article')
    #plt.show()

    # Populate y_train
    #print("Populating y_train...")
    start = time.time()
    get_data(filename=tr_labels, filetype='xml', mode="y", data=y_train)
    finish = time.time()
    print("get_data(y_train) took:", finish-start)

    # Populate x_val
    #print("Populating x_val...")
    start = time.time()
    #get_data(filename=val, labelfile=val_labels, filetype='xml', mode="x", data=x_val, word_index=validation_widx)
    get_data(filename=val, labelfile=val_labels, filetype='xml', mode="x", data=x_val, word_index=training_widx)

    finish = time.time()
    print("get_data(x_val) took:", finish-start)

    # Populate y_val
    #print("Populating y_val...")
    start = time.time()
    get_data(filename=val_labels, filetype='xml', mode="y", data=y_val)
    finish = time.time()
    print("get_data(y_val) took:", finish-start)

    # Populate x_test
    start = time.time()
    #get_data(filename=te, labelfile=te_labels, filetype='xml', mode='x', data=x_test, word_index=test_widx)
    get_data(filename=te, labelfile=te_labels, filetype='xml', mode='x', data=x_test, word_index=training_widx)
    finish = time.time()
    print("get_data(x_test) took:", finish-start)

    # Populate y_test
    start = time.time()
    get_data(filename=te_labels, filetype='xml', mode='y', data=y_test)
    finish = time.time()
    print("get_data(y_test) took:", finish-start)


    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    x_val = np.array(x_val)
    y_val = np.array(y_val)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    _remove_long_seq = sequence._remove_long_seq

    # Makes random numbers predictable based on (seed)
    np.random.seed(seed)

    # Returns an array of evenly spaced values ranged [0, len(x_train))
    # In english = it's getting an array of the indices of x_train
    # E.G. if len(x_train) = 3, indices => [0,1,2]
    indices = np.arange(len(x_train))

    # Shuffles the contents of indices
    np.random.shuffle(indices)

    # Rearranges x_train to match ordering of indices
    # x_train is normally x_train[0], x_train[1], x_train[2]
    # if indices = [0,2,1], then x_train[indices] => x_train[0], x_train[2], x_train[1]
    x_train = x_train[indices]
    y_train = y_train[indices]

    # Repeat above for validation set
    indices = np.arange(len(x_val))
    np.random.shuffle(indices)
    x_val = x_val[indices]
    y_val = y_val[indices]

    # Repeat above for the test set
    indices = np.arange(len(x_test))
    np.random.shuffle(indices)
    x_test = x_test[indices]
    y_test = y_test[indices]

    # Append all datasets
    xs = np.concatenate([x_train, x_val, x_test])
    ys = np.concatenate([y_train, y_val, y_test])
    #print_word_from_idx(xs[0])
    if start_char is not None:
        # Adds a start_char to the beginning of each sentence
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        # This shifts the indexes by index_from
        xs = [[w + index_from for w in x] for x in xs]
    #print_word_from_idx(xs[0], index_from)
    # Trims sentences down to maxlen
    if maxlen:
        xs, ys = _remove_long_seq(maxlen, xs, ys)
        if not xs:
            raise ValueError('After filtering for sequences shorter than maxlen=' +
                             str(maxlen) + ', no sequence was kept. Increase maxlen.')
 
    # Calculates the max val in xs
    if not num_words:
        num_words = max([max(x) for x in xs])

    # By convention, use 2 as OOV word
    # Reserve 'index_from' (3 by default) characters:
    # 0 => padding, 1 => start, 2 => OOV
    if oov_char is not None:
        # If a word is OOV, replace it w/ 2
        # Also remove any words that are < skip_top or > num_words
        xs = [[w if (skip_top <= w < num_words) else oov_char for w in x] for x in xs]
    else:
        # Only remove words that are < skip_top or > num_words
        xs = [[w for w in x if skip_top <= w < num_words] for x in xs]

    train_idx = len(x_train)
    val_idx = len(x_val)

    # Partition the newly preprocessed instances back into their respective arrays
    x_train, y_train = np.array(xs[:train_idx]), np.array(ys[:train_idx])
    x_val, y_val = np.array(xs[train_idx:(train_idx+val_idx)]), np.array(ys[train_idx:(train_idx+val_idx)])
    x_test, y_test = np.array(xs[(train_idx+val_idx):]), np.array(ys[(train_idx+val_idx):])
 
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
