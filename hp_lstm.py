#!/usr/bin/env python

# Parameters:
# --training_data=<directory>
#   Directory that contains the articles XML file with the articles for which a prediction should be made.
# --outputFile=<file>
#   File to which the term frequency vectors will be written. Will be overwritten if it exists.

# Output is one article per line:
# <article id> <token>:<count> <token>:<count> ...

from __future__ import print_function

import getopt
import json
import codecs
import os
import re
import sys
import time
import xml.sax
from array import array
from collections import Counter, OrderedDict
from tqdm import tqdm

import lxml.etree
import lxml.sax
import matplotlib.pyplot as plt
import numpy as np

from gensim.models.keyedvectors import KeyedVectors
from keras.datasets import imdb
from keras.layers import LSTM, Dense, Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import (Tokenizer, hashing_trick, one_hot,
                                      text_to_word_sequence)

#from gensim import KeyedVectors

def print_usage(filename, message):
    print(message)
    print("Usage: python %s --training_data <file> --training_labels <file> --validation_data <file> --validation_labels <file> --test_data <file> --test_labels <file>" % filename)
    print ("Optional args:")
    print ("-n <num>\tSpecify the maximum number of training articles to use. Must be greater than 1.")
    print ("-v <num>\tSpecify the maximum number of validation articles to use. Must be greater than 1.")
    print ("-t <num>\tSpecify the maximum number of test articles to use. Must be greater than 1.")
    print ("-o <FILE>\tSpecify a file to print output to")

########## OPTIONS HANDLING ##########
def parse_options():
    """Parses the command line options."""
    try:
        long_options = ["training_data=", "training_labels=", "validation_data=", "validation_labels=", "test_data=", "test_labels=", "outputFile=", "max_size="]
        opts, _ = getopt.getopt(sys.argv[1:], "o:n:v:t:", long_options)
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    training_data = "undefined"
    training_labels = "undefined"
    validation_data = "undefined"
    validation_labels = "undefined"
    test_data = "undefined"
    test_labels = "undefined"
    outputFile = "undefined"
    max_training_size = sys.maxsize
    max_validation_size = sys.maxsize
    max_test_size = sys.maxsize

    for opt, arg in opts:
        if opt in ("-trd", "--training_data"):
            training_data = arg
        elif opt in ("-trl", "--training_labels"):
            training_labels = arg
        elif opt in ("-vd", "--validation_data"):
            validation_data = arg
        elif opt in ("-vl", "--validation_labels"):
            validation_labels = arg
        elif opt in ("-ted", "--test_data"):
            test_data = arg
        elif opt in ("-tel", "--test_labels"):
            test_labels = arg
        elif opt in ("-o", "--outputFile"):
            outputFile = arg
        elif opt in "-n":
            max_training_size = arg
        elif opt in "-v":
            max_validation_size = arg
        elif opt in "-t":
            max_test_size = arg
        else:
            assert False, "Unknown option."

    if training_data == "undefined":
        message = "training_data, the directory that contains the training articles XML file, is undefined."
        print_usage(sys.argv[0], message)
        sys.exit()
    elif not os.path.exists(training_data):
        sys.exit("The input dataset folder does not exist (%s)." % training_data)

    if training_labels == "undefined":
        message = "Label directory, the directory that contains the articles label datafile, is undefined. Use option -l or --training_labels."
        print_usage(sys.argv[0], message)
        sys.exit()
    elif not os.path.exists(training_labels):
        sys.exit("The label folder does not exist (%s)." % training_labels)
    
    if validation_data == "undefined":
        message = "validation_data is undefined"
        print_usage(sys.argv[0], message)
        sys.exit()
    elif not os.path.exists(validation_data):
        sys.exit("the validation dataset file does not exist (%s)." % validation_data)

    if validation_labels == "undefined":
        message = "validation_labels is undefined"
        print_usage(sys.argv[0], message)
        sys.exit()
    elif not os.path.exists(validation_labels):
        sys.exit("the validation_labels file does not exist (%s)." % validation_labels)

    if test_data == "undefined":
        message = "Test data directory is undefined. Use --test_data option."
        print(sys.argv[0], message)
        sys.exit()
    elif not os.path.exists(test_data):
        sys.exit("The test data folder does not exist (%s)." % test_data)

    if test_labels == "undefined":
        message = "Test label directory is undefined. Use --test_labels option."
        print(sys.argv[0], message)
        sys.exit()
    elif not os.path.exists(test_labels):
        sys.exit("The test label folder does not exist (%s)." % test_labels)

    if outputFile != "undefined" and not os.path.exists(outputFile):
        sys.exit("The output folder does not exist (%s)." % outputFile)

    return (training_data, training_labels, validation_data, validation_labels, test_data, test_labels, outputFile, max_training_size, max_validation_size, max_test_size)

def clean_and_count(article, data):
    #text = lxml.etree.tostring(article, encoding="unicode", method="text")
    #textcleaned = re.sub('[^a-z ]', '', text.lower())

    for token in article.text.split():
        if token in data.keys():
            data[token] += 1
        else:
            data[token] = 1

class customException(Exception):
    pass

def parse(datafile, labelfile, mode, max_articles, word_index={}, data=[], outFile=""):

    count = 0
    left_count = 0
    right_count = 0
    neutral_count = 0
    max_articles = int(max_articles)

    tree = lxml.etree.parse(datafile)
    articles = tree.getroot().getchildren()

    # If both are provided
    # ie. The case of any data parsing, or word indexing
    if datafile != "" and labelfile != "":
        label_tree = lxml.etree.parse(labelfile)
        labels = label_tree.getroot().getchildren()

    # If only the datafile is provided
    # ie. The case of training_labels.xml
    elif labelfile == "":
        labels = articles        

    #print("max_articles= ", max_articles)
    #print("len(articles)", len(articles))
    #print("len(labels)", len(labels))
    print("mode= ", mode)
    for article, label in zip(articles, labels):
        if mode == 'y':
            print('max_articles=', max_articles)
            print("\tMode = " + mode)
            print("Total count:", count)
            print("\tleft:\t\t", left_count)
            print("\tright:\t\t", right_count)
            print("\tneutral:\t", neutral_count)

        # Once we've maintained our distribution, stop parsing
        if (count == max_articles and 
           left_count == max_articles/4 and 
           right_count == max_articles/4 and 
           neutral_count == max_articles/2):

            print("\tMode = " + mode)
            print("Total count:", count)
            print("\tleft:\t\t", left_count)
            print("\tright:\t\t", right_count)
            print("\tneutral:\t", neutral_count)

            return
        else:
            b = label.get('bias')
            #print("bias=", b)
            if left_count == max_articles/4 and b == 'left':
                continue
            elif right_count == max_articles/4 and b == 'right':
                continue
            elif neutral_count == max_articles/2 and b not in ['left', 'right']:
                continue
            else:
                if b == 'left':
                    left_count += 1
                elif b == 'right':
                    right_count += 1
                else:
                    neutral_count += 1

                # Do stuff
                if mode == 'widx':
                    clean_and_count(article, data)
                elif mode == "x":

                    row = []
                    
                    # Split into sequence of words
                    textcleaned = article.text.split()

                    # Look up each word's index in freq index and append
                    for word in textcleaned:
                        idx = word_index[word]
                        row.append(idx)
                    
                    # Append to sequence array
                    data.append(row)

                elif mode == "y":

                    bias = article.get('bias')

                    if (bias == 'left'):
                        left_count += 1
                    elif (bias == 'right'):
                        right_count += 1
                    else:
                        neutral_count += 1

                    hp = article.get('hyperpartisan')

                    if hp in ['true', 'True', 'TRUE']:
                        data.append(1)
                    elif hp in ['false', 'False', 'FALSE']:
                        data.append(0)
                    else:
                        err = "Mislabeled or unlabeled data found: " + hp
                        raise Exception(err)

                count += 1

########## SAX FOR STREAM PARSING ##########
class HyperpartisanNewsParser(xml.sax.ContentHandler):
    def __init__(self, mode, max_articles, word_index={}, data=[], outFile=""):
        xml.sax.ContentHandler.__init__(self)
        self.outFile = outFile
        self.mode = mode
        self.lxmlhandler = "undefined"
        self.data = data
        self.word_index = word_index
        self.max_articles = int(max_articles)
        self.counter = 0
        self.left_count = 0
        self.right_count = 0
        self.neutral_count = 0

    def startElement(self, name, attrs):
        #if self.counter == self.max_articles:
            #err = ''
            #if self.mode == 'y':
            #    print("Total count:", self.counter)
            #    print("\tleft:\t\t", self.left_count)
            #    print("\tright:\t\t", self.right_count)
            #    print("\tneutral:\t", self.neutral_count)
            #raise customException(err)
        if name != "articles":
            if name == "article":
                self.lxmlhandler = lxml.sax.ElementTreeContentHandler()

            self.lxmlhandler.startElement(name, attrs)

    def characters(self, data):
        if self.lxmlhandler != "undefined":
            self.lxmlhandler.characters(data)

    def endElement(self, name):
        if self.lxmlhandler != "undefined":
            self.lxmlhandler.endElement(name)
            if name == "article":
                if self.mode == "widx":
                    clean_and_count(self.lxmlhandler.etree.getroot(), self.data)
                elif self.mode == "x":

                    article = self.lxmlhandler.etree.getroot()

                    row = []
                    
                    # Get and clean text
                    #text = lxml.etree.tostring(article, encoding="unicode", method="text")
                    #textcleaned = re.sub('[^a-z ]', '', text.lower())
                    
                    # Split into sequence of words
                    #textcleaned = textcleaned.split()
                    textcleaned = article.text.split()

                    # Look up each word's index in freq index and append
                    for word in textcleaned:
                        idx = self.word_index[word]
                        row.append(idx)
                    
                    # Append to sequence array
                    self.data.append(row)

                elif self.mode == "y":
                    article = self.lxmlhandler.etree.getroot()

                    bias = article.get('bias')

                    if (bias == 'left'):
                        self.left_count += 1
                    elif (bias == 'right'):
                        self.right_count += 1
                    else:
                        self.neutral_count += 1

                    hp = article.get('hyperpartisan')

                    if hp in ['true', 'True', 'TRUE']:
                        self.data.append(1)
                    elif hp in ['false', 'False', 'FALSE']:
                        self.data.append(0)
                    else:
                        err = "Mislabeled or unlabeled data found: " + hp
                        raise Exception(err)
                    
                self.counter += 1
                self.lxmlhandler = "undefined"

    def endDocument(self):
        print("Total articles parsed:", self.counter)
        if self.mode == 'y':
                print("Total count:", self.counter)
                print("\tleft:\t\t", self.left_count)
                print("\tright:\t\t", self.right_count)
                print("\tneutral:\t", self.neutral_count)

def create_word_index(datafile, labelfile, mode, max_articles=sys.maxsize):

    # Create a new file with a blank dictionary
    # training.json
    # test.json
    idx_file = "data/word_indexes/" + mode + ".json"
    with open(idx_file, 'w') as f:
        data = {}
        json.dump(data,f)
        f.close()
   
   # Retrieve dictionary of {"word": count}
   # This is why the data must be in separate directories, may fix later on
    #for file in os.listdir(inputDir):
        #if file.endswith(".xml"):
            #with open(inputDir + "/" + file) as inputRunFile:
    with open(datafile) as inputRunFile:
        try:
            xml.sax.parse(inputRunFile, HyperpartisanNewsParser(mode="widx", data=data, max_articles=max_articles))
        except customException as e:
            print(e, end='')
            #break
    #try:
    #parse(datafile, labelfile, mode="widx", max_articles=max_articles, data=data)
    #except Exception:
    #    pass

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

# Reads in data files
def get_data(filename, filetype, mode, data, labelfile="", max_articles=sys.maxsize, word_index={}):
    #for file in os.listdir(directory):
        #if file.endswith('.' + filetype):
            #with open(directory + "/" + file) as iFile:
    
    with open(filename) as iFile:
        try:
            xml.sax.parse(  iFile, 
                            HyperpartisanNewsParser(
                                    mode=mode,
                                    word_index=word_index,
                                    data=data,
                                    max_articles=max_articles))
        except customException as e:
            if max_articles != sys.maxsize:
                print(e, end='')
                return
    '''      
    #try:
    parse(filename, labelfile, mode, 
          max_articles=max_articles, word_index=word_index,
          data=data)
    #except Exception:
    #    pass
    '''

# Loads data from xml files and transforms them for use with keras
def load_data(tr, tr_labels, val, val_labels, te, te_labels, max_tr_articles, max_val_articles, max_te_articles, num_words=None, skip_top=0, maxlen=None,
              seed=113, start_char=1, oov_char=2, index_from=3):

    start = time.time()
    with open('data/word_indexes/training.json', 'r') as f:
        training_widx = {}
        training_widx = json.load(f)
        f.close()
    finish = time.time()
    print("Loading training_widx took:", finish-start)

    start = time.time()
    with open('data/word_indexes/validation.json', 'r') as f:
        validation_widx = {}
        validation_widx = json.load(f)
        f.close()
    finish = time.time()
    print("loading validation_widx took:", finish-start)

    start = time.time()
    with open('data/word_indexes/test.json', 'r') as f:
        test_widx = {}
        test_widx = json.load(f)
        f.close()
    finish = time.time()
    print("Loading test_widx took:", finish-start)

    print('len(training_widx)= ', len(training_widx))
    #print('len(validation_widx)= ', len(validation_widx))
    #print('len(test_widx)= ', len(test_widx), "\n")

    # Start with python lists, then convert to numpy when finished for better runtime
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []

    # Populate x_train
    #print("Populating x_train...")
    start = time.time()
    get_data(filename=tr, labelfile=tr_labels, filetype='xml', mode="x", data=x_train, max_articles=max_tr_articles, word_index=training_widx)
    finish = time.time()
    print("get_data(x_train) took:", finish-start)

    # Populate y_train
    #print("Populating y_train...")
    start = time.time()
    get_data(filename=tr_labels, filetype='xml', mode="y", data=y_train, max_articles=max_tr_articles)
    finish = time.time()
    print("get_data(y_train) took:", finish-start)

    # Populate x_val
    #print("Populating x_val...")
    start = time.time()
    get_data(filename=val, labelfile=val_labels, filetype='xml', mode="x", data=x_val, max_articles=max_val_articles, word_index=validation_widx)
    finish = time.time()
    print("get_data(x_val) took:", finish-start)

    # Populate y_val
    #print("Populating y_val...")
    start = time.time()
    get_data(filename=val_labels, filetype='xml', mode="y", data=y_val, max_articles=max_val_articles)
    finish = time.time()
    print("get_data(y_val) took:", finish-start)

    # Populate x_test
    #print("Populating x_test...")
    start = time.time()
    get_data(filename=te, labelfile=te_labels, filetype='xml', mode='x', data=x_test, word_index=test_widx, max_articles=max_te_articles)
    finish = time.time()
    print("get_data(x_test) took:", finish-start)

    # Populate y_test
    #print("Populating y_test...\n")
    start = time.time()
    get_data(filename=te_labels, filetype='xml', mode='y', data=y_test, max_articles=max_te_articles)
    finish = time.time()
    print("get_data(y_test) took:", finish-start)

    print("Shuffling data...", end='')
    start = time.time()
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

    finish = time.time()
    print(finish-start)

    # Append all datasets
    print("Concatenating...", end='')
    start = time.time()
    xs = np.concatenate([x_train, x_val, x_test])
    ys = np.concatenate([y_train, y_val, y_test])
    finish = time.time()
    print(finish-start)

    print("start_char/index_from...", end='')
    start = time.time()
    if start_char is not None:
        # Adds a start_char to the beginning of each sentence
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        # This shifts the indexes by index_from
        xs = [[w + index_from for w in x] for x in xs]
    finish = time.time()
    print(finish-start)

    # Trims sentences down to maxlen
    print("Maxlen...", end='')
    start = time.time()
    if maxlen:
        xs, ys = _remove_long_seq(maxlen, xs, ys)
        if not xs:
            raise ValueError('After filtering for sequences shorter than maxlen=' +
                             str(maxlen) + ', no sequence was kept. Increase maxlen.')
    finish = time.time()
    print(finish-start)

    # Calculates the max val in xs
    print("Num_words = max...", end='')
    start = time.time()
    if not num_words:
        num_words = max([max(x) for x in xs])
    finish = time.time()
    print(finish-start)

    # By convention, use 2 as OOV word
    # Reserve 'index_from' (3 by default) characters:
    # 0 => padding, 1 => start, 2 => OOV
    print("Skip_top/num_words/oov...", end='')
    start = time.time()
    if oov_char is not None:
        # If a word is OOV, replace it w/ 2
        # Also remove any words that are < skip_top or > num_words
        xs = [[w if (skip_top <= w < num_words) else oov_char for w in x] for x in xs]
    else:
        # Only remove words that are < skip_top or > num_words
        xs = [[w for w in x if skip_top <= w < num_words] for x in xs]
    finish = time.time()
    print(finish-start)

    train_idx = len(x_train)
    val_idx = len(x_val)

    print("Partitioning...", end='')
    start = time.time()
    # Partition the newly preprocessed instances back into their respective arrays
    x_train, y_train = np.array(xs[:train_idx]), np.array(ys[:train_idx])
    x_val, y_val = np.array(xs[train_idx:(train_idx+val_idx)]), np.array(ys[train_idx:(train_idx+val_idx)])
    x_test, y_test = np.array(xs[(train_idx+val_idx):]), np.array(ys[(train_idx+val_idx):])
    finish = time.time()
    print(finish-start)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def get_pretrained_embeddings(tr_widx, embedding_file):

    with open(tr_widx, 'r') as w:
        word_index = {}
        word_index = json.load(w)
    
    # First, read in the embedding_index
    # Reference begin: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    EMBEDDING_DIM = 300

    start = time.time()
    # Takes about 50 seconds
    embedding = KeyedVectors.load_word2vec_format(embedding_file, binary=True)
    finish = time.time()
    print("Loading word2vec embeddings took:", finish-start, "\n")
    
    start = time.time()

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    with tqdm(total=len(word_index), unit='it', unit_scale=True, unit_divisor=1024) as pbar:
        i = 0
        unk_words = []
        for word in word_index:
            if word in embedding.wv.vocab:
                embedding_vector = embedding.wv[word]
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
            else:
                unk_words.append(word)
            
            pbar.update()
            i += 1

    print("Words not found:", len(unk_words))
    finish = time.time()
    print("Calculating embedding_matrix too:", finish-start)

    return embedding_matrix
    # Reference end

def main(tr, tr_labels, val, val_labels, te, te_labels, outFile, max_tr_articles, max_val_articles, max_te_articles):
    start = time.time()
    # Build training set word index
    print("\nBuilding training word index...")
    create_word_index(datafile=tr, labelfile=tr_labels, mode="training", max_articles=max_tr_articles)
    finish = time.time()
    print("building training widx took:", finish - start, "\n")

    start = time.time()
    # Build validation set word index
    print("Building validation word index...")
    create_word_index(datafile=val, labelfile=val_labels, mode="validation", max_articles=max_val_articles)
    finish = time.time()
    print("building validation widx took:", finish-start, "\n")

    start = time.time()
    # Build test set word index
    print("Building test word index...\n")
    create_word_index(datafile=te, labelfile=te_labels, mode="test", max_articles=max_te_articles)
    finish = time.time()
    print('building test widx took:', finish-start, "\n")

    # Load configuration
    with open('run.json', 'r') as j:
        config = {}
        config = json.load(j)

    max_features = config['max_features']           # Word Embedding                                 #default 20000
    skip_top = config['skip_top']                   # Skip the most common words                     #default 0
    num_words = config['num_words']                 # Upper limit for word commonality               #default 0
    maxlen = config['maxlen']                       # Maximum length of a sequence (sentence)        #default 80

    start = time.time()
    # Load and preprocess data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(tr, tr_labels, val, val_labels, te, te_labels, max_tr_articles, max_val_articles, max_te_articles,
                                                     skip_top=skip_top, num_words=num_words, maxlen=None)
    finish = time.time()
    print("Load_data took a total of:", finish-start)
    
    # ML Stuff now
    print(len(x_train), 'train sequences')
    print(len(x_val), 'validation sequences')
    print(len(x_test), 'test sequences\n')

    print('Pad sequences (samples x time)...', end='')
    start = time.time()
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_val = sequence.pad_sequences(x_val, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    finish = time.time()
    print(finish-start)

    batch_size = config['batch_size']               # Number of instances before updating weights    #default 32
    epochs = config['epochs']                       # Number of epochs                               #default 15

    # If we don't want any of these optional args,
    # we will have to remove them from the LSTM call itself
    go_backwards = True if (config['go_backwards'] == "True") else False
    dropout = config['dropout']
    recurrent_dropout = config['recurrent_dropout']
    #bias_regularizer = config['bias_regularizer']

    print('Build model...')

    # Obtain and compute embedding matrix
    print("Get pretrained word embeddings...")
    embedding_matrix = get_pretrained_embeddings(  'data/word_indexes/training.json',
                                                    'data/embeddings/GoogleNews-vectors-negative300.bin')

    model = Sequential()

    start = time.time()
    #model.add(Embedding(max_features, 128))
    
    with open('data/word_indexes/training.json', 'r') as f:
        word_index = {}
        word_index = json.load(f)

    EMBEDDING_DIM = 300
    model.add(Embedding(len(word_index) + 1,
                        EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=maxlen,
                        trainable=False))
    
    finish = time.time()
    print("adding embedding layer took:", finish-start)

    start = time.time()
    model.add(LSTM( 128, #bias_regularizer = bias_regularizer, 
                    dropout=dropout, recurrent_dropout=recurrent_dropout, 
                    go_backwards=go_backwards))
    finish = time.time()
    print("adding LSTM layer took:", finish-start)

    start = time.time()
    model.add(Dense(1, activation='sigmoid'))
    finish = time.time()
    print("adding Dense sigmoid:", finish-start)

    start = time.time()
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    finish = time.time()
    print("Compiling model took:", finish-start)

    print(model.summary())

    start = time.time()
    history = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            verbose=1)
    finish = time.time()
    print("Fitting model took:", finish-start)
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.yticks(np.arange(0.4, 1.0, 0.05))


    plot_prefix = str(int(tr.split('/')[-1].split('.')[0]) + int(val.split('/')[-1].split('.')[0]) + int(te.split('/')[-1].split('.')[0]))
    #plot_prefix = plot_prefix.split('.')[0]
    plot_name = plot_prefix + '.png'
    plot_name = 'results/runs/' + plot_name
    plot_config = 'results/runs/' + plot_prefix + '.config'

    i = 1
    jpg = '.jpg'
    conf = '.config'
    if os.path.exists(plot_name):
        n = plot_name.split('.')[0]
        plot_name = n + '_' + str(i) + jpg
        plot_config = n + '_' + str(i) + conf
        if os.path.exists(plot_name):
            while os.path.exists(plot_name):
                n = plot_name.split('.')[0]
                n = n.split('_')[0]
                plot_name = n + '_' + str(i) + jpg
                plot_config = n + '_' + str(i) + conf
                i += 1
    
    plt.savefig(plot_name)
    with open(plot_config, 'w') as f:
        json.dump(config, f, indent=1)
    #plt.show()

    # Plot training & validation loss values
    #plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    #plt.title('Model loss')
    #plt.ylabel('Loss')
    #plt.xlabel('Epoch')
    #plt.legend(['Train', 'Validation'], loc='upper left')
    #plt.show()
    #print("val_loss:\t", history['val_loss'])
    #print("val_acc:\t", history['val_acc'])
    #print("loss:\t\t", history['loss'])
    #print("acc:\t\t", history['acc'])

    # DO NOT UNCOMMENT THIS
    #score, acc = model.evaluate(x_test, y_test,
    #                            batch_size=batch_size)
    #print('Test score:', score)
    #print('Test accuracy:', acc)

if __name__ == '__main__':
    main(*parse_options())
