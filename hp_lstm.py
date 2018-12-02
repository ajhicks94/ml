#!/usr/bin/env python

# Parameters:
# --training_data=<directory>
#   Directory that contains the articles XML file with the articles for which a prediction should be made.
# --outputFile=<file>
#   File to which the term frequency vectors will be written. Will be overwritten if it exists.

# Output is one article per line:
# <article id> <token>:<count> <token>:<count> ...

from __future__ import print_function

import json
import os
import getopt
import sys
import xml.sax
import lxml.sax
import lxml.etree
import re
import numpy as np
import time

from collections import Counter, OrderedDict
from array import array
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.text import hashing_trick, text_to_word_sequence
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM

def print_usage(filename, message):
    print(message)
    print("Usage: python %s --training_data <DIR> --training_labels <DIR> --test_data <DIR> --test_labels <DIR>" % filename)
    print ("Optional args:")
    print ("-n <num>\tSpecify the maximum number of training articles to use. Must be greater than 1.")
    print ("-t <num>\tSpecify the maximum number of test articles to use. Must be greater than 1.")
    print ("-o <FILE>\tSpecify a file to print output to")

########## OPTIONS HANDLING ##########
def parse_options():
    """Parses the command line options."""
    try:
        long_options = ["training_data=", "training_labels=", "test_data=", "test_labels=", "outputFile=", "max_size="]
        opts, _ = getopt.getopt(sys.argv[1:], "o:n:t:", long_options)
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    training_data = "undefined"
    training_labels = "undefined"
    test_data = "undefined"
    test_labels = "undefined"
    outputFile = "undefined"
    max_training_size = sys.maxsize
    max_test_size = sys.maxsize

    for opt, arg in opts:
        if opt in ("-trd", "--training_data"):
            training_data = arg
        elif opt in ("-trl", "--training_labels"):
            training_labels = arg
        elif opt in ("-ted", "--test_data"):
            test_data = arg
        elif opt in ("-tel", "--test_labels"):
            test_labels = arg
        elif opt in ("-o", "--outputFile"):
            outputFile = arg
        elif opt in "-n":
            max_training_size = arg
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

    return (training_data, training_labels, test_data, test_labels, outputFile, max_training_size, max_test_size)

def clean_and_count(article, data):
    text = lxml.etree.tostring(article, encoding="unicode", method="text")
    textcleaned = re.sub('[^a-z ]', '', text.lower())

    for token in textcleaned.split():
        if token in data.keys():
            data[token] += 1
        else:
            data[token] = 1

class customException(Exception):
    pass

########## SAX FOR STREAM PARSING ##########
class HyperpartisanNewsTFExtractor(xml.sax.ContentHandler):
    def __init__(self, mode, max_articles, word_index={}, data=[], outFile=""):
        xml.sax.ContentHandler.__init__(self)
        self.outFile = outFile
        self.mode = mode
        self.lxmlhandler = "undefined"
        self.data = data
        self.word_index = word_index
        self.max_articles = int(max_articles)
        self.counter = 0

    def startElement(self, name, attrs):
        if self.counter == self.max_articles:
            err = ''
            raise customException(err)
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
                    x = self.data
                    row = []
                    
                    # Get and clean text
                    text = lxml.etree.tostring(article, encoding="unicode", method="text")
                    textcleaned = re.sub('[^a-z ]', '', text.lower())
                    
                    # Split into sequence of words
                    textcleaned = textcleaned.split()

                    # Look up each word's index in freq index and append
                    for word in textcleaned:
                        idx = self.word_index[word]
                        row.append(idx)
                    
                    # Append to sequence array
                    x.append(row)

                    #print("x: " + article.get("id") + " completed.")
                    self.data = x

                elif self.mode == "y":
                    article = self.lxmlhandler.etree.getroot()

                    y = self.data

                    hp = article.get('hyperpartisan')

                    if hp in ['true', 'True', 'TRUE']:
                        y.append(1)
                    elif hp in ['false', 'False', 'FALSE']:
                        y.append(0)
                    else:
                        err = "Mislabeled or unlabeled data found: " + hp
                        raise Exception(err)
                    
                    #print("y: " + article.get("id") + " completed.")
                    self.data = y

                self.counter += 1
                #print("Count= ", self.counter)
                self.lxmlhandler = "undefined"

def create_word_index(inputDir, mode, max_articles=sys.maxsize):

    # Create a new file with a blank dictionary
    # training.json
    # test.json
    idx_file = "Data/Word_Indexes/" + mode + ".json"
    with open(idx_file, 'w') as f:
        data = {}
        json.dump(data,f)
        f.close()
   
   # Retrieve dictionary of {"word": count}
   # This is why the data must be in separate directories, may fix later on
    for file in os.listdir(inputDir):
        if file.endswith(".xml"):
            with open(inputDir + "/" + file) as inputRunFile:
                try:
                    xml.sax.parse(inputRunFile, HyperpartisanNewsTFExtractor(mode="widx", data=data, max_articles=max_articles))
                except customException as e:
                    print(e, end='')
                    break

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
def get_data(directory, filetype, mode, data, max_articles=sys.maxsize, word_index={}):
    for file in os.listdir(directory):
        if file.endswith('.' + filetype):
            with open(directory + "/" + file) as iFile:
                try:
                    xml.sax.parse(  iFile, 
                                    HyperpartisanNewsTFExtractor(
                                            mode=mode,
                                            word_index=word_index,
                                            data=data,
                                            max_articles=max_articles))
                except customException as e:
                    if max_articles != sys.maxsize:
                        print(e, end='')
                        break

# Loads data from xml files and transforms them for use with keras
def load_data(training_data, training_labels, test_data, test_labels, max_training_articles, max_test_articles, num_words=None, skip_top=0, maxlen=None,
              seed=113, start_char=1, oov_char=2, index_from=3):

    with open('Data/Word_Indexes/training.json', 'r') as f:
        training_widx = {}
        training_widx = json.load(f)
        f.close()

    with open('Data/Word_Indexes/test.json', 'r') as f:
        test_widx = {}
        test_widx = json.load(f)
        f.close()

    print('len(training_widx)= ', len(training_widx))
    print('len(test_widx)= ', len(test_widx))
    # Start with python lists, then convert to numpy when finished for better runtime
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    # Populate x_train
    print("Populating x_train...")
    get_data(directory=training_data, filetype='xml', mode="x", data=x_train, max_articles=max_training_articles, word_index=training_widx)
    
    # Populate y_train
    print("Populating y_train...")
    get_data(directory=training_labels, filetype='xml', mode="y", data=y_train, max_articles=max_training_articles)
    
    # Populate x_test
    print("Populating x_test...")
    # TODO TRY WITH test data not same shape as training, possible?
    get_data(directory=test_data, filetype='xml', mode='x', data=x_test, word_index=test_widx, max_articles=max_test_articles)
    
    # Populate y_test
    print("Populating y_test...\n")
    get_data(directory=test_labels, filetype='xml', mode='y', data=y_test, max_articles=max_test_articles)

    # Transform Data TODO: PUT IN FUNCTION
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    print ("x_train.shape= ", x_train.shape)
    print ("y_train.shape= ", y_train.shape)
    print ("x_test.shape= ", x_test.shape)
    print ("y_test.shape= ", y_test.shape)
    
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

    # Repeat above for the test set
    indices = np.arange(len(x_test))
    np.random.shuffle(indices)
    x_test = x_test[indices]
    y_test = y_test[indices]

    #print("x_test.shape= ", x_test.shape)

    # Append test to train
    xs = np.concatenate([x_train, x_test])
    ys = np.concatenate([y_train, y_test])

    if start_char is not None:
        # Adds a start_char to the beginning of each sentence
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        # This shifts the indexes by index_from
        xs = [[w + index_from for w in x] for x in xs]

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
    
    idx = len(x_train)

    # Partition the newly preprocessed instances back into their respective arrays
    x_train, y_train = np.array(xs[:idx]), np.array(ys[:idx])
    x_test, y_test = np.array(xs[idx:]), np.array(ys[idx:])

    return (x_train, y_train), (x_test, y_test)

def main(training_data, training_labels, test_data, test_labels, outFile, max_training_articles, max_test_articles):
    max_features = 20000        # Word Embedding                                 #default 20000
    skip_top = 30                # Skip the most common words                     #default 0
    num_words = 500             # Upper limit for word commonality               #default 0
    maxlen = 80                 # Maximum length of a sequence (sentence)        #default 80
    batch_size = 50             # Number of instances before updating weights    #default 32
    epochs = 50                 # Number of epochs                               #default 15

    # Build training set word index
    print("\nBuilding training word index...")
    create_word_index(inputDir=training_data, mode="training", max_articles=max_training_articles)
    
    # Build test set word index
    print("Building test word index...\n")
    # Is it creating the word index based on the training data and not the test data?
    create_word_index(inputDir=test_data, mode="test", max_articles=max_test_articles)
    
    (x_train, y_train), (x_test, y_test) = load_data(training_data, training_labels, test_data, test_labels, max_training_articles, max_test_articles,
                                                     skip_top=skip_top, num_words=num_words, maxlen=None)
    
    # ML Stuff now
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences`')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, go_backwards=True))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            # Use validation_split = 0.2 instead of validating on the test set and then evaluating on the test set
            validation_split=0.2)
            #validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

if __name__ == '__main__':
    main(*parse_options())
