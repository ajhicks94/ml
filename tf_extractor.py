#!/usr/bin/env python

"""Term frequency extractor for the PAN19 hyperpartisan news detection task"""
# Version: 2018-10-09

# Parameters:
# --x_train_dir=<directory>
#   Directory that contains the articles XML file with the articles for which a prediction should be made.
# --outputFile=<file>
#   File to which the term frequency vectors will be written. Will be overwritten if it exists.

# Output is one article per line:
# <article id> <token>:<count> <token>:<count> ...

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

########## OPTIONS HANDLING ##########
def parse_options():
    """Parses the command line options."""
    try:
        long_options = ["x_train_dir=", "labelDir=", "outputFile=", "max_size="]
        opts, _ = getopt.getopt(sys.argv[1:], "d:l:o:n:", long_options)
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    x_train_dir = "undefined"
    labelDir = "undefined"
    outputFile = "undefined"
    max_size = sys.maxsize

    for opt, arg in opts:
        if opt in ("-d", "--x_train_dir"):
            x_train_dir = arg
        elif opt in ("-l", "--labelDir"):
            labelDir = arg
        elif opt in ("-o", "--outputFile"):
            outputFile = arg
        elif opt in "-n":
            max_size = arg
        else:
            assert False, "Unknown option."
    if x_train_dir == "undefined":
        sys.exit("Input dataset, the directory that contains the articles XML file, is undefined. Use option -d or --x_train_dir.")
    elif not os.path.exists(x_train_dir):
        sys.exit("The input dataset folder does not exist (%s)." % x_train_dir)

    if labelDir == "undefined":
        sys.exit("Label directory, the directory that contains the articles label datafile, is undefined. Use option -l or --labelDir.")
    elif not os.path.exists(labelDir):
        sys.exit("The label folder does not exist (%s)." % labelDir)
    if outputFile == "undefined":
        sys.exit("Output file, the file to which the vectors should be written, is undefined. Use option -o or --outputFile.")

    return (x_train_dir, labelDir, outputFile, max_size)

def clean_and_count(article, data):
    text = lxml.etree.tostring(article, encoding="unicode", method="text")
    textcleaned = re.sub('[^a-z ]', '', text.lower())

    for token in textcleaned.split():
        if token in data.keys():
            data[token] += 1
        else:
            data[token] = 1
    
########## ARTICLE HANDLING ##########
def handleArticle(article, outFile, data):
    text = lxml.etree.tostring(article, encoding="unicode", method="text")
    textcleaned = re.sub('[^a-z ]', '', text.lower())
  
    for token in textcleaned.split():
        if token in data.keys():
            data[token] += 1
        else:
            data[token] = 1

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
            if self.mode == 'y':
                err = "\nMaximum of " + str(self.counter) + " articles has been reached."
            raise Exception(err)
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
                        
                        # Remember, test will build it's own word_index too
                        # This line is causing major slowdowns in runtime
                        idx = self.word_index[word]
                        row.append(idx)
                    
                    # Append to sequence array
                    x.append(row)

                    #print(article.get("id") + " completed.")
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
                    
                    self.data = y

                self.counter += 1
                self.lxmlhandler = "undefined"

def create_word_index(inputDir, mode, max_articles=sys.maxsize):
    idx_file = mode + ".json"
    with open(idx_file, 'w') as f:
        data = {}
        json.dump(data,f)
        f.close()
   
    for file in os.listdir(inputDir):
        if file.endswith(".xml"):
            with open(inputDir + "/" + file) as inputRunFile:
                try:
                    xml.sax.parse(inputRunFile, HyperpartisanNewsTFExtractor(mode="widx", data=data, max_articles=max_articles))
                except Exception as e:
                    print(e)
                    break

    f = open(idx_file, 'w+')

    # Creates a sorted dictionary
    # {'brown': 28, 'blue': 56, 'red': 24} => {'blue': 56, 'brown': 28, 'red': 24}
    o = OrderedDict(Counter(data).most_common(len(data)))

    # Replaces dict value with its index
    # {'blue': 56, 'brown': 28, 'red': 24} => {'blue': 0, 'brown': 1, 'red': 2}
    # This decreases runtime DRAMATICALLY.
    for w in enumerate(o):
        o[w[1]] = w[0]

    json.dump(o, f)
    f.close()
    #print("The word counts have been written to the output file in descending order.")

def get_data(directory, filetype, mode, data, max_articles, word_index={}):
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
                except Exception as e:
                    print(e, end='')
                    break

def load_data(x_train_dir, labelDir, max_articles, num_words=None, skip_top=0, maxlen=None,
              seed=113, start_char=1, oov_char=2, index_from=3):

    with open('training.json', 'r') as f:
        word_index = {}
        word_index = json.load(f)
        f.close()

    # Start with python lists, then convert to numpy when finished for better runtime
    x_train = []
    y_train = []

    # Populate x_train
    get_data(directory=x_train_dir, filetype='xml', mode="x", data=x_train, max_articles=max_articles, word_index=word_index)
    
    # Populate y_train
    get_data(directory=labelDir, filetype='xml', mode="y", data=y_train, max_articles=max_articles)
    
    #print(y_train)
    # Create test_word_index
    # create_word_index(inputDir=testDir, mode="test", max_articles=max_articles)

    # Get test data => x_test, y_test
    x_test = [[0, 89, 100, 201, 3, 240, 1],[1],[1],[1],[2]]
    x_test = np.array(x_test)
    y_test = [1,1,1,0,1]
    y_test = np.array(y_test)
    # get_data(directory=testDir, filetype='xml', mode="x", data=x_test, max_articles=max_articles)
    
    # Transform Data TODO: PUT IN FUNCTION
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    #print("len(x_train)= ", len(x_train))
    #print("x_train.shape= ", x_train.shape)
    #print("y_train.shape= ", y_train.shape)

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
        # Since the word_index is sotted by count, this omits the top (index_from) most frequent words
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
        # Also remove any words that are < skip_top
        xs = [[w if (skip_top <= w < num_words) else oov_char for w in x] for x in xs]
    else:
        # Only remove words that are < skip_top
        xs = [[w for w in x if skip_top <= w < num_words] for x in xs]
    
    idx = len(x_train)

    # Partition the newly preprocessed instances back into their respective arrays
    x_train, y_train = np.array(xs[:idx]), np.array(ys[:idx])
    x_test, y_test = np.array(xs[idx:]), np.array(ys[idx:])

    return (x_train, y_train), (x_test, y_test)

def main(x_train_dir, labelDir, outFile, max_articles):
    total_time = 0

    start = time.time()
    # Create word index for the training set
    create_word_index(inputDir=x_train_dir, mode="training", max_articles=max_articles)
    end = time.time()

    print("Creating training word index took: ", end - start)
    total_time = end - start
    
    start = time.time()
    #
    (x_train, y_train), (x_test, y_test) = load_data(x_train_dir, labelDir, max_articles)
    end = time.time()

    print("\n\nLoading and transforming data took: ", end - start)
    total_time += (end - start)
    print("\nTotal elapsed time: %s for %s records" %(total_time, max_articles))

if __name__ == '__main__':
    main(*parse_options())
