#!/usr/bin/env python

"""Term frequency extractor for the PAN19 hyperpartisan news detection task"""
# Version: 2018-10-09

# Parameters:
# --inputDataset=<directory>
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
        long_options = ["inputDataset=", "outputFile=", "max_size=", "training_widx"]
        opts, _ = getopt.getopt(sys.argv[1:], "d:o:n:w", long_options)
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    inputDataset = "undefined"
    outputFile = "undefined"
    max_size = sys.maxsize
    training_widx = False

    for opt, arg in opts:
        if opt in ("-d", "--inputDataset"):
            inputDataset = arg
        elif opt in ("-o", "--outputFile"):
            outputFile = arg
        elif opt in "-n":
            max_size = arg
        elif opt in "-w":
            training_widx = arg
        else:
            assert False, "Unknown option."
    if inputDataset == "undefined":
        sys.exit("Input dataset, the directory that contains the articles XML file, is undefined. Use option -d or --inputDataset.")
    elif not os.path.exists(inputDataset):
        sys.exit("The input dataset folder does not exist (%s)." % inputDataset)

    if outputFile == "undefined":
        sys.exit("Output file, the file to which the vectors should be written, is undefined. Use option -o or --outputFile.")

    return (inputDataset, outputFile, max_size, training_widx)

def clean_and_count(article, data):
    text = lxml.etree.tostring(article, encoding="unicode", method="text")
    textcleaned = re.sub('[^a-z ]', '', text.lower())

    for token in textcleaned.split():
        if token in data.keys():
            data[token] += 1
        else:
            data[token] = 1
    
    #print(article.get("id") + " completed.")

########## ARTICLE HANDLING ##########
def handleArticle(article, outFile, data):
    termfrequencies = {}

    # get text from article
    text = lxml.etree.tostring(article, encoding="unicode", method="text")
    textcleaned = re.sub('[^a-z ]', '', text.lower())

    #maxlen = 2000
    #word_index = imdb.get_word_index()
    #words = set(text_to_word_sequence(textcleaned))
    #x_test = [[word_index[w] for w in words if w in word_index]]
    #print(x_test)
    #x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    #vector = np.array([x_test.flatten()])

  
    for token in textcleaned.split():
        if token in data.keys():
            data[token] += 1
        else:
            data[token] = 1
        #if token in termfrequencies:
        #    termfrequencies[token] += 1
        #else:
        #    termfrequencies[token] = 1
    #f = open('data.json', 'w+')
    #json.dump(data, f)
    #f.close()
    #words = set(text_to_word_sequence(textcleaned))
    #vocab_size = len(words)
    #oh = one_hot(textcleaned, round(1.3*vocab_size))
    #print('oh= \n\n', oh)
    #res = hashing_trick(textcleaned, round(vocab_size*1.3), hash_function='md5')
    #print(res)
    #print(termfrequencies, '\n')
    #terms = sorted(termfrequencies, key=termfrequencies.get, reverse=True)
    #d = dict()
    #for w in terms:
    #    d[w] = termfrequencies[w]
    #print('\n\n')
    #print(d)
    #print(textcleaned)
    #termfrequencies = d
    #wvec = []
    #for token in textcleaned.split():
    #    i = 1
    #    for x in d.keys():
    #        if x == token:
    #            break
    #        i +=1
    #    wvec.append(i)

    #print(wvec)
    #sys.exit()
    # writing counts: <article id> <token>:<count> <token>:<count> ...
    #outFile.write(article.get("id"))
    #for token, count in termfrequencies.items():
     #   outFile.write(" " + str(token) + ":" + str(count))
    #outFile.write("\n")
    print(article.get("id") + " completed.")



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
                    x = self.data[0]
                    y = self.data[1]
                    row = []
                    
                    # Get and clean text
                    text = lxml.etree.tostring(article, encoding="unicode", method="text")
                    textcleaned = re.sub('[^a-z ]', '', text.lower())

                    # Split into sequence of words
                    textcleaned = textcleaned.split()

                    # Look up each word's index in freq index and append
                    for word in textcleaned:
                        
                        # Remember, test will build it's own word_index too
                        idx = list(self.word_index).index(word)
                        row.append(idx)
                    
                    # Append to x array (data)
                    x.append(row)

                    # Since article doesn't have whether it is hyperpartisan or not, let's get y_train afterwards
                    print("article.items()= ", article.items())
                    #y.append(article.items())
                self.counter += 1
                self.lxmlhandler = "undefined"

def create_word_index(inputDataset, max_articles=sys.maxsize):
    with open('data.json', 'w') as f:
        data = {}
        json.dump(data,f)
        f.close()
   
    for file in os.listdir(inputDataset):
        if file.endswith(".xml"):
            with open(inputDataset + "/" + file) as inputRunFile:
                try:
                    xml.sax.parse(inputRunFile, HyperpartisanNewsTFExtractor(mode="widx", data=data, max_articles=max_articles))
                except Exception as e:
                    print(e)
                    break

    f = open('data.json', 'w+')
    o = OrderedDict(Counter(data).most_common(len(data)))
    json.dump(o, f)
    f.close()
    print("The word counts have been written to the output file in descending order.")

def main(inputDataset, outFile, max_articles, tr_word_index_created):
    # Give a True/False if word_index is created (from cmdline)
    print("tr_word_index_created= ", tr_word_index_created)
    if not tr_word_index_created:
        create_word_index(inputDataset, max_articles)

    with open('data.json', 'r') as f:
        word_index = {}
        word_index = json.load(f)
        f.close()

    # Start with python lists, then convert to numpy when finished for better runtime
    x_train = []
    y_train = []

    # Populate x_train
    for file in os.listdir(inputDataset):
        if file.endswith('.xml'):
            with open(inputDataset + "/" + file) as inputRunFile:
                try:
                    xml.sax.parse(  inputRunFile, 
                                    HyperpartisanNewsTFExtractor(
                                        mode="x",
                                        word_index=word_index,
                                        data=(x_train, y_train),
                                        max_articles=max_articles))
                except Exception as e:
                    print(e)
                    break

    # Populate y_train

    # Create test_word_index

    # Get test data => x_test, y_test
    x_test = []
    y_test = []

########## MAIN ##########
def old_main(inputDataset, outputFile, max_articles=sys.maxsize):

    with open('data.json', 'w') as f:
        data = {}
        json.dump(data,f)
        f.close()
   
    with open(outputFile, 'w') as outFile:
        for file in os.listdir(inputDataset):
            if file.endswith(".xml"):
                with open(inputDataset + "/" + file) as inputRunFile:
                    try:
                        xml.sax.parse(inputRunFile, HyperpartisanNewsTFExtractor(mode="x",outFile=outFile, data=data, max_articles=max_articles))
                    except Exception as e:
                        print(e)
                        break

    f = open('data.json', 'w+')
    o = OrderedDict(Counter(data).most_common(len(data)))
    json.dump(o, f)
    f.close()
    print("The word counts have been written to the output file in descending order.")

if __name__ == '__main__':
    main(*parse_options())
