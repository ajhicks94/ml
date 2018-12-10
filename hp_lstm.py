from __future__ import print_function

import getopt
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import LSTM, Dense, Embedding, BatchNormalization
from keras.models import Sequential
from keras.preprocessing import sequence
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

from hp import create_word_index, get_pretrained_embeddings, load_data


def print_usage(filename, message):
    print(message)
    print("Usage: python %s --training_data <file> --training_labels <file> --validation_data <file> --validation_labels <file> --test_data <file> --test_labels <file>" % filename)

def parse_options():
    try:
        long_options = ["training_data=", "training_labels=", "validation_data=", "validation_labels=", "test_data=", "test_labels="]
        opts, _ = getopt.getopt(sys.argv[1:], "", long_options)
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    training_data = "undefined"
    training_labels = "undefined"
    validation_data = "undefined"
    validation_labels = "undefined"
    test_data = "undefined"
    test_labels = "undefined"

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

    return (training_data, training_labels, validation_data, validation_labels, test_data, test_labels)

def create_word_indexes(tr, tr_l, val, val_l, te, te_l):
    create_word_index(datafile=tr, labelfile=tr_l, mode="training")
    create_word_index(datafile=val, labelfile=val_l, mode="validation")
    create_word_index(datafile=te, labelfile=te_l, mode="test")

def data(tr, tr_labels, val, val_labels, te, te_labels):
    MAXLEN = 208
    NUM_WORDS = 40000
    SKIP_TOP = 0

    tr = 'data/training/medium/18000.xml'
    tr_labels = 'data/training/medium/18000_labels.xml'
    val = 'data/validation/medium/6000.xml'
    val_labels = 'data/validation/medium/6000_labels.xml'
    te = 'data/test/medium/6000.xml'
    te_labels = 'data/test/medium/6000_labels.xml'
    
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(tr, tr_labels, val, val_labels, te, te_labels,
                                                     skip_top=SKIP_TOP, num_words=NUM_WORDS, maxlen=None)
    
    x_train = sequence.pad_sequences(x_train, maxlen=MAXLEN)
    x_val = sequence.pad_sequences(x_val, maxlen=MAXLEN)
    x_test = sequence.pad_sequences(x_test, maxlen=MAXLEN)

    embedding_matrix = get_pretrained_embeddings(  'data/word_indexes/training.json',
                                                    'data/embeddings/GoogleNews-vectors-negative300.bin')


    return (x_train, y_train), (x_val, y_val), embedding_matrix

def create_model(x_train, y_train, x_val, y_val, embedding_matrix):
    with open('data/word_indexes/training.json', 'r') as f:
        word_index = {}
        word_index = json.load(f)

    EMBEDDING_DIM = 300
    EPOCHS = 10
    MAXLEN = 208
    BATCH_SIZE = 32

    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                        EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=MAXLEN,
                        trainable=False))

    model.add(LSTM( 128, 
                    dropout={{uniform(0,1)}}, recurrent_dropout=0.1))
    
    model.add(Dense(1, activation='sigmoid'))
 
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    history = model.fit(x_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(x_val, y_val),
            verbose=1)

    validation_acc = np.amax(history.history['val_acc']) 
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

def main(tr, tr_labels, val, val_labels, te, te_labels):
    pass

if __name__ == '__main__':
    
    tr = 'data/training/medium/18000.xml'
    tr_labels = 'data/training/medium/18000_labels.xml'
    val = 'data/validation/medium/6000.xml'
    val_labels = 'data/validation/medium/6000_labels.xml'
    te = 'data/test/medium/6000.xml'
    te_labels = 'data/test/medium/6000_labels.xml'
    
    create_word_indexes(tr, tr_labels, val, val_labels, te, te_labels)

    trial = Trials()

    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=trial)

    (X_train, Y_train) , (X_test, Y_test), embedding_matrix = data()

    print("----------trials-------------")
    for i in trials.trials:
        vals = i.get('misc').get('vals')
        results = i.get('result').get('loss')
        print(vals,results)

    print("Evaluation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)