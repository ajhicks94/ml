#!/usr/bin/env python

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

from hp import create_word_index, get_pretrained_embeddings, load_data


def print_usage(filename, message):
    print(message)
    print("Usage: python %s --training_data <file> --training_labels <file> --validation_data <file> --validation_labels <file> --test_data <file> --test_labels <file>" % filename)

########## OPTIONS HANDLING ##########
def parse_options():
    """Parses the command line options."""
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

def main(tr, tr_labels, val, val_labels, te, te_labels):
    
    
    start = time.time()
    create_word_indexes(tr, tr_labels, val, val_labels, te, te_labels)
    finish = time.time()
    print("Building word indexes:", finish-start)

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
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(tr, tr_labels, val, val_labels, te, te_labels,
                                                     skip_top=skip_top, num_words=num_words, maxlen=None)
    finish = time.time()
    print("Load_data:", finish-start)
    
    # ML Stuff now
    print(len(x_train), 'train sequences')
    print(len(x_val), 'validation sequences')
    print(len(x_test), 'test sequences\n')

    print('Pad sequences...', end='')
    start = time.time()
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_val = sequence.pad_sequences(x_val, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    finish = time.time()
    print(finish-start)

    batch_size = config['batch_size']               # Number of instances before updating weights    #default 32
    epochs = config['epochs']                       # Number of epochs                               #default 15

    go_backwards = True if (config['go_backwards'] == "True") else False
    dropout = config['dropout']
    recurrent_dropout = config['recurrent_dropout']
    #bias_regularizer = config['bias_regularizer']

    # Word Embeddings
    start = time.time()
    embedding_matrix = get_pretrained_embeddings(  'data/word_indexes/training.json',
                                                    'data/embeddings/GoogleNews-vectors-negative300.bin')
    finish = time.time()
    print(finish-start)

    with open('data/word_indexes/training.json', 'r') as f:
        word_index = {}
        word_index = json.load(f)


    EMBEDDING_DIM = 300
    
    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                        EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=maxlen,
                        trainable=False))

    model.add(LSTM( 128, #bias_regularizer = bias_regularizer, 
                    dropout=dropout, recurrent_dropout=recurrent_dropout, 
                    go_backwards=go_backwards))

    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
 
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

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

    # Use string interpolation here instead of this nonsense
    plot_prefix = str(int(tr.split('/')[-1].split('.')[0]) + int(val.split('/')[-1].split('.')[0]) + int(te.split('/')[-1].split('.')[0]))
    #plot_prefix = plot_prefix.split('.')[0]
    plot_name = plot_prefix + '.png'
    plot_name = 'results/runs/' + plot_name
    plot_config = 'results/runs/' + plot_prefix + '.config'

    i = 1
    jpg = '.jpg'
    conf = '.config'
    # Just use a simple loop and check if 'filename_%s.jpg' % i exists
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
        json.dump(config, f, indent=4)
        f.write('\nModel Configuration\n')
        f.write(model.to_json(indent=4))

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
