'''Trains an LSTM model on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
# Notes
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

from keras_preprocessing import sequence

import sys
import numpy as np
import json
import warnings

def load_data(path='imdb.npz', num_words=None, skip_top=0, maxlen=None, seed=113,
                    start_char=1, oov_char=2, index_from=3, **kwargs):

    # load data from data.json
    # since it is already in order, preserve order w/ OrderedDict
    # load into OrderedDict
    # in order to get index by the key, use list(odict).index(word)
    # odict_list = list(odict)
    # for w in words: odict_list.index(word)

    # Start with numpy arrays
    with np.load(path) as f:
        x_train, labels_train = f['x_train'], f['y_train']
        x_test, labels_test = f['x_test'], f['y_test']

    
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

    # Do the same to the training labels
    labels_train = labels_train[indices]

    # Do all the same stuff for the test(validation?) set
    indices = np.arange(len(x_test))
    np.random.shuffle(indices)
    x_test = x_test[indices]
    labels_test = labels_test[indices]

    # Append x_test to x_train
    xs = np.concatenate([x_train, x_test])
    # And for the labels
    labels = np.concatenate([labels_train, labels_test])

    if start_char is not None:
        # Adds a start_char to the beginning of each sentence
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        # for each word_index in each sentence, add index_from to the value
        # Maybe since the word_index is sorted by count, this omits the
        #     top index_from most frequent words?
        xs = [[w + index_from for w in x] for x in xs]

    # Trims sentences down to maxlen
    if maxlen:
        xs, labels = _remove_long_seq(maxlen, xs, labels)
        if not xs:
            raise ValueError('After filtering for sequences shorter than maxlen=' +
                             str(maxlen) + ', no sequence was kept. '
                             'Increase maxlen.')
    # Calculates the max val in xs
    # Which means the least frequent term if word_index is sorted by desc freq
    if not num_words:
        num_words = max([max(x) for x in xs])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        # If a word is out-of-vocab, replace it w/ '2'/oov_char
        # Also remove any words that are less-than skip_top
        xs = [[w if (skip_top <= w < num_words) else oov_char for w in x]
              for x in xs]
    else:
        # Just remove words that are less-than skip_top
        xs = [[w for w in x if skip_top <= w < num_words]
              for x in xs]

    # Remember x_train only had 3 sentences, right?
    idx = len(x_train)

    # Partition the newly preprocessed training instances back into x_train
    x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])

    # Same for test
    x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])

    # Return tuple of training and tuple of test
    return (x_train, y_train), (x_test, y_test)


def get_word_index(path='imdb_word_index.json'):
    """Retrieves the dictionary mapping words to word indices.
    # Arguments
        path: where to cache the data (relative to `~/.keras/dataset`).
    # Returns
        The word index dictionary.
    """
    # Get_file just downloads a file from a URL
    # it returns the path to the downloaded file
    path = get_file(
        path,
        origin='https://s3.amazonaws.com/text-datasets/imdb_word_index.json',
        file_hash='bfafd718b763782e994055a2d397834f')
    with open(path) as f:
        return json.load(f)

if __name__ == "__main__":
    max_features = 20000
    # cut texts after this number of words (among top max_features most common words)
    maxlen = 80
    batch_size = 32

    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    wid = imdb.get_word_index()
    print(wid)
    id2w = {i: word for word, i in wid.items()}
    print([id2w.get(i, ' ') for i in x_train[0]])
    print(x_train[0])
    print('Positive? (1 is yes): ', y_train[0])
    #print(imdb.get_word_index())
    sys.exit()
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
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=1, #epochs=15
            validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)