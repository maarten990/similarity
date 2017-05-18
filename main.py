import argparse
import os.path
import pickle
import re
import sys
import threading
import zlib
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers import Conv1D, Embedding, MaxPool1D
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from scipy import sparse
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import corpus
import seaborn as sns
from tabulate import tabulate
import treetaggerwrapper

# alternatieve waarden voor links/rechtsheid genomen uit
# http://www.nyu.edu/gsas/dept/politics/faculty/laver/PPMD_draft.pdf
# parties = [('CDU/CSU', 13.6),
#           ('DIE LINKE', 3.6),
#           ('SPD', 8.4),
#           ('BÜNDNIS 90/DIE GRÜNEN', 7.1)]

# The available political parties, with a score indicating their political leaning.
# 0 = extreme left
# 10 = extreme right
# source: Chapel Hill Expert Survey (www.chesdata.eu)
parties = ['DIE LINKE', 'BÜNDNIS 90/DIE GRÜNEN', 'SPD', 'CDU/CSU']

newspapers = ['diewelt', 'taz', 'spiegel', 'rheinischepost', 'diezeit']
newspaper_leanings = ['CDU/conversative right', 'left', 'SPD/center left',
                      'CDU/CSU', 'FDP/center left']

# convenient datastructure to hold training and test data
Data = namedtuple('Data', ['X_train', 'X_test', 'y_train', 'y_test'])


class TokenizingTransformer(object):
    def __init__(self, max_words, vocab_size):
        self.tokenizer = Tokenizer(vocab_size)
        self.max_words = max_words

    def fit(self, X, y=None):
        self.tokenizer.fit_on_texts(X)
        return self

    def transform(self, X):
        tokens = self.tokenizer.texts_to_sequences(X)
        return pad_sequences(tokens, maxlen=self.max_words)


class POSTaggerTransformer(object):
    def __init__(self, lemmatize=True, tagfilter=[]):
        self.tagger = treetaggerwrapper.TreeTagger(TAGLANG='de',
                                                   TAGDIR='treetagger/tree-tagger-MacOSX-3.2')
        self.lemmatize = lemmatize
        self.filter = tagfilter
        self.first_transform = True

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        checksum = zlib.adler32(bytes(str(tuple(X)), 'UTF-8'))
        pickle_name = f'tagged_{"".join(self.filter)}_{checksum}.pkl'

        # check if the initial corpus has been saved
        if os.path.exists(pickle_name):
            with open(pickle_name, 'rb') as f:
                return pickle.load(f)

        out = []
        for i, speech in enumerate(X):
            if i % 1000 == 0:
                print(f'{i}/{len(X)}')

            speech = re.sub(r'\.([^ ])', r'. \1', speech)
            tags = treetaggerwrapper.make_tags(self.tagger.tag_text(speech))
            words = []
            for tag in tags:
                if isinstance(tag, treetaggerwrapper.NotTag):
                    print(tag)
                    continue

                if tag.pos not in self.filter:
                    words.append(tag.lemma if self.lemmatize else tag.word)

            out.append(' '.join(words))

        with open(pickle_name, 'wb') as f:
            pickle.dump(out, f)

        return out


class ContainsAllBut(object):
    def __init__(self, item):
        self.item = item

    def __contains__(self, item):
        return item != self.item

    def __iter__(self):
        self.done = False
        return self

    def __next__(self):
        if self.done:
            raise StopIteration

        self.done = True
        return 'Everything!'


def print_best_words(pipeline):
    print('k best features:')
    print('---')

    feature_names = pipeline.named_steps['tfidfvectorizer'].get_feature_names()
    for feature in pipeline.named_steps['selectkbest'].get_support(indices=True):
        print(feature_names[feature])

    print('---')


def get_train_test_data(folder, test_size, max_words=None):
    """
    Return the raw input data and labels, split into training and test data.
    folder: the folder containing the xml files to learn on
    test_size: the ratio of testing data (i.e. between 0 and 1)
    """
    all_speeches = []
    all_labels = []
    for i, party in enumerate(parties):
        speeches = corpus.get_by_party(folder, party, max_words)
        labels = [i for _ in speeches]

        all_speeches += speeches
        all_labels += labels

    X = np.array(all_speeches)
    y = np.array(all_labels)

    data = Data(*train_test_split(X, y, test_size=test_size, random_state=12))

    return data


def ensure_dense(X, *args, **kwargs):
    """ If the input is a sparse matrix, convert it to a dense one. """
    if sparse.issparse(X):
        # todense() returns a matrix, so convert it to an array
        return np.asarray(X.todense())
    else:
        return X


def create_neuralnet(k, dropout):
    """ Create a simple feedforward Keras neural net with k inputs """
    model = Sequential([
        Dense(500, input_dim=k, activation='relu'),
        Dropout(dropout),
        Dense(50, activation='relu'),
        Dropout(dropout),
        Dense(len(parties), activation='softmax')
    ])

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def create_rnn(timesteps, n, dropout):
    model = Sequential([
        Embedding(n, 64, input_length=timesteps, mask_zero=True),
        LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
        Dense(4),
        Activation('softmax')
    ])

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def create_cnn(timesteps, n, dropout):
    model = Sequential([
        Embedding(n, 128, input_length=timesteps, mask_zero=False),
        Conv1D(100, 5, activation='relu'),
        MaxPool1D(2),
        Dropout(dropout),

        Flatten(),
        Dense(4),
        Activation('softmax')
    ])

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def create_deep_model(func, timesteps, n, epochs, dropout):
    embedder = TokenizingTransformer(timesteps, n)
    model = KerasClassifier(func, timesteps=timesteps, n=n, dropout=dropout,
                            epochs=epochs, batch_size=32)

    return make_pipeline(embedder, model)


def create_model(k, epochs, dropout, only_nn):
    """
    Return an sklearn pipeline.
    k: the number of features to select
    """

    # preprocessing steps: TFIDF vectorizer, dimensionality reduction,
    # conversion from sparse to dense matrices because Keras doesn't support
    # sparse, and input scaling
    tagger = POSTaggerTransformer(tagfilter=ContainsAllBut('NN') if only_nn else [])
    vectorizer = TfidfVectorizer()
    kbest = SelectKBest(chi2, k=k)
    unsparse = FunctionTransformer(ensure_dense, accept_sparse=True)
    scaler = StandardScaler()

    model = KerasClassifier(create_neuralnet, k=k, dropout=dropout,
                            epochs=epochs, batch_size=32)

    return make_pipeline(tagger, vectorizer, kbest, unsparse, scaler, model)


def save_pipeline(model, data, path, name):
    if 'postaggertransformer' in model.named_steps:
        model.named_steps['postaggertransformer'].tagger.taggerlock = None

    nnet = model.named_steps['kerasclassifier'].model
    nnet.save(f'{name}.h5')
    model.named_steps['kerasclassifier'].model = None
    joblib.dump((model, data), path)
    model.named_steps['kerasclassifier'].model = nnet

    if 'postaggertransformer' in model.named_steps:
        model.named_steps['postaggertransformer'].tagger.taggerlock = threading.Lock()


def load_pipeline(path, name):
    model, data = joblib.load(path)
    model.named_steps['kerasclassifier'].model = load_model(f'{name}.h5')

    if 'postaggertransformer' in model.named_steps:
        model.named_steps['postaggertransformer'].tagger.taggerlock = threading.Lock()

    return model, data


def test_newspapers(model):
    counts = []

    for paper in newspapers:
        plt.figure()
        X = corpus.get_newspaper(paper)
        y = model.predict(X)

        # get a normalized histogram
        h, _ = np.histogram(y, bins=list(range(len(parties) + 1)), density=True)
        counts.append(h)

    counts = np.array(counts, dtype='float64')
    means = np.mean(counts, axis=0)

    # print the deviations in table form
    rows = []
    for i in range(np.shape(counts)[0]):
        row = counts[i, :]

        # do a significance test
        _, p = scipy.stats.chisquare(row * 100, means * 100)

        row = ((row - means) / means) * 100
        rows.append([newspapers[i]] + row.tolist() + [newspaper_leanings[i], p])

    print()
    print('Percentage increase over mean per party')
    print(tabulate(rows, headers=parties + ['expected leaning', 'p'], floatfmt=".3f"))


def get_args():
    parser = argparse.ArgumentParser(description='Predict political left-rightness')
    parser.add_argument('folder', help='folder containing the training data')
    parser.add_argument('-k', type=int, default=50000,
                        help='number of best features to select')

    parser.add_argument('-e', '--epochs', type=int, default=5,
                        help='number of epochs to train for')

    parser.add_argument('--neural_net', '-n', choices=['keras', 'cnn', 'rnn'],
                        default='sklearn', help='The neural net implementation to use')

    parser.add_argument('--dropout', type=float, default=0.25,
                        help='the dropout ratio (between 0 and 1)')

    parser.add_argument('--load_from_disk', '-l', action='store_true',
                        help='load a previously trained model from disk')

    parser.add_argument('--retrain', action='store_true',
                        help='retrain after loading')

    parser.add_argument('--only_nn', action='store_true',
                        help='only use nouns')

    parser.add_argument('--max_words', '-m', type=int, default=None,
                        help='the maximum number of words to use')

    return parser.parse_args()


def main():
    args = get_args()
    k = args.k

    model_path = f'model_{args.neural_net}.pkl'

    if os.path.exists(model_path) and args.load_from_disk:
        print('Loading model from disk')
        model, data = load_pipeline(model_path, args.neural_net)

        if args.retrain:
            model.fit(data.X_train, to_categorical(data.y_train))
            save_pipeline(model, data, model_path, args.neural_net + args.only_nn)
    else:
        data = get_train_test_data(sys.argv[1], 0.20, args.max_words)

        if args.neural_net == 'cnn':
            model = create_deep_model(create_cnn, args.max_words, 10000,
                                      args.epochs, args.dropout)
        elif args.neural_net == 'rnn':
            model = create_deep_model(create_rnn, args.max_words, 10000,
                                      args.epochs, args.dropout)
        else:
            model = create_model(k, args.epochs, args.dropout, args.only_nn)

        print(f'Training model on data {len(data.X_train)} samples')
        model.fit(data.X_train, data.y_train)
        print_best_words(model)
        save_pipeline(model, data, model_path, args.neural_net)

    print(f'Testing model on {len(data.y_test)} samples')
    y_predicted = model.predict(data.X_test)

    acc = accuracy_score(data.y_test, y_predicted)

    print()
    print(f'accuracy on testset: \t{acc:.2f}')
    print()

    test_newspapers(model)


if __name__ == '__main__':
    main()
