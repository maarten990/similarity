import argparse
import itertools
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
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

import corpus
import seaborn as sns
from tabulate import tabulate
# import treetaggerwrapper

LANG = 'de'

if LANG == 'de':
    parties = ['DIE LINKE', 'BÜNDNIS 90/DIE GRÜNEN', 'SPD', 'CDU/CSU']
    newspapers = ['diewelt', 'taz', 'spiegel', 'rheinischepost', 'diezeit']
    newspaper_leanings = ['CDU/conversative right', 'left', 'SPD/center left',
                          'CDU/CSU', 'FDP/center left']

if LANG == 'nl':
    parties = ['CDA', 'ChristenUnie', 'D66', 'GroenLinks', 'PVV',
               'PvdA', 'SGP', 'SP', 'VVD']
    newspapers = ['telegraaf', 'trouw', 'volkskrant']
    newspaper_leanings = ['eh', 'eh', 'eh']

# convenient datastructure to hold training and test data
Data = namedtuple('Data', ['X_train', 'X_test', 'y_train', 'y_test'])


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


def get_train_test_data(folder, test_size, avg_proc=False):
    """
    Return the raw input data and labels, split into training and test data.
    folder: the folder containing the xml files to learn on
    test_size: the ratio of testing data (i.e. between 0 and 1)
    """
    all_speeches = []
    all_labels = []
    for i, party in enumerate(parties):
        if LANG == 'de':
            speeches = corpus.get_by_party(folder, party, avg_proc)
        if LANG == 'nl':
            speeches = corpus.get_dutch_proceedings('proceedings_NL', party, avg_proc)

        if len(speeches) > 5000:
            speeches = speeches[:5000]
        print(f'{party}: {len(speeches)} speeches')
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
        Dense(500, input_dim=k, activation='tanh'),
        Dropout(dropout),
        Dense(50, activation='tanh'),
        Dropout(dropout),
        Dense(len(parties), activation='softmax')
    ])

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def create_svm_model(k):
    vectorizer = TfidfVectorizer()
    kbest = SelectKBest(chi2, k=k)
    scaler = StandardScaler(with_mean=False)
    svc = SVC(probability=True)

    return make_pipeline(vectorizer, kbest, scaler), svc


def create_model(k, epochs, dropout, only_nouns):
    """
    Return an sklearn pipeline.
    k: the number of features to select
    """

    # preprocessing steps: TFIDF vectorizer, dimensionality reduction,
    # conversion from sparse to dense matrices because Keras doesn't support
    # sparse, and input scaling
    # tagger = POSTaggerTransformer(tagfilter=ContainsAllBut('NN') if only_nouns else [])
    vectorizer = TfidfVectorizer()
    kbest = SelectKBest(chi2, k=k)
    unsparse = FunctionTransformer(ensure_dense, accept_sparse=True)
    scaler = StandardScaler()

    model = KerasClassifier(create_neuralnet, k=k, dropout=dropout,
                            epochs=epochs, batch_size=32)

    return make_pipeline(vectorizer, kbest, unsparse, scaler), model


def save_pipeline(pipeline, model, path, name):
    if 'postaggertransformer' in pipeline.named_steps:
        pipeline.named_steps['postaggertransformer'].tagger.taggerlock = None

    if isinstance(model, KerasClassifier):
        nnet = model.named_steps['kerasclassifier'].model
        nnet.save(f'{name}.h5')
        model.model = None

    joblib.dump((pipeline, model), path)

    if 'postaggertransformer' in pipeline.named_steps:
        pipeline.named_steps['postaggertransformer'].tagger.taggerlock = threading.Lock()

    if isinstance(model, KerasClassifier):
        model.model = nnet


def load_pipeline(path, name):
    pipeline, model = joblib.load(path)
    if isinstance(model, KerasClassifier):
        model.model = load_model(f'{name}.h5')

    if 'postaggertransformer' in pipeline.named_steps:
        pipeline.named_steps['postaggertransformer'].tagger.taggerlock = threading.Lock()

    return pipeline, model


def test_newspapers(preprocess, model, concat=False):
    counts = []

    for paper in newspapers:
        plt.figure()
        X = corpus.get_newspaper(paper, concat)
        y = model.predict_proba(preprocess.transform(X))

        # get a normalized histogram
        if not concat:
            h, _ = np.histogram(y, bins=list(range(len(parties) + 1)), density=True)
            counts.append(h)
        else:
            counts.append(y[0])

    if not concat:
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
    else:
        # print the deviations in table form
        rows = []
        for i in range(len(counts)):
            row = counts[i]
            rows.append([newspapers[i]] + row.tolist() + [newspaper_leanings[i]])

        print()
        print(tabulate(rows, headers=parties + ['expected leaning'], floatfmt=".2f"))


def plot_confusion_matrix(model, y_true, y_predicted):
    cm = confusion_matrix(y_true, y_predicted)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(parties))
    plt.xticks(tick_marks, parties, rotation=45)
    plt.yticks(tick_marks, parties)

    # normalize the confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # ensure the text is readable
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.1f}'.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('true label')
    plt.xlabel('predicted label')
    plt.savefig('confusion_matrix.png')


def get_args():
    parser = argparse.ArgumentParser(description='Predict political left-rightness')
    parser.add_argument('folder', help='folder containing the training data')

    parser.add_argument('-k', type=int, default=50000,
                        help='number of best features to select')

    parser.add_argument('-e', '--epochs', type=int, default=5,
                        help='number of epochs to train for')

    parser.add_argument('--neural_net', '-n', choices=['keras', 'svm'],
                        default='sklearn', help='The neural net implementation to use')

    parser.add_argument('--dropout', type=float, default=0.25,
                        help='the dropout ratio (between 0 and 1)')

    parser.add_argument('--load_from_disk', '-l', action='store_true',
                        help='load a previously trained model from disk')

    parser.add_argument('--only_nouns', action='store_true',
                        help='only use nouns')

    parser.add_argument('--avg_proc', action='store_true',
                        help='average over proceedings')

    return parser.parse_args()


def main():
    args = get_args()
    k = args.k

    model_path = f'model_{args.neural_net}.pkl'
    data = get_train_test_data(sys.argv[1], 0.20, args.avg_proc)

    if os.path.exists(model_path) and args.load_from_disk:
        print('Loading model from disk')
        preprocess, model = load_pipeline(model_path, args.neural_net)

        if args.retrain:
            model.fit(data.X_train, to_categorical(data.y_train))
            save_pipeline(preprocess, model, model_path, args.neural_net + args.only_nouns)
    else:
        if args.neural_net == 'svm':
            preprocess, model = create_svm_model(k)
        else:
            preprocess, model = create_model(k, args.epochs, args.dropout, args.only_nouns)

        print(f'Training model on data {len(data.X_train)} samples')
        X_trans = preprocess.fit_transform(data.X_train, data.y_train)
        model.fit(X_trans, data.y_train)
        save_pipeline(preprocess, model, model_path, args.neural_net)
        # print_best_words(model)

    print(f'Testing model on {len(data.y_test)} samples')
    y_predicted = model.predict(preprocess.transform(data.X_test))

    acc = accuracy_score(data.y_test, y_predicted)
    plot_confusion_matrix(model, data.y_test, y_predicted)

    print()
    print(f'accuracy on testset: \t{acc:.2f}')
    print()

    test_newspapers(preprocess, model, args.avg_proc)


if __name__ == '__main__':
    main()
