import argparse
import os.path
import sys
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from scipy import sparse
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

import corpus
import seaborn as sns

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

newspapers = ['diewelt', 'taz', 'spiegel']

# convenient datastructure to hold training and test data
Data = namedtuple('Data', ['X_train', 'X_test', 'y_train', 'y_test'])


def get_train_test_data(folder, test_size):
    """
    Return the raw input data and labels, split into training and test data.
    folder: the folder containing the xml files to learn on
    test_size: the ratio of testing data (i.e. between 0 and 1)
    """
    all_speeches = []
    all_labels = []
    for i, party in enumerate(parties):
        speeches = corpus.get_by_party(folder, party)
        labels = [i for _ in speeches]

        all_speeches += speeches
        all_labels += labels

    X = np.array(all_speeches)
    y = np.array(all_labels)

    data = Data(*train_test_split(X, y, test_size=test_size))

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


def create_model(k, epochs, dropout, use_keras):
    """
    Return an sklearn pipeline.
    k: the number of features to select
    """

    # preprocessing steps: TFIDF vectorizer, dimensionality reduction,
    # conversion from sparse to dense matrices because Keras doesn't support
    # sparse, and input scaling
    vectorizer = TfidfVectorizer()
    kbest = SelectKBest(chi2, k=k)
    unsparse = FunctionTransformer(ensure_dense, accept_sparse=True)
    scaler = StandardScaler()

    if use_keras:
        model = KerasClassifier(create_neuralnet, k=k, dropout=dropout,
                                epochs=epochs, batch_size=32)
    else:
        model = LinearSVC()

    return make_pipeline(vectorizer, kbest, unsparse, scaler, model)


def save_pipeline(model, data, path):
    if 'kerasclassifier' in model.named_steps:
        nnet = model.named_steps['kerasclassifier'].model
        nnet.save('nnet.h5')
        model.named_steps['kerasclassifier'].model = None
        joblib.dump((model, data), path)
        model.named_steps['kerasclassifier'].model = nnet

    else:
        joblib.dump((model, data), path)


def load_pipeline(path, keras=False):
    model, data = joblib.load(path)

    if keras:
        model.named_steps['kerasclassifier'].model = load_model('nnet.h5')

    return model, data


def test_newspapers(model):
    for paper in newspapers:
        plt.figure()
        X = corpus.get_newspaper(paper)
        y = model.predict(X)
        sns.barplot(parties, y)
        plt.savefig(f'classification_{paper}.png')


def get_args():
    parser = argparse.ArgumentParser(description='Predict political left-rightness')
    parser.add_argument('folder', help='folder containing the training data')
    parser.add_argument('-k', type=int, default=50000,
                        help='number of best features to select')

    parser.add_argument('-e', '--epochs', type=int, default=5,
                        help='number of epochs to train for')

    parser.add_argument('--neural_net', '-n', choices=['sklearn', 'keras'],
                        default='sklearn', help='The neural net implementation to use')

    parser.add_argument('--dropout', type=float, default=0.2,
                        help='the dropout ratio (between 0 and 1)')

    parser.add_argument('--load_from_disk', '-l', action='store_true',
                        help='load a previously trained model from disk')

    return parser.parse_args()


def main():
    args = get_args()
    k = args.k
    use_keras = True if args.neural_net == 'keras' else False

    if use_keras:
        data_transform = to_categorical
    else:
        data_transform = lambda x: x

    model_path = f'model_{args.neural_net}.pkl'

    if os.path.exists(model_path) and args.load_from_disk:
        print('Loading model from disk')
        model, data = load_pipeline(model_path, use_keras)
    else:
        data = get_train_test_data(sys.argv[1], 0.20)
        model = create_model(k, args.epochs, args.dropout, use_keras)

        print(f'Training model on data {len(data.X_train)} samples')
        model.fit(data.X_train, data_transform(data.y_train))
        save_pipeline(model, data, model_path)

    print(f'Testing model on {len(data.y_test)} samples')
    y_predicted = model.predict(data.X_test)

    acc = accuracy_score(data.y_test, y_predicted)

    print()
    print(f'accuracy on testset: \t{acc:.2f}')
    print()

    test_newspapers(model)


if __name__ == '__main__':
    main()
