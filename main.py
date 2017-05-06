import argparse
import os.path
import sys
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from keras.wrappers.scikit_learn import KerasRegressor
from scipy import sparse
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import (explained_variance_score, mean_squared_error,
                             r2_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import corpus
import seaborn as sns
from tabulate import tabulate

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
parties = [('CDU/CSU', 5.92),
           ('DIE LINKE', 1.23),
           ('SPD', 3.77),
           ('BÜNDNIS 90/DIE GRÜNEN', 3.61)]

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
    for i, (party, score) in enumerate(parties):
        speeches = corpus.get_by_party(folder, party)
        labels = [score for _ in speeches]

        all_speeches += speeches
        all_labels += labels

    X = np.array(all_speeches)
    y = np.array(all_labels)

    data = Data(*train_test_split(X, y, test_size=test_size))

    return data


def uncentered_f_regression(a, b):
    """
    Needed because lambdas can't be pickled and sparse matrices can't be
    centered.
    """
    return f_regression(a, b, center=False)


def ensure_dense(X, *args, **kwargs):
    """ If the input is a sparse matrix, convert it to a dense one. """
    if sparse.issparse(X):
        # todense() returns a matrix, so convert it to an array
        return np.asarray(X.todense())
    else:
        return X


def create_neuralnet(k):
    """ Create a simple feedforward Keras neural net with k inputs """
    model = Sequential([
        Dense(500, input_dim=k, activation='relu'),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='relu')
    ])

    model.compile(optimizer='rmsprop', loss='mse')
    return model


def create_model(k, epochs, use_keras):
    """
    Return an sklearn pipeline.
    k: the number of features to select
    """

    # preprocessing steps: TFIDF vectorizer, dimensionality reduction,
    # conversion from sparse to dense matrices because Keras doesn't support
    # sparse, and input scaling
    vectorizer = TfidfVectorizer()
    kbest = SelectKBest(uncentered_f_regression, k=k)
    unsparse = FunctionTransformer(ensure_dense, accept_sparse=True)
    scaler = StandardScaler()

    if use_keras:
        model = KerasRegressor(create_neuralnet, k=k, epochs=epochs, batch_size=32)
    else:
        model = MLPRegressor(hidden_layer_sizes=(500,), verbose=True)

    return make_pipeline(vectorizer, kbest, unsparse, scaler, model)


def save_pipeline(model, data, path):
    if 'kerasregressor' in model.named_steps:
        nnet = model.named_steps['kerasregressor'].model
        nnet.save('nnet.h5')
        model.named_steps['kerasregressor'].model = None

    joblib.dump((model, data), path)
    model.named_steps['kerasregressor'].model = nnet


def load_pipeline(path, keras=False):
    model, data = joblib.load(path)

    if keras:
        model.named_steps['kerasregressor'].model = load_model('nnet.h5')

    return model, data


def get_rightness(model, X):
    """
    Return the mean of the predictions, giving a measure of how rightwing the
    given texts are.
    """
    predictions = model.predict(X)
    return np.mean(predictions)


def plot_predictions(names, predictions, filename):
    plt.figure()

    for label, y in zip(names, predictions):
        ax = sns.distplot(y, label=label, axlabel='Political rightness')

    ax.legend()
    fig = ax.get_figure()
    fig.savefig(filename)


def test_parties(model, X, y):
    """ Calculate the average score per party """

    predictions = [model.predict(X[y == label])
                   for _, label in parties]

    rows = [[parties[i][0], np.mean(predictions[i], axis=0), parties[i][1]]
            for i in range(len(parties))]
    print(tabulate(rows, headers=['Average rightness', 'Expected rightness']))

    # sort the parties from left to right based on both the known scores and the
    # predicted scores
    left_to_right = sorted([(np.mean(prediction, axis=0), name) for prediction, (name, _)
                            in zip(predictions, parties)])
    expected = sorted([(rightness, name) for name, rightness
                       in parties])

    print()
    print(f'Sorted from left to right: \t{", ".join([a[1] for a in left_to_right])}')
    print(f'Expected order: \t\t{", ".join([a[1] for a in expected])}')
    print()

    plot_predictions([name for name, _ in parties], predictions, 'parties.png')


def test_newspapers(model):
    paper_table = []
    predictions = [model.predict(corpus.get_newspaper(name))
                   for name in newspapers]

    # print the average predictions
    paper_table = [[name, np.mean(X, axis=0)]
                   for name, X in zip(newspapers, predictions)]
    print(tabulate(paper_table))
    print()

    # plot the histogram
    plot_predictions(newspapers, predictions, 'sources.png')


def get_args():
    parser = argparse.ArgumentParser(description='Predict political left-rightness')
    parser.add_argument('folder', help='folder containing the training data')
    parser.add_argument('-k', type=int, default=50000,
                        help='number of best features to select')

    parser.add_argument('-e', '--epochs', type=int, default=5,
                        help='number of epochs to train for')

    parser.add_argument('--neural_net', '-n', choices=['sklearn', 'keras'],
                        default='sklearn', help='The neural net implementation to use')

    return parser.parse_args()


def main():
    args = get_args()
    k = args.k
    use_keras = True if args.neural_net == 'keras' else False

    model_path = f'model_{args.neural_net}.pkl'

    if os.path.exists(model_path):
        print('Loading model from disk')
        model, data = load_pipeline(model_path, use_keras)
    else:
        data = get_train_test_data(sys.argv[1], 0.20)
        model = create_model(k, args.epochs, use_keras)

        print(f'Training model on data {len(data.X_train)} samples')
        model.fit(data.X_train, data.y_train)
        save_pipeline(model, data, model_path)

    print(f'Testing model on {len(data.y_test)} samples')
    y_predicted = model.predict(data.X_test)

    mse = mean_squared_error(data.y_test, y_predicted)
    var = explained_variance_score(data.y_test, y_predicted)
    r2 = r2_score(data.y_test, y_predicted)

    print()
    print(f'mean squared error on testset: \t{mse:.2f}')
    print(f'explained variance on testset: \t{var:.2f}')
    print(f'r2 score on testset: \t\t{r2:.2f}')
    print()

    test_parties(model, data.X_test, data.y_test)
    test_newspapers(model)


if __name__ == '__main__':
    main()
