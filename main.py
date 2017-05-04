import os.path
import sys
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import (explained_variance_score, mean_squared_error,
                             r2_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasRegressor

import corpus
from tabulate import tabulate


plt.style.use('ggplot')


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

newspapers = ['taz']

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
    """ Needed because lambda's can't be pickled. """
    return f_regression(a, b, center=False)


def create_neuralnet(k):
    model = Sequential([
        Dense(500, input_dim=k, dropout=0.2),
        Dense(1, dropout=0.2),
        Activation('relu'),
    ])

    model.compile(optimizer='rmsprop', loss='mse')
    return model


def create_model(k):
    """
    Return an sklearn pipeline.
    k: the number of features to select
    """

    # preprocessing steps: TFID vectorizers and dimensionality reducting
    vectorizer = TfidfVectorizer()
    kbest = SelectKBest(uncentered_f_regression, k=k)

    # model = LinearSVR()
    model = MLPRegressor(hidden_layer_sizes=(500,), verbose=True)
    # model = KerasRegressor(create_neuralnet, k=k, epochs=10, batch_size=32)

    return make_pipeline(vectorizer, kbest, model)


def get_rightness(model, X):
    """
    Return the mean of the predictions, giving a measure of how rightwing the
    given texts are.
    """
    predictions = model.predict(X)
    return np.mean(predictions)


def plot_party_predictions(model, X, y):
    plt.figure()

    for party, label in parties:
        # get this party's predictions
        predictions = model.predict(X[y == label])

        # create and plot histogram
        count, bin_edges = np.histogram(predictions, bins=20, density=True)
        bincenters = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        plt.plot(bincenters, count, label=party)

    plt.legend()
    plt.xlabel('score')
    plt.ylabel('count')
    plt.savefig('histogram.png')


def predict_party_rightness(model, X, y):
    """ Calculate the average score per party """

    rightnesses = [get_rightness(model, X[y == label])
                   for _, label in parties]

    # get the data in a nice table for pretty printing
    rows = [[parties[i][0], rightnesses[i], parties[i][1]]
            for i in range(len(parties))]
    print(tabulate(rows, headers=['Average rightness', 'Expected rightness']))

    # sort the parties from left to right based on both the known scores and the
    # predicted scores
    left_to_right = sorted([(rightness, name) for rightness, (name, _)
                            in zip(rightnesses, parties)])
    expected = sorted([(rightness, name) for name, rightness
                       in parties])

    print()
    print(f'Sorted from left to right: \t{", ".join([a[1] for a in left_to_right])}')
    print(f'Expected order: \t\t{", ".join([a[1] for a in expected])}')
    print()


def test_newspapers(model):
    paper_table = []
    for newspaper in newspapers:
        X = corpus.get_newspaper(newspaper)
        rightness = get_rightness(model, X)
        paper_table.append([newspaper, rightness])

    print(tabulate(paper_table))
    print()


def main():
    k = 50000

    model_path = 'model.pkl'

    if os.path.exists(model_path):
        print('Loading model from disk')
        model, data = joblib.load(model_path)
    else:
        data = get_train_test_data(sys.argv[1], 0.20)
        model = create_model(k)

        print(f'Training model on data {len(data.X_train)} samples')
        model.fit(data.X_train, data.y_train)
        joblib.dump((model, data), model_path)

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

    predict_party_rightness(model, data.X_test, data.y_test)
    plot_party_predictions(model, data.X_test, data.y_test)
    test_newspapers(model)


if __name__ == '__main__':
    main()
