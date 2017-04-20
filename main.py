import sys
import corpus

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error

from tabulate import tabulate
from collections import namedtuple
import numpy as np
import scipy
import matplotlib.pyplot as plt
from IPython import embed

plt.style.use('ggplot')


# boek
#parties = [('CDU/CSU', 13.6),
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

newspapers = ['Die Welt', 'Frankfurter Neue Presse', 'Taz, die Tageszeitung']

# convenient datastructure to hold training and test data
Data = namedtuple('Data', ['X_train', 'X_test', 'y_train', 'y_test'])

def get_train_test_data(folder, test_size):
    """
    Return the raw input data and labels, split into training and test data.
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


def create_model(k):
    """
    Return an sklearn pipeline.
    k: the number of features to select
    """

    # preprocessing steps: TFID vectorizers and dimensionality reducting
    vectorizer = TfidfVectorizer()
    kbest = SelectKBest(lambda a, b: f_regression(a, b, center=False), k=k)

    # The linear support vector regression seems to offer both the best
    # performance and quickest training, but can probably be improved upon with
    # some more experimentation.
    model = LinearSVR()

    return make_pipeline(vectorizer, kbest, model)
    

def get_rightness(model, X):
    """ Return the mean of the predictions """
    predictions = model.predict(X)
    return np.mean(predictions)


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


def test_k():
    """ Plot the MSE for selecting the K best features """
    mses = []
    ks = [10, 100, 500, 1000, 2500, 5000, 7500, 10000, 50000, 100000]
    for k in ks:
        data = get_train_test_data(sys.argv[1], k)
        model.fit(data.X_train, data.y_train)
        y_predicted = model.predict(data.X_test)

        mses.append(mean_squared_error(data.y_test, y_predicted))

    plt.plot(ks, mses, marker='o')
    plt.xlabel('K best features')
    plt.ylabel('Mean Squared Error')
    plt.show()


def test_newspapers(model):
    paper_table = []
    for newspaper in newspapers:
        X = corpus.get_newspaper(newspaper)
        rightness = get_rightness(model, X)
        paper_table.append([newspaper, rightness])

    print(tabulate(paper_table))
    print()


def main():
    if len(sys.argv) > 2 and sys.argv[2] == '--testk':
        test_k()
    else:
        k = 50000

    model = create_model(k)
    data = get_train_test_data(sys.argv[1], 0.20)

    print(f'Training model on data {len(data.X_train)} samples')
    model.fit(data.X_train, data.y_train)

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
    test_newspapers(model)


if __name__ == '__main__':
    main()
