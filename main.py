import argparse
import itertools
import os.path
import sys
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize

import corpus
import models
from tabulate import tabulate

LANG = 'de'
parties = []
newspapers = []
newspaper_leanings = []


def set_lang(lang):
    global LANG
    global parties
    global newspapers
    global newspaper_leanings

    LANG = lang

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


def print_best_words(data, labels, k, pipeline):
    print(f'{k} best features:')
    print('---')

    feature_names = pipeline.named_steps['tfidfvectorizer'].get_feature_names()
    indices = pipeline.named_steps['selectkbest'].get_support(indices=True).tolist()
    X = pipeline.transform(data)
    chi_scores, _ = chi2(X, labels)

    idx_scores = sorted(indices, key=lambda i: chi_scores[indices.index(i)], reverse=True)

    for i, idx in enumerate(idx_scores[:k]):
        print(f'{i}.\t{feature_names[idx]}')

    print('---')


def get_train_test_data(folder, test_size, avg_proc=False):
    """
    Return the raw input data and labels, split into training and test data.
    folder: the folder containing the xml files to learn on
    test_size: the ratio of testing data (i.e. between 0 and 1)
    avg_proc: concatenate each party's speeches per proceeding
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

    # filter party and speaker names from the speeches
    names = corpus.get_pronouns(folder)
    names |= set(['grüne', 'sozialdemokraten', 'sozialdemokratinnen',
                  'grünensowie', 'grünenund'])
    X_filtered = []
    for speech in X:
        tokens = word_tokenize(speech)
        X_filtered.append(' '.join(t for t in tokens if t.lower() not in names))

    data = Data(*train_test_split(X_filtered, y, test_size=test_size, random_state=12))

    return data


def test_newspapers(preprocess, model, concat=False):
    counts = []

    for paper in newspapers:
        X = corpus.get_newspaper(paper, concat)
        y = model.predict_proba(preprocess.transform(X))

        # get a normalized histogram
        if not concat:
            mean_prob = np.mean(y, axis=0)
            counts.append(mean_prob)
        else:
            counts.append(y[0])

    if not concat and False:
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

    parser.add_argument('--method', '-m', choices=['keras', 'svm', 'nb',
                                                   'auto', 'cossim'],
                        default='sklearn', help='The method to use')

    parser.add_argument('--dropout', type=float, default=0.25,
                        help='the dropout ratio (between 0 and 1)')

    parser.add_argument('--load_from_disk', '-l', action='store_true',
                        help='load a previously trained model from disk')

    parser.add_argument('--avg_proc', action='store_true',
                        help='average over proceedings')

    parser.add_argument('--avg_paper', action='store_true',
                        help='average over newspapers')

    lang = parser.add_mutually_exclusive_group(required=True)
    lang.add_argument('-nl', action='store_true')
    lang.add_argument('-de', action='store_true')

    return parser.parse_args()


def main():
    args = get_args()
    k = args.k

    # set language
    global LANG
    if args.nl:
        set_lang('nl')
    else:
        set_lang('de')

    model_path = f'model_{args.method}_{LANG}.pkl'
    data = get_train_test_data(sys.argv[1], 0.20, args.avg_proc)

    if os.path.exists(model_path) and args.load_from_disk:
        print('Loading model from disk')
        preprocess, model = models.load_pipeline(model_path, args.method)
    else:
        if args.method == 'svm':
            preprocess, model = models.create_svm_model(k)
        elif args.method == 'nb':
            preprocess, model = models.create_nb_model(k)
        elif args.method == 'auto':
            preprocess, model = models.create_auto_model(k)
        else:
            preprocess, model = models.create_model(k, args.epochs,
                                                    args.dropout, len(parties))

        print(f'Training model on data {len(data.X_train)} samples')
        X_trans = preprocess.fit_transform(data.X_train, data.y_train)
        model.fit(X_trans, data.y_train)
        models.save_pipeline(preprocess, model, model_path, args.method)

    if args.method != 'keras':
        print_best_words(data.X_train, data.y_train, 20, preprocess)

    print(f'Testing model on {len(data.y_test)} samples')
    y_predicted = model.predict(preprocess.transform(data.X_test))

    acc = accuracy_score(data.y_test, y_predicted)
    plot_confusion_matrix(model, data.y_test, y_predicted)

    if args.method == 'auto':
        print(model.show_models())

    print()
    print(f'accuracy on testset: \t{acc:.2f}')
    print()

    test_newspapers(preprocess, model, args.avg_paper)


if __name__ == '__main__':
    main()
