import argparse
import itertools
import os.path
import pickle
import sys
from collections import namedtuple, Counter
from operator import itemgetter

import gensim
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem.snowball import GermanStemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

import corpus
import models
import requests
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
        newspapers = ['diewelt', 'taz', 'spiegel', 'rheinischepost', 'diezeit', 'bild']
        newspaper_leanings = ['CDU/conversative right', 'left', 'SPD/center left',
                              'CDU/CSU', 'FDP/center left', 'right']

    if LANG == 'nl':
        parties = ['CDA', 'ChristenUnie', 'D66', 'GroenLinks', 'PVV',
                   'PvdA', 'SGP', 'SP', 'VVD']
        newspapers = ['telegraaf', 'trouw', 'volkskrant']
        newspaper_leanings = ['eh', 'eh', 'eh']


# convenient datastructure to hold training and test data
Data = namedtuple('Data', ['X_train', 'X_test', 'y_train', 'y_test'])


def message(msg):
    requests.post('http://maarten.sexy:5000/',
                  {'message': msg})


def pickle_results(func):
    def pickled_func(*args, **kwargs):
        if 'pkl_path' not in kwargs:
            return func(*args, **kwargs)
        else:
            path = kwargs['pkl_path']

            if os.path.exists(path):
                print(f'Loading {path} from disk')
                with open(path, 'rb') as f:
                    return pickle.load(f)

            output = func(*args, **kwargs)
            print(f'Saving {path} to disk')
            with open(path, 'wb') as f:
                pickle.dump(output, f)

            return output

    return pickled_func


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


def get_train_test_data(folder, test_size, avg_proc=False, filter_pronouns=False):
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
    if filter_pronouns:
        names = corpus.get_pronouns(folder)
        names |= set(['grüne', 'sozialdemokraten', 'sozialdemokratinnen',
                      'grünensowie', 'grünenund'])
        X_filtered = []
        for speech in X:
            tokens = word_tokenize(speech)
            X_filtered.append(' '.join(t for t in tokens if t.lower() not in names))

        data = Data(*train_test_split(X_filtered, y, test_size=test_size, random_state=12))
    else:
        data = Data(*train_test_split(X, y, test_size=test_size, random_state=12))

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


@pickle_results
def cosine_preprocess(texts, pkl_path=None):
    processed = []
    for text in tqdm(texts):
        stemmer = GermanStemmer()
        words = stopwords.words('german')

        tokens = [stemmer.stem(token) for token in word_tokenize(text)
                  if token not in words]

        processed.append(' '.join(tokens))

    return processed


def cosine_method(data):
    # concatenate the training and test data together, since we're not learning
    X = np.concatenate((data.X_train, data.X_test), axis=0)
    y = np.concatenate((data.y_train, data.y_test), axis=0)
    X = cosine_preprocess(X, pkl_path='preprocessed.pkl')

    # vectorize the documents
    # vectorizer = TfidfVectorizer(max_df=1.0)
    vectorizer = models.SentimentVectorizer(topics)
    document_vectors = vectorizer.fit_transform(X)

    # filter all-zero samples
    non_zero_indices = np.any(document_vectors, axis=1)
    document_vectors = document_vectors[non_zero_indices, :]
    y = y[non_zero_indices]

    # group by party and take the mean
    party_vectors = [document_vectors[(y == label), :]
                     for label in range(len(parties))]

    mean_party_vectors = np.array([np.mean(vectors, axis=0) for vectors in party_vectors])
    for i in range(mean_party_vectors.shape[0]):
        if np.nan in mean_party_vectors[i, :]:
            mean_party_vectors[i, :] = np.zeros(mean_party_vectors.shape[1])

    # print cosine similarity between parties
    for i1 in range(len(parties)):
        for i2 in range(i1, len(parties)):
            sim = cosine_similarity([mean_party_vectors[i1, :]],
                                    [mean_party_vectors[i2, :]])

            print(f'{parties[i1]}, {parties[i2]}: {sim[0, 0]}')

    # get the newspaper data, vectorize it and get the mean
    paper_vectors = [vectorizer.transform(corpus.get_newspaper(paper, True))
                     for paper in newspapers]

    rows = []
    for i, paper in enumerate(paper_vectors):
        row = [newspapers[i]]
        for party in range(len(parties)):
            row.append(cosine_similarity([mean_party_vectors[party, :]], paper)[0, 0])

        # add a softmaxed version to amplify differences
        # scores = row[1:]
        # softmax = np.exp(scores) / np.sum(np.exp(scores))

        rows.append(row)
        # table_rows.append([row[0] + ' softmax'] + softmax.tolist())

    print(tabulate(rows, headers=parties, floatfmt=".3f"))

    # subtract each newspaper's mean to account for overall more political
    # use of language
    value_matrix = np.array(rows)[:, 1:].astype(np.float32)
    means = np.mean(value_matrix, axis=1)
    means_subtracted = (value_matrix.T - means).T
    for i in range(len(rows)):
        for j in range(1, len(rows[i])):
            rows[i][j] = means_subtracted[i, j - 1]

    print(tabulate(rows, headers=parties, floatfmt=".3f"))


def doc2vec_method(data):
    X_train = cosine_preprocess(data.X_train, pkl_path='doc2vec_xtrain_prepr.pkl')
    X_test = cosine_preprocess(data.X_test, pkl_path='doc2vec_xtest_prepr.pkl')

    pkl_path = 'doc_embeddings.model'
    if os.path.exists(pkl_path):
        model = gensim.models.Doc2Vec.load(pkl_path)
    else:
        # create a tagged corpus
        train_corpus = [gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(doc),
                                                             [data.y_train[i]])
                        for i, doc in enumerate(X_train)]

        model = gensim.models.doc2vec.Doc2Vec(dm=0, size=64, min_count=2, iter=20)
        model.build_vocab(train_corpus)
        model.train(train_corpus, total_examples=model.corpus_count)
        model.save(pkl_path)

    # test on testset
    message('Testing')
    rows = []
    for i, party in enumerate(parties):
        docs = [gensim.utils.simple_preprocess(x)
                for x, y in zip(X_test, data.y_test)
                if y == i]

        best_matches = []
        for doc in docs:
            best_matches.append(model.docvecs.most_similar([model.infer_vector(doc)])[0][0])

        c = Counter(best_matches)
        acc = c[i] / np.sum([c.get(pred, 0) for pred in range(len(parties))])
        row = [party] + [c.get(pred, 0) for pred in range(len(parties))] + [acc]
        rows.append(row)

    print(tabulate(rows, headers=parties + ['prec'], floatfmt=".3f"))

    # test on newspapers
    rows = []
    for i, paper in enumerate(newspapers):
        docs = [gensim.utils.simple_preprocess(x)
                for x in corpus.get_newspaper(paper, False)]

        similarities = []
        for doc in docs:
            # sort by label
            sims = sorted(model.docvecs.most_similar([model.infer_vector(doc)]))
            similarities.append(list(map(itemgetter(1), sims)))

        values = np.mean(similarities, axis=0)
        row = [paper] + values.tolist()
        rows.append(row)

    print(tabulate(rows, headers=parties, floatfmt=".3f"))

    # subtract each newspaper's mean to account for overall more political
    # use of language
    value_matrix = np.array(rows)[:, 1:].astype(np.float32)
    means = np.mean(value_matrix, axis=1)
    means_subtracted = (value_matrix.T - means).T
    for i in range(len(rows)):
        for j in range(1, len(rows[i])):
            rows[i][j] = means_subtracted[i, j - 1]

    print(tabulate(rows, headers=parties, floatfmt=".3f"))


def get_args():
    parser = argparse.ArgumentParser(description='Predict political left-rightness')
    parser.add_argument('folder', help='folder containing the training data')

    parser.add_argument('-k', type=int, default=50000,
                        help='number of best features to select')

    parser.add_argument('-e', '--epochs', type=int, default=5,
                        help='number of epochs to train for')

    parser.add_argument('--method', '-m', choices=['keras', 'svm', 'nb', 'svm_sent',
                                                   'auto', 'cossim', 'doc2vec'],
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


topics = ['profite',
          'vermögenden',
          'erwerbslosen',
          'kurdinnen',
          'vermögensteuer',
          'profitinteressen',
          'sozialabbau',
          'ostfriesland',
          'kürzungspolitik',
          'zukunftsvergessen',
          'integrationskraft',
          'superreichen',
          'klimakrise',
          'terrorbanden']


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

    if args.method == 'cossim':
        return cosine_method(data)
    if args.method == 'doc2vec':
        return doc2vec_method(data)

    if os.path.exists(model_path) and args.load_from_disk:
        print('Loading model from disk')
        preprocess, model = models.load_pipeline(model_path, args.method)
    else:
        preprocess, model = {
            'svm': lambda: models.create_svm_model(k),
            'svm_sent': lambda: models.create_svm_sent_model(k, topics),
            'nb': lambda: models.create_nb_model(k),
            'auto': lambda: models.create_auto_model(k),
            'keras': lambda: models.create_model(k, args.epochs, args.dropout,
                                                 len(parties)),
        }.get(args.method)()

        print(f'Training model on data {len(data.X_train)} samples')
        X_trans = preprocess.fit_transform(data.X_train, data.y_train)
        print(f'Number of features: {X_trans.shape[1]}')
        model.fit(X_trans, data.y_train)
        models.save_pipeline(preprocess, model, model_path, args.method)

    if args.method != 'keras':
        print_best_words(data.X_train, data.y_train, 30, preprocess)

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
