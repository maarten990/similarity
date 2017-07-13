import zlib
import os.path
import threading
import pickle

import numpy as np
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from nltk.tokenize import word_tokenize
from tabulate import tabulate
from tqdm import tqdm

import corpus

try:
    from sentistrength.senti_client import sentistrength
except:
    print('Warning: sentistrength not found')


class WeightedWordCountVectorizer:
    """
    Wordcount vectorizer, weighed by the difference in vocabulary between the
    newspapers and the parliamentary data. Words get a higher weight if their
    frequency is similar in both corpora.
    """
    def __init__(self, newspaper_data):
        self.countvec = CountVectorizer(min_df=0.2, max_df=0.8)
        self.countvec_papers = CountVectorizer()
        self.paper_counts = self.countvec_papers.fit_transform(newspaper_data)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def fit(self, X, y=None):
        self.countvec.fit(X)

        # get the overlapping set of words
        words = self.countvec.get_feature_names()
        common_words_idx = np.in1d(words, self.countvec_papers.get_feature_names())
        common_words_idx = np.nonzero(common_words_idx)[0]
        self.common_words = np.array(words)[common_words_idx]
        self.common_word_idx = [self.countvec.vocabulary_[w] for w in self.common_words]
        self.common_word_idx_papers = [self.countvec_papers.vocabulary_[w]
                                       for w in self.common_words]

        # filter for only the common words
        counts = self.countvec.transform(X)
        vec = counts[:, self.common_word_idx]
        vec_paper = self.paper_counts[:, self.common_word_idx_papers]

        # average the counts for each word for both the bundestag and the newspapers
        avg = np.asarray(np.mean(vec, axis=0)).squeeze()
        paper_avg = np.asarray(np.mean(vec_paper, axis=0)).squeeze()

        # the weights are then a measure of difference between the 2 'distributions'
        self.weights = np.divide(1, np.abs(avg - paper_avg) + 1)

        # print the heighest and lowest weighted words
        sorted_indices = sorted(list(range(len(self.weights))),
                                key=lambda i: self.weights[i])
        names = self.countvec.get_feature_names()
        lowest = [[names[self.common_word_idx[i]], self.weights[i], avg[i], paper_avg[i]]
                  for i in sorted_indices[:10]]
        highest = [[names[self.common_word_idx[i]], self.weights[i], avg[i], paper_avg[i]]
                   for i in sorted_indices[-10:]]
        print('Lowest weights:')
        print(tabulate(lowest, headers=['Word', 'Weight', 'Count parliament', 'Count papers']))
        print()
        print('Heighest weights:')
        print(tabulate(highest, headers=['Word', 'Weight', 'Count parliament', 'Count papers']))
        print()

        return self

    def transform(self, X):
        # restrict the counts to the common vocabulary and convert it to a
        # dense representation to speed up the multiplication
        counts = self.countvec.transform(X)
        counts = counts[:, self.common_word_idx]
        counts = np.asarray(counts.todense())

        # repeat the weights in the 0th dimension to match the size of counts
        n_samples = counts.shape[0]
        weights = np.tile(self.weights, (n_samples, 1))

        # elementwise multiplication
        return np.multiply(weights, counts)


class SentimentVectorizer:
    """
    Vectorize the data based on sentiment values.
    """
    def __init__(self, topics, lang='DE'):
        self.topics = topics
        self.senti = sentistrength(lang)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # use a deterministic hash of the data to create the pickle path
        h = zlib.adler32(''.join(X).encode('utf-8'))
        pkl_path = f'pickle/sentiment_{h}.pkl'
        print(pkl_path)

        if os.path.exists(pkl_path):
            print(f'Loading {pkl_path} from disk')
            with open(pkl_path, 'rb') as f:
                return pickle.load(f)

        sentiments = []

        for sample in tqdm(X, desc='Sentimenting'):
            sample_sentiment = []

            for topic in self.topics:
                # collect the average sentiment for this topic in the current
                # document
                topic_sentiment = []
                words = word_tokenize(sample)

                indices = [i for i, w in enumerate(words) if w == topic]
                for idx in indices:
                    start = max(0, idx - 5)
                    end = min(len(sample), idx + 6)
                    sent_dict = self.senti.get_sentiment(' '.join(words[start:end]))
                    topic_sentiment.append([int(sent_dict[key])
                                            for key in ['positive', 'negative', 'neutral']])

                if len(topic_sentiment) > 0:
                    avg = np.mean(np.array(topic_sentiment), axis=0)
                else:
                    avg = np.array([0, 0, 0])

                sample_sentiment.extend(avg.tolist())

            sentiments.append(sample_sentiment)

        print(f'Saving {pkl_path} to disk')
        with open(pkl_path, 'wb') as f:
            pickle.dump(np.array(sentiments), f)

        return np.array(sentiments)


def create_svm_model(k, method='tfidf', **kwargs):
    """
    Create an svm model with corresponding preprocessing pipeline.
    The method parameter specifies how to vectorize the documents:
      - tfidf: standard TF-IDF vectorization
      - weighted: word counts weighted on vocabularity similarity between
        2 corpora
      - sentiment: vectorized by sentiment values for a specified set of words
    """
    if method == 'tfidf':
        vectorizer = TfidfVectorizer()
    elif method == 'weighted':
        vectorizer = WeightedWordCountVectorizer(
            [a for articles in [corpus.get_newspaper(paper, False)
                                for paper in kwargs['newspapers']]
             for a in articles])
        k = 'all'
    elif method == 'sentiment':
        vectorizer = SentimentVectorizer(kwargs['topics'])

    kbest = SelectKBest(chi2, k=k)
    scaler = StandardScaler(with_mean=False)
    svc = SVC(kernel='linear', probability=True)

    if method == 'sentiment':
        return make_pipeline(vectorizer, scaler), svc
    else:
        return make_pipeline(vectorizer, kbest, scaler), svc


def save_pipeline(pipeline, model, path, name):
    """ Save a model and its preprocessing pipeline. """

    # the postagger object has a lock which can't be pickled, so we need to
    # remove it before saving and restore it afterwards
    if 'postaggertransformer' in pipeline.named_steps:
        pipeline.named_steps['postaggertransformer'].tagger.taggerlock = None

    joblib.dump((pipeline, model), path)

    if 'postaggertransformer' in pipeline.named_steps:
        pipeline.named_steps['postaggertransformer'].tagger.taggerlock = threading.Lock()


def load_pipeline(path, name):
    """ Load a model and its preprocessing pipeline. """
    pipeline, model = joblib.load(path)

    # restore the tagger's lock, which couldn't be save
    if 'postaggertransformer' in pipeline.named_steps:
        pipeline.named_steps['postaggertransformer'].tagger.taggerlock = threading.Lock()

    return pipeline, model
