import os.path
import threading
import pickle

import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from keras.wrappers.scikit_learn import KerasClassifier
from scipy import sparse
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

import autosklearn.classification
from nltk.tokenize import word_tokenize
from sentistrength.senti_client import sentistrength
from tqdm import tqdm


class SentimentVectorizer:
    def __init__(self, topics, lang='DE'):
        self.topics = topics
        self.senti = sentistrength(lang)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        pkl_path = 'sentiment_avg.pkl'

        if os.path.exists(pkl_path):
            print(f'Loading {pkl_path} from disk')
            with open(pkl_path, 'rb') as f:
                return pickle.load(f)

        sentiments = []

        for sample in tqdm(X):
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


def ensure_dense(X):
    """ If the input is a sparse matrix, convert it to a dense one. """
    if sparse.issparse(X):
        # todense() returns a matrix, so convert it to an array
        return np.asarray(X.todense())
    else:
        return X


def create_neuralnet(k, dropout, num_parties):
    """ Create a simple feedforward Keras neural net with k inputs """
    model = Sequential([
        Dense(100, input_dim=k, activation='tanh'),
        Dropout(dropout),
        Dense(num_parties, activation='softmax'),
    ])

    model.compile(optimizer='nadam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def create_svm_model(k):
    vectorizer = TfidfVectorizer()
    kbest = SelectKBest(chi2, k=k)

    scaler = StandardScaler(with_mean=False)
    svc = SVC(kernel='linear', probability=True)

    return make_pipeline(vectorizer, kbest, scaler), svc


def create_svm_sent_model(k, topics):
    vectorizer = SentimentVectorizer(topics)
    scaler = StandardScaler(with_mean=False)
    svc = SVC(kernel='linear', probability=True)

    return make_pipeline(vectorizer, scaler), svc


def create_nb_model(k):
    vectorizer = TfidfVectorizer()
    kbest = SelectKBest(chi2, k=k)
    unsparse = FunctionTransformer(ensure_dense, accept_sparse=True)
    nb = MultinomialNB()

    return make_pipeline(vectorizer, kbest, unsparse), nb


def create_auto_model(k):
    vectorizer = TfidfVectorizer()
    kbest = SelectKBest(chi2, k=k)
    auto = autosklearn.classification.AutoSklearnClassifier()

    return make_pipeline(vectorizer, kbest), auto


def create_model(k, epochs, dropout, num_parties):
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

    model = KerasClassifier(create_neuralnet, k=k, dropout=dropout,
                            num_parties=num_parties,
                            epochs=epochs, batch_size=32)

    return make_pipeline(vectorizer, kbest, unsparse, scaler), model


def save_pipeline(pipeline, model, path, name):
    if 'postaggertransformer' in pipeline.named_steps:
        pipeline.named_steps['postaggertransformer'].tagger.taggerlock = None

    if isinstance(model, KerasClassifier):
        nnet = model.model
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
