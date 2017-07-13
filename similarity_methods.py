import os
import os.path
import pickle
from collections import Counter
from operator import itemgetter

import gensim
import numpy as np
import tqdm
from nltk.corpus import stopwords
from nltk.stem.snowball import GermanStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

import corpus
from models import SentimentVectorizer, WeightedWordCountVectorizer


def cosine_preprocess(texts, pickle_name, pickle_folder='pickle'):
    pickle_path = os.path.join(pickle_folder, pickle_name)

    # Return from disk if possible for efficiency reasons
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)

    processed = []
    for text in tqdm(texts):
        stemmer = GermanStemmer()
        words = stopwords.words('german')

        tokens = [stemmer.stem(token) for token in word_tokenize(text)
                  if token not in words]

        processed.append(' '.join(tokens))

    # Pickle the output
    if not os.path.exists(pickle_folder):
        os.makedirs(pickle_folder)

    with open(pickle_path, 'wb') as f:
        pickle.dump(processed, f)

    return processed


def cosine_method(data, parties, newspapers, lang, method, topics):
    # concatenate the training and test data together, since we're not learning
    X = np.concatenate((data.X_train, data.X_test), axis=0)
    y = np.concatenate((data.y_train, data.y_test), axis=0)
    X = cosine_preprocess(X, pickle_name=f'preprocessed_{lang}.pkl')

    # vectorize the documents
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(min_df=0.1, max_df=0.9)
    elif method == 'sentiment':
        vectorizer = SentimentVectorizer(topics)
    else:
        vectorizer = WeightedWordCountVectorizer(
            [a for articles in [corpus.get_newspaper(paper, False) for paper in newspapers]
             for a in articles])
    document_vectors = vectorizer.fit_transform(X)

    # filter all-zero samples
    # non_zero_indices = np.any(document_vectors, axis=1)
    # document_vectors = document_vectors[non_zero_indices, :]
    # y = y[non_zero_indices]

    # group by party and take the mean
    party_vectors = [document_vectors[(y == label), :]
                     for label in range(len(parties))]

    mean_party_vectors = np.array([np.mean(vectors, axis=0) for vectors in party_vectors]).squeeze()
    for i in range(mean_party_vectors.shape[0]):
        if np.nan in mean_party_vectors[i, :]:
            mean_party_vectors[i, :] = np.zeros(mean_party_vectors.shape[1])

    # print cosine similarity between parties
    # for i1 in range(len(parties)):
    #     for i2 in range(i1, len(parties)):
    #         sim = cosine_similarity([mean_party_vectors[i1, :]],
    #                                 [mean_party_vectors[i2, :]])

    #         print(f'{parties[i1]}, {parties[i2]}: {sim[0, 0]}')

    # get the newspaper data, vectorize it and get the mean
    paper_vectors = [vectorizer.transform(corpus.get_newspaper(paper, True))
                     for paper in newspapers]

    rows = []
    for i, paper in enumerate(paper_vectors):
        row = [newspapers[i]]
        for party in range(len(parties)):
            row.append(cosine_similarity([mean_party_vectors[party, :]], paper)[0, 0])

        rows.append(row)

    print()
    print('Similarity scores:')
    print(tabulate(rows, headers=parties, floatfmt=".3f"))

    # subtract each newspaper's mean to account for overall more political
    # use of language
    value_matrix = np.array(rows)[:, 1:].astype(np.float32)
    means = np.mean(value_matrix, axis=1)
    means_subtracted = (value_matrix.T - means).T
    for i in range(len(rows)):
        for j in range(1, len(rows[i])):
            rows[i][j] = means_subtracted[i, j - 1]

    print()
    print('Mean subtracted per newspaper:')
    print(tabulate(rows, headers=parties, floatfmt=".3f"))

    # subtract each party's mean
    value_matrix = np.array(rows)[:, 1:].astype(np.float32)
    means = np.mean(value_matrix, axis=0)
    means_subtracted = value_matrix - means
    for i in range(len(rows)):
        for j in range(1, len(rows[i])):
            rows[i][j] = means_subtracted[i, j - 1]

    print()
    print('Mean subtracted per party:')
    print(tabulate(rows, headers=parties, floatfmt=".3f"))


def doc2vec_method(data, parties, newspapers, lang):
    X_train = cosine_preprocess(data.X_train,
                                pickle_name=f'doc2vec_xtrain_prepr_{lang}.pkl')
    X_test = cosine_preprocess(data.X_test,
                               pickle_name=f'doc2vec_xtest_prepr_{lang}.pkl')

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
