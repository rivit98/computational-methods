from collections import Counter
from typing import List, Any
from scipy import sparse
from scipy.sparse.linalg import svds
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import os
import pickle
import re
import numpy as np
import operator

nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

data_dir = "./data"


class CacheManager:
    cache_dir = "./cache"  # place for storing calculated matrices, etc

    def __init__(self):
        self.loaded = set()

        if not os.path.exists(CacheManager.cache_dir):
            os.makedirs(CacheManager.cache_dir)

    def was_loaded(self, filename):
        return filename in self.loaded

    def save(self, filename, object):
        if self.was_loaded(filename):
            return

        try:
            with open('{}/{}'.format(CacheManager.cache_dir, filename), "wb") as f:
                pickle.dump(object, f, protocol=pickle.HIGHEST_PROTOCOL)
                print("> caching " + filename)
        except:
            return

    def load(self, filename):
        try:
            with open('{}/{}'.format(CacheManager.cache_dir, filename), "rb") as f:
                res = pickle.load(f)
                print("> using cached " + filename)
                self.loaded.add(filename)
                return res
        except:
            return None


class ArticleData:
    def __init__(self, title):
        self.title = title.split('.')[0]
        self.bag_of_words = Counter()
        self.words_vec = None
        self.words_vec_norm = None

    def load_bag_of_words(self, path):
        with open(path, "rt", encoding='utf-8') as f:
            lemmatizer = WordNetLemmatizer()
            words = re.findall(r'\w+', f.read().lower())
            loaded_words = [lemmatizer.lemmatize(word) for word in words if len(word) > 2 and word not in stop_words]
            self.bag_of_words.update(loaded_words)

    def create_full_bag_of_words(self, keyset, size):
        self.words_vec = np.zeros(size)  # d_j
        for i, k in enumerate(keyset):
            self.words_vec[i] = self.bag_of_words[k]

        self.words_vec_norm = np.linalg.norm(self.words_vec)

    def print_contents(self):
        with open('{}/{}.txt'.format(data_dir, self.title), "rt", encoding='utf-8') as f:
            print(f.read())

    def normalize_word_vec(self):
        self.words_vec = self.words_vec / np.linalg.norm(self.words_vec)


def getIDF(wordset, articles_data):
    articles_num = len(articles_data)
    idf = []
    for word in wordset:
        cnt = 0
        for article in articles_data:
            if article.bag_of_words[word] != 0:
                cnt += 1

        idf.append(np.log10(articles_num / cnt))

    return idf


def create_sparse(articles_data, sizeof_total, idf):
    row = []
    column = []
    data = []

    for i in range(len(articles_data)):
        article = articles_data[i]
        for j in range(sizeof_total):
            if article.words_vec[j] != 0:
                row.append(j)
                column.append(i)
                data.append(article.words_vec[j] * idf[j])

    term_by_document_matirx = sparse.csr_matrix((data, (row, column)), shape=(sizeof_total, len(articles_data)))
    return term_by_document_matirx


def prepare_data(cache):
    articles_data: List[ArticleData] = cache.load('articles_data.dump')
    if articles_data is None:
        articles_data = []
        for file in os.listdir(data_dir):
            a_data = ArticleData(file)
            a_data.load_bag_of_words("{}/{}".format(data_dir, file))
            articles_data.append(a_data)
    print("total number of articles {}".format(len(articles_data)))

    total_bag_of_words: Counter = cache.load('total_bag_of_words.dump')
    if total_bag_of_words is None:
        total_bag_of_words = Counter()
        for article in articles_data:
            total_bag_of_words += article.bag_of_words

    sizeof_total = len(total_bag_of_words)
    wordset: List[Any] = cache.load('wordset.dump')
    if wordset is None:
        wordset = list(total_bag_of_words.keys())
    print("total number of words: {}".format(sizeof_total))

    if not cache.was_loaded('articles_data.dump'):
        print("creating bag of words for every article")
        for article in articles_data:
            article.create_full_bag_of_words(wordset, sizeof_total)
    print("created {} bags, every has {} elements".format(len(articles_data), sizeof_total))

    idf: List[Any] = cache.load('idf.dump')
    if idf is None:
        print('calculating idf')
        idf = getIDF(wordset, articles_data)

    term_by_document_matirx: sparse.csr_matrix = cache.load('term_by_document_sparse_matrix.dump')
    if term_by_document_matirx is None:
        print('creating sparse matrix')
        term_by_document_matirx = create_sparse(articles_data, sizeof_total, idf)
    print("term by document matrix size: {}x{}".format(term_by_document_matirx.shape[0],
                                                       term_by_document_matirx.shape[1]))

    cache.save('articles_data.dump', articles_data)
    cache.save('wordset.dump', wordset)
    cache.save('term_by_document_sparse_matrix.dump', term_by_document_matirx)
    cache.save('total_bag_of_words.dump', total_bag_of_words)
    cache.save('idf.dump', idf)

    return articles_data, wordset, term_by_document_matirx, idf


def parse_query(query, word_list):
    query = query.lower()
    words_dict = {word: index for index, word in enumerate(word_list)}
    words = re.findall(r'\w+', query)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    vec_query = np.zeros(len(word_list), dtype=int)
    for w in words:
        if w in words_dict.keys():
            vec_query[words_dict[w]] += 1

    if not np.any(vec_query):
        print("No results")
        return

    return vec_query


def print_search_results(res, k, query):
    res.sort(key=operator.itemgetter(0), reverse=True)
    print("Found articles for query [{}]:".format(query))
    for res_entry in res[:k]:
        print('> ' + res_entry[1].title.replace("_", " "))

    print()

    # print("\n\nFull articles:")
    # for res_entry in res[:k]:
    #     print(res_entry[1].print_contents())
    #     print('\n')
    #     print('-' * 40)


def do_query(query, k, word_list, articles):
    vec_query = parse_query(query, word_list)

    q_norm = np.linalg.norm(vec_query)
    vec_query = vec_query.T
    res = []
    for a in articles:
        divider = q_norm * a.words_vec_norm
        prod = vec_query @ a.words_vec
        cos_theta = prod / divider
        res.append((cos_theta, a))

    print_search_results(res, k, query)


def normalize_vectors(articles):
    for a in articles:
        a.normalize_word_vec()


def do_query2(query, k, word_list, articles, A):
    vec_query = parse_query(query, word_list)
    vec_query = vec_query / np.linalg.norm(vec_query)

    res = vec_query.T @ A
    probabilities = []
    for i, cos_theta in enumerate(res):
        probabilities.append((cos_theta, articles[i]))

    print_search_results(probabilities, k, query)


def getSVD(A, rank):
    U, S, VT = sparse.linalg.svds(A, rank)
    return U @ np.diag(S) @ VT


def do_query3(query, k, word_list, articles, A, rank):
    vec_query = parse_query(query, word_list)
    q_norm = np.linalg.norm(vec_query)

    res = []
    for i, ak_row in enumerate(getSVD(A, rank).T):
        prod = vec_query.T @ ak_row
        cos_fi = prod / (q_norm * np.linalg.norm(ak_row))
        res.append((cos_fi, articles[i]))

    print_search_results(res, k, query)


if __name__ == "__main__":
    cache = CacheManager()
    # articles - list with all of documents (words vectors + bag of words)
    # word_list - bag_of_words_dict.keys()
    # A - sparse matrix, columns are words vectors from articles_data
    articles, word_list, A, idf = prepare_data(cache)



    normalize_vectors(articles)
    A_normalized: sparse.csr_matrix = cache.load('A_normalized.dump')
    if A_normalized is None:
        print('calculating new sparse matrix with new vectors')
        A_normalized = create_sparse(articles, len(word_list), idf)
        cache.save('A_normalized.dump', A_normalized)

    rank = 120

    do_query("Action film", 5, word_list, articles)
    do_query2("Action film", 5, word_list, articles, A_normalized)
    do_query3("Action film", 5, word_list, articles, A, rank)

    do_query2("Winston Churchill", 5, word_list, articles, A_normalized)
    do_query("Winston Churchill", 5, word_list, articles)
    do_query3("Winston Churchill", 5, word_list, articles, A, rank)

    do_query("Beautiful places on earth", 5, word_list, articles)
    do_query2("Beautiful places on earth", 5, word_list, articles, A_normalized)
    do_query3("Beautiful places on earth", 5, word_list, articles, A, rank)


