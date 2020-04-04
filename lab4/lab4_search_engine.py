import pickle
from typing import Any, List
from scipy import sparse
import numpy as np
import re
import operator

cache_dir = "./cache"


class ArticleData:
    def __init__(self):
        self.title = None
        self.bag_of_words = None
        self.words_vec = None
        self.words_vec_norm = None


def load_cached_data():
    cache_file_path = cache_dir + '/{}'
    with open(cache_file_path.format('articles_data.dump'), "rb") as f:
        _articles_data: List[ArticleData] = pickle.load(f)

    with open(cache_file_path.format('wordset.dump'), "rb") as f:
        _word_set: List[Any] = pickle.load(f)

    _A: sparse.csr_matrix = sparse.load_npz(cache_file_path.format('term_by_document_matirx.npz'))
    _bag_of_words: np.array = np.load(cache_file_path.format('total_bag_of_words_vector.npy'))

    return _articles_data, _word_set, _A, _bag_of_words


def do_query(query, k, word_list, articles):
    query = query.lower()
    words_dict = {word: index for index, word in enumerate(word_list)}
    words = re.findall(r'\w+', query)

    vec_query = np.zeros(len(word_list), dtype=int)
    for w in words:
        if w in words_dict.keys():
            vec_query[words_dict[w]] += 1

    if not np.any(vec_query):
        print("No results")
        return

    q_norm = np.linalg.norm(vec_query)
    res = []
    for a in articles:
        divider = q_norm * a.words_vec_norm
        prod = vec_query @ a.words_vec
        cos_theta = prod/divider
        res.append((cos_theta, a))

    res.sort(key=operator.itemgetter(0), reverse=True)
    for res_entry in res[:k]:
        print(res_entry[1].title)


if __name__ == "__main__":
    # articles - list with all of documents (words vectors + bag of words)
    # word_list - bag_of_words_dict.keys()
    # A - sparse matrix, columns are words vectors from articles_data
    # bag_of_words - total bag_of_words vector (index 0 is occurences of the word word_list[0])
    articles, word_list, A, bag_of_words = load_cached_data()

    do_query('Nikola Jean Caro', 10, word_list, articles)
