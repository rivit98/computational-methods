from collections import Counter
from scipy import sparse
import os
import pickle
import re
import numpy as np

data_dir = "./data"
cache_dir = "./cache"  # place for storing calculated matrices, etc

class ArticleData:
    ignored_words = ["a", "the", "of", "is"]  # and probably more

    def __init__(self, title):
        self.title = title.split('.')[0]
        self.bag_of_words = Counter()
        self.words_vec = None
        self.words_vec_norm = None

    def load_bag_of_words(self, path):
        with open(path, "rt", encoding='utf-8') as f:
            words = re.findall(r'\w+', f.read().lower())
            loaded_words = [word for word in words if len(word) > 2]
            self.bag_of_words.update(loaded_words)

        for ignore_token in ArticleData.ignored_words:
            del self.bag_of_words[ignore_token]

    def create_full_bag_of_words(self, keyset, size):
        self.words_vec = np.zeros(size)  # d_j
        for i, k in enumerate(keyset):
            self.words_vec[i] = self.bag_of_words[k]

        self.words_vec_norm = np.linalg.norm(self.words_vec)



def getIDF(wordset, articles_data):
    articles_num = len(articles_data)
    idf = []
    for word in wordset:
        cnt = 0
        for article in articles_data:
            if article.bag_of_words[word] != 0:
                cnt += 1

        idf.append(np.log10(articles_num/cnt))

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


if __name__ == "__main__":
    articles_data = []
    for file in os.listdir(data_dir):
        a_data = ArticleData(file)
        a_data.load_bag_of_words("{}/{}".format(data_dir, file))
        articles_data.append(a_data)

    total_bag_of_words = Counter()
    for article in articles_data:
        total_bag_of_words += article.bag_of_words

    sizeof_total = len(total_bag_of_words)
    wordset = list(total_bag_of_words.keys())
    print("bag-of-words: {} words".format(sizeof_total))

    total_bag_of_words_vector = np.zeros(sizeof_total, dtype=int)

    # convert Counter to vectors
    for counter, key in enumerate(wordset):
        total_bag_of_words_vector[counter] = total_bag_of_words[key]

    for article in articles_data:
        article.create_full_bag_of_words(wordset, sizeof_total)

    print('calculating idf')
    idf = getIDF(wordset, articles_data)

    print('creating sparse matrix')
    term_by_document_matirx = create_sparse(articles_data, sizeof_total, idf)
    print("term-by-document matrix: {}x{}".format(term_by_document_matirx.shape[0], term_by_document_matirx.shape[1]))



    # caching data

    cache_file_path = cache_dir + '/{}'
    print("Caching data...")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    print("> articles data")
    with open(cache_file_path.format('articles_data.dump'), "wb") as f:
        pickle.dump(articles_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("> wordlist")
    with open(cache_file_path.format('wordset.dump'), 'wb') as f:
        pickle.dump(wordset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("> bag of words vector")
    np.save(cache_file_path.format('total_bag_of_words_vector'), total_bag_of_words_vector)

    print("> term by document sparse matrix")
    sparse.save_npz(cache_file_path.format('term_by_document_matirx'), term_by_document_matirx)

