import nltk
import csv
from collections import Counter
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from flask import Flask, request, jsonify
from inverted_index_gcp import *
import gensim.downloader
### remember to install this in the instance -> pip install gensim
# model_glove = gensim.downloader.load('glove-wiki-gigaword-50')
# model_fasttext = gensim.downloader.load('fasttext-wiki-news-subwords-300')
# model_word2vec = gensim.downloader.load('word2vec-google-news-300')

# # dict {(doc_id:title)}
# doc2title = pd.read_pickle('doc2title/doc2title_dict.pkl')
# # paths
# base_path = "bins/"
# title_path_bins = "Title_data/"
# body_path_bins = "Body_data/"
# anchor_path_bins = "Anchor_data/"
# # index paths
# indices_path = "indices/"
#
# # reading 3 Indexes: title, body and anchor
# title_index = InvertedIndex.read_index(indices_path, "Title_index")
# body_index = InvertedIndex.read_index(indices_path, "Body_index")
# anchor_index = InvertedIndex.read_index(indices_path, "Anchor_index")

# dict {(doc_id:title)}
doc2title = pd.read_pickle('/home/omer6/DocID_Title_Dict/doc2title_dict.pkl')

# paths
base_path = '/home/omer6/Indices'
title_path_bins = '/home/omer6/bins/Title_Bins/Title_data/'
body_path_bins = '/home/omer6/bins/Body_Bins/Body_data/'
anchor_path_bins = '/home/omer6/bins/Anchor_Bins/Anchor_data/'

# reading 3 Indexes: title, body and anchor
title_index = InvertedIndex.read_index( '/home/omer6/Indices', "Title_index")
body_index = InvertedIndex.read_index('/home/omer6/Indices', "Body_index")
anchor_index = InvertedIndex.read_index('/home/omer6/Indices', "Anchor_index")


# reading the page view
page_views = pd.read_pickle('/home/omer6/PageViews/pageviews-202108-user.pkl')

# reading the page rank
with open('/home/omer6/PageRank/page_rank.csv','r') as f:
  csv_reader = csv.reader(f)
  pagerank_list = list(csv_reader)
  pagerank_dict = dict([(int(doc_id),float(score)) for doc_id,score in pagerank_list])


# getting average  doc len
sum = 0
for d in body_index.DL:
    sum += body_index.DL[d]
avgdl = sum / body_index.corpus_size

# *********************FLASK**************************
class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# *********************Reading the posting list**************************
# We're going to pack the doc_id and tf values in this many bytes.
TUPLE_SIZE = 6
# Masking the 16 low bits of an integer
TF_MASK = 2 ** 16 - 1


def read_posting_list(index, w, f_name):
    with closing(MultiFileReader()) as reader:
        locs = index.posting_locs[w]
        locs = [(f_name + lo[0], lo[1]) for lo in locs]
        b = reader.read(locs, index.df[w] * TUPLE_SIZE)
        posting_list = []
        for i in range(index.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list

# **************************************** TFIDF DICT FOR MODELS ****************************************
# this dictionary will help us to pick the best terms from the query
# (according to their tfidf score)
# tfidf_dict_body = {}
# for term in body_index.term_total.keys():
#     pl = read_posting_list(body_index, term,  body_path_bins)
#     for doc_id, tf in pl:
#         tfidf_dict_body = tf * body_index.idf_dict[doc_id]

# **************************************** TOKENIZE ****************************************
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
all_stopwords = frozenset(stopwords.words('english'))


def tokenize(text):
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filters stopwords.

    Parameters:
    -----------
    text: string , separating the text to tokenize.

    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if
                      token.group() not in all_stopwords]
    return list_of_tokens


# **************************************** RANKING ***************************************
def get_candidate_documents_and_scores(query_to_search, index, path):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
    and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
    Then it will populate the dictionary 'candidates.'
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the document.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']
    index:           inverted index loaded from the corresponding files.
    words,pls: iterator for working with posting.
    Returns:
    -----------
    dictionary of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """
    DL = index.DL
    candidates = {}
    for term in np.unique(query_to_search):
        if term in index.term_total.keys():
            list_of_doc = read_posting_list(index, term, path)[:7100]
            normalized_tfidf = []
            for doc_id, freq in list_of_doc:
                formula = (freq / DL[doc_id]) * index.idf_dict[term]
                id_tfidf = (doc_id, formula)
                normalized_tfidf.append(id_tfidf)

            for doc_id, tfidf in normalized_tfidf:
                if (doc_id, term) in candidates:
                    candidates[(doc_id, term)] += tfidf
                else:
                    candidates[(doc_id, term)] = tfidf
    return candidates


def get_top_n(sim_dict, N=100):
    """
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores
    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

    N: Integer (how many documents to retrieve). By default N = 100
    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """
    dict_sorted = sorted([(doc_id, np.round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1],
                         reverse=True)[:N]
    return dict_sorted


# ****************************************  QUERY EXPANSIONS ****************************************
def expand_query(query_to_search, index,model):
    """
        Taking the tokens with the 3 best tfidf score and finding them similar words, adding them to the
        query to search list and then returns it to the search metric that was chosen ( we can change to more words)
    Args:
        query_to_search: List of tokens (str)

    Returns:
        extended_query_to_search: extended query_to_search list of tokens (str)
    """
    terms_to_expend = {}
    for term in query_to_search:
        terms_to_expend[term] = index.term_total[term]
    terms_to_expend = sorted(terms_to_expend.keys())[:3]
    threshold = 0.8
    for term in terms_to_expend:
        similarities = [word for word, sim in model.most_similar(term) if sim > threshold]
        query_to_search.extend(similarities)
    return query_to_search

# ****************************************  COSINE ****************************************
def cosine_similarity(candidates, search_query, index):
    """
    Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    Generate a dictionary of cosine similarity scores
    key: doc_id
    value: cosine similarity score

    Parameters:
    -----------
    D: DataFrame of tfidf scores.
    Q: vectorized query with tfidf scores

    Returns:
    -----------
    dictionary of cosine similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: cosine similarty score.
    """
    candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    dict_cosine_sim = {}
    for id_term, normalized_tfidf in candidates:
        doc_len = index.DL.get(id_term[0])
        if id_term[0] in dict_cosine_sim:
            dict_cosine_sim[id_term[0]] += normalized_tfidf / (len(search_query) * doc_len)
        else:
            dict_cosine_sim[id_term[0]] = normalized_tfidf / (len(search_query) * doc_len)
    return dict_cosine_sim


# **************************************** BINARY ****************************************
def binary_docs(query_to_search, index, path):
    #normalize by the query length
    res = {}
    for term in np.unique(query_to_search):
        pl = read_posting_list(index, term, path)
        for doc_id, tf in pl:
            if doc_id in res:
                res[doc_id] = res[doc_id]+1
            else:
                res[doc_id] = 1
    return res

# **************************************** PUBLISH ****************************************
def publish_results(id_score_dict):
    res = []
    srtd_list = sorted(id_score_dict.items(), key= lambda x: x[1], reverse=True)
    for doc_id, score in srtd_list:
        if doc_id in doc2title:
            res.append((doc_id, doc2title[doc_id]))
    return res

# **************************************** BM25 ****************************************
class BM25:
    """
    Best Match 25.

    Parameters to tune
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    Attributes
    ----------
    index :
    """

    def __init__(self, index, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.doc_len_ = index.DL
        self.df_ = index.df
        self.N_ = index.corpus_size
        self.avgdl_ = avgdl
        self.idf = index.idf_dict

    def bm25_score(self, tf, term, doc_id):
        """
        This function calculate the bm25 score for given query and document.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        B = 1 - self.b + self.b * (self.doc_len_[doc_id] / self.avgdl_)
        BM25_sum = (((self.k1 + 1) * tf) / (B * self.k1 + tf)) * self.idf[term]
        return BM25_sum

    def search(self, query, index, path):
        """
        This function calculate the bm25 score for given query and document.
        We need to check only documents which are 'candidates' for a given query.
        This function return a dictionary of scores as the following:
                                                                    key: query_id
                                                                    value: a ranked list of pairs (doc_id, score) in the length of N.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        scores_dict: a dictionary of N scores as the following:
                                                                    key: doc_id
                                                                    value: bm25 score
        """
        res = {}
        for term in query:
            list_of_doc = read_posting_list(index, term, path)
            for doc, tf in list_of_doc:
                if doc in res:
                    res[doc] += self.bm25_score(tf, term, doc)
                else:
                    res[doc] = self.bm25_score(tf, term, doc)
        return res

# **************************************** PAGE RANK ****************************************
def page_rank_res(doc_lst):
    res = []
    for doc_id in doc_lst:
        res.append(pagerank_dict[doc_id])
    return res

# **************************************** PAGE VIEW ****************************************
def page_view_res(doc_lst):
    res = []
    for doc_id in doc_lst:
        res.append(page_views[doc_id])
    return res

# **************************************** SEARCHES ****************************************
@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query = tokenize(query)
    if len(query) > 2:
        bimi = BM25(body_index)
        BM = bimi.search(query, body_index, body_path_bins)
        for doc_id in list(BM.keys()):
            BM[doc_id] += pagerank_dict.get(doc_id, 0)
            BM[doc_id] += page_views.get(doc_id, 0)
        res = publish_results(BM)[:25]
    else:
        title = binary_docs(query, title_index, title_path_bins)
        res = publish_results(title)[:25]
    # END SOLUTION
    return jsonify(res)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query = tokenize(query)
    # query = expand_query(query, body_index, model_fasttext)
    doc2score_dict = cosine_similarity(get_candidate_documents_and_scores(query, body_index, body_path_bins), query, body_index)
    res = publish_results(doc2score_dict)[:100]
    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query = tokenize(query)
    res = publish_results(binary_docs(query, title_index, title_path_bins))
    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query = tokenize(query)
    res = publish_results(binary_docs(query, anchor_index, anchor_path_bins))
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = page_rank_res(wiki_ids)
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = page_view_res(wiki_ids)
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=False)
