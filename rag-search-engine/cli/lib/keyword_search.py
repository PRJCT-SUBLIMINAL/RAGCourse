import argparse
import json
import string
import collections
import pickle
import os
import sys
import collections
import math

from nltk.stem import PorterStemmer
from lib.search_utils import format_search_result

BM25_K1 = 1.5
BM25_B = 0.75

class InvertedIndex:
    def __init__(self):
        self.index = collections.defaultdict(set)
        self.index_path = "cache/index.pkl"
        self.docmap = dict()
        self.term_frequencies = dict()
        self.doc_lengths = dict()
        self.doc_lengths_path = os.path.join("cache", "doc_lengths.pkl")
    
    def __add_document(self, doc_id, text):
        tokens = createTokens(text)
        self.doc_lengths[doc_id] = len(tokens)
        c = collections.Counter()
        for i in range(len(tokens)):
            self.index[tokens[i]].add(doc_id)
            c[tokens[i]] += 1
        self.term_frequencies[doc_id] = c

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0

        total_of_lengths = 0
        for doc_id in self.doc_lengths:
            total_of_lengths += self.doc_lengths[doc_id]
        avg_doc_length = total_of_lengths / len(self.doc_lengths)
        
        return float(avg_doc_length)

    def get_documents(self, term):
        return sorted(list(self.index[term]))

    def get_tf(self, doc_id, term):
        c = self.term_frequencies.get(doc_id)
        if c is None:
            return 0
        return c[term]

    def get_bm25_idf(self, term: str) -> float:
        total_documents = len(self.docmap)
        df = len(self.get_documents(term))

        return math.log((total_documents - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        tf = self.get_tf(doc_id, term)
        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return bm25_tf

    def bm25(self, doc_id, term):
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        return tf * idf

    def bm25_search(self, query, limit=5):
        tokens = createTokens(query)
        scores = dict()
        for doc_id in self.docmap:
            total = 0
            for token in tokens:
                total += self.bm25(doc_id, token)
            scores[doc_id] = total
        ranked_scores = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        
        results = []
        
        for i in range(len(ranked_scores)):
            ranked_score = ranked_scores[i]
            doc_id = ranked_score[0]
            doc = self.docmap[doc_id]
            score = ranked_score[1]

            results.append(format_search_result(doc["id"], doc["title"], doc["description"][:100], score))

        return results[:limit]

    def build(self):
        movies = load_movies()
        for m in movies:
            self.docmap[m["id"]] = m
            self.__add_document(m["id"], f"{m['title']} {m['description']}")

    def save(self):
        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open("cache/doc_lengths.pkl", "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)
        with open("cache/term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open("cache/doc_lengths.pkl", "rb") as f:
            self.doc_lengths = pickle.load(f)

translator = str.maketrans("", "", string.punctuation)

words = []
index = InvertedIndex()

def load_movies():
    with open("data/movies.json", "r", encoding="utf-8") as file:
        jsonFile = json.load(file)
        return jsonFile["movies"]

def preprocessWords():
    with open("data/stopwords.txt", "r", encoding="utf-8") as f:
        file = f.read().splitlines()

    for i in range(len(file)):
        words.append(file[i].translate(translator))

preprocessWords()

def createTokens(text):
    stemmer = PorterStemmer()

    text = text.lower().translate(translator)

    tokens = text.split()

    tokens = [token for token in tokens if token not in words]

    return [stemmer.stem(token) for token in tokens]

def tokenizeTerm(term):
    tokens = createTokens(term)
    if len(tokens) != 1:
        raise Exception("Term holds more than 1 token")
    return tokens[0]

def loadIndex():
    try:
        index.load()
            
    except FileNotFoundError:
        print("Failed to load index")
        sys.exit(1)

def bm25_idf_command(term):
    loadIndex()
    token = tokenizeTerm(term)

    bm25_idf = index.get_bm25_idf(token)
    return float(bm25_idf)

def bm25_tf_command(doc_id, term, k1=BM25_K1, b=BM25_B):
    loadIndex()
    token = tokenizeTerm(term)
    bm25_tf = index.get_bm25_tf(doc_id, token, k1, b)
    return float(bm25_tf)