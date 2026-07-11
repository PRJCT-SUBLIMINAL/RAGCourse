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

BM25_K1 = 1.5
BM25_B = 0.75

translator = str.maketrans("", "", string.punctuation)

words = []

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

def load_movies():
    with open("data/movies.json", "r", encoding="utf-8") as file:
        jsonFile = json.load(file)
        return jsonFile["movies"]

class InvertedIndex:
    def __init__(self):
        self.index = collections.defaultdict(set)
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
        return ranked_scores[:limit]

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

index = InvertedIndex()

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

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build the inverted index")

    tf_parser = subparsers.add_parser("tf", help="Search with a specific term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="The search term")

    idf_parser = subparsers.add_parser("idf", help="Inverse search term")
    idf_parser.add_argument("term", type=str, help="The search term")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get the relevance score")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="The search term")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default=5, help="Limit the result to N amount")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f'Searching for: {args.query}')
            loadIndex()

            results = []
            seen_ids = set()

            tokens = createTokens(args.query)
            for token in tokens:
                if len(results) == 5:
                    break
                documents = index.get_documents(token)
                for doc_id in documents:
                    if len(results) == 5:
                        break
                    if doc_id in seen_ids:
                        continue
                    seen_ids.add(doc_id)
                    results.append(index.docmap[doc_id])

            for result in results:
                print(f"{result['id']}. {result['title']}")
        
        case "build":
            index.build()
            index.save()
        
        case "tf":
            loadIndex()

            token = tokenizeTerm(args.term)

            result = index.get_tf(args.doc_id, token)
            print(result)

        case "idf":
            loadIndex()
            
            token = tokenizeTerm(args.term)

            total_doc_count = len(index.docmap)
            term_match_doc_count = len(index.get_documents(token))

            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))

            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            loadIndex()

            token = tokenizeTerm(args.term)

            tf = index.get_tf(args.doc_id, token)

            total_doc_count = len(index.docmap)
            term_match_doc_count = len(index.get_documents(token))

            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))

            tf_idf = tf * idf

            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case "bm25idf":
            bm25_idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")

        case "bm25tf":
            bm25_tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf:.2f}")

        case "bm25search":
            loadIndex()
            count = 0
            search_results = index.bm25_search(args.query, args.limit)
            for result in search_results:
                count += 1
                doc_id = result[0]
                doc_score = result[1]
                doc_title = index.docmap[doc_id]["title"]
                print(f"{count}. ({doc_id}) {doc_title} - Score: {doc_score:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()