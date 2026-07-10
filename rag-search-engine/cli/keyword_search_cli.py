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
    
    def __add_document(self, doc_id, text):
        tokens = createTokens(text)
        c = collections.Counter()
        for i in range(len(tokens)):
            self.index[tokens[i]].add(doc_id)
            c[tokens[i]] += 1
        self.term_frequencies[doc_id] = c

    def get_documents(self, term):
        return sorted(list(self.index[term]))

    def get_tf(self, doc_id, term):
        c = self.term_frequencies.get(doc_id)
        if c is None:
            return 0
        return c[term]

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

    def load(self):
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)
        with open("cache/term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)

index = InvertedIndex()

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

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f'Searching for: {args.query}')
            try:
                index.load()
            
            except FileNotFoundError:
                print("Failed to load index")
                sys.exit(1)

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
            try:
                index.load()
            
            except FileNotFoundError:
                print("Failed to load index")
                sys.exit(1)

            token = tokenizeTerm(args.term)

            result = index.get_tf(args.doc_id, token)
            print(result)

        case "idf":
            try:
                index.load()
            
            except:
                print("Failed to load index")
                sys.exit(1)
            
            token = tokenizeTerm(args.term)

            total_doc_count = len(index.docmap)
            term_match_doc_count = len(index.get_documents(token))

            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))

            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()