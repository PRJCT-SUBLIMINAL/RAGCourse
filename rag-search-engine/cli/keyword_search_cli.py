import argparse
import json
import string
import collections
import pickle
import os
import sys

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

def load_movies():
    with open("data/movies.json", "r", encoding="utf-8") as file:
        jsonFile = json.load(file)
        return jsonFile["movies"]

class InvertedIndex:
    def __init__(self):
        self.index = collections.defaultdict(set)
        self.docmap = dict()
    
    def __add_document(self, doc_id, text):
        tokens = createTokens(text)
        for i in range(len(tokens)):
            self.index[tokens[i]].add(doc_id)

    def get_documents(self, term):
        return sorted(list(self.index[term]))

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

    def load(self):
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)

index = InvertedIndex()

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build the inverted index")

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

            # docs = index.get_documents("merida")
            # print(f"First document for token 'merida' = {docs[0]}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()