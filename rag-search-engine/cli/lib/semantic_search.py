from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
from lib.keyword_search import load_movies

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self.embeddings = None
        self.documents = None
        self.document_map = dict()

    def generate_embedding(self, text):
        text = text.strip()
        if text == "":
            raise ValueError("Text is blank.")
        
        embedding = self.model.encode([f"{text}"])
        return embedding[0]

    def build_embeddings(self, documents):
        movie_list = []
        self.documents = documents
        for i in range(len(documents)):
            doc = documents[i]
            self.document_map[doc["id"]] = doc
            movie_list.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(movie_list)
        np.save("cache/movie_embeddings.npy", self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for i in range(len(documents)):
            doc = documents[i]
            self.document_map[doc["id"]] = doc
        
        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy")
            if len(self.embeddings) == len(documents):
                return self.embeddings
        
        return self.build_embeddings(documents)

    def search(self, query, limit=5):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        results = []

        embedding = self.generate_embedding(query)
        for i in range(len(self.document_map)):
            similarity = cosine_similarity(self.embeddings[i], embedding)
            results.append((similarity, self.documents[i]))

        final_results = sorted(results, key=lambda pair: pair[0], reverse=True)

        final_list = []

        for i in range(len(final_results[:limit])):
            result = final_results[i]
            final_list.append({"score": result[0], "title": result[1]["title"], "description": result[1]["description"]})

        return final_list

def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")

def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    ss = SemanticSearch()
    movies = []
    with open("data/movies.json", "r", encoding="utf-8") as f:
        jsonFile = json.load(f)
        movies = jsonFile["movies"]
    
    embeddings = ss.load_or_create_embeddings(movies)

    print(f"Number of docs: {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def semantic_search(query, limit=5):
    ss = SemanticSearch()
    movies = load_movies()
    ss.load_or_create_embeddings(movies)
    
    results = ss.search(query, limit)

    for i in range(len(results)):
        result = results[i]
        print(f"{i + 1}. {result["title"]} (score: {result["score"]:.4f})\n\t{result["description"]}")

def chunk_text(text, chunk_size=200):
    words = text.split(" ")

    chunks = []

    for i in range(0, len(words), chunk_size):
        string = " ".join(words[i:i+chunk_size])
        chunks.append(string)

    print(f"Chunking {len(text)} characters")

    for i in range(len(chunks)):
        print(f"{i+1}. {chunks[i]}")