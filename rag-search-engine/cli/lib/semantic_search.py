from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
import re
from lib.keyword_search import load_movies
from lib.search_utils import format_search_result

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
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

        os.makedirs("cache", exist_ok=True)
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

def chunk_text(text, chunk_size=200, overlap=0):
    words = text.split(" ")

    chunks = []

    for i in range(0, len(words), chunk_size):
        string = ""
        if overlap > 0 and i > 0:
            string = " ".join(words[i - overlap : i + chunk_size])
        else:
            string = " ".join(words[i : i + chunk_size])
        chunks.append(string)

    print(f"Chunking {len(text)} characters")

    for i in range(len(chunks)):
        print(f"{i+1}. {chunks[i]}")

def semantic_chunk_text(text, max_chunk_size=4, overlap=0):
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    
    for i in range(0, len(sentences), max_chunk_size - overlap):
        chunks.append(" ".join(sentences[i : i + max_chunk_size]))
        if i >= len(sentences) - max_chunk_size:
            break

    print(f"Semantically chunking {len(text)} characters")

    # for i in range(len(chunks)):
    #     print(f"{i+1}. {chunks[i]}")

    return chunks

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.documents = None
        self.document_map = dict()

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        
        chunks = []
        metas = []

        for i in range(len(documents)):
            doc = documents[i]
            self.document_map[doc["id"]] = doc
            
            if doc["description"] is None or doc["description"].strip() == "":
                continue
            
            chunked_text = semantic_chunk_text(doc["description"], 4, 1)
            chunks += chunked_text

            for j in range(len(chunked_text)):
                movie_idx = i
                chunk_idx = j
                total_chunks = len(chunked_text)

                metas.append({"movie_idx": movie_idx, "chunk_idx": chunk_idx, "total_chunks": total_chunks})

        self.chunk_embeddings = self.model.encode(chunks)
        self.chunk_metadata = metas

        os.makedirs("cache", exist_ok=True)
        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)

        with open("cache/chunk_metadata.json", "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(chunks)}, f, indent=2)

        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for i in range(len(self.documents)):
            doc = documents[i]
            self.document_map[doc["id"]] = doc
        
        if os.path.exists("cache/chunk_embeddings.npy") and os.path.exists("cache/chunk_metadata.json"):
            self.chunk_embeddings = np.load("cache/chunk_embeddings.npy")
            
            with open("cache/chunk_metadata.json", "r", encoding="utf-8") as f:
                jsonFile = json.load(f)
                self.chunk_metadata = jsonFile["chunks"]

            return self.chunk_embeddings
        
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 5):
        embedding = self.generate_embedding(query)

        chunk_scores = []

        for i in range(len(self.chunk_embeddings)):
            current_embedding = self.chunk_embeddings[i]

            cs = cosine_similarity(current_embedding, embedding)
            chunk_scores.append({"chunk_idx": i, "movie_idx": self.chunk_metadata[i]["movie_idx"], "score": cs})
        
        best_chunk_scores = dict()

        for i in range(len(chunk_scores)):
            chunk_score = chunk_scores[i]
            score = chunk_score["score"]
            movie_idx = chunk_score["movie_idx"]

            if movie_idx in best_chunk_scores:
                if score < best_chunk_scores[movie_idx]["score"]:
                    continue

            
            
            best_chunk_scores[movie_idx] = chunk_score

        sorted_scores = sorted(best_chunk_scores.values(), key=lambda pair: pair["score"], reverse=True)

        top_scores = sorted_scores[:limit]

        results = []

        for chunk_score in top_scores:
            movie_idx = chunk_score["movie_idx"]
            doc = self.documents[movie_idx]

            results.append(format_search_result(doc["id"], doc["title"], doc["description"][:100], chunk_score["score"]))

        return results

def embed_chunks():
    movies = load_movies()
    css = ChunkedSemanticSearch()
    embeddings = css.load_or_create_chunk_embeddings(movies)

    print(f"Generated {len(embeddings)} chunked embeddings")

    return css

def search_chunked(query: str, limit: int = 5):
    css = embed_chunks()
    results = css.search_chunks(query, limit)

    for result in results:
        title = result["title"]
        score = result["score"]
        document = result["document"]

        print(f"\n{result["id"]}. {title} (score: {score:.4f})")
        print(f"    {document}...")
    