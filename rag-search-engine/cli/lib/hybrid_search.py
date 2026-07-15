import os

from .keyword_search import InvertedIndex, load_movies
from .semantic_search import ChunkedSemanticSearch

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        bm25_scores = []
        semantic_scores = []

        for result in bm25_results:
            score = result["score"]
            bm25_scores.append(score)

        for result in semantic_results:
            score = result["score"]
            semantic_scores.append(score)

        normalized_bm25_scores = normalize_scores(bm25_scores)
        normalized_semantic_scores = normalize_scores(semantic_scores)

        combined_scores = dict()

        for result, normalized_bm25_score in zip(bm25_results, normalized_bm25_scores):
            if result["id"] not in combined_scores:
                combined_scores[result["id"]] = {
                    "title": result["title"],
                    "document": result["document"],
                    "bm25_score": normalized_bm25_score,
                    "semantic_score": 0.0
                }
            else:
                combined_score = combined_scores[result["id"]]
                combined_score["bm25_score"] = normalized_bm25_score

        for result, normalized_semantic_score in zip(semantic_results, normalized_semantic_scores):
            if result["id"] not in combined_scores:
                combined_scores[result["id"]] = {
                    "title": result["title"],
                    "document": result["document"],
                    "bm25_score": 0.0,
                    "semantic_score": normalized_semantic_score
                }
            else:
                combined_score = combined_scores[result["id"]]
                combined_score["semantic_score"] = normalized_semantic_score

        combined_scores_list = []

        for item in combined_scores.items():
            doc_id = item[0]
            values = item[1]

            values["hybrid_score"] = hybrid_score(values["bm25_score"], values["semantic_score"], alpha)

            combined_scores_list.append(values)

        sorted_combined_scores_list = sorted(combined_scores_list, key=lambda result: result["hybrid_score"], reverse=True)

        return sorted_combined_scores_list[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

def normalize_scores(scores_list):
    if scores_list is None:
        raise ValueError("No scores were given.")

    min_score = min(scores_list)
    max_score = max(scores_list)

    values = []

    for score in scores_list:
        if min_score == max_score:
            values.append(1.0)
        else:
            values.append((score - min_score) / (max_score - min_score))

    return values

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def weighted_search(query, alpha: float = 0.5, limit: int = 5) -> list:
    movies = load_movies()
    hs = HybridSearch(movies)

    results = hs.weighted_search(query, alpha, limit)
    for i in range(len(results)):
        result = results[i]
        print(f"{i+1}. {result['title']}")
        print(f"  Hybrid Score: {result['hybrid_score']}")
        print(f"  BM25: {result['bm25_score']:.4f}, Semantic: {result['semantic_score']:.4f}")
        print(f"  {result['document']}...")