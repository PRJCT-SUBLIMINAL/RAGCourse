import os

from .keyword_search import InvertedIndex, load_movies
from .semantic_search import ChunkedSemanticSearch
from .search_utils import format_search_result
from .prompt_utils import perform_prompt

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

    def rrf_search(self, query: str, k: int = 60, limit: int = 5, enhance=None) -> list[dict]:
        if enhance is not None and enhance == "spell":
            enhanced_query = perform_prompt(f"""Fix any spelling error in the user-provided movie search query below.
            Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
            Preserve punctuation and capitalization unless a change is require for the typo fix.
            If there are no spelling errors, or if you're unsure, output the original query unchanged.
            Output only the final query text, nothing else.
            User query: "{query}"
            """)
            print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")
            query = enhanced_query

        bm25_results = self._bm25_search(query, limit * 500)
        semantic_chunk_results = self.semantic_search.search_chunks(query, limit * 500)

        combined_ranks = dict()

        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result["id"]
            if doc_id not in combined_ranks:
                combined_ranks[doc_id] = {
                    "title": result["title"],
                    "document": result["document"],
                    "rrf_score": 0.0,
                    "bm25_rank": None,
                    "semantic_rank": None
                }
            combined_ranks[doc_id]["bm25_rank"] = rank
            combined_ranks[doc_id]["rrf_score"] += rrf_score(rank, k)

        for rank, result in enumerate(semantic_chunk_results, start=1):
            doc_id = result["id"]
            if doc_id not in combined_ranks:
                combined_ranks[doc_id] = {
                    "title": result["title"],
                    "document": result["document"],
                    "rrf_score": 0.0,
                    "bm25_rank": None,
                    "semantic_rank": None
                }
            combined_ranks[doc_id]["semantic_rank"] = rank
            combined_ranks[doc_id]["rrf_score"] += rrf_score(rank, k)
        
        combined_ranks_list = []

        for item in combined_ranks.items():
            doc_id = item[0]
            values = item[1]

            combined_ranks_list.append(values)

        sorted_combined_ranks_list = sorted(combined_ranks_list, key=lambda result: result["rrf_score"], reverse=True)

        return sorted_combined_ranks_list[:limit]


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

def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)

def rrf_search(query, k: int = 60, limit: int = 5, enhance=""):
    movies = load_movies()
    hs = HybridSearch(movies)

    results = hs.rrf_search(query, k, limit, enhance)

    for i in range(len(results)):
        result = results[i]
        print(f"{i+1}. {result["title"]}")
        print(f"  RRF Score: {result["rrf_score"]:.3f}")
        print(f"  BM25 Rank: {result["bm25_rank"]}, Semantic Rank: {result["semantic_rank"]}")
        print(f"  {result["document"]}...")