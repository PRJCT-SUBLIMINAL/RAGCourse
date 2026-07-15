import argparse
from lib.hybrid_search import normalize_scores, weighted_search, rrf_search

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of scores")
    normalize_parser.add_argument("scores_list", nargs="*", type=float, help="A list of scores, inf args")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Get a weighted hybrid score based on query")
    weighted_search_parser.add_argument("query", type=str, help="The query to run the weighted search with")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="A float value used to dynamically control the weighting between 2 scores")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="Limit the search results to this value")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="Get an rrf score based on query")
    rrf_search_parser.add_argument("query", type=str, help="The query to run the rrf search with")
    rrf_search_parser.add_argument("-k", type=int, default=60, help="The k parameter to control how much more weight to give")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="Limit the search results to this value")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite"], help="Query enhancement method")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_scores(args.scores_list)

        case "weighted-search":
            weighted_search(args.query, args.alpha, args.limit)

        case "rrf-search":
            rrf_search(args.query, args.k, args.limit, args.enhance)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()