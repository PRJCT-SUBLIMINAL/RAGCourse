import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, semantic_search, chunk_text, semantic_chunk_text, embed_chunks, search_chunked

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify the embedded model")

    embed_parser = subparsers.add_parser("embed_text", help="Embed text into the embedded model")
    embed_parser.add_argument("text", type=str, default="", help="The string/text to embed")

    verify_embed_parser = subparsers.add_parser("verify_embeddings", help="Verify the models embeddings")

    embed_query_parser = subparsers.add_parser("embed_query", help="Embed a query into th model")
    embed_query_parser.add_argument("query", type=str, help="The query string/text to embed")

    semantic_search_parser = subparsers.add_parser("search", help="Seach using semantics")
    semantic_search_parser.add_argument("query", type=str, help="A query to search")
    semantic_search_parser.add_argument("--limit", type=int, default=5, help="An optional limit for the output results")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk text")
    chunk_parser.add_argument("text", type=str, help="The text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="The optional chunking size. Defaults to 200")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="The amount of words to overlap")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Semantically chunk text")
    semantic_chunk_parser.add_argument("text", type=str, help="The text to semantically chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="The max chunk size of each chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="The amount of words to overlap")

    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Embed the documents data into chunks semantically")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search chunked embeddings")
    search_chunked_parser.add_argument("query", type=str, help="The query to search")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="The limit number of search results to print")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embed_query":
            embed_query_text(args.query)

        case "search":
            semantic_search(args.query, args.limit)

        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)

        case "semantic_chunk":
            semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)

        case "embed_chunks":
            embed_chunks()

        case "search_chunked":
            search_chunked(args.query, args.limit)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()