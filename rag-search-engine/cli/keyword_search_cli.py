import argparse
import json
import string

translator = str.maketrans("", "", string.punctuation)

def preprocessWords():
    words = []
    with open("data/stopwords.txt", "r", encoding="utf-8") as f:
        words = f.read().splitlines()

    for i in range(len(words)):
        words[i] = words[i].translate(translator)
    
    return words

def createTokens(text, stopWords):
    tokens = text.split(" ")

    tokens = [token for token in tokens if token.strip()]
    tokens = [token for token in tokens if token not in stopWords]

    return tokens

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
             print(f'Searching for: {args.query}')
             with open("data/movies.json", "r", encoding="utf-8") as file:
                jsonFile = json.load(file)
                movies = jsonFile["movies"]

                results = []

                stopWords = preprocessWords()

                for i in range(len(movies)):
                    if len(results) == 5: break
                    movie = movies[i]
                    query = createTokens(args.query.lower(), stopWords)
                    title = createTokens(movie.get("title", "").translate(translator).lower(), stopWords)

                    found = False
                    for queryToken in query:
                        for titleToken in title:
                            if found:
                                break
                            if queryToken in titleToken:
                                found = True
                                break
                    if found:
                        results.append(movie)

                for i in range(len(results)):
                    print(f'{i+1}. {results[i]["title"]}')
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()