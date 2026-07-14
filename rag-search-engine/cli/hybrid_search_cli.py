import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of scores")
    normalize_parser.add_argument("scores_list", nargs="*", type=float, help="A list of scores, inf args")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            if args.scores_list is None:
                raise ValueError("No scores were given.")

            min_score = min(args.scores_list)
            max_score = max(args.scores_list)

            values = []

            for score in args.scores_list:
                if min_score == max_score:
                    values.append(1.0)
                else:
                    values.append((score - min_score) / (max_score - min_score))

            for value in values:
                print(f"* {value:.4f}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()