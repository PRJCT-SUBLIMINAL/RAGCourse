import json

def load_movies():
    with open("data/movies.json", "r", encoding="utf-8") as file:
        jsonFile = json.load(file)
        return jsonFile["movies"]