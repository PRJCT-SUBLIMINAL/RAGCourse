import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

# response = client.chat.completions.create(
#     model="openrouter/free",
#     messages=[
#         {
#             "role": "user",
#             "content": "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum"
#         }
#     ]
# )

# print(response.choices[0].message.content)
# print(f"Prompt tokens: {response.usage.prompt_tokens}")
# print(f"Response tokens: {response.usage.completion_tokens}")

def perform_prompt(query):
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": query
            }
        ]
    )
    return response.choices[0].message.content

def enhance_query(query: str, enhance: str) -> str:

    match enhance:

        case "spell":
            enhanced_query = perform_prompt(f"""Fix any spelling error in the user-provided movie search query below.
                Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
                Preserve punctuation and capitalization unless a change is require for the typo fix.
                If there are no spelling errors, or if you're unsure, output the original query unchanged.
                Output only the final query text, nothing else.

                User query: "{query}"
                """)

        case "rewrite":
            enhanced_query = perform_prompt(f"""Rewrite the user-provided movie search query below to be more specific and searchable.

                Consider:
                - Common movie knowledge (famous actors, popular films)
                - Genre conventions (horror = scary, animation = cartoon)
                - Keep the rewritten query concise (under 10 words)
                - It should be a Google-style search query, specific enough to yield relevant results
                - Don't use boolean logic

                Examples:
                - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                If you cannot improve the query, output the original unchanged.
                Output only the rewritten query text, nothing else.

                User query: "{query}"
                """)

        case "expand":
            enhanced_query = perform_prompt(f"""Expand the user-provided movie search query below with related terms.
                
                Add synonyms and related concepts that might appear in movie descriptions.
                Keep expansions relevant and focused.
                Output only the additional terms; they will be appended to the original query.

                Examples:
                - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                - "action movie with bear" -> "action thriller bear chase fight adventure"
                - "comedy with bear" -> "comedy funny bear humor lighthearted"

                User query: "{query}"
                """)

        case _:
            raise ValueError("Wrong/No enhance method provided")

    return enhanced_query