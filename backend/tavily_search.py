from tavily import TavilyClient

from .config import TAVILY_API_KEY


def search_web(query: str):

    client = TavilyClient(
        api_key=TAVILY_API_KEY
    )

    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=5,
        include_answer=True
    )

    results = []


    # Tavily generated answer
    if response.get("answer"):

        results.append({
            "title": "Tavily Answer",

            "content": response["answer"],

            "url": ""
        })


    # Search results
    for result in response.get(
        "results",
        []
    ):

        results.append({

            "title": result.get(
                "title",
                ""
            ),

            "content": result.get(
                "content",
                ""
            ),

            "url": result.get(
                "url",
                ""
            )
        })


    return results