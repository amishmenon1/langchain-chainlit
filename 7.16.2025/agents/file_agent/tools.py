"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

import asyncio
from typing import Any, List, Optional, cast

from langchain_core.tools import tool
from langchain_tavily import TavilySearch  # type: ignore[import-not-found]

from agents.analysis_agent.configuration import Configuration


@tool
def search(query: str) -> Optional[dict[str, Any]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.

    Args:
        query: The search query to find relevant information

    Returns:
        Dictionary containing search results with relevant information
    """
    print(f"performing search with query: {query}")
    configuration = Configuration.from_context()
    wrapped = TavilySearch(max_results=configuration.max_search_results)

    # Use asyncio.run to handle the async call in a sync context
    loop = None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        pass

    if loop is not None:
        # We're in an async context, need to use a new thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                asyncio.run, wrapped.ainvoke({"query": query}))
            result = future.result()
    else:
        # We're not in an async context, can use asyncio.run directly
        result = asyncio.run(wrapped.ainvoke({"query": query}))

    return cast(dict[str, Any], result)


TOOLS = [search]
