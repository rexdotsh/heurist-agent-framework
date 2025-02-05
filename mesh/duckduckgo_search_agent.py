import json
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from duckduckgo_search import DDGS  # pip install duckduckgo_search

from core.llm import call_llm_async, call_llm_with_tools_async

from .mesh_agent import MeshAgent, monitor_execution, with_cache, with_retry

load_dotenv()


class DuckDuckGoSearchAgent(MeshAgent):
    def __init__(self):
        super().__init__()
        self.metadata.update(
            {
                "name": "DuckDuckGo Search Agent",
                "version": "1.0.0",
                "author": "Heurist Team",
                "author_address": "0x7d9d1821d15B9e0b8Ab98A058361233E255E405D",
                "description": (
                    "Fetch and analyze web search results using DuckDuckGo API. "
                    "Analyze content relevance, source credibility, information completeness, "
                    "and provide intelligent summaries. This agent helps you gather and understand "
                    "information from across the web with AI-powered analysis."
                ),
                "inputs": [
                    {
                        "name": "query",
                        "description": "The search query to analyze",
                        "type": "str",
                    }
                ],
                "outputs": [
                    {
                        "name": "response",
                        "description": "Analysis and explanation of search results",
                        "type": "str",
                    },
                    {
                        "name": "data",
                        "description": "The raw search results data from DuckDuckGo API",
                        "type": "dict",
                    },
                ],
                "external_apis": ["duckduckgo"],
                "tags": ["Search", "Web", "AI"],
            }
        )

    @monitor_execution()
    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def fetch_search_results(
        self, query: str, max_results: int = 5
    ) -> Optional[Dict]:
        """
        Fetch search results from DuckDuckGo API with retry and caching
        """
        try:
            with DDGS() as ddgs:
                results = []
                for r in ddgs.text(query, max_results=max_results):
                    results.append(
                        {"title": r["title"], "link": r["href"], "snippet": r["body"]}
                    )
                return {"results": results}

        except Exception as e:
            print(f"Error fetching search results: {e}")
            return None

    def get_system_prompt(self) -> str:
        return """You are an AI assistant that analyzes web search results.
        You should analyze the search results and provide insights about:
        1. Content relevance to the query
        2. Source credibility and diversity
        3. Information completeness and potential gaps
        4. Key themes and findings
        5. Any conflicting information or biases
        If the results are insufficient or irrelevant, explain why.
        Provide a clear and concise analysis focusing on the most important insights."""

    def get_tool_schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": "fetch_search_results",
                "description": "Fetch web search results",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    @monitor_execution()
    @with_retry(max_retries=3)
    async def handle_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query")
        if not query:
            raise ValueError("Query parameter is required")

        max_results = params.get("max_results", 5)

        response = await call_llm_with_tools_async(
            base_url=self.heurist_base_url,
            api_key=self.heurist_api_key,
            model_id=self.large_model_id,
            system_prompt=self.get_system_prompt(),
            user_prompt=query,
            temperature=0.1,
            tools=[self.get_tool_schema()],
        )

        if not response or not response.get("tool_calls"):
            return {"response": response.get("content")}

        tool_call = response["tool_calls"]
        function_args = json.loads(tool_call.function.arguments)
        tool_result = await self.fetch_search_results(
            function_args["query"], max_results
        )

        explanation = await call_llm_async(
            base_url=self.heurist_base_url,
            api_key=self.heurist_api_key,
            model_id=self.large_model_id,
            messages=[
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": query},
                {
                    "role": "tool",
                    "content": str(tool_result),
                    "tool_call_id": tool_call.id,
                },
            ],
            temperature=0.7,
        )

        search_data = tool_result.get("results", []) if tool_result else []

        essential_search_info = {
            "query_info": {"query": query, "result_count": len(search_data)},
            "results": search_data,
        }

        return {"response": explanation, "data": essential_search_info}
