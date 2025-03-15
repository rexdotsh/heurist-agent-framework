import asyncio
import json
import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from firecrawl import FirecrawlApp

from core.llm import call_llm_async, call_llm_with_tools_async
from decorators import monitor_execution, with_cache, with_retry

from .mesh_agent import MeshAgent

load_dotenv()
logger = logging.getLogger(__name__)


class FirecrawlSearchAgent(MeshAgent):
    def __init__(self):
        super().__init__()
        self.metadata.update(
            {
                "name": "Firecrawl Search Agent",
                "version": "1.0.0",
                "author": "Heurist Team",
                "author_address": "0x7d9d1821d15B9e0b8Ab98A058361233E255E405D",
                "description": "Advanced search agent that uses Firecrawl to perform research with intelligent query generation and content analysis.",
                "inputs": [
                    {
                        "name": "query",
                        "description": "Natural language research query to analyze",
                        "type": "str",
                        "required": False,
                    },
                    {
                        "name": "raw_data_only",
                        "description": "If true, returns only raw data without analysis",
                        "type": "bool",
                        "required": False,
                        "default": False,
                    },
                ],
                "outputs": [
                    {"name": "response", "description": "Natural language analysis of search results", "type": "str"},
                    {"name": "data", "description": "Structured search results and metadata", "type": "dict"},
                ],
                "external_apis": ["Firecrawl"],
                "tags": ["Search", "Research", "Analysis"],
            }
        )
        self.app = FirecrawlApp(api_key=os.environ.get("FIRECRAWL_KEY", ""))

    def get_system_prompt(self) -> str:
        return """You are an expert research analyst that processes web search results.

        Your capabilities:
        1. Execute targeted web searches on specific topics
        2. Analyze search results for key findings and patterns

        For search results:
        1. Only use relevant, good quality, credible information
        2. Extract key facts and statistics
        3. Present the information like a human, not a robot

        Return analyses in clear natural language with concrete findings. Do not make up any information."""

    def get_tool_schemas(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "execute_web_search",
                    "description": "Execute a web search query by reading the web pages. It provides more comprehensive information than standard web search by extracting the full contents from the pages. Use this when you need in-depth information on a topic. Data comes from Firecrawl search API. It may fail to find information of niche topics such like small cap crypto projects.",
                    "parameters": {
                        "type": "object",
                        "properties": {"search_term": {"type": "string", "description": "The search term to execute"}},
                        "required": ["search_term"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "extract_web_data",
                    "description": "Extract structured data from one or multiple web pages using natural language instructions. This tool can process single URLs or entire domains (using wildcards like example.com/*). Use this when you need specific information from websites rather than general search results. You must specify what data to extract from the pages using the 'extraction_prompt' parameter.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "urls": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of URLs to extract data from. Can include wildcards (e.g., 'example.com/*') to crawl entire domains."
                            },
                            "extraction_prompt": {
                                "type": "string",
                                "description": "Natural language description of what data to extract from the pages."
                            },
                            # "enable_web_search": {
                            #     "type": "boolean",
                            #     "description": "When true, extraction can follow links outside the specified domain.",
                            #     "default": False
                            # }
                        },
                        "required": ["urls", "extraction_prompt"],
                    },
                },
            },
        ]

    # ------------------------------------------------------------------------
    #                       SHARED / UTILITY METHODS
    # ------------------------------------------------------------------------
    async def _respond_with_llm(self, query: str, tool_call_id: str, data: dict, temperature: float) -> str:
        """Generate a natural language response using the LLM"""
        return await call_llm_async(
            base_url=self.heurist_base_url,
            api_key=self.heurist_api_key,
            model_id=self.metadata["large_model_id"],
            messages=[
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": query},
                {"role": "tool", "content": str(data), "tool_call_id": tool_call_id},
            ],
            temperature=temperature,
        )

    def _handle_error(self, maybe_error: dict) -> dict:
        """Check for and return any errors in the response"""
        if isinstance(maybe_error, dict) and "error" in maybe_error:
            return {"error": maybe_error["error"]}
        return {}

    # ------------------------------------------------------------------------
    #                      FIRECRAWL-SPECIFIC METHODS
    # ------------------------------------------------------------------------
    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def execute_web_search(self, query: str) -> Dict:
        """Execute a search with error handling"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.app.search(query=query, params={"scrapeOptions": {"formats": ["markdown"]}})
            )

            if isinstance(response, dict) and "data" in response:
                return response
            elif isinstance(response, dict) and "success" in response:
                return {"data": response.get("data", [])}
            elif isinstance(response, list):
                formatted_data = []
                for item in response:
                    if isinstance(item, dict):
                        formatted_data.append(item)
                    else:
                        formatted_data.append(
                            {
                                "url": getattr(item, "url", ""),
                                "markdown": getattr(item, "markdown", "") or getattr(item, "content", ""),
                                "title": getattr(item, "title", "") or getattr(item, "metadata", {}).get("title", ""),
                            }
                        )
                return {"data": formatted_data}
            else:
                return {"data": []}

        except Exception as e:
            logger.error(f"Search error: {e}")
            return {"error": f"Failed to execute search: {str(e)}"}

    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def extract_web_data(self, urls: List[str], extraction_prompt: str, enable_web_search: bool = False) -> Dict:
        """Extract structured data from web pages using natural language instructions"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.app.extract(
                    urls=urls, 
                    params={
                        "prompt": extraction_prompt,
                        "enableWebSearch": enable_web_search
                    }
                )
            )
            
            if isinstance(response, dict):
                if "data" in response:
                    return response
                elif "success" in response and response.get("success"):
                    return {"data": response.get("data", {})}
                else:
                    return {"error": "Extraction failed", "details": response}
            else:
                return {"data": response}
                
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            return {"error": f"Failed to extract data: {str(e)}"}

    # ------------------------------------------------------------------------
    #                      COMMON HANDLER LOGIC
    # ------------------------------------------------------------------------
    async def _handle_tool_logic(
        self, tool_name: str, function_args: dict, query: str, tool_call_id: str, raw_data_only: bool
    ) -> Dict[str, Any]:
        """Handle execution of specific tools and format responses"""

        if tool_name == "execute_web_search":
            search_term = function_args.get("search_term")
            if not search_term:
                return {"error": "Missing 'search_term' in tool_arguments"}

            result = await self.execute_web_search(search_term)
        elif tool_name == "extract_web_data":
            urls = function_args.get("urls")
            extraction_prompt = function_args.get("extraction_prompt")
            enable_web_search = function_args.get("enable_web_search", False)
            
            if not urls:
                return {"error": "Missing 'urls' in tool_arguments"}
            if not extraction_prompt:
                return {"error": "Missing 'extraction_prompt' in tool_arguments"}
                
            result = await self.extract_web_data(urls, extraction_prompt, enable_web_search)
        else:
            return {"error": f"Unsupported tool: {tool_name}"}

        errors = self._handle_error(result)
        if errors:
            return errors

        if raw_data_only:
            return {"response": "", "data": result}

        explanation = await self._respond_with_llm(query=query, tool_call_id=tool_call_id, data=result, temperature=0.7)

        return {"response": explanation, "data": result}

    @monitor_execution()
    @with_retry(max_retries=3)
    async def handle_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming messages, supporting both direct tool calls and natural language queries.

        Either 'query' or 'tool' is required in params.
          - If 'tool' is provided, call that tool directly with 'tool_arguments' (bypassing the LLM).
          - If 'query' is provided, route via LLM for dynamic tool selection.
        """
        query = params.get("query")
        tool_name = params.get("tool")
        tool_args = params.get("tool_arguments", {})
        raw_data_only = params.get("raw_data_only", False)

        # ---------------------
        # 1) DIRECT TOOL CALL
        # ---------------------
        if tool_name:
            return await self._handle_tool_logic(
                tool_name=tool_name,
                function_args=tool_args,
                query=query or "Direct tool call without LLM.",
                tool_call_id="direct_tool",
                raw_data_only=raw_data_only,
            )

        # ---------------------
        # 2) NATURAL LANGUAGE QUERY (LLM decides the tool)
        # ---------------------
        if query:
            response = await call_llm_with_tools_async(
                base_url=self.heurist_base_url,
                api_key=self.heurist_api_key,
                model_id=self.metadata["large_model_id"],
                system_prompt=self.get_system_prompt(),
                user_prompt=query,
                temperature=0.1,
                tools=self.get_tool_schemas(),
            )

            if not response:
                return {"error": "Failed to process query"}
            if not response.get("tool_calls"):
                return {"response": response["content"], "data": {}}

            tool_call = response["tool_calls"]
            tool_call_name = tool_call.function.name
            tool_call_args = json.loads(tool_call.function.arguments)

            return await self._handle_tool_logic(
                tool_name=tool_call_name,
                function_args=tool_call_args,
                query=query,
                tool_call_id=tool_call.id,
                raw_data_only=raw_data_only,
            )

        # ---------------------
        # 3) NEITHER query NOR tool
        # ---------------------
        return {"error": "Either 'query' or 'tool' must be provided in the parameters."}
