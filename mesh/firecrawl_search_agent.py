import json
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
import os
from core.llm import call_llm_async, call_llm_with_tools_async
from core.utils.text_splitter import trim_prompt
from .mesh_agent import MeshAgent, monitor_execution, with_cache, with_retry
import asyncio
import logging

logger = logging.getLogger(__name__)
load_dotenv()

@dataclass
class SearchQuery:
    query: str
    research_goal: str

class FirecrawlSearchAgent(MeshAgent):
    def __init__(self):
        super().__init__()
        self.metadata.update({
            "name": "Firecrawl Search Agent",
            "version": "1.0.0",
            "author": "Heurist Team",
            "author_address": "0x7d9d1821d15B9e0b8Ab98A058361233E255E405D",
            "description": (
                "Advanced search agent that uses Firecrawl to perform deep research "
                "with intelligent query generation, content analysis, and detailed reporting. "
                "Supports recursive search patterns and concurrent processing for thorough "
                "topic exploration."
            ),
            "inputs": [
                {
                    "name": "query",
                    "description": "The main research query or topic to analyze",
                    "type": "str",
                    "required": False
                },
                {
                    "name": "tool",
                    "description": "Direct tool name to call",
                    "type": "str", 
                    "required": False
                },
                {
                    "name": "tool_arguments",
                    "description": "Arguments for direct tool call",
                    "type": "dict",
                    "required": False
                },
                {
                    "name": "depth",
                    "description": "How many levels deep to research (1-3)",
                    "type": "int",
                    "required": False,
                    "default": 1
                },
                {
                    "name": "breadth",
                    "description": "Number of parallel searches per level (1-5)",
                    "type": "int",
                    "required": False,
                    "default": 3
                },
                {
                    "name": "raw_data_only",
                    "description": "If true, returns only raw search data without analysis",
                    "type": "bool",
                    "required": False,
                    "default": False
                }
            ],
            "outputs": [
                {
                    "name": "response",
                    "description": "Detailed analysis and research findings",
                    "type": "str",
                },
                {
                    "name": "data",
                    "description": "Raw search results and metadata",
                    "type": "dict",
                }
            ],
            "external_apis": ["Firecrawl"],
            "tags": ["Search", "Research", "Analysis"],
        })
        self.app = FirecrawlApp(api_key=os.environ.get("FIRECRAWL_KEY", ""))
        self._last_request_time = 0

    def get_system_prompt(self) -> str:
        return """You are an expert research analyst that processes web search results.
        Analyze the content and provide insights about:
        1. Key findings and main themes
        2. Source credibility and diversity
        3. Information completeness and gaps
        4. Emerging patterns and trends
        5. Potential biases or conflicting information
        
        Be thorough and detailed in your analysis. Focus on extracting concrete facts,
        statistics, and verifiable information. Highlight any uncertainties or areas
        needing further research.
        
        Return your analysis in a clear, structured format with sections for key findings,
        detailed analysis, and recommendations for further research."""

    def get_tool_schemas(self) -> List[Dict]:
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'search',
                    'description': 'Execute a search query using Firecrawl',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'query': {
                                'type': 'string',
                                'description': 'The search query to execute'
                            },
                            'limit': {
                                'type': 'integer',
                                'description': 'Maximum number of results to return',
                                'default': 5
                            }
                        },
                        'required': ['query'],
                    },
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'generate_search_queries',
                    'description': 'Generate multiple search queries for a topic',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'query': {
                                'type': 'string',
                                'description': 'The main topic to generate queries for'
                            },
                            'num_queries': {
                                'type': 'integer',
                                'description': 'Number of queries to generate',
                                'default': 3
                            }
                        },
                        'required': ['query'],
                    },
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'analyze_results',
                    'description': 'Analyze search results and generate insights',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'query': {
                                'type': 'string',
                                'description': 'The original search query'
                            },
                            'search_results': {
                                'type': 'object',
                                'description': 'The search results to analyze'
                            }
                        },
                        'required': ['query', 'search_results'],
                    },
                }
            }
        ]

    async def _respond_with_llm(
        self, query: str, tool_call_id: str, data: dict, temperature: float
    ) -> str:
        return await call_llm_async(
            base_url=self.heurist_base_url,
            api_key=self.heurist_api_key,
            model_id=self.metadata['large_model_id'],
            messages=[
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": query},
                {"role": "tool", "content": str(data), "tool_call_id": tool_call_id}
            ],
            temperature=temperature
        )

    def _handle_error(self, maybe_error: dict) -> dict:
        if 'error' in maybe_error:
            return {"error": maybe_error['error']}
        return {}

    @monitor_execution()
    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def search(self, query: str, limit: int = 5) -> Dict:
        """Execute search with rate limiting and error handling"""
        try:
            # Run the synchronous SDK call in a thread pool
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.app.search(
                    query=query,
                    params={"scrapeOptions": {"formats": ["markdown"]}}
                )
            )
            
            # Handle the response format from the SDK
            if isinstance(response, dict) and "data" in response:
                # Response is already in the right format
                return response
            elif isinstance(response, dict) and "success" in response:
                # Response is in the documented format
                return {"data": response.get("data", [])}
            elif isinstance(response, list):
                # Response is a list of results
                formatted_data = []
                for item in response:
                    if isinstance(item, dict):
                        formatted_data.append(item)
                    else:
                        # Handle non-dict items (like objects)
                        formatted_data.append({
                            "url": getattr(item, "url", ""),
                            "markdown": getattr(item, "markdown", "") or getattr(item, "content", ""),
                            "title": getattr(item, "title", "") or getattr(item, "metadata", {}).get("title", "")
                        })
                return {"data": formatted_data}
            else:
                print(f"Unexpected response format from Firecrawl: {type(response)}")
                return {"data": []}
            
        except Exception as e:
            print(f"Search error: {e}")
            print(f"Response type: {type(response) if 'response' in locals() else 'N/A'}")
            return {"data": []}

    async def generate_search_queries(self, query: str, num_queries: int = 3) -> List[SearchQuery]:
        """Generate intelligent search queries based on the input topic"""
        prompt = f"Generate {num_queries} specific search queries to investigate this topic: {query}"
        
        tools = [{
            "type": "function",
            "function": {
                "name": "generate_queries",
                "description": "Generate search queries for a topic",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "research_goal": {"type": "string"}
                                },
                                "required": ["query", "research_goal"]
                            }
                        }
                    },
                    "required": ["queries"]
                }
            }
        }]

        response = await call_llm_with_tools_async(
            base_url=self.heurist_base_url,
            api_key=self.heurist_api_key,
            model_id=self.metadata['large_model_id'],
            messages=[
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "generate_queries"}},
            temperature=0.7
        )

        try:
            # Extract the arguments from tool_calls
            if isinstance(response, dict) and 'tool_calls' in response:
                tool_call = response['tool_calls']
                if hasattr(tool_call, 'function'):
                    arguments = tool_call.function.arguments
                    if isinstance(arguments, str):
                        result = json.loads(arguments)
                        queries = result.get("queries", [])
                        return [SearchQuery(**q) for q in queries][:num_queries]
        except Exception as e:
            print(f"Error generating queries: {e}")
            print(f"Raw response: {response}")
            return [SearchQuery(query=query, research_goal="Main topic research")]
        
        return [SearchQuery(query=query, research_goal="Main topic research")]

    async def analyze_results(self, query: str, search_results: Dict) -> Dict[str, Any]:
        """Analyze search results and generate insights"""
        contents = [
            trim_prompt(item.get("markdown", ""), 25000)
            for item in search_results.get("data", [])
            if item.get("markdown")
        ]

        if not contents:
            return {
                "analysis": "No search results found to analyze.",
                "key_findings": [],
                "recommendations": []
            }

        prompt = (
            f"Analyze these search results for the query: {query}\n\n"
            f"Content:\n{' '.join(contents)}\n\n"
            f"Provide a detailed analysis including key findings, main themes, "
            f"and recommendations for further research. Return as JSON with "
            f"'analysis', 'key_findings', and 'recommendations' fields."
        )
        prompt_example = """
        IMPORTANT: MAKE SURE YOU RETURN THE JSON ONLY, NO OTHER TEXT OR MARKUP AND A VALID JSON.
        DONT ADD ANY COMMENTS OR MARKUP TO THE JSON. Example NO # or /* */ or /* */ or // or any other comments or markup.
        MAKE SURE YOU RETURN THE JSON ONLY, JSON SHOULD BE PERFECTLY FORMATTED. ALL KEYS SHOULD BE OPENED AND CLOSED.
        {
            "analysis": "Analysis of the search results",
            "key_findings": ["Key finding 1", "Key finding 2"],
            "recommendations": ["Recommendation 1", "Recommendation 2"]
        }
        """
        prompt = prompt + prompt_example
        response = await call_llm_async(
            base_url=self.heurist_base_url,
            api_key=self.heurist_api_key,
            model_id=self.metadata['large_model_id'],
            messages=[
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        try:
            return json.loads(response)
        except Exception as e:
            print(f"Error analyzing results: {e}")
            return {
                "analysis": "Error processing search results.",
                "key_findings": [],
                "recommendations": []
            }
    
    async def _handle_tool_logic(
        self, tool_name: str, function_args: dict, query: str, tool_call_id: str, raw_data_only: bool
    ) -> Dict[str, Any]:
        """Handle direct tool calls with proper error handling and response formatting"""
        
        if tool_name == 'search':
            search_query = function_args.get('query')
            limit = function_args.get('limit', 5)
            
            if not search_query:
                return {"error": "Missing 'query' in tool_arguments"}
                
            logger.info(f"Executing search for: {search_query}")
            result = await self.search(search_query, limit)
            
            errors = self._handle_error(result)
            if errors:
                return errors

            if raw_data_only:
                return {"response": "", "data": result}

            analysis = await self.analyze_results(search_query, result)
            explanation = await self._respond_with_llm(
                query=query,
                tool_call_id=tool_call_id,
                data={"search_results": result, "analysis": analysis},
                temperature=0.3
            )
            return {"response": explanation, "data": {"results": result, "analysis": analysis}}

        elif tool_name == 'generate_search_queries':
            main_query = function_args.get('query')
            num_queries = function_args.get('num_queries', 3)
            
            if not main_query:
                return {"error": "Missing 'query' in tool_arguments"}
                
            logger.info(f"Generating search queries for: {main_query}")
            queries = await self.generate_search_queries(main_query, num_queries)
            
            formatted_data = {
                "generated_queries": [
                    {"query": q.query, "research_goal": q.research_goal}
                    for q in queries
                ]
            }
            
            if raw_data_only:
                return {"response": "", "data": formatted_data}

            explanation = await self._respond_with_llm(
                query=query,
                tool_call_id=tool_call_id,
                data=formatted_data,
                temperature=0.3
            )
            return {"response": explanation, "data": formatted_data}

        elif tool_name == 'analyze_results':
            analysis_query = function_args.get('query')
            search_results = function_args.get('search_results')
            
            if not analysis_query or not search_results:
                return {"error": "Both 'query' and 'search_results' are required in tool_arguments"}
                
            logger.info(f"Analyzing results for query: {analysis_query}")
            analysis = await self.analyze_results(analysis_query, search_results)
            
            if raw_data_only:
                return {"response": "", "data": analysis}

            explanation = await self._respond_with_llm(
                query=query,
                tool_call_id=tool_call_id,
                data=analysis,
                temperature=0.3
            )
            return {"response": explanation, "data": analysis}

        else:
            return {"error": f"Unsupported tool '{tool_name}'"}

    @monitor_execution()
    @with_retry(max_retries=3)
    async def handle_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle both direct tool calls and natural language queries.
        Processes search queries with specified depth and breadth.
        """
        query = params.get("query")
        depth = min(max(params.get("depth", 1), 1), 3)
        breadth = min(max(params.get("breadth", 3), 1), 5)
        raw_data_only = params.get("raw_data_only", False)
        
        # For direct tool calls
        tool_name = params.get("tool")
        tool_args = params.get("tool_arguments", {})
        
        # ---------------------
        # 1) DIRECT TOOL CALL
        # ---------------------
        if tool_name:
            return await self._handle_tool_logic(
                tool_name=tool_name,
                function_args=tool_args,
                query=query or "Direct tool call without LLM.",
                tool_call_id="direct_tool",
                raw_data_only=raw_data_only
            )

        # ---------------------
        # 2) NATURAL LANGUAGE QUERY
        # ---------------------
        if query:
            # First, perform the main search with increased limit
            search_results = await self.search(query, limit=10)
            
            # Format function call response
            function_call = {
                "function": "search",
                "arguments": {
                    "query": query,
                    "limit": 10
                }
            }
            
            # Format as string
            function_response = f'<function={function_call["function"]}{json.dumps(function_call["arguments"])}></function>'
            
            # If raw_data_only, return just the results
            if raw_data_only:
                return {
                    "response": function_response,
                    "data": search_results
                }
            
            # Otherwise, analyze results
            analysis = await self.analyze_results(query, search_results)
            
            return {
                "response": function_response,
                "data": {
                    "results": search_results,
                    "analysis": analysis,
                    "metadata": {
                        "depth": depth,
                        "breadth": breadth,
                        "query": query
                    }
                }
            }

        # ---------------------
        # 3) NEITHER query NOR tool
        # ---------------------
        return {"error": "Either 'query' or 'tool' must be provided in the parameters."}