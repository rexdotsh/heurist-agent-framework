import json
import logging
import os
from typing import Any, Dict, List, Optional

import aiohttp
from dotenv import load_dotenv

from core.llm import call_llm_async, call_llm_with_tools_async
from decorators import monitor_execution, with_cache, with_retry

from .mesh_agent import MeshAgent

logger = logging.getLogger(__name__)
load_dotenv()


class SpaceAndTimeAgent(MeshAgent):
    def __init__(self):
        super().__init__()
        self.session = None
        self.access_token = None
        self.refresh_token = None
        self.base_url = "https://proxy.api.spaceandtime.dev"

        # Initialize the session during initialization to ensure it exists
        self.session = aiohttp.ClientSession()

        self.metadata.update(
            {
                "name": "Space and Time Agent",
                "version": "1.0.0",
                "author": "Heurist Team",
                "author_address": "0x7d9d1821d15B9e0b8Ab98A058361233E255E405D",
                "description": "This agent can query blockchain and web3 data using natural language through the Space and Time API.",
                "inputs": [
                    {
                        "name": "query",
                        "description": "Natural language query about blockchain data.",
                        "type": "str",
                        "required": False,
                    },
                    {
                        "name": "raw_data_only",
                        "description": "If true, the agent will only return the raw data without LLM explanation",
                        "type": "bool",
                        "required": False,
                        "default": False,
                    },
                ],
                "outputs": [
                    {
                        "name": "response",
                        "description": "Natural language explanation of the blockchain data",
                        "type": "str",
                    },
                    {
                        "name": "data",
                        "description": "Structured blockchain data response",
                        "type": "dict",
                    },
                ],
                "external_apis": ["Space and Time"],
                "tags": ["Blockchain", "SQL", "Data Analytics"],
                "image_url": "https://raw.githubusercontent.com/heurist-network/space-and-time-agent/main/images/SpaceAndTime.png",
                "examples": [
                    "Get the number of blocks created on Ethereum per day over the last month",
                    "What are the top 10 NFT collections by trading volume on Ethereum?",
                    "Show me daily transaction counts for Solana over the past week",
                ],
            }
        )

    async def __aenter__(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    async def cleanup(self):
        """Cleanup method to ensure session is closed properly"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    def get_system_prompt(self) -> str:
        return """You are a helpful blockchain data analyst that can query blockchain and web3 data using natural language.
        You have access to the Space and Time data platform, which allows you to analyze on-chain data across multiple blockchains.

        When answering questions:
        - Focus on providing accurate, data-driven insights
        - Translate natural language questions into appropriate SQL queries
        - Present data in a clear, concise format
        - Explain blockchain metrics in terms that are understandable to both technical and non-technical users
        - If a query is not possible with the available data, explain why rather than attempting to answer

        You can query data across various schemas and tables, with a focus on blockchain transaction data,
        token metrics, NFT collections, DeFi protocols, and more. When the data is returned, summarize
        the key findings and provide relevant context.
        """

    def get_tool_schemas(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "query_space_and_time",
                    "description": "Generate and execute SQL queries for blockchain data using AI through the Space and Time API.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "A natural language query describing the blockchain data request.",
                            },
                            "schema": {
                                "type": "string",
                                "description": "Optional: The schema to query. If not provided, it will be inferred.",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_blockchains",
                    "description": "List all supported blockchains in the Space and Time platform.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_schemas",
                    "description": "List all available schemas in the Space and Time platform.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_tables",
                    "description": "List all tables in a specific schema.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "schema": {
                                "type": "string",
                                "description": "The name of the schema to list tables from.",
                            },
                        },
                        "required": ["schema"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_table_columns",
                    "description": "List all columns in a specific table.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "schema": {
                                "type": "string",
                                "description": "The name of the schema the table belongs to.",
                            },
                            "table": {
                                "type": "string",
                                "description": "The name of the table to list columns from.",
                            },
                        },
                        "required": ["schema", "table"],
                    },
                },
            },
        ]

    # ------------------------------------------------------------------------
    #                       SHARED / UTILITY METHODS
    # ------------------------------------------------------------------------
    async def _respond_with_llm(self, query: str, tool_call_id: str, data: dict, temperature: float) -> str:
        """
        Reusable helper to ask the LLM to generate a user-friendly explanation
        given a piece of data from a tool call.
        """
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
        """
        Small helper to return the error if present in
        a dictionary with the 'error' key.
        """
        if "error" in maybe_error:
            return {"error": maybe_error["error"]}
        return {}

    # ------------------------------------------------------------------------
    #                      AUTHENTICATION METHODS
    # ------------------------------------------------------------------------
    async def _authenticate(self) -> bool:
        """
        Authenticate with the Space and Time API using the API key.
        Returns True if authentication was successful, False otherwise.
        """
        if self.access_token:
            return True

        api_key = os.getenv("SPACE_AND_TIME_API_KEY")
        if not api_key:
            logger.error("SPACE_AND_TIME_API_KEY environment variable not set")
            return False

        # Ensure we have a valid session
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

        # Using the direct API URL for authentication instead of proxy URL
        url = f"{self.base_url}/auth/apikey"

        headers = {"Content-Type": "application/json", "apiKey": api_key}

        try:
            # Log authentication attempt
            logger.info(f"Attempting to authenticate with Space and Time API at: {url}")

            async with self.session.post(url, headers=headers) as response:
                response_text = await response.text()

                if response.status != 200:
                    logger.error(f"Authentication failed with status {response.status}: {response_text}")
                    return False

                try:
                    auth_data = json.loads(response_text)
                    self.access_token = auth_data.get("accessToken")
                    self.refresh_token = auth_data.get("refreshToken")

                    if self.access_token and self.refresh_token:
                        logger.info("Successfully authenticated with Space and Time API")
                        return True
                    else:
                        logger.error("Authentication response missing tokens")
                        return False
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse authentication response as JSON: {response_text}")
                    return False

        except Exception as e:
            logger.error(f"Error during authentication: {str(e)}")
            return False

    async def _refresh_token(self) -> bool:
        """
        Refresh the access token using the refresh token.
        Returns True if refresh was successful, False otherwise.
        """
        if not self.refresh_token:
            return False

        # Ensure we have a valid session
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

        # Updated to use the direct API URL
        url = f"{self.base_url}/v1/auth/refresh"

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.refresh_token}"}

        try:
            logger.info("Attempting to refresh token")

            async with self.session.post(url, headers=headers) as response:
                response_text = await response.text()

                if response.status != 200:
                    logger.error(f"Token refresh failed with status {response.status}: {response_text}")
                    return False

                try:
                    refresh_data = json.loads(response_text)
                    self.access_token = refresh_data.get("accessToken")
                    self.refresh_token = refresh_data.get("refreshToken")

                    if self.access_token and self.refresh_token:
                        logger.info("Successfully refreshed token")
                        return True
                    else:
                        logger.error("Refresh response missing tokens")
                        return False
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse refresh response as JSON: {response_text}")
                    return False

        except Exception as e:
            logger.error(f"Error during token refresh: {str(e)}")
            return False

    async def _ensure_authenticated(self) -> bool:
        """
        Ensure that we have a valid authentication token.
        Tries to authenticate if we don't have a token, or refresh if we do.
        Returns True if we have a valid token, False otherwise.
        """
        if not self.access_token:
            return await self._authenticate()

        # For simplicity, we'll just try to re-authenticate rather than checking token expiry
        # In a production system, you'd want to check the token expiry and refresh only when needed
        return await self._refresh_token() or await self._authenticate()

    # ------------------------------------------------------------------------
    #                      SPACE AND TIME API METHODS
    # ------------------------------------------------------------------------
    @monitor_execution()
    @with_cache(ttl_seconds=300)  # Cache for 5 minutes
    @with_retry(max_retries=3)
    async def query_space_and_time(self, query: str, schema: Optional[str] = None) -> Dict:
        """
        Generate and execute an SQL query based on natural language using the Space and Time AI feature.

        Args:
            query: Natural language query to convert to SQL
            schema: Optional schema name to use (if known)

        Returns:
            Dictionary containing the results or an error message
        """
        if not await self._ensure_authenticated():
            return {"error": "Failed to authenticate with Space and Time API"}

        try:
            # Step 1: Generate SQL from natural language
            sql_endpoint = "https://api.spaceandtime.dev/v1/ai/sql/generate"
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.access_token}"}

            # Prepare the AI query with schema hint if provided
            ai_query = query
            if schema:
                ai_query = f"Using schema {schema}, {query}"

            sql_data = {"prompt": ai_query, "metadata": {}}

            logger.info(f"Sending SQL generation request to: {sql_endpoint}")

            async with self.session.post(sql_endpoint, json=sql_data, headers=headers) as response:
                response_text = await response.text()

                if response.status != 200:
                    logger.error(f"SQL generation failed with status {response.status}: {response_text}")
                    return {"error": f"SQL generation failed: {response_text}"}

                try:
                    sql_response = json.loads(response_text)
                    generated_sql = sql_response.get("sql")

                    if not generated_sql:
                        return {"error": "No SQL generated from the query"}

                    # Step 2: Execute the generated SQL
                    execute_endpoint = f"{self.base_url}/v2/query/sql/proxy"
                    execute_data = {
                        "sqlText": generated_sql,
                        "resourceId": schema or "SUBSCRIPTION",  # Use provided schema or default
                    }

                    logger.info(f"Executing generated SQL query: {generated_sql}")

                    async with self.session.post(execute_endpoint, json=execute_data, headers=headers) as exec_response:
                        exec_response_text = await exec_response.text()

                        if exec_response.status != 200:
                            logger.error(
                                f"SQL execution failed with status {exec_response.status}: {exec_response_text}"
                            )
                            return {
                                "error": f"SQL execution failed: {exec_response_text}",
                                "generated_sql": generated_sql,
                            }

                        try:
                            query_result = json.loads(exec_response_text)

                            return {
                                "results": query_result,
                                "generated_sql": generated_sql,
                                "natural_language_query": query,
                            }
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse SQL execution response as JSON: {exec_response_text}")
                            return {"error": f"Failed to parse SQL execution result: {exec_response_text}"}

                except json.JSONDecodeError:
                    logger.error(f"Failed to parse SQL generation response as JSON: {response_text}")
                    return {"error": f"Failed to parse SQL generation result: {response_text}"}

        except Exception as e:
            logger.error(f"Error querying Space and Time: {str(e)}")
            return {"error": f"Failed to query Space and Time: {str(e)}"}

    @monitor_execution()
    @with_cache(ttl_seconds=3600)  # Cache for 1 hour
    @with_retry(max_retries=3)
    async def list_blockchains(self) -> Dict:
        """
        List all supported blockchains in the Space and Time platform.
        """
        if not await self._ensure_authenticated():
            return {"error": "Failed to authenticate with Space and Time API"}

        try:
            url = f"{self.base_url}/v2/discover/blockchains"
            headers = {"Authorization": f"Bearer {self.access_token}"}

            logger.info(f"Listing blockchains from: {url}")

            async with self.session.get(url, headers=headers) as response:
                response_text = await response.text()

                if response.status != 200:
                    logger.error(f"Failed to list blockchains with status {response.status}: {response_text}")
                    return {"error": f"Failed to list blockchains: {response_text}"}

                try:
                    result = json.loads(response_text)
                    return {"blockchains": result}
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse blockchain list response as JSON: {response_text}")
                    return {"error": f"Failed to parse blockchain list: {response_text}"}

        except Exception as e:
            logger.error(f"Error listing blockchains: {str(e)}")
            return {"error": f"Failed to list blockchains: {str(e)}"}

    @monitor_execution()
    @with_cache(ttl_seconds=3600)  # Cache for 1 hour
    @with_retry(max_retries=3)
    async def list_schemas(self) -> Dict:
        """
        List all available schemas in the Space and Time platform.
        """
        if not await self._ensure_authenticated():
            return {"error": "Failed to authenticate with Space and Time API"}

        try:
            url = f"{self.base_url}/v2/discover/schema"
            headers = {"Authorization": f"Bearer {self.access_token}"}

            logger.info(f"Listing schemas from: {url}")

            async with self.session.get(url, headers=headers) as response:
                response_text = await response.text()

                if response.status != 200:
                    logger.error(f"Failed to list schemas with status {response.status}: {response_text}")
                    return {"error": f"Failed to list schemas: {response_text}"}

                try:
                    result = json.loads(response_text)
                    return {"schemas": result}
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse schema list response as JSON: {response_text}")
                    return {"error": f"Failed to parse schema list: {response_text}"}

        except Exception as e:
            logger.error(f"Error listing schemas: {str(e)}")
            return {"error": f"Failed to list schemas: {str(e)}"}

    @monitor_execution()
    @with_cache(ttl_seconds=3600)  # Cache for 1 hour
    @with_retry(max_retries=3)
    async def list_tables(self, schema: str) -> Dict:
        """
        List all tables in a specific schema.

        Args:
            schema: The name of the schema to list tables from

        Returns:
            Dictionary containing table information or an error message
        """
        if not await self._ensure_authenticated():
            return {"error": "Failed to authenticate with Space and Time API"}

        try:
            url = f"{self.base_url}/v2/discover/table?scope=SUBSCRIPTION&schema={schema}"
            headers = {"Authorization": f"Bearer {self.access_token}"}

            logger.info(f"Listing tables for schema {schema} from: {url}")

            async with self.session.get(url, headers=headers) as response:
                response_text = await response.text()

                if response.status != 200:
                    logger.error(f"Failed to list tables with status {response.status}: {response_text}")
                    return {"error": f"Failed to list tables: {response_text}"}

                try:
                    result = json.loads(response_text)
                    return {"tables": result, "schema": schema}
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse table list response as JSON: {response_text}")
                    return {"error": f"Failed to parse table list: {response_text}"}

        except Exception as e:
            logger.error(f"Error listing tables: {str(e)}")
            return {"error": f"Failed to list tables: {str(e)}"}

    @monitor_execution()
    @with_cache(ttl_seconds=3600)  # Cache for 1 hour
    @with_retry(max_retries=3)
    async def list_table_columns(self, schema: str, table: str) -> Dict:
        """
        List all columns in a specific table.

        Args:
            schema: The name of the schema the table belongs to
            table: The name of the table to list columns from

        Returns:
            Dictionary containing column information or an error message
        """
        if not await self._ensure_authenticated():
            return {"error": "Failed to authenticate with Space and Time API"}

        try:
            url = f"{self.base_url}/v2/discover/table/column?schema={schema}&table={table}"
            headers = {"Authorization": f"Bearer {self.access_token}"}

            logger.info(f"Listing columns for {schema}.{table} from: {url}")

            async with self.session.get(url, headers=headers) as response:
                response_text = await response.text()

                if response.status != 200:
                    logger.error(f"Failed to list table columns with status {response.status}: {response_text}")
                    return {"error": f"Failed to list table columns: {response_text}"}

                try:
                    result = json.loads(response_text)
                    return {"columns": result, "schema": schema, "table": table}
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse column list response as JSON: {response_text}")
                    return {"error": f"Failed to parse column list: {response_text}"}

        except Exception as e:
            logger.error(f"Error listing table columns: {str(e)}")
            return {"error": f"Failed to list table columns: {str(e)}"}

    # ------------------------------------------------------------------------
    #                      TOOL HANDLING LOGIC
    # ------------------------------------------------------------------------
    async def _handle_tool_logic(self, tool_name: str, function_args: dict) -> Dict[str, Any]:
        """
        Handle tool execution without LLM involvement
        """
        if tool_name == "query_space_and_time":
            return await self.query_space_and_time(query=function_args.get("query"), schema=function_args.get("schema"))
        elif tool_name == "list_blockchains":
            return await self.list_blockchains()
        elif tool_name == "list_schemas":
            return await self.list_schemas()
        elif tool_name == "list_tables":
            return await self.list_tables(schema=function_args.get("schema"))
        elif tool_name == "list_table_columns":
            return await self.list_table_columns(schema=function_args.get("schema"), table=function_args.get("table"))
        else:
            return {"error": f"Unsupported tool '{tool_name}'"}

    # ------------------------------------------------------------------------
    #                      MAIN HANDLER
    # ------------------------------------------------------------------------
    @monitor_execution()
    @with_retry(max_retries=3)
    async def handle_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming messages, supporting both direct tool calls and natural language queries.

        Either 'query' or 'tool' is required in params.
        - If 'query' is present, it means "agent mode", we use LLM to interpret the query and call tools
          - if 'raw_data_only' is present, we return tool results without another LLM call
        - If 'tool' is present, it means "direct tool call mode", we bypass LLM and directly call the API
          - never run another LLM call, this minimizes latency and reduces error
        """
        query = params.get("query")
        tool_name = params.get("tool")
        tool_args = params.get("tool_arguments", {})
        raw_data_only = params.get("raw_data_only", False)

        # Ensure we have a valid session
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

        try:
            # ---------------------
            # 1) DIRECT TOOL CALL
            # ---------------------
            if tool_name:
                logger.info(f"Processing direct tool call: {tool_name}")
                data = await self._handle_tool_logic(tool_name=tool_name, function_args=tool_args)
                return {"response": "", "data": data}

            # ---------------------
            # 2) NATURAL LANGUAGE QUERY (LLM decides the tool)
            # ---------------------
            if query:
                logger.info(f"Processing natural language query: {query}")
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
                    logger.error("Failed to get response from LLM")
                    return {"error": "Failed to process query"}

                if not response.get("tool_calls"):
                    logger.info("LLM response contains no tool calls")
                    return {"response": response["content"], "data": {}}

                tool_call = response["tool_calls"]
                tool_call_name = tool_call.function.name
                tool_call_args = json.loads(tool_call.function.arguments)

                logger.info(f"LLM selected tool: {tool_call_name}")
                data = await self._handle_tool_logic(
                    tool_name=tool_call_name,
                    function_args=tool_call_args,
                )

                if raw_data_only:
                    logger.info("Returning raw data without LLM explanation")
                    return {"response": "", "data": data}

                logger.info("Generating LLM explanation of the data")
                explanation = await self._respond_with_llm(
                    query=query, tool_call_id=tool_call.id, data=data, temperature=0.1
                )
                return {"response": explanation, "data": data}

            # ---------------------
            # 3) NEITHER query NOR tool
            # ---------------------
            logger.error("Request missing both 'query' and 'tool' parameters")
            return {"error": "Either 'query' or 'tool' must be provided in the parameters."}

        except Exception as e:
            logger.error(f"Unexpected error in handle_message: {str(e)}")
            return {"error": f"An unexpected error occurred: {str(e)}"}
        finally:
            # We don't close the session here as it may be reused
            pass
