import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List

from dotenv import load_dotenv

# Import the MCP client
from clients.mcp_client import MCPClient

# Import simple LLM interface
from core.llm import call_llm_async, call_llm_with_tools_async

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class SimpleMCPClient:
    """A simplified client that uses MCP tools directly"""

    def __init__(self, server_url: str):
        self.server_url = server_url
        self.mcp_client = MCPClient()
        self.available_tools = []

        # LLM configuration
        self.base_url = os.getenv("HEURIST_BASE_URL")
        self.api_key = os.getenv("HEURIST_API_KEY")
        self.model_id = os.getenv("LARGE_MODEL_ID")

        if not self.api_key:
            logger.warning("HEURIST_API_KEY environment variable not set. LLM functionality will not work.")

    async def initialize(self):
        """Initialize the MCP client and fetch available tools"""
        try:
            # Connect to the MCP server
            tools = await self.mcp_client.connect_to_sse_server(server_url=self.server_url)
            self.available_tools = tools
            logger.info(f"Connected to MCP server with {len(tools)} tools")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize MCP Client: {str(e)}")
            return False

    async def list_available_tools(self):
        """List all available tools"""
        self.mcp_client.print_available_tools()

    async def get_available_tools_json(self):
        """Get all available tools formatted for LLM consumption"""
        return self.mcp_client.get_available_tools_json()

    def get_tools_by_category(self, category: str):
        """Get tools by category"""
        return self.mcp_client.get_tools_by_category(category)

    async def call_tool(self, tool_name: str, params: Dict[str, Any] = None):
        """Call a tool and return the result"""
        if not params:
            params = {}

        result = await self.mcp_client.call_tool(tool_name, params)
        if result:
            formatted_result = self.mcp_client.format_result(result.content)
            return result.content, formatted_result
        return None, None

    async def process_with_llm(self, message: str, system_prompt: str = None):
        """Process a message using the LLM without tools"""
        if not self.api_key:
            return "API key not configured for LLM access."

        if not system_prompt:
            system_prompt = "You are a helpful assistant."

        try:
            response = await call_llm_async(
                base_url=self.base_url,
                api_key=self.api_key,
                model_id=self.model_id,
                system_prompt=system_prompt,
                user_prompt=message,
                temperature=0.7,
            )
            return response
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            return f"Error: {str(e)}"

    async def process_with_tools(self, message: str, system_prompt: str = None, tools: List[Dict] = None):
        """Process a message using the LLM with tools"""
        if not self.api_key:
            return "API key not configured for LLM access."

        if not system_prompt:
            system_prompt = "You are a helpful assistant with access to tools."

        try:
            response = await call_llm_with_tools_async(
                base_url=self.base_url,
                api_key=self.api_key,
                model_id=self.model_id,
                system_prompt=system_prompt,
                user_prompt=message,
                temperature=0.7,
                tools=tools,
            )

            # Check if the response contains tool calls
            if isinstance(response, dict) and "tool_calls" in response:
                tool_calls = response["tool_calls"]
                content = response.get("content", "")

                # Process each tool call
                for tool_call in tool_calls if isinstance(tool_calls, list) else [tool_calls]:
                    try:
                        # Extract tool name and parameters
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)

                        logger.info(f"Executing tool call: {tool_name} with args: {tool_args}")

                        # Call the tool using MCP client
                        raw_result, formatted_result = await self.call_tool(tool_name, tool_args)

                        # Append the tool result to the content
                        tool_result = formatted_result or raw_result
                        if content:
                            content += f"\n\nTool result for {tool_name}:\n{tool_result}"
                        else:
                            content = f"Tool result for {tool_name}:\n{tool_result}"
                    except Exception as e:
                        logger.error(f"Error executing tool call {tool_name}: {str(e)}")
                        if content:
                            content += f"\n\nError executing tool {tool_name}: {str(e)}"
                        else:
                            content = f"Error executing tool {tool_name}: {str(e)}"

                return content

            return response
        except Exception as e:
            logger.error(f"Error calling LLM with tools: {str(e)}")
            return f"Error: {str(e)}"

    async def cleanup(self):
        """Clean up resources"""
        await self.mcp_client.cleanup()


async def test_coingecko_tools(client: SimpleMCPClient):
    """Test CoinGecko tools directly through the MCP client"""
    # Get CoinGecko tools
    coingecko_tools = client.get_tools_by_category("coingecko")

    if not coingecko_tools:
        print("No CoinGecko tools found!")
        return

    print(f"\nFound {len(coingecko_tools)} CoinGecko tools:")
    for i, tool in enumerate(coingecko_tools):
        print(f"{i + 1}. {tool.name}: {tool.description}")

    # Find the price tool for Bitcoin
    price_tools = [
        tool for tool in coingecko_tools if "price" in tool.name.lower() or "price" in tool.description.lower()
    ]

    if price_tools:
        price_tool = price_tools[0]
        print(f"\nCalling CoinGecko price tool: {price_tool.name}")

        # Determine the correct parameter based on the tool schema
        params = {}
        if "token_name" in str(price_tool.inputSchema):
            params = {"token_name": "bitcoin"}
        elif "coingecko_id" in str(price_tool.inputSchema):
            params = {"coingecko_id": "bitcoin"}
        else:
            # Try to parse the input schema to find the right parameter
            try:
                schema = (
                    json.loads(price_tool.inputSchema)
                    if isinstance(price_tool.inputSchema, str)
                    else price_tool.inputSchema
                )
                required_params = schema.get("required", [])
                if required_params:
                    params = {required_params[0]: "bitcoin"}
            except Exception:
                pass

        print(f"Using parameters: {params}")

        # Call the tool
        raw_result, formatted_result = await client.call_tool(price_tool.name, params)

        if formatted_result:
            print("\nFormatted result:")
            print(formatted_result)
        else:
            print("\nRaw result:")
            print(raw_result)

    else:
        # If no price tool is found, try the trending tool or any other CoinGecko tool
        trending_tool = next((tool for tool in coingecko_tools if "trending" in tool.name.lower()), None)

        if trending_tool:
            print(f"\nCalling CoinGecko trending tool: {trending_tool.name}")
            raw_result, formatted_result = await client.call_tool(trending_tool.name, {})

            if formatted_result:
                print("\nFormatted result:")
                print(formatted_result)
            else:
                print("\nRaw result:")
                print(raw_result)
        else:
            # Use the first available tool
            tool = coingecko_tools[0]
            print(f"\nCalling CoinGecko tool: {tool.name}")

            # Try to determine parameters
            params = {}
            raw_result, formatted_result = await client.call_tool(tool.name, params)

            if formatted_result:
                print("\nFormatted result:")
                print(formatted_result)
            else:
                print("\nRaw result:")
                print(raw_result)


async def main():
    if len(sys.argv) < 2:
        print("Usage: python main_mcp.py <URL of SSE MCP server (i.e. http://localhost:8000/sse)>")
        sys.exit(1)

    server_url = sys.argv[1]
    client = SimpleMCPClient(server_url)

    try:
        # Initialize the client
        print(f"Connecting to MCP server at {server_url}...")
        success = await client.initialize()
        if not success:
            print("Failed to initialize the MCP Client. Exiting.")
            return

        print("MCP Client initialized successfully!")

        # List available tools
        print("\nAvailable tools:")
        await client.list_available_tools()
        # Test the CoinGecko tools directly
        await test_coingecko_tools(client)

        # Interactive mode
        print("\n\nEntering interactive mode. Type 'exit' to quit.")
        print("Options:")
        print("  1. Type a message to process with LLM")
        print("  2. Type 'tool:<tool_name>' to directly call a tool")
        print("  3. Type 'exit' to quit")

        # Get tools once before the loop
        tools_json = await client.get_available_tools_json()

        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                break

            if user_input.lower().startswith("tool:"):
                # Extract tool name and parameters
                parts = user_input[5:].strip().split(" ", 1)
                tool_name = parts[0]
                params_str = parts[1] if len(parts) > 1 else "{}"

                try:
                    params = json.loads(params_str)
                    print(f"Calling tool '{tool_name}' with parameters: {params}")
                    raw_result, formatted_result = await client.call_tool(tool_name, params)

                    if formatted_result:
                        print("\nFormatted result:")
                        print(formatted_result)
                    else:
                        print("\nRaw result:")
                        print(raw_result)
                except json.JSONDecodeError:
                    print(f"Invalid JSON parameters: {params_str}")
                except Exception as e:
                    print(f"Error calling tool: {str(e)}")
            else:
                # Process with LLM
                response = await client.process_with_tools(
                    user_input, "You are a helpful assistant specializing in cryptocurrency information.", tools_json
                )
                print(f"\nAssistant: {response}")

    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
