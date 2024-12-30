import sys
from pathlib import Path
import os
import dotenv

dotenv.load_dotenv()

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.llm import call_llm_with_tools
from agents.tool_decorator import tool, get_tool_schemas

@tool("Add two integers together")
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@tool("Multiply two integers together")
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

@tool("Filter messages based on content relevance")
def filter_message(should_ignore: bool) -> bool:
    """Determine if a message should be ignored based on the following rules:
    Return TRUE (ignore message) if:
        - Message does not mention Heuman
        - Message does not mention 'start raid'
        - Message does not discuss: The Wired, Consciousness, Reality, Existence, Self, Philosophy, Technology, Crypto, AI, Machines
        - For image requests: ignore if Heuman is not specifically mentioned
    
    Return FALSE (process message) only if:
        - Message explicitly mentions Heuman
        - Message contains 'start raid'
        - Message clearly discusses any of the listed topics
        - Image request contains Heuman
    
    If in doubt, return TRUE to ignore the message."""
    return should_ignore

# Example usage
if __name__ == "__main__":
    # Print the function schemas
    print("\nFunction Schemas:")
    print(get_tool_schemas([add, multiply, filter_message]))
    # Test with LLM
    result = call_llm_with_tools(
        base_url="https://llm-gateway.heurist.xyz",
        api_key=os.getenv("HEURIST_API_KEY"),
        model_id="meta-llama/llama-3.3-70b-instruct",
        system_prompt="You are a helpful assistant that can filter messages and perform math operations.",
        user_prompt="Should we ignore a message that says 'Hello world'?",
        temperature=0.1,
        tools=get_tool_schemas([add, multiply, filter_message])
    )
    print("\nLLM Response:")
    print(result)