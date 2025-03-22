import asyncio
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent.parent))
from mesh.bitquery_solana_token_info_agent import BitquerySolanaTokenInfoAgent  # noqa: E402

load_dotenv()


async def run_agent():
    agent = BitquerySolanaTokenInfoAgent()
    try:
        # Test with a query that mentions a token mint address for trading info
        agent_input = {"query": "Get token info for HeLp6NuQkmYB4pYWo2zYs22mESHXPQYzXbB8n4V98jwC"}
        agent_output = await agent.handle_message(agent_input)
        print(f"Result of handle_message (by token address): {agent_output}")

        # Test with a query for trending tokens
        agent_input_trending = {"query": "Get trending tokens"}
        agent_output_trending = await agent.handle_message(agent_input_trending)
        print(f"Result of handle_message (trending tokens): {agent_output_trending}")

        # Test direct tool call for token trading info
        agent_input_direct_tool = {
            "tool": "get_token_trading_info",
            "tool_arguments": {"token_address": "HeLp6NuQkmYB4pYWo2zYs22mESHXPQYzXbB8n4V98jwC"},
        }
        agent_output_direct_tool = await agent.handle_message(agent_input_direct_tool)
        print(f"Result of direct tool call (token trading info): {agent_output_direct_tool}")

        # Test direct tool call for top trending tokens
        agent_input_direct_trending = {"tool": "get_top_trending_tokens", "tool_arguments": {}}
        agent_output_direct_trending = await agent.handle_message(agent_input_direct_trending)
        print(f"Result of direct tool call (trending tokens): {agent_output_direct_trending}")

        # Test with raw_data_only flag
        agent_input_raw_data = {
            "query": "Get token info for HeLp6NuQkmYB4pYWo2zYs22mESHXPQYzXbB8n4V98jwC",
            "raw_data_only": True,
        }
        agent_output_raw_data = await agent.handle_message(agent_input_raw_data)
        print(f"Result with raw_data_only=True: {agent_output_raw_data}")

        # Save the test inputs and outputs to a YAML file for further inspection
        script_dir = Path(__file__).parent
        current_file = Path(__file__).stem
        base_filename = f"{current_file}_example"
        output_file = script_dir / f"{base_filename}.yaml"

        yaml_content = {
            "input_by_token_address": agent_input,
            "output_by_token_address": agent_output,
            "input_trending": agent_input_trending,
            "output_trending": agent_output_trending,
            "input_direct_tool": agent_input_direct_tool,
            "output_direct_tool": agent_output_direct_tool,
            "input_direct_trending": agent_input_direct_trending,
            "output_direct_trending": agent_output_direct_trending,
            "input_raw_data": agent_input_raw_data,
            "output_raw_data": agent_output_raw_data,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)

        print(f"Results saved to {output_file}")

    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(run_agent())
