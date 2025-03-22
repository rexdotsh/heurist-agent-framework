import asyncio
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent.parent))
from mesh.coingecko_token_info_agent import CoinGeckoTokenInfoAgent  # noqa: E402

load_dotenv()


async def run_agent():
    agent = CoinGeckoTokenInfoAgent()
    try:
        # Test with a query mentioning CoinGecko ID
        agent_input = {"query": "Get information about MONA"}

        agent_output = await agent.handle_message(agent_input)
        print(f"Result of handle_message: {agent_output}")

        # Test with query mentioning the token name
        agent_input_name = {"query": "analyze HEU"}

        agent_output_name = await agent.handle_message(agent_input_name)
        print(f"Result of handle_message when token name is provided: {agent_output_name}")

        # Test with trending coins
        agent_input_trending = {"query": "Get information about trending coins"}
        agent_output_trending = await agent.handle_message(agent_input_trending)
        print(f"Result of handle_message when trending coins is provided: {agent_output_trending}")

        agent_input_direct = {
            "tool": "get_token_info",
            "tool_arguments": {"coingecko_id": "bitcoin"},
            "raw_data_only": True,
        }
        agent_output_direct = await agent.handle_message(agent_input_direct)
        print(f"Result of direct tool call: {agent_output_direct}")

        # Test the new get_token_price_multi tool
        agent_input_price_multi = {
            "tool": "get_token_price_multi",
            "tool_arguments": {
                "ids": "bitcoin,ethereum,solana",
                "vs_currencies": "usd",
                "include_market_cap": True,
                "include_24hr_vol": True,
                "include_24hr_change": True,
            },
        }
        agent_output_price_multi = await agent.handle_message(agent_input_price_multi)
        print(f"Result of get_token_price_multi tool call: {agent_output_price_multi}")

        # Test query for comparing multiple tokens
        agent_input_compare = {"query": "Compare Bitcoin and Ethereum"}
        agent_output_compare = await agent.handle_message(agent_input_compare)
        print(f"Result of token comparison query: {agent_output_compare}")

        script_dir = Path(__file__).parent
        current_file = Path(__file__).stem
        base_filename = f"{current_file}_example"
        output_file = script_dir / f"{base_filename}.yaml"

        yaml_content = {
            "input_by_id": agent_input,
            "output_by_id": agent_output,
            "input_by_name": agent_input_name,
            "output_by_name": agent_output_name,
            "input_by_trending": agent_input_trending,
            "output_by_trending": agent_output_trending,
            "input_direct_tool": agent_input_direct,
            "output_direct_tool": agent_output_direct,
            "input_price_multi": agent_input_price_multi,
            "output_price_multi": agent_output_price_multi,
            "input_comparison": agent_input_compare,
            "output_comparison": agent_output_compare,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)

        print(f"Results saved to {output_file}")

    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(run_agent())
