import asyncio
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent.parent))
from mesh.twitter_insight_agent import MoniTwitterProfileAgent  # noqa: E402

load_dotenv()


async def run_agent():
    agent = MoniTwitterProfileAgent()
    try:
        # Test with a query for account analysis
        agent_input_profile = {"query": "Analyze the Twitter account @heurist_ai"}
        agent_output_profile = await agent.handle_message(agent_input_profile)
        print(f"Result of handle_message (profile analysis): {agent_output_profile}")

        # # Test with a query for smart followers
        # agent_input_followers = {"query": "Who are the smart followers of heurist_ai?"}
        # agent_output_followers = await agent.handle_message(agent_input_followers)
        # print(f"Result of handle_message (smart followers): {agent_output_followers}")

        # # Test with a query for mentions history
        # agent_input_mentions = {
        #     "query": "Show me the mention history trends for ethereum",
        # }
        # agent_output_mentions = await agent.handle_message(agent_input_mentions)
        # print(f"Result of handle_message (mentions history): {agent_output_mentions}")

        # # Test direct tool call for smart profile
        # agent_input_direct_profile = {
        #     "tool": "get_account_full_info",  # Changed from get_smart_profile to get_account_full_info
        #     "tool_arguments": {"username": "artoriatech"},
        # }

        # agent_output_direct_profile = await agent.handle_message(agent_input_direct_profile)
        # print(f"Result of direct tool call (account full info): {agent_output_direct_profile}")

        # # Test direct tool call for smart followers history with timeframe
        # agent_input_followers_history = {
        #     "tool": "get_smart_followers_history",
        #     "tool_arguments": {"username": "heurist_ai", "timeframe": "D7"},
        # }
        # agent_output_followers_history = await agent.handle_message(agent_input_followers_history)
        # print(f"Result of direct tool call (followers history): {agent_output_followers_history}")

        # # Test direct tool call for smart followers categories
        # agent_input_categories = {
        #     "tool": "get_smart_followers_categories",
        #     "tool_arguments": {"username": "heurist_ai"},
        # }
        # agent_output_categories = await agent.handle_message(agent_input_categories)
        # print(f"Result of direct tool call (follower categories): {agent_output_categories}")

        # # Test direct tool call for smart followers full with parameters
        # agent_input_followers_full = {
        #     "tool": "get_smart_followers_full",
        #     "tool_arguments": {
        #         "username": "heurist_ai",
        #         "limit": 100,
        #         "offset": 0,
        #         "orderBy": "CREATED_AT",
        #         "orderByDirection": "DESC",
        #     },
        # }
        # agent_output_followers_full = await agent.handle_message(agent_input_followers_full)
        # print(f"Result of direct tool call (followers full): {agent_output_followers_full}")

        # # Test direct tool call for smart mentions feed with parameters
        # agent_input_mentions_feed = {
        #     "tool": "get_smart_mentions_feed",
        #     "tool_arguments": {"username": "heurist_ai", "limit": 100},
        # }
        # agent_output_mentions_feed = await agent.handle_message(agent_input_mentions_feed)
        # print(f"Result of direct tool call (mentions feed): {agent_output_mentions_feed}")

        # # Test with raw_data_only flag
        # agent_input_raw_data = {
        #     "query": "Get smart mentions feed for bitcoin",
        #     "raw_data_only": True,
        # }
        # agent_output_raw_data = await agent.handle_message(agent_input_raw_data)
        # print(f"Result with raw_data_only=True: {agent_output_raw_data}")

        # Save the test inputs and outputs to a YAML file for further inspection
        script_dir = Path(__file__).parent
        current_file = Path(__file__).stem
        base_filename = f"{current_file}_example"
        output_file = script_dir / f"{base_filename}.yaml"

        yaml_content = {
            "input_profile": agent_input_profile,
            "output_profile": agent_output_profile,
            # "input_followers": agent_input_followers,
            # "output_followers": agent_output_followers,
            # "input_mentions": agent_input_mentions,
            # "output_mentions": agent_output_mentions,
            # "input_direct_profile": agent_input_direct_profile,
            # "output_direct_profile": agent_output_direct_profile,
            # "input_followers_history": agent_input_followers_history,
            # "output_followers_history": agent_output_followers_history,
            # "input_categories": agent_input_categories,
            # "output_categories": agent_output_categories,
            # "input_followers_full": agent_input_followers_full,
            # "output_followers_full": agent_output_followers_full,
            # "input_mentions_feed": agent_input_mentions_feed,
            # "output_mentions_feed": agent_output_mentions_feed,
            # "input_raw_data": agent_input_raw_data,
            # "output_raw_data": agent_output_raw_data,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)

        print(f"Results saved to {output_file}")
    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(run_agent())
