import sys
from pathlib import Path
import yaml
import os
from dotenv import load_dotenv
import asyncio

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from mesh.moni_twitter_smart_agent import MoniTwitterSmartAgent

load_dotenv()

async def run_agent():
    agent = MoniTwitterSmartAgent()
    try:
        test_cases = {
            'smart_followers': {
                'query': 'Get smart followers for @heurist_ai'
            },
            'smart_followers_distribution': {
                'query': 'Get the distribution of smart followers by level for @heurist_ai'
            },
            'smart_followers_categories': {
                'query': 'Get the categories of smart followers for @heurist_ai'
            },
            'timeline': {
                'query': 'Get timeline for @heurist_ai'
            },
            'tweet_tracking': {
                'query': 'Start tweet tracking for @heurist_ai'
            }
        }

        results = {}
        for test_name, test_input in test_cases.items():
            print(f"\nTesting {test_name}...")
            agent_output = await agent.handle_message(test_input)
            print(f"Result for {test_name}: {agent_output}")
            results[test_name] = {
                'input': test_input,
                'output': agent_output
            }

        # Save the test inputs and outputs to a YAML file
        script_dir = Path(__file__).parent
        current_file = Path(__file__).stem
        base_filename = f"{current_file}_example"
        output_file = script_dir / f"{base_filename}.yaml"

        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(results, f, allow_unicode=True, sort_keys=False)

        print(f"\nResults saved to {output_file}")

    finally:
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(run_agent()) 