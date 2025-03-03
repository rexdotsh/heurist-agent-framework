import sys
from pathlib import Path
import yaml
import os
import asyncio
from dotenv import load_dotenv
from threading import Thread

sys.path.append(str(Path(__file__).parent.parent.parent))

from mesh.masa_twitter_intellgence_agent import MasaTwitterSearchAgent

load_dotenv()

# DEBUG Mode:
# Set DEBUG=True to run the script normally (blocking execution until completion).
# Set DEBUG=False to execute processing in the background and exit early to avoid long wait times.
DEBUG = False

def run_in_background(coro):
    """
    Runs an asyncio coroutine in a separate thread.
    This is used when DEBUG is set to False to ensure the script does not block execution
    while processing continues in the background.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(coro)

def save_results(output_file, yaml_content):
    """
    Saves the results to a YAML file.
    If DEBUG is False, this function is executed in the background to allow the script to exit early.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)
    print(f"Results saved to {output_file}")

async def run_agent():
    """
    Runs the MasaTwitterSearchAgent to fetch data and store results.
    If DEBUG is False, file writing is executed in the background to avoid blocking script execution.
    """
    agent = MasaTwitterSearchAgent()
    try:
        agent_input = {'query': '@heurist_ai', 'max_results': 100}
        agent_output = await agent.handle_message(agent_input)
        
        agent_input_specific = {'query': '$HEU', 'max_results': 100}
        agent_output_specific = await agent.handle_message(agent_input_specific)
        
        agent_input_direct = {
            'tool': 'search_twitter',
            'tool_arguments': {'query': 'Heurist crypto', 'max_results': 30},
            'raw_data_only': True
        }
        agent_output_direct = await agent.handle_message(agent_input_direct)
        
        script_dir = Path(__file__).parent
        current_file = Path(__file__).stem
        base_filename = f"{current_file}_example"
        output_file = script_dir / f"{base_filename}.yaml"

        yaml_content = {
            'input_basic': agent_input,
            'output_basic': agent_output,
            'input_specific': agent_input_specific,
            'output_specific': agent_output_specific,
            'input_direct': agent_input_direct,
            'output_direct': agent_output_direct
        }
        
        if DEBUG:
            save_results(output_file, yaml_content)
        else:
            thread = Thread(target=run_in_background, args=(save_results(output_file, yaml_content),))
            thread.start()
            print("Processing in background... Exiting early!")
    
    finally:
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(run_agent())