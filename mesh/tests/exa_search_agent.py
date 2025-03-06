import sys
from pathlib import Path
import yaml
import os
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent.parent))

from mesh.exa_search_agent import ExaSearchAgent
import asyncio

load_dotenv()

async def run_agent():
    agent = ExaSearchAgent()
    try:
        # Test with a search query
        agent_input_search = {
            'query': 'What is Heurist.ai'
        }
        
        agent_output_search = await agent.handle_message(agent_input_search)

        # Test with direct answer query
        agent_input_answer = {
            'tool': 'answer',
            'tool_arguments': {
                'query': 'What is market value of $HEU coin'
            }
        }
        
        agent_output_answer = await agent.handle_message(agent_input_answer)

        # Test with combined search and answer query
        agent_input_combined = {
            'tool': 'search_and_answer',
            'tool_arguments': {
                'query': 'Tell me about global crysis in 2030'
            }
        }
        
        agent_output_combined = await agent.handle_message(agent_input_combined)
        
        script_dir = Path(__file__).parent
        current_file = Path(__file__).stem
        base_filename = f"{current_file}_example"
        output_file = script_dir / f"{base_filename}.yaml"

        yaml_content = {
            'input_search': agent_input_search,
            'output_search': agent_output_search,
            'input_answer': agent_input_answer,
            'output_answer': agent_output_answer,
            'input_combined': agent_input_combined,
            'output_combined': agent_output_combined
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)
            
        print(f"Results saved to {output_file}")
        
    finally:
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(run_agent())