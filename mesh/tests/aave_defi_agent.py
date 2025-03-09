import asyncio
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from mesh.aave_defi_agent import Aaveagent

load_dotenv()


async def run_agent():
    agent = Aaveagent()
    try:
        # Test with a query to fetch default reserve data.
        agent_input = {"query": "Get Aave V3 reserve data"}
        agent_output = await agent.handle_message(agent_input)

        # Test with a query including a block identifier.
        agent_input_block = {
            "query": "Get Aave V3 reserve data at block 14568297 ",
            "tool_arguments": {"block_identifier": "14568297 "},
        }
        agent_output_block = await agent.handle_message(agent_input_block)

        # Save the test inputs and outputs to a YAML file for inspection.
        script_dir = Path(__file__).parent
        current_file = Path(__file__).stem
        base_filename = f"{current_file}_example"
        output_file = script_dir / f"{base_filename}.yaml"

        yaml_content = {
            "input_reserve_data": agent_input,
            "output_reserve_data": agent_output,
            "input_reserve_data_block": agent_input_block,
            "output_reserve_data_block": agent_output_block,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)

        print(f"Results saved to {output_file}")

    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(run_agent())
