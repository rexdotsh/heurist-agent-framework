import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).parent.parent.parent))

import asyncio

from mesh.aave_defi_agent import AaveReserveAgent


async def run_agent():
    agent = AaveReserveAgent()
    try:
        agent_input = {"query": "What are the current borrow rates for USDC on Polygon?"}
        agent_output = await agent.handle_message(agent_input)
        
        direct_input = {
            "tool": "get_aave_reserves",
            "tool_arguments": {
                "chain_id": 137,  # Polygon
                "asset_filter": "USDC"
            }
        }
        direct_output = await agent.handle_message(direct_input)
        
        print("\nExample 3: Raw data only")
        raw_input = {
            "query": "Show me all Aave assets on Polygon with their liquidity rates",
            "raw_data_only": True
        }
        raw_output = await agent.handle_message(raw_input)

        script_dir = Path(__file__).parent
        current_file = Path(__file__).stem
        base_filename = f"{current_file}_example"
        output_file = script_dir / f"{base_filename}.yaml"

        yaml_content = {
            "example1": {
                "input": agent_input,
                "output": agent_output
            },
            "example2": {
                "input": direct_input,
                "output": direct_output
            },
            "example3": {
                "input": raw_input,
                "output": raw_output
            }
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)

        print(f"Results saved to {output_file}")

    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(run_agent())