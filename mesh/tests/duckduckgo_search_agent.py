import asyncio
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent.parent))
from mesh.duckduckgo_search_agent import DuckDuckGoSearchAgent  # noqa: E402

load_dotenv()


async def run_agent():
    agent = DuckDuckGoSearchAgent()
    try:
        agent_input = {
            "query": "What are the latest developments in artificial intelligence?",
            "max_results": 3,
        }

        agent_output = await agent.handle_message(agent_input)

        script_dir = Path(__file__).parent
        current_file = Path(__file__).stem
        base_filename = f"{current_file}_example"
        output_file = script_dir / f"{base_filename}.yaml"

        yaml_content = {"input": agent_input, "output": agent_output}

        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)

        print(f"Results saved to {output_file}")

    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(run_agent())
