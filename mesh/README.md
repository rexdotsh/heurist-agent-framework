# Heurist Mesh

**Heurist Mesh** is a new open network of AI agents designed to work together like DeFi smart contracts—modular, composable, and contributed by the community. Each agent is a specialized unit that can process data, generate reports, or engage in conversations, while collectively forming an intelligent swarm to tackle complex tasks. Built on decentralized compute and powered by diverse open-source AI models, Mesh agents can be combined into powerful workflows for cost-efficient and highly flexible solutions.

## Table of Contents
1. [How It Works](#how-it-works)
2. [Agent Deployment](#agent-deployment)
3. [Using Mesh Agents](#using-mesh-agents)
   - [Synchronous API](#synchronous-api)
   - [Asynchronous API](#asynchronous-api)
4. [Folder Structure](#folder-structure)
5. [How to Build a New Mesh Agent](#how-to-build-a-new-mesh-agent)
   - [Prerequisites](#prerequisites)
   - [Creating a New Mesh Agent](#creating-a-new-mesh-agent)
   - [Testing Your Agent](#testing-your-agent)
   - [Start a Local Server](#start-a-local-server)
6. [Contributor Guidelines](#contributor-guidelines)
   - [Pull Request Checklist](#pull-request-checklist)
   - [Coding Style and Best Practices](#coding-style-and-best-practices)
   - [Metadata Requirements](#metadata-requirements)
   - [Testing Requirements](#testing-requirements)
7. [Examples](#examples)
8. [Contact & Support](#contact--support)

---

## How It Works

- **Mesh Agents** can process information from external APIs, or access other mesh agents.
- Agents run on a decentralized compute layer, and each agent can optionally use external APIs, Large Language Models, or other tools provided by Heurist and 3rd parties.
- **Agent Developers** can contribute by adding specialized agents to the network. Each invocation of an agent can generate pay-per-use revenue for the agent's author.  
- **Users or Developers** get access to a rich library of pre-built, purpose-driven AI agents they can seamlessly integrate into their products or workflows via REST APIs or frontend interface usage.

## Agent Deployment

Mesh agents are deployed on Heurist's compute layer. Agents are deployed after a pull request is merged into the main branch, and will be available for use in the Heurist Mesh via API or frontend interface. Heurist team will take care of the API keys and other environment variables used by the agents.

We plan to enable local hosting of agents in the future where you can deploy your agents on your own servers without giving up any sensitive data, while being composable with the Heurist Mesh.

## Using Mesh Agents

Mesh agents hosted by Heurist can be accessed via two interfaces:

- Synchronous API - Direct, immediate responses
- Asynchronous API - For longer-running tasks, or when you want to query the reasoning process of the agent execution

To use any Mesh agent, you'll need a Heurist API key. Get one at https://www.heurist.ai/credits 

### Synchronous API

Make a POST request to URL: `https://sequencer-v2.heurist.xyz/mesh_request`

#### Natural Language Mode

Use this mode to interact with agents using human language. The agent will interpret the query and return a response. Example:

```bash
curl -X POST https://sequencer-v2.heurist.xyz/mesh_request \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "CoinGeckoTokenInfoAgent",
    "input": {
      "query": "Get info about ADA",
      "api_key": "YOUR_API_KEY"
    }
  }'
```

Response format:
```json
{
  "response": "Natural language explanation of results",
  "data": {
    // Structured data specific to the agent
  }
}
```

#### Direct Tool Mode

Some agents provides direct tool access. You can bypass the language model and call specific agent tools. Tools are typically functions that wrap external APIs. This mode is useful if you know the exact API schema and want to reduce the response time. Example:

```bash
curl -X POST https://sequencer-v2.heurist.xyz/mesh_request \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "CoinGeckoTokenInfoAgent",
    "input": {
      "tool": "get_token_info",
      "tool_arguments": {
        "coingecko_id": "cardano"
      },
      "raw_data_only": true,
      "api_key": "YOUR_API_KEY"
    }
  }'
```

You may set `raw_data_only: true` to receive only structured data without natural language explanations.

### Asynchronous API

For tasks that may take longer to complete, use the asynchronous API flow:

1. Create a task:

```bash
curl -X POST https://sequencer-v2.heurist.xyz/mesh_task_create \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "ZkIgniteAnalystAgent",
    "agent_type": "AGENT",
    "task_details": {
      "query": "Compare the TVL and yield of ZeroLend and Syncswap"
    },
    "api_key": "YOUR_API_KEY"
  }'
```

2. Query task status using the returned task_id:

```bash
curl -X POST https://sequencer-v2.heurist.xyz/mesh_task_query \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "RETURNED_TASK_ID",
    "api_key": "YOUR_API_KEY"
  }'
```

---

## Folder Structure

Inside the `mesh` folder, you will find:

```
mesh/
├─ __init__.py
├─ mesh_agent.py            # Base class for all Mesh Agents
├─ your_agent.py            # Your agent class
├─ ...
└─ tests/
   ├─ your_agent.py         # Test script for your agent
   └─ ...
```

- **`mesh_agent.py`** – The base `MeshAgent` class providing a common interface and shared utilities (metadata handling, lifecycle management, default model references, etc.).  
- **`{your_agent_name}_agent.py`** – Individual agent classes extending `MeshAgent` to implement specialized functionalities.  
- **`tests/`** – Contains test scripts for each agent, demonstrating example inputs and outputs.  

---

## How to Build a New Mesh Agent

### Prerequisites

- Python 3.11+
- Heurist API key

### Creating a New Mesh Agent

1. **Create Your Agent File**  
   - Name it logically, e.g., `my_special_agent.py`.  
   - Place it inside the `mesh` folder at the same level as the other agent files.

2. **Inherit from `MeshAgent`**  
   ```python
   from .mesh_agent import MeshAgent
   from typing import Dict, Any

   class MySpecialAgent(MeshAgent):
       def __init__(self):
           super().__init__()
           self.metadata.update({
               'name': 'My Special Agent',
               'version': '1.0.0',
               'author': 'Your Name',
               'author_address': '0xYourEthereumAddress',
               'description': 'This agent can do...',
               'inputs': [
                   {'name': 'query', 'description': 'User input or data', 'type': 'str'}
                   # Add more inputs as needed
               ],
               'outputs': [
                   {'name': 'response', 'description': 'Agent response', 'type': 'str'}
                   # Add more outputs as needed. We recommend adding 'data' output for raw data from external APIs.
               ],
               'external_apis': [],
               'tags': ['DeFi', 'Trading']
           })

       async def handle_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
           # Implement your agent logic here
           user_input = params.get('query', '')
           response_text = f"This is a response from MySpecialAgent for query: {user_input}"
           return {"response": response_text}
   ```

3. **Implement Your Custom Logic**  
   - You can call external APIs, orchestrate LLM calls, call other mesh agents, or do any specialized processing.
   - You should use os.environ to store any sensitive information. Such data should not be hardcoded in the agent file and should not be committed to the repository.
   - Make sure to wrap significant network operations with any relevant decorators (e.g., `@with_retry`, `@with_cache`) if needed.

4. **Add Required Metadata**  
   - Ensure you update `self.metadata` with all relevant fields (e.g., `name`, `description`, `inputs`, `outputs`, `tags`, and any external APIs used).

### Testing Your Agent

1. **Create a Test Script**  
   - In `mesh/tests`, create a file named `my_special_agent.py` (or similar).  
   - Import your agent and run a small demonstration of `handle_message`.

   ```python
   # mesh/tests/my_special_agent.py
   import sys
   from pathlib import Path
   import yaml
   import asyncio

   sys.path.append(str(Path(__file__).parent.parent.parent))

   from mesh.my_special_agent import MySpecialAgent

   async def run_agent():
       agent = MySpecialAgent()
       try:
           input_data = {'query': 'Say something special'}
           output_data = await agent.handle_message(input_data)

           # Save to YAML as an example
           test_dir = Path(__file__).parent
           output_file = test_dir / "my_special_agent_example.yaml"

           yaml_content = {
               'input': input_data,
               'output': output_data
           }
           with open(output_file, 'w', encoding='utf-8') as f:
               yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)

           print(f"Results saved to {output_file}")
       finally:
           await agent.cleanup()

   if __name__ == "__main__":
       asyncio.run(run_agent())
   ```

2. **Run the Test**  
   ```bash
   cd mesh/tests
   python my_special_agent.py
   ```
   - This will produce a YAML file (e.g., `my_special_agent_example.yaml`) showing the input and output.

3. **Check and Update**  
   - Verify the output is as expected.  
   - Update your agent logic or test script if necessary.  

### Start a local server

`mesh_api.py` is a FastAPI server that you can use to test your agent with synchronous API. It will automatically reload the agent when you make changes to the code.

```bash
python3 -m uvicorn mesh_api:app --reload
```

You can now test your agent by calling `http://localhost:8000/mesh_request` with the same input as in the test script.

---

## Contributor Guidelines

We welcome community contributions to develop the Heurist Mesh.

### Pull Request Checklist

1. **Fork the Repository**  
   - Fork [heurist-agent-framework](https://github.com/YOUR-ORG/heurist-agent-framework) to your own GitHub account.  

2. **Create a Branch**  
   - For a new agent, use a descriptive branch name, such as `feature/my-special-agent`.  

3. **Add Your Agent**  
   - Create your `.py` agent file under `mesh/`.  
   - Write a corresponding test script under `mesh/tests/`.  

4. **Test**  
   - Run your test script locally to confirm no errors occur and that your agent works as expected.  

5. **Open a Pull Request**  
   - Summarize what your agent does and why it's valuable.  
   - Include any relevant info (e.g., external APIs used, example test output).  

### Coding Style and Best Practices

- Use Python **type hints** (`typing.Dict`, `typing.Any`, etc.) where feasible.  
- Use **docstrings** for important methods or classes to clarify behavior.  
- **Modularity** is key: break large tasks into smaller methods, especially if you rely on external services. Every agent should focus on a single domain.
- Add proper [decorators](https://github.com/heurist-network/heurist-agent-framework/blob/main/decorators.py) to functions using external APIs for caching and retry.
- `query` should be the first parameter of `handle_message`. It should support natural language queries and does not make assumptions on matching an exact format. (for example, your agent should accept both "tell me about "Ethereum" and "analyze ETH" - we don't strictly match the token name or ticker but let LLM do the job)

### Metadata Requirements

Each agent's `metadata` dictionary should at least contain:
- **`name`**: Human-readable name of the agent.  
- **`version`**: Agent version (e.g., `1.0.0`).  
- **`author`**: Name or handle of the contributor.  
- **`author_address`**: Ethereum address (or any relevant address) for potential revenue share.  
- **`description`**: Short, clear summary of your agent's purpose.  
- **`inputs`**: List of inputs with `name`, `description`, and `type`.
- **`outputs`**: List of outputs with `name`, `description`, and `type`.  
- **`external_apis`**: Any external service your agent accesses (e.g., `['DefiLlama']`).  
- **`tags`**: Keywords or categories to help users discover your agent.  
- **`mcp_tool_name` (optional)**: If you want your agent interoperable through Claude's MCP interface, specify a unique tool name.

### Testing Requirements

- Provide one **test script** in `mesh/tests/` that:  
  1. Instantiates your agent.  
  2. Calls its `handle_message` with example input.  
  3. Outputs results to a `.yaml` file as an example.  

---

## Examples

We have included example agents in this folder:

1. **Allora Price Prediction Agent** (`allora_price_prediction_agent.py`)  
   - Fetches and predicts short-term crypto prices using Allora's API.  
   - Demonstrates how to integrate external APIs, handle asynchronous calls, and structure multi-step logic.

2. **Token Contract Security Agent** (`goplus_analysis_agent.py`)  
   - Fetches security details for blockchain token contracts using the GoPlus API.  
   - Showcases best practices for validating user queries, calling external tools, and returning structured data.  

Each example agent has a corresponding test script in `mesh/tests/` that demonstrates how to run the agent and produce an example output file (in YAML).

---

## Contact & Support

- **Issues**: If you find bugs or have questions, open an issue on the [GitHub repository](https://github.com/heurist-network/heurist-agent-framework/issues).
- **Community Chat**: Join our [Discord](https://discord.com/invite/heuristai) or [Telegram Builder Group](https://t.me/heuristsupport) for real-time support or to showcase your new agents.  

Thank you for contributing to **Heurist Mesh** and helping build a diverse ecosystem of AI agents! We're excited to see the specialized solutions you create. 

> **Happy Hacking & Welcome to the Mesh!**  

---  

*This document is a work-in-progress. Please feel free to update and improve it as the system evolves.*
