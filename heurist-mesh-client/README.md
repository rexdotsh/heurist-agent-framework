# Heurist Mesh API

### Overview

Heurist Mesh provides a powerful REST API that enables developers to seamlessly interact with various AI agents and tools programmatically. This document outlines how to use the API client to submit jobs and retrieve results from different specialized agents.

### Installation

You can install the Heurist Mesh client directly from GitHub:

```bash
pip install git+https://github.com/heurist-network/heurist-agent-framework.git#subdirectory=heurist-mesh-client
```

### Authentication

The API requires authentication using an API key. Set your API key as an environment variable:

```
HEURIST_API_KEY=your_api_key_here
```

Alternatively, you can use a `.env` file in your project directory:

```
# .env file
HEURIST_API_KEY=your_api_key_here
```

Then load it in your application:

```python
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

# the client will automatically use the HEURIST_API_KEY from environment
from heurist_mesh_client import MeshClient
client = MeshClient()
```

### Client Setup

The client supports both synchronous requests and asynchronous task patterns. Here's how to initialize the client:

```python
from heurist_mesh_client import MeshClient

# initialize the client
client = MeshClient()
```

### Request Patterns

The client supports two main request patterns:

1. **Synchronous Requests**: Get immediate results
2. **Asynchronous Tasks**: Create a task and poll for results

#### Synchronous Request Example

```python
# synchronous request
response = client.sync_request(
    agent_id="CoinGeckoTokenInfoAgent",
    query="What is the current price of Bitcoin?"
)
print(json.dumps(response, indent=2))
```

#### Asynchronous Task Example

```python
# create an asynchronous task
task = client.create_task(
    agent_id="CoinGeckoTokenInfoAgent",
    query="What is the current price of Bitcoin?"
)
print(f"Task created with ID: {task.task_id}")

# poll for results
while True:
    result = client.query_task(task_id=task.task_id)
    if result.status == "finished":
        print("Task completed:")
        print(json.dumps(result.model_dump(), indent=2))
        break
    print("Task in progress...")
    time.sleep(1)
```

### Request Types

The client handles two main types of requests:

1. **Natural Language Queries**: Requires `agent_id` and `query`
2. **Tool-based Requests**: Requires `agent_id` and `tool` (some tools may need `tool_arguments`)

#### Natural Language Example

```python
# natural language query
response = client.sync_request(
    agent_id="CoinGeckoTokenInfoAgent",
    query="What is the current price of Bitcoin?"
)
```

#### Tool-based Request Example

```python
# tool-based request with specific arguments
response = client.sync_request(
    agent_id="CoinGeckoTokenInfoAgent",
    tool="get_token_price",
    tool_arguments={"coingecko_id": "solana"},
)
```

### Parameters

| Parameter        | Type    | Required | Description                                                                       |
| ---------------- | ------- | -------- | --------------------------------------------------------------------------------- |
| `agent_id`       | String  | Yes      | The identifier for the Heurist agent you want to interact with                    |
| `query`          | String  | No†      | Natural language query for the agent (\*Required if not using tool-based request) |
| `tool`           | String  | No†      | Name of the specific tool to invoke (\*Required for tool-based requests)          |
| `tool_arguments` | Dict    | No‡      | Arguments for the tool being invoked (\*Required for tool-based requests)         |
| `raw_data_only`  | Boolean | No       | Whether to return raw data without natural language processing (Default: `false`) |

† You must provide either `query` OR `tool`, but not both  
‡ Only required for tools that accept arguments. Check the agent's documentation for tool-specific requirements

### Response Structure

The API returns responses in JSON format containing:

```python
# for sync_request:
{
    "result": {
        # agent-specific response data
    },
    "status": "success",  # or "error"
    "message": "optional status message"
}

# for query_task:
{
    "status": "finished",  # or "in_progress", "error"
    "reasoning_steps": [
        {
            "timestamp": 1234567890,
            "content": "step description",
            "is_sent": true
        }
    ],
    "result": {
        "response": {
            # agent-specific response data
        },
        "success": true  # or false
    }
}
```

### Available Agents

The Mesh API provides access to a growing ecosystem of specialized AI agents:

| Agent                        | Description                                           |
| ---------------------------- | ----------------------------------------------------- |
| AlloraPricePredictionAgent   | ETH/BTC price predictions with confidence intervals   |
| BitquerySolanaTokenInfoAgent | Solana token trading data and trending tokens         |
| CoinGeckoTokenInfoAgent      | Token information, market data, and trending coins    |
| DexScreenerTokenInfoAgent    | Real-time DEX trading data across multiple chains     |
| ExaSearchAgent               | Web search with content relevance assessment          |
| ElfaTwitterIntelligenceAgent | Twitter mentions and trends for crypto tokens         |
| FirecrawlSearchAgent         | Advanced search with intelligent query analysis       |
| GoplusAnalysisAgent          | Security analysis of blockchain token contracts       |
| PumpFunTokenAgent            | Solana token analysis with market cap and holder data |
| ZkIgniteAnalystAgent         | zkSync Era DeFi analysis with yield and TVL data      |
| ZerionWalletAnalysisAgent    | Token and NFT holdings of a crypto wallet             |

For a complete and up-to-date list of available agents and their capabilities, visit the [Mesh Agents Metadata Endpoint](https://mesh.heurist.ai/mesh_agents_metadata.json).
