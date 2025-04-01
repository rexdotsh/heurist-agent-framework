<div align="center">
    <img src="./docs/img/agent-framework-poster.jpg" alt="Heurist Agent Framework Banner" width="100%" />
</div>

# Heurist Agent Framework

A flexible multi-interface AI agent framework that can interact through various platforms including Telegram, Discord, Twitter, Farcaster, REST API, and MCP.

Grab a Heurist API Key instantly for free by using the code 'agent' while submitting the form on https://heurist.ai/dev-access

---

## Overview

The Heurist Agent Framework is built on a modular architecture that allows an AI agent to:
- Process text and voice messages
- Generate images and videos
- Interact across multiple platforms with consistent behavior
- Fetch and store information in a knowledge base (Postgres and SQLite supported)
- Access external APIs, tools, and a wide range of [Mesh Agents](./mesh/README.md) to compose complex workflows

## Features

- 🤖 Core Agent functionality with LLM integration
- 🖼️ Image generation capabilities
- 🎤 Voice processing (transcription and TTS)
- 🔌 Multiple interface support:
  - Telegram bot
  - Discord bot
  - Twitter automation
  - Farcaster integration
  - REST API

## Heurist Mesh
![mesh](https://github.com/user-attachments/assets/77a2ab3b-e35c-4313-8a5b-a0e751cac879)

**Heurist Mesh** is an open network where AI agents are contributed by the community and used modularly—similar to DeFi smart contracts. Each agent is a specialized unit that can process data, generate reports, or perform actions while collectively forming an intelligent swarm to tackle complex tasks. Each agent is accessible via a unified REST API interface, and can be used in conjunction with any agent framework or application.

Want to contribute your own agent? Check out the [Mesh README](./mesh/README.md) for detailed guidelines, examples, and best practices.

### MCP Support
**Just In:** All Heurist Mesh agents are accessible via MCP! This means you can access them from your favorite MCP client, including [Claude Desktop](https://claude.ai/download), [Cursor](https://www.cursor.com/), and [Windsurf](https://codeium.com/windsurf).

Head to [heurist-mesh-mcp-server](https://github.com/heurist-network/heurist-mesh-mcp-server) to set up a server and give your AI assistant a powerup.

### Recommended Mesh Agents

• [BitquerySolanaTokenInfoAgent](./mesh/bitquery_solana_token_info_agent.py) - Provides comprehensive analysis of Solana tokens including metrics, holders, trading activity, and trending token discovery

• [CoinGeckoTokenInfoAgent](./mesh/coingecko_token_info_agent.py) - Fetches token information, market data, trending coins, and category data from CoinGecko

• [DexScreenerTokenInfoAgent](./mesh/dexscreener_token_info_agent.py) - Fetches real-time DEX trading data and token information across multiple chains

• [ElfaTwitterIntelligenceAgent](./mesh/elfa_twitter_intelligence_agent.py) - Analyzes tokens, topics or Twitter accounts using Twitter data, highlighting smart influencers

• [ExaSearchAgent](./mesh/exa_search_agent.py) - Searches the web and provides direct answers to questions

• [GoplusAnalysisAgent](./mesh/goplus_analysis_agent.py) - Fetches and analyzes security details of blockchain token contracts

• [MetaSleuthSolTokenWalletClusterAgent](./mesh/metasleuth_sol_token_wallet_cluster_agent.py) - Analyzes wallet clusters holding Solana tokens to identify holder behavior, concentration, and potential market manipulation

• [PumpFunTokenAgent](./mesh/pumpfun_token_agent.py) - Analyzes Pump.fun tokens on Solana, tracking token creation and graduation events

• [SolWalletAgent](./mesh/sol_wallet_agent.py) - Queries Solana wallet assets and recent swap transactions

### Full List of Mesh Agents

[View the full list](./mesh/README.md#appendix-all-available-mesh-agents)

### Usage and Development Guide

[Read the Mesh documentation](./mesh/README.md)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/heurist-network/heurist-agent-framework.git
cd heurist-agent-framework
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
see `.env.example`


## Usage

### Running Different Interfaces

1. Telegram Agent:
```bash
python main_telegram.py
```

2. Discord Agent:
```bash
python main_discord.py
```

3. REST API:
```bash
python main_api.py
```

4. Twitter Bot (Posting):
```bash
python main_twitter.py
```

### API Endpoints

The REST API provides the following endpoints:

- POST `/message`
  - Request body: `{"message": "Your message here"}`
  - Response: `{"text": "Response text", "image_url": "Optional image URL"}`

Example:
```bash
curl -X POST http://localhost:5005/message \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about artificial intelligence"}'
```

## Architecture

The framework follows a modular design:

1. Core Agent (`core_agent.py`)
   - Handles core functionality
   - Manages LLM interactions
   - Processes images and voice

2. Interfaces
   - Telegram (`interfaces/telegram_agent.py`)
   - Discord (`interfaces/discord_agent.py`)
   - API (`interfaces/flask_agent.py`)
   - Twitter (`interfaces/twitter_agent.py`)
   - Farcaster (`interfaces/farcaster_agent.py`)

Each interface inherits from the CoreAgent and implements platform-specific handling.

<div align="center">
<img src="./docs/img/HFA_1.png" alt="Heurist Agent Framework" width="500">
</div>

### Main Loop

<div align="center">
<img src="./docs/img/HFA_2.png" alt="Heurist Agent Framework" width="500">
</div>

## Configuration

The framework uses YAML configuration for prompts and agent behavior. Configure these in:
```
config/prompts.yaml
```

## Development

To add a new interface:

1. Create a new class inheriting from `CoreAgent`
2. Implement platform-specific handlers
3. Use core agent methods for processing:
   - `handle_message()`
   - `handle_image_generation()`
   - `transcribe_audio()`

## How to Use GitHub Issues

We encourage the community to open **GitHub issues** whenever you have a new idea or find something that needs attention. When creating an issue, please use our [Issue Template](./.github/ISSUE_TEMPLATE/general_issue_template.md) and select one of the following categories:

1. **Integration Request**  
   - For requests to integrate with a new data source (e.g., CoinGecko, arXiv) or a new AI use case.  
   - **Most important** for the community, as these issues help drive the direction of our framework's evolution.  
   - If you have an idea but aren't sure how to implement it, open an issue under this label so others can pick it up or offer suggestions.

2. **Bug**  
   - For reporting errors or unexpected behavior in the framework.  
   - Provide as much detail as possible (logs, steps to reproduce, environment, etc.).

3. **Question**  
   - For inquiries about usage, best practices, or clarifications on existing features.

4. **Bounty**  
   - For tasks with a **reward** (e.g., tokens, NFTs, or other benefits).  
   - The bounty label indicates that Heurist team or another community member are offering a reward to whoever resolves the issue.  
   - **Bounty Rules**:
     - Make sure to read the issue description carefully for scope and acceptance criteria.  
     - Once your Pull Request addressing the bounty is merged, we'll follow up on fulfilling the reward.  
     - Additional instructions (e.g., contact method) may be included in the issue itself.

### Picking Up an Issue

- Look for **Integration Requests** or **Bounty** issues if you want to contribute new features or earn rewards.  
- Feel free to discuss approaches in the comments. If you're ready to tackle it, mention "I'm working on this!" so others know it's in progress.

This process helps us stay organized, encourages community involvement, and keeps development transparent.

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

For Heurist Mesh agents or to learn about contributing specialized community agents, please refer to the [Mesh README](./mesh/README.md)

## Support

For support, please open an issue in the GitHub repository or contact the maintainers. Join the Heurist Ecosystem Builder telegram https://t.me/heuristsupport

## WIP

More features and refinement on the way!

Example structure for finalized tweet flow on the works:

<div align="center">
<img src="./docs/img/TwitterFinalFLow.png" alt="Heurist Agent Framework" width="500">
</div>

*"_eval" param indicates requires agent to evaluate if it should respond*

*"_HITL" param indicates requirement to activate Human In The Loop flow*

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=heurist-network/heurist-agent-framework&type=Date)](https://star-history.com/#heurist-network/heurist-agent-framework&Date)
