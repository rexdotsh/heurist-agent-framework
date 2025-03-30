<div align="center">
  <img src="./docs/img/agent-framework-poster.jpg" alt="Heurist Agent Framework Banner" width="100%" />
  <h1>Heurist Agent Framework</h1>
  <p><strong>A flexible multi-interface AI agent framework with cross-platform capabilities</strong></p>
  
  <p>
    <a href="https://github.com/heurist-network/heurist-agent-framework/stargazers">
      <img src="https://img.shields.io/github/stars/heurist-network/heurist-agent-framework?style=for-the-badge" alt="Stars" />
    </a>
    <a href="https://github.com/heurist-network/heurist-agent-framework/network/members">
      <img src="https://img.shields.io/github/forks/heurist-network/heurist-agent-framework?style=for-the-badge" alt="Forks" />
    </a>
    <a href="https://github.com/heurist-network/heurist-agent-framework/issues">
      <img src="https://img.shields.io/github/issues/heurist-network/heurist-agent-framework?style=for-the-badge" alt="Issues" />
    </a>
    <a href="https://github.com/heurist-network/heurist-agent-framework/blob/main/LICENSE">
      <img src="https://img.shields.io/github/license/heurist-network/heurist-agent-framework?style=for-the-badge" alt="License" />
    </a>
  </p>
  
  <p>
    <a href="#overview">Overview</a> •
    <a href="#features">Features</a> •
    <a href="#heurist-mesh">Heurist Mesh</a> •
    <a href="#installation">Installation</a> •
    <a href="#usage">Usage</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#development">Development</a> •
    <a href="#contributing">Contributing</a>
  </p>
</div>

## 📋 Overview

The **Heurist Agent Framework** is built on a modular architecture that enables AI agents to interact across multiple platforms while maintaining consistent behavior. With this framework, agents can:

- Process text and voice messages
- Generate images and videos
- Interact seamlessly across Telegram, Discord, Twitter, Farcaster and more
- Fetch and store information in knowledge bases (Postgres and SQLite)
- Access external APIs, tools, and [Mesh Agents](#heurist-mesh) for complex workflows

> 🚀 **Get Started Quickly**: Grab a Heurist API Key instantly for free by using the code `agent` while submitting the form on [heurist.ai/dev-access](https://heurist.ai/dev-access)

## ✨ Features

<table>
  <tr>
    <td width="33%">
      <h3 align="center">🤖 Core Agent</h3>
      <p align="center">Advanced LLM integration with powerful reasoning capabilities</p>
    </td>
    <td width="33%">
      <h3 align="center">🖼️ Media Generation</h3>
      <p align="center">Create images and videos based on natural language prompts</p>
    </td>
    <td width="33%">
      <h3 align="center">🎤 Voice Processing</h3>
      <p align="center">Transcription and text-to-speech capabilities</p>
    </td>
  </tr>
  <tr>
    <td width="33%">
      <h3 align="center">🔌 Multi-Interface</h3>
      <p align="center">Deploy on Telegram, Discord, Twitter, Farcaster, and REST API</p>
    </td>
    <td width="33%">
      <h3 align="center">📊 Data Storage</h3>
      <p align="center">Flexible knowledge base with Postgres and SQLite support</p>
    </td>
    <td width="33%">
      <h3 align="center">🧩 Modular Design</h3>
      <p align="center">Easily extend with new capabilities and interfaces</p>
    </td>
  </tr>
</table>

## 🌐 Heurist Mesh

<div align="center">
  <img src="https://github.com/user-attachments/assets/77a2ab3b-e35c-4313-8a5b-a0e751cac879" alt="Heurist Mesh" width="80%" />
</div>

**Heurist Mesh** is an open network where AI agents contributed by the community can be used modularly—similar to DeFi smart contracts. Each agent is a specialized unit that can:

- Process data and generate reports
- Perform specific actions in its domain of expertise
- Collectively form an intelligent swarm to tackle complex tasks

All agents are accessible via a unified REST API interface and can be used with any agent framework or application.

> 👉 Want to contribute your own agent? Check out the [Mesh README](./mesh/README.md) for guidelines, examples, and best practices.

### 🖥️ MCP Support

**Just In:** All Heurist Mesh agents are accessible via MCP! Access them from your favorite MCP client, including:
- [Claude Desktop](https://claude.ai/download)
- [Cursor](https://www.cursor.com/)
- [Windsurf](https://codeium.com/windsurf)

Head to [heurist-mesh-mcp-server](https://github.com/heurist-network/heurist-mesh-mcp-server) to set up a server and give your AI assistant a powerup.

### 🧠 Available Mesh Agents

<details>
<summary><b>Click to view all available Mesh Agents (25+)</b></summary>
| Agent | Description | Tools | Source |
|-------|-------------|-------|--------|
| **AaveAgent** | Reports on Aave v3 protocols (Ethereum, Polygon, Avalanche, Arbitrum) | • get_aave_reserves | [Source](./mesh/aave_agent.py) |
| **AlloraPricePredictionAgent** | Predicts ETH/BTC prices with confidence intervals | • get_allora_prediction | [Source](./mesh/allora_price_prediction_agent.py) |
| **BitquerySolanaTokenInfoAgent** | Comprehensive Solana token analysis (metrics, holders, trading activity) | • query_token_metrics<br>• query_token_holders<br>• query_token_buyers<br>• query_top_traders<br>• query_holder_status<br>• get_top_trending_tokens | [Source](./mesh/bitquery_solana_token_info_agent.py) |
| **CarvOnchainDataAgent** | Natural language queries for blockchain metrics | • query_onchain_data | [Source](./mesh/carv_onchain_data_agent.py) |
| **CoinGeckoTokenInfoAgent** | Token information, market data, trending coins | • get_coingecko_id<br>• get_token_info<br>• get_trending_coins<br>• get_token_price_multi<br>• get_categories_list<br>• get_category_data<br>• get_tokens_by_category | [Source](./mesh/coingecko_token_info_agent.py) |
| **DeepResearchAgent** | Multi-level web research with recursive exploration | • deep_research | [Source](./mesh/deep_research_agent.py) |
| **DexScreenerTokenInfoAgent** | Real-time DEX trading data across multiple chains | • search_pairs<br>• get_specific_pair_info<br>• get_token_pairs | [Source](./mesh/dexscreener_token_info_agent.py) |
| **DuckDuckGoSearchAgent** | Web search results with intelligent summaries | • search_web | [Source](./mesh/duckduckgo_search_agent.py) |
| **ElfaTwitterIntelligenceAgent** | Token/topic/account analysis with smart influencer highlighting | • search_mentions<br>• search_account<br>• get_trending_tokens | [Source](./mesh/elfa_twitter_intelligence_agent.py) |
| **ExaSearchAgent** | Direct answers from web search results | • exa_web_search<br>• exa_answer_question | [Source](./mesh/exa_search_agent.py) |
| **FirecrawlSearchAgent** | Advanced research with intelligent query generation | • firecrawl_web_search<br>• firecrawl_extract_web_data | [Source](./mesh/firecrawl_search_agent.py) |
| **FundingRateAgent** | Fetches funding rate data and identifies arbitrage opportunities | • get_all_funding_rates<br>• get_symbol_funding_rates<br>• find_cross_exchange_opportunities<br>• find_spot_futures_opportunities | [Source](./mesh/funding_rate_agent.py) |
| **GoplusAnalysisAgent** | Fetches and analyzes security details of blockchain token contracts | • fetch_security_details | [Source](./mesh/goplus_analysis_agent.py) |
| **MasaTwitterSearchAgent** | Searches Twitter for trending topics and sentiment | • search_twitter | [Source](./mesh/masa_twitter_search_agent.py) |
| **MetaSleuthSolTokenWalletClusterAgent** | Analyzes wallet clusters holding Solana tokens | • fetch_token_clusters<br>• fetch_cluster_details | [Source](./mesh/metasleuth_sol_token_wallet_cluster_agent.py) |
| **PumpFunTokenAgent** | Analyzes Pump.fun tokens on Solana | • query_recent_token_creation<br>• query_latest_graduated_tokens | [Source](./mesh/pumpfun_token_agent.py) |
| **SolWalletAgent** | Queries Solana wallet assets and transactions | • get_wallet_assets<br>• analyze_common_holdings_of_top_holders<br>• get_tx_history | [Source](./mesh/sol_wallet_agent.py) |
| **ZerionWalletAnalysisAgent** | Analyzes token and NFT holdings for EVM wallets | • fetch_wallet_tokens<br>• fetch_wallet_nfts | [Source](./mesh/zerion_wallet_analysis_agent.py) |
| **ZkIgniteAnalystAgent** | Analyzes zkSync Era DeFi opportunities | - | [Source](./mesh/zkignite_analyst_agent.py) |
| **MoniTwitterProfileAgent** | Analyzes Twitter accounts for insights | • get_smart_profile<br>• get_smart_followers_history<br>• get_smart_mentions_history<br>• get_smart_followers_categories<br>• get_smart_followers_full<br>• get_smart_mentions_feed<br>• get_account_full_info | [Source](./mesh/twitter_insight_agent.py) |
| **TwitterInsightAgent** | Provides Twitter account insights | • get_smart_followers_history<br>• get_smart_followers_categories<br>• get_smart_mentions_feed | [Source](./mesh/twitter_insight_agent.py) |

## 🚀 Installation

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

```bash
# Copy the example file
cp .env.example .env

# Edit with your configuration
nano .env
```

## 🔧 Usage

### Running Different Interfaces

<table>
  <tr>
    <th>Interface</th>
    <th>Command</th>
  </tr>
  <tr>
    <td>Telegram Agent</td>
    <td><code>python main_telegram.py</code></td>
  </tr>
  <tr>
    <td>Discord Agent</td>
    <td><code>python main_discord.py</code></td>
  </tr>
  <tr>
    <td>REST API</td>
    <td><code>python main_api.py</code></td>
  </tr>
  <tr>
    <td>Twitter Bot</td>
    <td><code>python main_twitter.py</code></td>
  </tr>
</table>

### API Endpoints

The REST API provides the following endpoints:

```
POST /message
  Request: {"message": "Your message here"}
  Response: {"text": "Response text", "image_url": "Optional image URL"}
```

Example:
```bash
curl -X POST http://localhost:5005/message \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about artificial intelligence"}'
```

## 🏗️ Architecture

<div align="center">
  <img src="./docs/img/HFA_1.png" alt="Heurist Agent Framework Architecture" width="70%" />
</div>

The framework follows a modular design:

1. **Core Agent** (`core_agent.py`)
   - Handles core functionality
   - Manages LLM interactions
   - Processes images and voice

2. **Interfaces**
   - Telegram (`interfaces/telegram_agent.py`)
   - Discord (`interfaces/discord_agent.py`)
   - API (`interfaces/flask_agent.py`)
   - Twitter (`interfaces/twitter_agent.py`)
   - Farcaster (`interfaces/farcaster_agent.py`)

### Main Processing Loop

<div align="center">
  <img src="./docs/img/HFA_2.png" alt="Main Processing Loop" width="70%" />
</div>

## ⚙️ Configuration

The framework uses YAML configuration for prompts and agent behavior:

```yaml
# config/prompts.yaml
system_prompt: |
  You are a helpful assistant...

user_prompt_template: |
  {user_message}
  
# Additional configuration...
```

## 💻 Development

To add a new interface:

1. Create a new class inheriting from `CoreAgent`
2. Implement platform-specific handlers
3. Use core agent methods for processing:
   - `handle_message()`
   - `handle_image_generation()`
   - `transcribe_audio()`

## 🤝 Contributing

We encourage community contributions! Here's how to get involved:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

### Using GitHub Issues

When creating an issue, please select one of the following categories:

- **Integration Request**: New data source or AI use case integrations
- **Bug**: Report errors or unexpected behavior
- **Question**: Inquiries about usage or features
- **Bounty**: Tasks with rewards (tokens, NFTs, etc.)

For Heurist Mesh agents or to learn about contributing specialized community agents, refer to the [Mesh README](./mesh/README.md).

## 📄 License

[MIT License](./LICENSE) - See LICENSE file for details.

## 🆘 Support

For support, please:
- Open an issue in the GitHub repository
- Contact maintainers
- Join the [Heurist Ecosystem Builder Telegram](https://t.me/heuristsupport)

## 📊 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=heurist-network/heurist-agent-framework&type=Date)](https://star-history.com/#heurist-network/heurist-agent-framework&Date)

---

<div align="center">
  <p>Built with ❤️ by the Heurist Network community</p>
</div>