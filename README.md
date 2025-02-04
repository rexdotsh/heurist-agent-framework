<div align="center">
    <img src="./docs/img/agent-framework-poster.jpg" alt="Heurist Agent Framework Banner" width="100%" />
</div>

# Heurist Agent Framework
*The Raspberry Pi of Agent Frameworks*

A flexible multi-interface AI agent framework that can interact through various platforms including Telegram, Discord, Twitter, Farcaster, and REST API.

Grab a Heurist API Key instantly for free by using the code 'agent' while submitting the form on https://heurist.ai/dev-access

---

## For Heurist Mesh Contributors

**Heurist Mesh** is our new open network where AI agents can be contributed by the community and used modularly—just like DeFi smart contracts. If you want to **add your own specialized agents**, **please see the [Mesh README](./mesh/README.md)** for detailed guidelines, examples, and best practices.

---

## Overview

The Heurist Agent Framework is built on a modular architecture that allows an AI agent to:
- Process text messages and generate responses
- Generate and handle images
- Process voice messages (transcription and text-to-speech)
- Interact across multiple platforms with consistent behavior

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

<div align="center">
<img src="./docs/img/HFA_2.png" alt="Heurist Agent Framework" width="500">
</div>

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
   - **Most important** for the community, as these issues help drive the direction of our framework’s evolution.  
   - If you have an idea but aren’t sure how to implement it, open an issue under this label so others can pick it up or offer suggestions.

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
     - Once your Pull Request addressing the bounty is merged, we’ll follow up on fulfilling the reward.  
     - Additional instructions (e.g., contact method) may be included in the issue itself.

### Picking Up an Issue

- Look for **Integration Requests** or **Bounty** issues if you want to contribute new features or earn rewards.  
- Feel free to discuss approaches in the comments. If you’re ready to tackle it, mention “I’m working on this!” so others know it’s in progress.

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
