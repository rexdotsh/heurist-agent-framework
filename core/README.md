# Heurist Core

Core components of the Heurist Agent Framework for building LLM-powered agents.

## Installation

```bash
pip install heurist-core
```

Or install directly from the repository:

```bash
pip install git+https://github.com/heurist-ai/heuman-agent-framework.git#subdirectory=core
```

## Features

- LLM interaction through various providers
- Embedding and vector storage utilities
- Image and video generation capabilities
- Voice synthesis and transcription
- Component-based architecture for agents
- Workflow management for complex agent tasks
- Tool integration framework

## Key Components

### LLM
Functions for interacting with large language models via various APIs.

```python
from heurist_core import call_llm, call_llm_with_tools
```

### Embedding
Vector storage and embedding utilities for semantic search and memory.

```python
from heurist_core import get_embedding, SQLiteVectorStorage, PostgresVectorStorage
```

### Image Generation
Tools for generating images from text prompts.

```python
from heurist_core import generate_image_with_retry_smartgen
```

### Voice
Speech synthesis and transcription utilities.

```python
from heurist_core import speak_text, transcribe_audio
```

### Components
Building blocks for agent architectures.

```python
from heurist_core.components import ConversationManager, LLMProvider, MediaHandler
```

### Tools
Framework for integrating tools with LLM agents.

```python
from heurist_core.tools import ToolBox, tool
```

### Workflows
Predefined workflows for complex agent tasks.

```python
from heurist_core.workflows import AugmentedLLMCall, ChainOfThoughtReasoning
```

## License

MIT
