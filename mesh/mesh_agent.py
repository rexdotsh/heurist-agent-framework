from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from decorators import with_cache, with_retry, monitor_execution
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

# By default, large and small models are the same
DEFAULT_MODEL_ID = "nvidia/llama-3.1-nemotron-70b-instruct"

HEURIST_BASE_URL = os.getenv('HEURIST_BASE_URL')
HEURIST_API_KEY = os.getenv('HEURIST_API_KEY')

class MeshAgent(ABC):
    """Base class for all mesh agents"""
    
    def __init__(self):
        self.agent_name: str = self.__class__.__name__
        self.metadata: Dict[str, Any] = {
            'name': self.agent_name,
            'version': '1.0.0',
            'author': 'unknown',
            'author_address': '0x0000000000000000000000000000000000000000',
            'description': '',
            'inputs': [],
            'outputs': [],
            'external_apis': [],
            'tags': [],
            'large_model_id': DEFAULT_MODEL_ID,
            'small_model_id': DEFAULT_MODEL_ID,
            'mcp_tool_name': None
        }
        self.heurist_base_url = HEURIST_BASE_URL
        self.heurist_api_key = HEURIST_API_KEY
        self._api_clients: Dict[str, Any] = {}
    
    @abstractmethod
    async def handle_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming message - must be implemented by subclasses"""
        pass

    def set_heurist_api_key(self, api_key: str) -> None:
        self.heurist_api_key = api_key

    async def cleanup(self):
        """Cleanup API clients"""
        for client in self._api_clients.values():
            await client.close()
        self._api_clients.clear()

    def __del__(self):
        """Destructor to ensure cleanup of resources"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.cleanup())
            else:
                loop.run_until_complete(self.cleanup())
        except Exception:
            pass
