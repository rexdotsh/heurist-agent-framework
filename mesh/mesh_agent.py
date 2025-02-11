from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from decorators import with_cache, with_retry, monitor_execution
import asyncio
import os
from dotenv import load_dotenv
from clients.mesh_client import MeshClient

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
        
        self.mesh_client = MeshClient(base_url=os.getenv("PROTOCOL_V2_SERVER_URL", "https://sequencer-v2.heurist.xyz"))
        self._api_clients['mesh'] = self.mesh_client
    
    @abstractmethod
    async def handle_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming message - must be implemented by subclasses"""
        pass
    
    async def _before_handle_message(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Hook called before message handling. Return modified params or None"""
        self.push_update(params, f"{self.agent_name} is thinking...")
        return None

    async def _after_handle_message(self, response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Hook called after message handling. Return modified response or None"""
        return None

    def set_heurist_api_key(self, api_key: str) -> None:
        self.heurist_api_key = api_key

    def push_update(self, params: Dict[str, Any], content: str) -> None:
        """Push an update for a running task. This is only invoked if a task_id or origin_task_id is provided in the params"""
        origin_task_id = params.get('origin_task_id')
        if not origin_task_id:
            origin_task_id = params.get('task_id')
        if origin_task_id:
            self.mesh_client.push_update(origin_task_id, content)

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

    async def call_agent(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point that handles the message flow with hooks"""
        result = await self._before_handle_message(params)
        params = result if result is not None else params
            
        response = await self.handle_message(params)
        
        after_result = await self._after_handle_message(response)
        return after_result if after_result is not None else response
