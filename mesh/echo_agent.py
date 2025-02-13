import logging
import asyncio
import random
from typing import Dict, Any
from .mesh_agent import MeshAgent  # Adjust this import to match your project structure

logger = logging.getLogger(__name__)

class EchoAgent(MeshAgent):
    def __init__(self):
        super().__init__()
        # You can update the metadata if you want
        self.metadata.update({
            'name': 'EchoAgent',
            'version': '1.0.0',
            'author': 'Heurist Team',
            'description': 'An agent that simply echoes user input after a random delay (1-4 seconds). This agent is for testing only.',
            'inputs': [
                {
                    'name': 'query',
                    'description': 'Any text to echo back.',
                    'type': 'str'
                }
            ],
            'outputs': [
                {
                    'name': 'response',
                    'description': 'Echoed text identical to the input.',
                    'type': 'str'
                }
            ],
            'tags': ['Test']
        })

    async def handle_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sleep for a random 1-4 seconds, then echo the input."""
        query = params.get("query", "")
        # Example of accessing auth info
        user_context = f"heurist_api_key: {self.heurist_api_key}"
        delay = random.uniform(1, 4)
        await asyncio.sleep(delay)

        return {"response": query}
