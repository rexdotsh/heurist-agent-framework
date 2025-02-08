# mesh/composable_echo_agent.py
from typing import Dict, Any
import logging
from .mesh_agent import MeshAgent
from clients.mesh_client import MeshClient

logger = logging.getLogger(__name__)

class ComposableEchoAgent(MeshAgent):
    def __init__(self):
        super().__init__()
        self.metadata.update({
            'name': 'ComposableEchoAgent',
            'version': '1.0.0',
            'author': 'Heurist Team',
            'description': 'An agent that calls EchoAgent and prefixes its response. This agent is for testing only.',
            'inputs': [
                {
                    'name': 'query',
                    'description': 'Text to echo back with prefix',
                    'type': 'str'
                }
            ],
            'outputs': [
                {
                    'name': 'response',
                    'description': 'Prefixed echo response',
                    'type': 'str'
                }
            ],
            'tags': ['Test']
        })
        self.mesh_client = MeshClient(self.heurist_base_url)
        if not self.heurist_base_url:
            raise ValueError("HEURIST_BASE_URL environment variable is not set")

    async def handle_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query", "")
        logger.info(f"[{self.agent_name}] Original query: {query}")

        # Create task for EchoAgent
        echo_response = await self.mesh_client.create_task(
            agent_id="EchoAgent",
            task_details={"query": query},
            api_key=self.heurist_api_key
        )
        
        # Get task ID from response
        echo_task_id = echo_response.get("task_id")
        if not echo_task_id:
            raise ValueError("No task_id in EchoAgent response")

        # Wait for EchoAgent result
        echo_result = await self.mesh_client.poll_result(echo_task_id)
        if not echo_result:
            raise RuntimeError(f"Failed to get result from EchoAgent task {echo_task_id}")

        # Get the echoed response
        echoed_text = echo_result.get("response", "")
        
        # Add prefix to the echoed response
        prefixed_response = f"COMPOSABLE: {echoed_text}"
        
        return {"response": prefixed_response}

    async def cleanup(self):
        await self.mesh_client.close()
        await super().cleanup()