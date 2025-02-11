# clients/mesh_client.py
import logging
import asyncio
import json
from .base_client import BaseAPIClient
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MeshClient(BaseAPIClient):
    """Client for invoking other agents through Protocol V2 Server"""
   
    async def create_task(
        self,
        agent_id: str,
        task_details: Dict[str, Any],
        api_key: str
    ) -> Dict[str, Any]:
        """Create a task for another agent
        
        Args:
            agent_id: ID of the agent to invoke (e.g. 'EchoAgent')
            task_details: Parameters to pass to the agent
            api_key: Heurist API key for authentication
            
        Returns:
            Dict containing {
                "task_id": str,
                "msg": str
            }
        """
        payload = {
            "agent_id": agent_id,
            "agent_type": "AGENT",  # Default type as defined in mesh_manager
            "task_details": task_details,
            "api_key": api_key
        }

        try:
            response = await self._async_request(
                method="post",
                endpoint="/mesh_task_create",
                json=payload
            )
            return response
            
        except Exception as e:
            logger.error(f"Failed to create task for agent {agent_id}: {e}")
            raise

    async def poll_result(
        self,
        task_id: str,
        max_retries: int = 30,
        retry_delay: float = 1.0
    ) -> Optional[Dict[str, Any]]:
        """Poll for task result with retries
        
        Args:
            task_id: The task ID to poll
            max_retries: Maximum number of polling attempts
            retry_delay: Delay between polling attempts in seconds
        """
        for attempt in range(max_retries):
            try:
                response = await self._async_request(
                    method="post",
                    endpoint="/task_result_query",
                    json={"task_id": task_id}
                )

                logger.info(f"Task {task_id} response: {response}")
                
                status = response.get("status")
                if status == "finished":
                    try:
                        return json.loads(response.get("result", "{}"))
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse task result: {response.get('result')}")
                        return None
                elif status in ["failed", "canceled"]:
                    logger.error(f"Task {task_id} {status}")
                    return None
                    
                # Task still running, wait and retry
                await asyncio.sleep(retry_delay)
                
            except Exception as e:
                logger.error(f"Error polling task {task_id}: {e}")
                await asyncio.sleep(retry_delay)
        
        logger.error(f"Task {task_id} polling timed out after {max_retries} attempts")
        return None

    def push_update(self, task_id: str, content: str):
        """Push an update for a running task. Use this to push reasoning steps, etc.
        
        Args:
            task_id: ID of the task to update
            content: Update message content
            
        Returns:
            Dict containing server response
        """

        # FIXME: DEBUG: print the content
        print(f"Pushing update for task {task_id}: {content}")
        return

        try:
            self._sync_request(
                method="post",
                endpoint="/mesh_task_update",
                json={
                    "task_id": task_id,
                    "content": content
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to push update for task {task_id}: {e}")
            # Non-critical error, don't raise

    async def mesh_request(
        self,
        agent_id: str,
        input_data: Dict[str, Any],
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make a direct request to an agent
        
        Args:
            agent_id: ID of the agent to invoke (e.g. 'EchoAgent')
            input_data: Input parameters for the agent
            api_key: Optional Heurist API key for authentication
            
        Returns:
            Dict containing the agent's response
        """
        payload = {
            "agent_id": agent_id,
            "input": input_data
        }
        
        if api_key:
            payload["heurist_api_key"] = api_key

        try:
            response = await self._async_request(
                method="post",
                endpoint="/mesh_request",
                json=payload
            )
            return response
            
        except Exception as e:
            logger.error(f"Failed to make mesh request to agent {agent_id}: {e}")
            raise
