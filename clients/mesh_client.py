# clients/mesh_client.py
import asyncio
import json
from .base_client import BaseAPIClient
from typing import Dict, Any, Optional
from loguru import logger

class MeshClient(BaseAPIClient):
    """Client for invoking other agents through Protocol V2 Server"""
   
    async def create_task(
        self,
        agent_id: str,
        task_details: Dict[str, Any],
        api_key: str
    ) -> Dict[str, Any]:
        """Create a task for another agent with proper task ID propagation"""
        task_details_copy = task_details.copy()
        origin_task_id = task_details_copy.get("origin_task_id")
        
        payload = {
            "agent_id": agent_id,
            "agent_type": "AGENT",
            "task_details": task_details_copy,
            "api_key": api_key,
        }
        
        if origin_task_id:
            payload["origin_task_id"] = origin_task_id
            task_details_copy["origin_task_id"] = origin_task_id
        
        try:
            response = await self._async_request(
                method="post",
                endpoint="/mesh_task_create",
                json=payload
            )
            logger.info(f"Task created | Agent: {agent_id} | Task ID: {response.get('task_id')}")
            return response
        except Exception as e:
            logger.error(f"Task creation failed | Agent: {agent_id} | Error: {str(e)}")
            raise

    async def poll_result(
        self,
        task_id: str,
        max_retries: int = 30,
        retry_delay: float = 1.0
    ) -> Optional[Dict[str, Any]]:
        """Poll for task result and reasoning steps"""
        seen_steps = set()
        logger.debug(f"Starting poll | Task: {task_id}")
        
        for attempt in range(max_retries):
            try:
                response = await self._async_request(
                    method="post",
                    endpoint="/mesh_task_query",
                    json={"task_id": task_id}
                )

                if not response:
                    logger.warning(f"Empty response | Task: {task_id} | Attempt: {attempt + 1}")
                    await asyncio.sleep(retry_delay)
                    continue

                # Handle reasoning steps
                reasoning_steps = response.get("reasoning_steps", []) or []
                for step in reasoning_steps:
                    step_content = step.get("content", "")
                    if step_content and step_content not in seen_steps:
                        logger.info(f"Reasoning step | Task: {task_id} | Content: {step_content}")
                        seen_steps.add(step_content)
                
                # Handle status
                status = response.get("status")
                if status == "finished":
                    return response.get("result")
                elif status in ["failed", "canceled"]:
                    logger.error(f"Task {status} | Task: {task_id} | Message: {response.get('message', '')}")
                    return response
                    
                await asyncio.sleep(retry_delay)
                
            except Exception as e:
                logger.error(f"Poll error | Task: {task_id} | Error: {str(e)}")
                await asyncio.sleep(retry_delay)
        
        logger.error(f"Poll timeout | Task: {task_id} | Max retries: {max_retries}")
        return None

    def push_update(self, task_id: str, content: str):
        """Push an update for a running task"""
        try:
            self._sync_request(
                method="post",
                endpoint="/mesh_task_update",
                json={
                    "task_id": task_id,
                    "content": content
                }
            )
            logger.debug(f"Update pushed | Task: {task_id} | Content: {content}")
            
        except Exception as e:
            logger.error(f"Update failed | Task: {task_id} | Error: {str(e)}")

    async def mesh_request(
        self,
        agent_id: str,
        input_data: Dict[str, Any],
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make a direct request to an agent"""
        payload = {
            "agent_id": agent_id,
            "input": input_data
        }
        
        if api_key:
            payload["heurist_api_key"] = api_key

        logger.debug(f"Direct request | Agent: {agent_id}")
        try:
            response = await self._async_request(
                method="post",
                endpoint="/mesh_request",
                json=payload
            )
            logger.info(f"Direct request completed | Agent: {agent_id}")
            return response
            
        except Exception as e:
            logger.error(f"Direct request failed | Agent: {agent_id} | Error: {str(e)}")
            raise