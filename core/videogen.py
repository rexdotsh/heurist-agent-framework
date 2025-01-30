from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Tuple
import random
import string
import time
import requests
from dataclasses import dataclass

class WorkflowTaskType(str, Enum):
    # UPSCALER = "upscaler"
    # FLUX_LORA = "flux-lora"
    TEXT2VIDEO = "txt2vid"

def parse_api_key_string(combined_key: str) -> Tuple[str, str]:
    """Split the combined API key into consumer ID and API key."""
    parts = combined_key.split('#')
    return parts[0] if parts else '', parts[1] if len(parts) > 1 else ''

@dataclass
class WorkflowTaskResult:
    task_id: str
    status: str  # 'waiting' | 'running' | 'finished' | 'failed' | 'canceled'
    result: Optional[Any] = None

class WorkflowTask(ABC):
    def __init__(
        self,
        consumer_id: Optional[str] = None,
        job_id_prefix: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        workflow_id: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.consumer_id = consumer_id
        self.job_id_prefix = job_id_prefix
        self.timeout_seconds = timeout_seconds
        self.workflow_id = workflow_id
        self.api_key = api_key

    @property
    @abstractmethod
    def task_type(self) -> WorkflowTaskType:
        pass

    @property
    @abstractmethod
    def task_details(self) -> Dict[str, Any]:
        pass

class Text2VideoTask(WorkflowTask):
    def __init__(
        self,
        prompt: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: Optional[int] = None,
        length: Optional[int] = None,
        seed: Optional[int] = None,
        fps: Optional[int] = None,
        quality: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.prompt = prompt
        self.width = width
        self.height = height
        self.steps = steps
        self.length = length
        self.seed = seed
        self.fps = fps
        self.quality = quality

    @property
    def task_type(self) -> WorkflowTaskType:
        return WorkflowTaskType.TEXT2VIDEO

    @property
    def task_details(self) -> Dict[str, Any]:
        parameters = {"prompt": self.prompt}
        
        optional_params = {
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "length": self.length,
            "seed": self.seed,
            "fps": self.fps,
            "quality": self.quality
        }
        
        parameters.update({k: v for k, v in optional_params.items() if v is not None})
        return {"parameters": parameters}

class Workflow:
    def __init__(self, api_key: str, workflow_url: str):
        self.workflow_url = workflow_url
        self.default_consumer_id, self.default_api_key = parse_api_key_string(api_key)

    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make an HTTP request to the workflow API."""
        url = f"{self.workflow_url}/{endpoint}"
        response = requests.post(url, json=data)
        
        if not response.ok:
            try:
                error_data = response.json()
                error_message = error_data.get('error') or error_data.get('message') or 'Unknown error'
            except ValueError:
                error_message = response.text or 'Unknown error'
            raise Exception(error_message)
            
        return response.json()

    async def resource_request(self, consumer_id: str, workflow_id: Optional[str] = None) -> str:
        """Request resources for a workflow."""
        data = {
            "consumer_id": consumer_id,
            "workflow_id": workflow_id
        }
        result = self._make_request("resource_request", data)
        return result["miner_id"]

    def _generate_random_id(self, length: int = 10) -> str:
        """Generate a random hexadecimal ID."""
        return ''.join(random.choices(string.hexdigits.lower(), k=length))

    async def create_task(self, task: WorkflowTask) -> str:
        """Create a new workflow task."""
        task_id = self._generate_random_id()
        data = {
            "consumer_id": task.consumer_id or self.default_consumer_id,
            "api_key": task.api_key or self.default_api_key,
            "task_type": task.task_type,
            "task_details": task.task_details,
            "job_id": f"{task.job_id_prefix or 'sdk-workflow'}-{task_id}",
            "workflow_id": task.workflow_id
        }
        
        if task.timeout_seconds:
            data["timeout_seconds"] = task.timeout_seconds
            
        result = self._make_request("task_create", data)
        return result["task_id"]

    async def execute_workflow(self, task: WorkflowTask) -> str:
        """Execute a workflow task."""
        await self.resource_request(
            task.consumer_id or self.default_consumer_id,
            task.workflow_id
        )
        return await self.create_task(task)

    async def query_task_result(self, task_id: str) -> WorkflowTaskResult:
        """Query the result of a task."""
        result = self._make_request("task_result_query", {
            "task_id": task_id
        })
        return WorkflowTaskResult(**result)

    async def execute_workflow_and_wait_for_result(
        self,
        task: WorkflowTask,
        timeout: int = 600000,
        interval: int = 10000,
        initial_wait: int = 120000
    ) -> WorkflowTaskResult:
        """Execute a workflow task and wait for its result.
        
        Args:
            task: The workflow task to execute
            timeout: Maximum time to wait in milliseconds
            interval: Time between status checks in milliseconds
            initial_wait: Time to wait before first status check in milliseconds
        """
        if interval < 1000:
            raise ValueError("Interval should be more than 1000 (1 second)")

        task_id = await self.execute_workflow(task)
        start_time = time.time() * 1000  # Convert to milliseconds
        
        # Initial wait before first query
        await asyncio.sleep(initial_wait / 1000)  # Convert to seconds

        while True:
            result = await self.query_task_result(task_id)
            if result.status in ("finished", "failed"):
                return result

            if (time.time() * 1000) - start_time > timeout:
                raise TimeoutError("Timeout waiting for task result")

            await asyncio.sleep(interval / 1000)  # Convert to seconds

    async def cancel_task(self, task_id: str) -> Dict[str, str]:
        """Cancel a running task."""
        return self._make_request("task_cancel", {
            "task_id": task_id,
            "api_key": self.default_api_key
        })