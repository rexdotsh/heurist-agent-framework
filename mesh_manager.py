import os
import asyncio
import aiohttp
import logging
import json
import boto3
from datetime import datetime

from typing import Dict, Type, List, Any
from importlib import import_module
from pkgutil import iter_modules
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# For logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("MeshManager")

# Read environment variables
PROTOCOL_V2_SERVER_URL = os.getenv("PROTOCOL_V2_SERVER_URL", "https://sequencer-v2.heurist.xyz")
POLL_INTERVAL_SECONDS = float(os.getenv("POLL_INTERVAL_SECONDS", "2.0"))

PROTOCOL_V2_AUTH_TOKEN = os.getenv("PROTOCOL_V2_AUTH_TOKEN", "test_key")

# TODO: this is unused for now
DEFAULT_AGENT_TYPE = "AGENT"

# Import the base agent so we can identify agent subclasses
# Make sure this path matches your actual package structure
from mesh.mesh_agent import MeshAgent

# Add these env vars at the top with other env vars
S3_ENDPOINT = os.getenv('S3_ENDPOINT')
S3_ACCESS_KEY = os.getenv('ACCESS_KEY')
S3_SECRET_KEY = os.getenv('SECRET_KEY')
S3_BUCKET = os.getenv('S3_BUCKET', 'mesh')

def load_agents_from_mesh_folder() -> Dict[str, Type[MeshAgent]]:
    """
    Dynamically imports all modules in the 'heurist_agent_framework.mesh' package
    and returns a dict of {agent_id: agent_class} for every subclass of MeshAgent found.

    Also uploads the metadata to S3.
    """
    agents_dict = {}
    agents_metadata = {
        "last_updated": datetime.utcnow().isoformat(),
        "agents": {}
    }

    # Configure S3 client
    s3_client = boto3.client(
        's3',
        region_name='enam',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY
    )

    # The package name of your mesh folder
    package_name = "mesh"

    # Get the path to that package on disk
    # (Assuming `heurist_agent_framework/mesh/__init__.py` exists or mesh is recognized as a package)
    package = import_module(package_name)
    package_path = Path(package.__file__).parent

    for _, module_name, is_pkg in iter_modules([str(package_path)]):
        if is_pkg:
            # If there are subpackages, you might traverse them recursively if needed
            continue

        # Import the module
        full_module_name = f"{package_name}.{module_name}"
        try:
            mod = import_module(full_module_name)
        except Exception as e:
            logger.warning(f"Failed to import module {full_module_name}: {e}")
            continue

        # Look through module members to find classes that inherit from MeshAgent
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            # Check if it's a subclass of MeshAgent (but not MeshAgent itself)
            if (
                isinstance(attr, type)
                and issubclass(attr, MeshAgent)
                and attr is not MeshAgent
            ):
                # Example: use the class name as agent_id
                agent_id = attr.__name__
                agents_dict[agent_id] = attr

                # Get agent metadata
                agent = attr()
                agents_metadata["agents"][agent_id] = {
                    "metadata": agent.metadata,
                    "module": module_name
                }
                logger.info(f"Found agent: {agent_id} in {module_name}")

    # Upload metadata to S3
    try:
        metadata_json = json.dumps(agents_metadata, indent=2)
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key='mesh_agents_metadata.json',
            Body=metadata_json,
            ContentType='application/json'
        )
        logger.info("Successfully uploaded agents metadata to S3")
    except Exception as e:
        logger.error(f"Failed to upload metadata to S3: {e}")

    return agents_dict


class MeshManager:
    """
    The MeshManager coordinates tasks between the Protocol V2 server
    and the various MeshAgent implementations. Each agent has its own poll loop.
    """

    def __init__(self):
        self.session: aiohttp.ClientSession = None
        self.agents_dict = load_agents_from_mesh_folder()
        self.active_tasks = {}
        self.tasks = {} # Tracking poll tasks

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        for task in self.tasks.values():
            task.cancel()
        try:
            await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        except Exception:
            pass
        if self.session:
            await self.session.close()
            self.session = None

    async def poll_for_tasks_for_agent(self, agent_id: str, agent_cls: Type[MeshAgent]):
        """
        Continuously polls the Protocol V2 server for new tasks addressed to this particular agent_id.
        If a new task arrives, we instantiate the agent, run handle_message, then submit results.
        Runs in a loop with a fixed polling interval.
        """
        self.active_tasks[agent_id] = set()
        headers = {
            "Authorization": PROTOCOL_V2_AUTH_TOKEN,
            "Content-Type": "application/json"
        }
        while True:
            try:
                # Example payload to poll for tasks for a specific agent
                poll_endpoint = f"{PROTOCOL_V2_SERVER_URL}/mesh_manager_poll"
                submit_endpoint = f"{PROTOCOL_V2_SERVER_URL}/mesh_manager_submit"
                payload = {
                    "agent_info": [
                        {
                            "agent_id": agent_id,
                            "agent_type": DEFAULT_AGENT_TYPE,
                        }
                    ]
                }

                logger.debug(f"Polling for tasks (agent_id={agent_id})...")
                async with self.session.post(poll_endpoint, json=payload, headers=headers) as resp:
                    resp_data = await resp.json()
                    logger.debug(f"Poll response (agent_id={agent_id}): {resp_data}")

                    # NOTE: we're expecting a response like this:
                    # {
                    #   "input": {
                    #     "query": "user's query",
                    #     "some_other_field": "additional inputs",
                    #   },
                    #   "task_id": "task_id",
                    #   "heurist_api_key": "heurist_api_key",
                    # }
                    # The "input" field is the input for the agent
                    # The other fields are added by the V2 server for internal use
                    # IMPORTANT: if a self-hosted server queries the V2 server, it will not have the heurist_api_key
                    # TODO: V2 server should accept a PROTOCOL_V2_AUTH_TOKEN variable that can be used to authenticate the request

                    if "input" in resp_data:
                        task_id = resp_data.get("task_id")
                        user_input = resp_data["input"]  # This is what we pass to the agent

                        # Log current concurrency
                        self.active_tasks[agent_id].add(task_id)
                        logger.info(f"[{agent_id}] Current active tasks: {len(self.active_tasks[agent_id])}")

                        agent = agent_cls()
                        if "heurist_api_key" in resp_data:
                            agent.set_heurist_api_key(resp_data["heurist_api_key"])
                        logger.info(f"[{agent_id}] Processing task_id={task_id} with input={user_input}")
                        try:
                            result = await agent.handle_message(user_input)
                            logger.info(f"[{agent_id}] Task result: {result}")
                        except Exception as e:
                            logger.error(f"[{agent_id}] Error in handle_message: {e}", exc_info=True)
                            result = {"error": str(e)}
                        finally:
                            self.active_tasks[agent_id].remove(task_id)
                            await agent.cleanup()


                        # Submit the result back
                        submit_data = {
                            "task_id": task_id,
                            "agent_id": agent_id,
                            "agent_type": DEFAULT_AGENT_TYPE,
                            "results": result
                        }
                        try:
                            async with self.session.post(submit_endpoint, json=submit_data, headers=headers) as submit_resp:
                                submit_resp_data = await submit_resp.json()
                                logger.info(f"[{agent_id}] Submitted result for task_id={task_id}: {submit_resp_data}")
                        except Exception as e:
                            logger.error(
                                f"[{agent_id}] Failed to submit result for task_id={task_id}: {e}",
                                exc_info=True
                            )
                    else:
                        # No new tasks for this agent
                        logger.debug(f"[{agent_id}] No tasks found, will poll again...")

            except Exception as e:
                logger.error(f"Error polling or processing tasks for agent {agent_id}: {e}")

            # Sleep before next poll
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

    async def run_forever(self):
        """
        Creates a polling task for each known agent ID and runs them in parallel.
        """
        if not self.agents_dict:
            logger.warning("No agents found to run.")
            return

        # Create tasks for each agent
        self.tasks = {}  # Reset tasks dict
        for agent_id, agent_cls in self.agents_dict.items():
            task = asyncio.create_task(self.poll_for_tasks_for_agent(agent_id, agent_cls))
            self.tasks[agent_id] = task  # Store in self.tasks instead of local list
            logger.info(f"Started polling loop for agent_id={agent_id}")

        try:
            await asyncio.gather(*self.tasks.values())
        except Exception as e:
            logger.error(f"Fatal error in run_forever: {e}", exc_info=True)
            # Cancel all tasks on fatal error
            for task in self.tasks.values():
                task.cancel()
            raise


async def main():
    async with MeshManager() as manager:
        await manager.run_forever()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("MeshManager stopped by user.")
