from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging
from mesh_manager import AgentLoader, Config

# Example call
# curl -X POST http://localhost:8000/mesh_request \
#   -H "Content-Type: application/json" \
#   -d '{
#     "agent_id": "EchoAgent",
#     "input": {
#       "message": "Hello, Echo!"
#     }
#   }'

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("MeshAPI")

app = FastAPI()

config = Config()
agents_dict = AgentLoader(config).load_agents()

class MeshRequest(BaseModel):
    agent_id: str
    input: Dict[str, Any]
    heurist_api_key: str | None = None

@app.post("/mesh_request")
async def process_mesh_request(request: MeshRequest):
    if request.agent_id not in agents_dict:
        raise HTTPException(status_code=404, detail=f"Agent {request.agent_id} not found")
    
    agent_cls = agents_dict[request.agent_id]
    agent = agent_cls()
    
    if request.heurist_api_key:
        agent.set_heurist_api_key(request.heurist_api_key)
    
    try:
        result = await agent.call_agent(request.input)
        await agent.cleanup()
        return result
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def list_agents():
    """Return list of available agents and their metadata"""
    agents_info = {}
    for agent_id, agent_cls in agents_dict.items():
        agent = agent_cls()
        agents_info[agent_id] = {
            "metadata": agent.metadata,
            "module": agent_cls.__module__.split('.')[-1]
        }
    return agents_info 