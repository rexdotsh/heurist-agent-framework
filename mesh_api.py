from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import logging
import json
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
        result = await agent.handle_message(request.input)
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

##############################################
# Helper functions for public tool conversion
##############################################
def convert_parameters(parameters: Dict) -> List[Dict[str, Any]]:
    """
    Converts a JSON schema (as defined in the tool function's parameters)
    into a list of input definitions matching the format seen in
    bitquery_sol_token_info_agent.py.
    """
    result = []
    properties = parameters.get("properties", {})
    required = parameters.get("required", [])
    for name, schema in properties.items():
        input_obj = {
            "name": name,
            "description": schema.get("description", ""),
            "type": schema.get("type", "str"),
            "required": name in required
        }
        if "default" in schema:
            input_obj["default"] = schema["default"]
        result.append(input_obj)
    return result

def get_tools() -> List[Dict[str, Any]]:
    """
    Retrieve the public tool definitions for client consumption,
    formatted to adhere to OpenAI's chat completions tool schema.
    Each tool includes:
      - A top-level "type": "function"
      - The function details nested under the "function" key.
      - The "agent_id" field set to a sanitized version of the agent's class name.
      
    Note:
      Due to current OpenAI requirements, all properties defined in the function's
      parameters must be supplied in the "required" field—even if they are intended to be optional.
    """
    tools = []
    # Map our shorthand types to valid JSON Schema types.
    type_mapping = {
        "str": "string",
        "bool": "boolean",
        "int": "integer",
        "float": "number"
    }
    for agent_id, agent_cls in agents_dict.items():
        try:
            agent_instance = agent_cls()
            if hasattr(agent_instance, "get_tool_schemas") and callable(agent_instance.get_tool_schemas):
                schemas = agent_instance.get_tool_schemas()
                # Use agent's metadata "inputs" if available.
                agent_inputs = agent_instance.metadata.get("inputs")
                for schema in schemas:
                    # Get inputs from metadata if available, otherwise convert using existing logic.
                    if agent_inputs is not None:
                        inputs_schema_list = agent_inputs
                    else:
                        inputs_schema_list = convert_parameters(schema["function"].get("parameters", {}))
    
                    # Build a JSON schema for the function's parameters.
                    properties = {}
                    for inp in inputs_schema_list:
                        param_type = type_mapping.get(inp["type"], inp["type"])
                        properties[inp["name"]] = {
                            "type": param_type,
                            "description": inp.get("description", "")
                        }
                    # Due to OpenAI requirements, include every property as required.
                    required = list(properties.keys())
    
                    # Instead of using metadata name which might have spaces, use the class name.
                    # This ensures consistency with agents_dict keys.
                    original_agent_name = agent_cls.__name__
                    # Sanitize function name if needed (though class names are often already valid)
                    function_name = "".join(
                        c if c.isalnum() or c in ["_", "-"] else "_" for c in original_agent_name
                    )
    
                    fn_schema = {
                        "name": function_name,
                        "description": schema["function"]["description"],
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                            "additionalProperties": False
                        },
                        "strict": True
                    }
    
                    public_tool = {
                        "type": "function",
                        "function": fn_schema,
                        "agent_id": function_name  # Using the sanitized class name.
                    }
                    tools.append(public_tool)
        except Exception as e:
            logger.error(f"Error retrieving tool schemas for agent {agent_id}: {e}")
    return tools

@app.get("/tools")
async def list_tools():
    """
    Endpoint to list the available tools.
    The public tool format includes the tool's name, description, and inputs (as defined in agent metadata).
    """
    return get_tools()

##############################################
# Helper function to run agents based on tool_calls
##############################################
async def run_agents(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Processes an LLM response (with tool_calls) and for each tool call:
      - Parses the function definition and its JSON arguments.
      - Finds the corresponding agent using the public tools (via "agent_id").
      - Executes the agent through process_mesh_request.
      - Returns two messages: one for the assistant (with the original tool_calls)
        and one for the tool (with the agent's result) matching the OpenAI chat format.
        
    This version supports both top-level "tool_calls" as well as nested ones within choices.
    """
    messages = []
    tool_calls = []
    logger.info(f"Response: {response}")
    
    # Check if tool_calls is directly in the response
    if response.get("tool_calls"):
        tool_calls = response["tool_calls"]
    # Otherwise, check inside choices (the client-side payload structure)
    elif "choices" in response:
        for choice in response["choices"]:
            message = choice.get("message", {})
            if "tool_calls" in message:
                if isinstance(message["tool_calls"], list):
                    tool_calls.extend(message["tool_calls"])
                else:
                    tool_calls.append(message["tool_calls"])
    
    if not tool_calls:
        return messages
    
    for call in tool_calls:
        try:
            function_call = call.get("function", {})
            function_name = function_call.get("name")
            arguments_str = function_call.get("arguments")
            if not function_name or not arguments_str:
                continue
    
            try:
                arguments = json.loads(arguments_str)
            except Exception as ex:
                logger.error(f"Error parsing arguments: {ex}")
                continue
    
            # Look up the corresponding tool using public tool definitions.
            available_tools = get_tools()
            matched_tool = None
            for tool in available_tools:
                if tool.get("function", {}).get("name") == function_name:
                    matched_tool = tool
                    break
    
            if not matched_tool:
                logger.error(f"No matching agent found for function {function_name}")
                continue
    
            agent_id = matched_tool.get("agent_id")
            mesh_request = MeshRequest(agent_id=agent_id, input=arguments)
            tool_result = await process_mesh_request(mesh_request)
            logger.info(f"Tool result: {tool_result}")
    
            # Build the messages in the expected format.
            assistant_message = {
                "role": "assistant",
                "content": None,  # Expected to be None as per the expected return format.
                "tool_calls": [call]
            }
    
            if isinstance(tool_result, dict):
                content = tool_result.get("response")
                if not content:
                    data_field = tool_result.get("data")
                    if isinstance(data_field, dict):
                        content = json.dumps(data_field, default=str)
                    elif data_field is not None:
                        content = str(data_field)
                    else:
                        content = ""
            else:
                content = str(tool_result)
    
            tool_message = {
                "role": "tool",
                "tool_call_id": call.get("id"),
                "name": function_name,
                "content": content
            }
            messages.extend([assistant_message, tool_message])
        except Exception as e:
            logger.error(f"Error processing tool call: {e}")
    return messages

##############################################
# New Endpoint to Execute run_agents
##############################################
class RunAgentsRequest(BaseModel):
    response: Dict[str, Any]

@app.post("/run_agents")
async def process_run_agents(request: RunAgentsRequest):
    """
    Endpoint for clients to process LLM responses with tool_calls.
    This endpoint accepts a payload containing a "response" (from client.chat.completions.create)
    and returns messages (assistant and tool messages) in a format compatible with the
    OpenAI chat interface.
    """
    try:
        messages = await run_agents(request.response)
        return messages
    except Exception as e:
        logger.error(f"Error processing run_agents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 