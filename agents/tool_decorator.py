import inspect
from typing import Callable, Dict, Any


def tool(description: str):
    """
    A decorator factory that creates a tool decorator with a specified description.
    """
    def decorator(func):
        # Add metadata to the function
        func.name = func.__name__
        func.description = func.__doc__ if func.__doc__ != inspect._empty else description

        # Generate the parameter schema from the original function
        signature = inspect.signature(func)
        func.args_schema = {
            "type": "object",
            "properties": {
                param: {
                    "type": str(param_type.annotation.__name__).lower(),
                    "description": str(param_type.annotation) if param_type.annotation != inspect._empty else "No type specified"
                }
                for param, param_type in signature.parameters.items()
            },
            "required": [
                param
                for param, param_type in signature.parameters.items()
                if param_type.default == inspect._empty
            ]
        }
        
        async def wrapper(args: Dict[str, Any], agent_context: Any):
            # Remove agent_context from args if it exists
            if "agent_context" in args:
                args["agent_context"] = agent_context   
            result = await func(**args) if inspect.iscoroutinefunction(func) else func(**args)
            return result
            
        wrapper.name = func.name
        wrapper.description = func.description
        wrapper.args_schema = func.args_schema
        wrapper.original = func
        
        return wrapper
    return decorator



def convert_to_function_schema(func: Callable) -> Dict[str, Any]:
    """
    Converts a decorated function into an OpenAI function schema format.
    """
    return {
        "type": "function",
        "function": {
            "name": func.name,
            "description": func.description,
            "parameters": func.args_schema
        }
    }

def get_tool_schemas(tools: list[Callable]) -> list[Dict[str, Any]]:
    """
    Convert a list of tool-decorated functions into OpenAI function schemas.
    
    Args:
        tools: List of functions decorated with @tool
        
    Returns:
        List of function schemas in OpenAI format
    """
    return [convert_to_function_schema(tool) for tool in tools]
