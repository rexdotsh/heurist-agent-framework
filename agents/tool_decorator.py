import inspect
from typing import Callable, Dict, Any


def tool(description: str) -> Callable[[Callable], Callable]:
    """
    A decorator factory that creates a tool decorator with a specified description.
    """
    def decorator(func: Callable) -> Callable:
        # Add metadata to the function
        func.name = func.__name__
        func.description = func.__doc__ if func.__doc__ != inspect._empty else description
        
        # Generate the parameter schema from the function signature
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
        
        return func
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
