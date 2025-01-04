from typing import List, Dict, Any, Optional, Callable
import logging
from .tool_decorator import get_tool_schemas

logger = logging.getLogger(__name__)
## YOUR TOOLS GO HERE

class ToolBox:
    """Base class containing tool configurations and handlers"""
    
    def __init__(self):
        # Base tools configuration
        self.tools_config = [
            {
                "type": "function",
                "function": {
                    "name": "generate_image",
                    "description": "Generate an image based on a text prompt, any request to create an image should be handled by this tool, only use this tool if the user asks to create an image",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string", "description": "The prompt to generate the image from"}
                        },
                        "required": ["prompt"]
                    }
                }
            },
        ]
        
        # Base handlers
        self.tool_handlers = {
            "generate_image": self.handle_image_generation
        }

        self.decorated_tools = []

    async def handle_image_generation(self, args: Dict[str, Any], agent_context: Any) -> Dict[str, Any]:
        """Handle image generation tool"""
        logger.info(args['prompt'])
        try:
            image_prompt = args['prompt']#await agent_context.generate_image_prompt(args['prompt'])
            image_url = await agent_context.handle_image_generation(image_prompt)
            return {"image_url": image_url}
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            return {"error": str(e)}
