from typing import List, Dict, Any, Optional
import logging
import requests
from core.imgen import generate_image_with_retry

logger = logging.getLogger(__name__)

class Tools:
    def __init__(self):
        self.tools_config = [
            {
                "type": "function",
                "function": {
                    "name": "generate_image",
                    "description": "Generate an image based on a text prompt...",
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
        self.tool_handlers = {
            "generate_image": self.handle_image_generation,
            "start_raid": self.handle_raid
        }

    def get_tools_config(self, filter_tools: List[str] = None) -> List[Dict[str, Any]]:
        """
        Get tool configurations, optionally filtered by tool names
        
        Args:
            filter_tools: Optional list of tool names to include
            
        Returns:
            List of tool configurations
        """
        if filter_tools:
            return [tool for tool in self.tools_config 
                    if tool["function"]["name"] in filter_tools]
        return self.tools_config

    async def execute_tool(self, tool_name: str, args: Dict[str, Any], agent_context: Any) -> Optional[Dict[str, Any]]:
        """Execute a tool by name with given arguments"""
        if tool_name not in self.tool_handlers:
            logger.error(f"Unknown tool: {tool_name}")
            return None
            
        return await self.tool_handlers[tool_name](args, agent_context)

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

