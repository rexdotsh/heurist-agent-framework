from typing import List, Dict, Any, Optional, Callable
import logging
from .tool_decorator import get_tool_schemas, tool
import aiohttp

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

        self.decorated_tools = [self.get_crypto_price]

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

    @staticmethod
    async def normalize_ticker(ticker: str) -> str:
        """Normalize various token names to Binance ticker format"""
        ticker_map = {
            "BTC": "BTC",
            "BITCOIN": "BTC",
            "ETH": "ETH",
            "ETHEREUM": "ETH",
            "MATIC": "MATIC",
            "POLYGON": "MATIC",
            "SOL": "SOL",
            "SOLANA": "SOL",
            # Add more mappings as needed
        }
        
        normalized = ticker_map.get(ticker.upper())
        if normalized:
            return f"{normalized}USDT"
        return f"{ticker.upper()}USDT"

    @staticmethod
    @tool("Get the current price of a cryptocurrency in USD")
    async def get_crypto_price(ticker: str) -> float:
        """
        Get the current price of a cryptocurrency in USD from Binance.
        
        Args:
            ticker: The cryptocurrency ticker symbol (e.g., BTC, ETH, SOL)
            
        Returns:
            float: Current price in USD
        """
        try:
            #normalized_ticker = await ToolBox.normalize_ticker(ticker)
            normalized_ticker = f"{ticker.upper()}USDT"
            async with aiohttp.ClientSession() as session:
                url = f"https://api.binance.com/api/v3/ticker/price?symbol={normalized_ticker}"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        price = float(data['price'])
                        logger.info(f"The current price for {normalized_ticker}: ${price:.2f}")
                        return {"message": f"The current price for {normalized_ticker}: ${price:.2f}"}
                    else:
                        error_msg = f"Failed to get price for {normalized_ticker}"
                        logger.error(error_msg)
                        return {"message": error_msg}
                        
        except Exception as e:
            error_msg = f"Error getting crypto price: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}