from typing import Dict, Any
from .mesh_agent import MeshAgent, monitor_execution, with_retry, with_cache
from core.llm import call_llm_with_tools_async, call_llm_async
import os
import requests
from dotenv import load_dotenv
import json
import aiohttp

load_dotenv()

class AlloraPricePredictionAgent(MeshAgent):
    def __init__(self):
        super().__init__()
        self.session = None  # Initialize as None
        self.metadata.update({
            'name': 'Allora Price Prediction Agent',
            'version': '1.0.0',
            'author': 'Heurist Team',
            'author_address': '0x7d9d1821d15B9e0b8Ab98A058361233E255E405D',
            'description': 'Get price predictions for ETH/BTC with confidence intervals from Allora price prediction API',
            'inputs': [
                {
                    'name': 'query',
                    'description': 'The cryptocurrency symbol (only ETH or BTC supported) and the time period (5m or 8h)',
                    'type': 'str'
                }
            ],
            'outputs': [
                {
                    'name': 'response',
                    'description': 'The price prediction with confidence intervals',
                    'type': 'str'
                }
            ],
            'external_apis': ['allora'],
            'tags': ['Trading', 'Allora'],
            'mcp_tool_name': 'get_allora_price_prediction'
        })

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()  # Create session when entering context
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()  # Close session when exiting context
            self.session = None

    @monitor_execution()
    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def get_allora_prediction(self, token: str, timeframe: str) -> Dict:
        should_close = False
        if not self.session:
            self.session = aiohttp.ClientSession()
            should_close = True
            
        try:
            base_url = "https://api.upshot.xyz/v2/allora/consumer/price/ethereum-11155111"
            url = f"{base_url}/{token.upper()}/{timeframe}"

            headers = {
                "accept": "application/json",
                "x-api-key": os.getenv("ALLORA_API_KEY"),
            }

            async with self.session.get(url, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
                # print('data', data)

                prediction = float(data["data"]["inference_data"]["network_inference_normalized"])
                confidence_intervals = data["data"]["inference_data"]["confidence_interval_percentiles_normalized"]
                confidence_interval_values_normalized = data["data"]["inference_data"]["confidence_interval_values_normalized"]

                return {
                    "prediction": prediction,
                    "confidence_intervals": confidence_intervals,
                    "confidence_interval_values_normalized": confidence_interval_values_normalized,
                }
        finally:
            if should_close and self.session:
                await self.session.close()
                self.session = None

    def get_system_prompt(self) -> str:
        return """You are a helpful assistant that can access external tools to provide Bitcoin and Ethereum price prediction data.
        The price prediction is provided by Allora. You only have access to BTC and ETH data with 5-minute and 8-hour time frames.
        You don't have the ability to tell anything else. If the user's query is out of your scope, return a brief error message.
        If the user's query doesn't mention the time frame, use 5-minute by default.
        If the tool call successfully returns the data, limit your response to 50 words like a professional financial analyst,
        and output in CLEAN text format with no markdown or other formatting. Only return your response, no other text."""

    def get_tool_schema(self) -> Dict:
        return {
            'type': 'function',
            'function': {
                'name': 'get_allora_prediction',
                'description': 'Get price prediction for ETH or BTC with confidence intervals',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'token': {
                            'type': 'string',
                            'description': 'The cryptocurrency symbol (ETH or BTC)',
                            'enum': ['ETH', 'BTC']
                        },
                        'timeframe': {
                            'type': 'string',
                            'description': 'Time period for prediction',
                            'enum': ['5m', '8h']
                        }
                    },
                    'required': ['token', 'timeframe'],
                },
            }
        }

    @monitor_execution()
    @with_retry(max_retries=3)
    async def handle_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get('query')
        if not query:
            raise ValueError("Query parameter is required")

        # Get LLM response with tool call
        response = await call_llm_with_tools_async(
            base_url=self.heurist_base_url,
            api_key=self.heurist_api_key,
            model_id=self.large_model_id,
            system_prompt=self.get_system_prompt(),
            user_prompt=query,
            temperature=0.1,
            tools=[self.get_tool_schema()]
        )

        print(response)

        if not response or not response.get('tool_calls'):
            return {"response": response.get('content')}

        tool_call = response['tool_calls']
        function_args = json.loads(tool_call.function.arguments)
        token = function_args['token']
        timeframe = function_args['timeframe']
        
        # Get prediction data
        result = await self.get_allora_prediction(token, timeframe)
        # print('result', result)
        
        tool_response = (
            f"Price prediction for {function_args['token']} ({function_args['timeframe']} timeframe):\n"
            f"Predicted price: {result['prediction']}\n"
            f"Confidence Intervals: {result['confidence_intervals']}\n"
            f"Confidence Interval Values Normalized: {result['confidence_interval_values_normalized']}\n"
        )

        # print('tool_response', tool_response)
        final_response = await call_llm_async(
            base_url=self.heurist_base_url,
            api_key=self.heurist_api_key,
            model_id=self.large_model_id,
            messages=[
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": query},
                {"role": "tool", "content": tool_response, "tool_call_id": tool_call.id}
            ],
            temperature=0.1
        )

        return {
            "response": final_response
        }