import requests
import os
from .mesh_agent import MeshAgent, with_cache, with_retry, monitor_execution
from core.llm import call_llm_async, call_llm_with_tools_async
from typing import List, Dict, Any
import json

class CoinGeckoTokenInfoAgent(MeshAgent):
    def __init__(self):
        super().__init__()
        self.api_url = "https://api.coingecko.com/api/v3"
        self.headers = {
            'Authorization': f'Bearer {os.getenv("COINGECKO_API_KEY")}'
        }
        
        # Add required metadata
        self.metadata.update({
            'name': 'Token Info Agent',
            'version': '1.0.0',
            'author': 'Your Name',
            'description': 'Fetches token information and trending coins from CoinGecko',
            'inputs': [
                {
                    'name': 'query',
                    'description': 'Natural language query about a token',
                    'type': 'str',
                    'optional': False
                }
            ],
            'outputs': [
                {
                    'name': 'response',
                    'description': 'Natural language explanation of the token information',
                    'type': 'str'
                },
                {
                    'name': 'data',
                    'description': 'Structured token information or trending coins data',
                    'type': 'dict'
                }
            ],
            'external_apis': ['coingecko'],
            'tags': ['DeFi', 'Market Data']
        })

    def get_system_prompt(self) -> str:
        return """You are a helpful assistant that can fetch token information from CoinGecko.
        You can:
        1. Search for specific tokens and get their details
        2. Get current trending coins in the market
        
        For specific token queries, identify whether the user provided a CoinGecko ID directly or needs to search by token name.
        For trending coins requests, use the get_trending_coins tool to fetch the current top trending cryptocurrencies."""

    def get_tool_schemas(self) -> List[Dict]:
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'get_coingecko_id',
                    'description': 'Search for a token by name to get its CoinGecko ID',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'token_name': {
                                'type': 'string',
                                'description': 'The token name to search for'
                            }
                        },
                        'required': ['token_name'],
                    },
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'get_token_info',
                    'description': 'Get detailed token information using CoinGecko ID',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'coingecko_id': {
                                'type': 'string',
                                'description': 'The CoinGecko ID of the token'
                            }
                        },
                        'required': ['coingecko_id'],
                    },
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'get_trending_coins',
                    'description': 'Get the current top trending cryptocurrencies',
                    'parameters': {
                        'type': 'object',
                        'properties': {},
                        'required': [],
                    },
                }
            }
        ]

    @with_cache(ttl_seconds=300)  # Cache for 5 minutes
    async def get_trending_coins(self):
        try:
            response = requests.get(
                f"{self.api_url}/search/trending",
                headers=self.headers
            )
            response.raise_for_status()
            trending_data = response.json()
            
            # Format the trending coins data
            formatted_trending = []
            for coin in trending_data.get('coins', [])[:10]:
                coin_info = coin['item']
                formatted_trending.append({
                    'name': coin_info['name'],
                    'symbol': coin_info['symbol'],
                    'market_cap_rank': coin_info.get('market_cap_rank', 'N/A'),
                    'price_usd': coin_info["data"].get('price', 'N/A')
                })
            return {'trending_coins': formatted_trending}
            
        except requests.RequestException as e:
            print(f"error: {e}")
            return {"error": f"Failed to fetch trending coins: {str(e)}"}

    @monitor_execution()
    @with_retry(max_retries=3)
    async def handle_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get('query')
        if not query:
            raise ValueError("Query parameter is required")

        response = await call_llm_with_tools_async(
            base_url=self.heurist_base_url,
            api_key=self.heurist_api_key,
            model_id=self.metadata['large_model_id'],
            system_prompt=self.get_system_prompt(),
            user_prompt=query,
            temperature=0.1,
            tools=self.get_tool_schemas()
        )

        if not response or not response.get('tool_calls'):
            return {"error": "Failed to process query"}

        tool_call = response['tool_calls']
        function_args = json.loads(tool_call.function.arguments)
        
        if tool_call.function.name == 'get_trending_coins':
            trending_results = await self.get_trending_coins()
            if 'error' in trending_results:
                return trending_results
                
            explanation = await call_llm_async(
                base_url=self.heurist_base_url,
                api_key=self.heurist_api_key,
                model_id=self.metadata['large_model_id'],
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": query},
                    {"role": "tool", "content": str(trending_results), "tool_call_id": tool_call.id}
                ],
                temperature=0.3
            )
            
            return {
                "response": explanation,
                "data": trending_results
            }
            
        elif tool_call.function.name == 'get_coingecko_id':
            token_name = function_args['token_name']
            coingecko_id = await self.get_coingecko_id(token_name)
            if isinstance(coingecko_id, dict) and 'error' in coingecko_id:
                return coingecko_id
            
            # Get token info using the found coingecko_id
            token_info = await self.get_token_info(coingecko_id)
            if 'error' in token_info:
                return token_info
                
            formatted_data = self.format_token_info(token_info)
            
            explanation = await call_llm_async(
                base_url=self.heurist_base_url,
                api_key=self.heurist_api_key,
                model_id=self.metadata['large_model_id'],
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": query},
                    {"role": "tool", "content": str(formatted_data), "tool_call_id": tool_call.id}
                ],
                temperature=0.7
            )
            
            return {
                "response": explanation,
                "data": formatted_data
            }
            
        elif tool_call.function.name == 'get_token_info':
            coingecko_id = function_args['coingecko_id']
            token_info = await self.get_token_info(coingecko_id)
            
            if 'error' in token_info:
                return token_info
                
            formatted_data = self.format_token_info(token_info)
            
            explanation = await call_llm_async(
                base_url=self.heurist_base_url,
                api_key=self.heurist_api_key,
                model_id=self.metadata['large_model_id'],
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": query},
                    {"role": "tool", "content": str(formatted_data), "tool_call_id": tool_call.id}
                ],
                temperature=0.7
            )
            
            return {
                "response": explanation,
                "data": formatted_data
            }

        return {"error": "Unsupported operation"}

    def format_token_info(self, data: Dict) -> Dict:
        """Format token information in a structured way"""
        market_data = data.get('market_data', {})
        return {
            "token_info": {
                "name": data.get('name', 'N/A'),
                "symbol": data.get('symbol', 'N/A').upper(),
                "market_cap_rank": data.get('market_cap_rank', 'N/A'),
                "categories": data.get('categories', []),
                "description": data.get('description', {}).get('en', 'N/A')
            },
            "market_metrics": {
                "current_price_usd": market_data.get('current_price', {}).get('usd', 'N/A'),
                "market_cap_usd": market_data.get('market_cap', {}).get('usd', 'N/A'),
                "fully_diluted_valuation_usd": market_data.get('fully_diluted_valuation', {}).get('usd', 'N/A'),
                "total_volume_usd": market_data.get('total_volume', {}).get('usd', 'N/A'),
            },
            "price_metrics": {
                "ath_usd": market_data.get('ath', {}).get('usd', 'N/A'),
                "ath_change_percentage": market_data.get('ath_change_percentage', {}).get('usd', 'N/A'),
                "ath_date": market_data.get('ath_date', {}).get('usd', 'N/A'),
                "high_24h_usd": market_data.get('high_24h', {}).get('usd', 'N/A'),
                "low_24h_usd": market_data.get('low_24h', {}).get('usd', 'N/A'),
                "price_change_24h": market_data.get('price_change_24h', 'N/A'),
                "price_change_percentage_24h": market_data.get('price_change_percentage_24h', 'N/A'),
            },
            "supply_info": {
                "total_supply": market_data.get('total_supply', 'N/A'),
                "max_supply": market_data.get('max_supply', 'N/A'),
                "circulating_supply": market_data.get('circulating_supply', 'N/A')
            }
        }

    @with_cache(ttl_seconds=3600)
    async def get_coingecko_id(self, token_name):
        try:
            # Use CoinGecko's search API instead of coins/list
            response = requests.get(
                f"{self.api_url}/search?query={token_name}",
                headers=self.headers
            )
            response.raise_for_status()
            search_results = response.json()
            print(f"search_results: {search_results}")
            # Get the first coin result if available
            if search_results.get('coins') and len(search_results['coins']) > 0:
                return search_results['coins'][0]['id']
            return None
            
        except requests.RequestException as e:
            print(f"error: {e}")
            return {"error": f"Failed to search for token: {str(e)}"}

    @with_cache(ttl_seconds=3600)
    async def get_token_info(self, coingecko_id):
        try:
            response = requests.get(f"{self.api_url}/coins/{coingecko_id}", headers=self.headers)
            response.raise_for_status()
            print(f"response: {response}")
            return response.json()
        except requests.RequestException as e:
            print(f"error: {e}")
            return {"error": f"Failed to fetch token info: {str(e)}"}