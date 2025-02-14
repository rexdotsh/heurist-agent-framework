import datetime
import requests
import os
from .mesh_agent import MeshAgent, with_cache, with_retry, monitor_execution
from core.llm import call_llm_async, call_llm_with_tools_async
from typing import List, Dict, Any
import json
from dotenv import load_dotenv

load_dotenv()

class DexScreenerTokenInfoAgent(MeshAgent):
    """
    An agent that integrates with DexScreener API to fetch real-time DEX trading data
    and token information across multiple chains.
    """

    def __init__(self):
        super().__init__()

        self.metadata.update({
            'name': 'DexScreener Token Info Agent',
            'version': '1.0.0',
            'author': 'dyt9qc',
            'created_at': '2025-02-13 07:43:15',
            'description': 'This agent fetches real-time DEX trading data and token information across multiple chains using DexScreener API',
            'inputs': [
                {
                    'name': 'query',
                    'description': 'Search query for token name, symbol or address',
                    'type': 'str',
                    'required': True
                },
                {
                    'name': 'raw_data_only',
                    'description': 'If true, return only raw data without natural language response',
                    'type': 'bool',
                    'required': False,
                    'default': False
                }
            ],
            'outputs': [
                {
                    'name': 'response',
                    'description': 'Natural language explanation of token/pair data',
                    'type': 'str'
                },
                {
                    'name': 'data',
                    'description': 'Structured token/pair data from DexScreener',
                    'type': 'dict'
                }
            ],
            'external_apis': ['DexScreener'],
            'tags': ['DeFi', 'Trading', 'Multi-chain', 'DEX']
        })

    def get_system_prompt(self) -> str:
        return (
            "You are DexScreener Assistant, providing real-time token and pair information.\n\n"

            "Data Analysis Capabilities:\n"
            "1. Token Pair Search\n"
            "2. Specific Pair Details\n"
            "3. Token Profile Information\n"
            "4. Multi-chain Token Pairs\n\n"

            "When presenting data, include:\n"

            "1. Core Token Information:\n"
            "   - Base/Quote token names and symbols\n"
            "   - Contract addresses\n"
            "   - Chain and DEX platform\n\n"

            "2. Market Metrics:\n"
            "   - Price (USD and native token)\n"
            "   - 24h Volume\n"
            "   - Liquidity\n"
            "   - Market Cap & FDV\n\n"

            "3. Trading Activity:\n"
            "   - 24h Price change\n"
            "   - Buy/Sell transactions\n"
            "   - Volume distribution\n\n"

            "4. Project Information:\n"
            "   - Website\n"
            "   - Social media links\n"
            "   - Documentation\n\n"

            "Response Format:\n"
            "- URLs: https://dexscreener.com/{chain}/{address}\n"
            "- Numbers: Standard decimal format\n"
            "- Percentages: Include % symbol (e.g., 5.25%)\n"
            "- Addresses: Short format (0x1234...abcd)\n"
            "- Lists: Bullet points for multiple items\n\n"
        )

    def get_tool_schemas(self) -> List[Dict]:
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'search_pairs',
                    'description': 'Search for trading pairs by token name, symbol, or address',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'query': {
                                'type': 'string',
                                'description': 'Search query (token name, symbol, or address)'
                            }
                        },
                        'required': ['query']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'get_specific_pair_info',
                    'description': 'Get pair info by chain and pair address',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'chain': {
                                'type': 'string',
                                'description': 'Chain identifier (e.g., solana、bsc、base、ethereum、pulsechain、ton、avalanche、sui、xrpl、sonic、polygon、hyperliquid、arbitrum、unichain、abstract、moonshot、optimism、algorand、cardano、zksync、apechain、icp、ink、multiversx、mantle、starknet、soneium、injective、dogechain、shibarium、merlinchain、ethereumpow、core、seiv2)'
                            },
                            'pair_address': {
                                'type': 'string',
                                'description': 'The pair contract address to look up'
                            }
                        },
                        'required': ['chain', 'pair_address']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'get_token_pairs',
                    'description': 'Get the trading pairs by chain and token address',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'chain': {
                                'type': 'string',
                                'description': 'Chain identifier (e.g., solana、bsc、base、ethereum、pulsechain、ton、avalanche、sui、xrpl、sonic、polygon、hyperliquid、arbitrum、unichain、abstract、moonshot、optimism、algorand、cardano、zksync、apechain、icp、ink、multiversx、mantle、starknet、soneium、injective、dogechain、shibarium、merlinchain、ethereumpow、core、seiv2)'
                            },
                            'token_address': {
                                'type': 'string',
                                'description': 'The token contract address to look up all pairs for'
                            }
                        },
                        'required': ['chain', 'token_address']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'get_token_profiles',
                    'description': 'Get the latest token profiles from DexScreener',
                }
            }
        ]

    @with_cache(ttl_seconds=300)
    async def search_pairs(self, query: str) -> Dict:
        """
        Search for trading pairs (up to 30) using DexScreener API.

        Args:
            query (str): Search query for token name, symbol, or address

        Returns:
            Dict: Top 30 matching pairs with status information
        """
        try:
            result = fetch_dex_pairs(query)

            if result['status'] == 'success':
                return {
                    'status': 'success',
                    'data': {
                        'pairs': result['pairs'],
                    }
                }

            return {
                'status': result['status'],
                'error': result.get('error', 'Unknown error occurred'),
                'data': None
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': f'Failed to search pairs: {str(e)}',
                'data': None
            }

    @with_cache(ttl_seconds=300)
    async def get_specific_pair_info(self, chain: str, pair_address: str) -> Dict:
        """
        Get detailed information for a specific trading pair.

        Args:
            chain (str): Chain identifier (e.g., solana, bsc, ethereum, etc.)
            pair_address (str): The pair contract address to look up

        Returns:
            Dict: Detailed pair information with status
        """
        try:
            # Get raw pair data using the fetch_pair_info helper
            result = fetch_pair_info(chain, pair_address)

            if result['status'] == 'success':
                if result.get('pair'):
                    return {
                        'status': 'success',
                        'data': {
                            'pair': result['pair'],
                        }
                    }
                return {
                    'status': 'no_data',
                    'error': 'No matching pair found',
                    'data': None
                }

            return {
                'status': 'error',
                'error': result.get('error', 'Unknown error occurred'),
                'data': None
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': f"Failed to get pair info: {str(e)}",
                'data': None
            }

    @with_cache(ttl_seconds=300)
    async def get_token_pairs(self, chain: str, token_address: str) -> Dict:
        """
        Get trading pairs (up to 30) for a specific token on a chain.

        Args:
            chain (str): Chain identifier (e.g., solana, bsc, ethereum)
            token_address (str): Token contract address

        Returns:
            Dict: Top 30 trading pairs for the token with status
        """
        try:
            result = fetch_token_pairs(chain, token_address)

            if result['status'] == 'success':
                return {
                    'status': 'success',
                    'data': {
                        'pairs': result['pairs'],
                        'dex_url': f"https://dexscreener.com/{chain}/{token_address}"
                    }
                }

            return {
                'status': result['status'],
                'error': result.get('error', 'Unknown error occurred'),
                'data': None
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': f'Failed to get token pairs: {str(e)}',
                'data': None
            }

    @with_cache(ttl_seconds=300)
    async def get_token_profiles(self) -> Dict:
        """
        Get the latest token profiles from DexScreener.

        Returns:
            Dict: Latest token profiles with status
        """
        try:
            result = fetch_token_profiles()

            if result['status'] == 'success':
                return {
                    'status': 'success',
                    'data': {
                        'profiles': result['profiles']
                    }
                }

            return {
                'status': result['status'],
                'error': result.get('error', 'Unknown error occurred'),
                'data': None
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': f'Failed to get token profiles: {str(e)}',
                'data': None
            }

    @monitor_execution()
    @with_retry(max_retries=3)
    async def handle_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main handler for processing incoming messages/queries.
        """
        query = params.get('query')
        if not query:
            raise ValueError("Query parameter is required")

        raw_data_only = params.get('raw_data_only', False)

        response = await call_llm_with_tools_async(
            base_url=self.heurist_base_url,
            api_key=self.heurist_api_key,
            model_id=self.metadata['large_model_id'],
            system_prompt=self.get_system_prompt(),
            user_prompt=query,
            temperature=0.1,
            tools=self.get_tool_schemas()
        )

        if not response:
            return {"error": "Failed to process query"}

        if not response.get('tool_calls'):
            return {"response": response['content'], "data": {}}

        tool_call = response['tool_calls']
        function_args = json.loads(tool_call.function.arguments)

        # Handle different tool calls
        result_data = None
        if tool_call.function.name == 'search_pairs':
            result_data = await self.search_pairs(query=function_args['query'])

        elif tool_call.function.name == 'get_specific_pair_info':
            result_data = await self.get_specific_pair_info(
                chain=function_args['chain'],
                pair_address=function_args['pair_address']
            )

        elif tool_call.function.name == 'get_token_pairs':
            result_data = await self.get_token_pairs(
                chain=function_args['chain'],
                token_address=function_args['token_address']
            )

        elif tool_call.function.name == 'get_token_profiles':
            result_data = await self.get_token_profiles()

        else :
            return {"error": "Unsupported operation"}

        if result_data and 'error' in result_data:
            return {"error": result_data['error']}

        if raw_data_only:
            return {"response": "", "data": result_data}

        explanation = await call_llm_async(
            base_url=self.heurist_base_url,
            api_key=self.heurist_api_key,
            model_id=self.metadata['large_model_id'],
            messages=[
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": query},
                {"role": "tool", "content": str(result_data), "tool_call_id": tool_call.id}
            ],
            temperature=0.3
        )

        return {
            "response": explanation,
            "data": result_data
        }

# ------------------------- Helper Functions ------------------------- #
def fetch_dex_pairs(query: str) -> Dict:
    """
    Fetches trading pair information from DexScreener API.

    Args:
        query (str): Search query (token name, symbol, or address)

    Returns:
        Dict: Processed pair information with status
    """
    try:
        response = requests.get(
            f"https://api.dexscreener.com/latest/dex/search/?q={query}",
            headers={},
        )

        if response.status_code != 200:
            return {
                'status': 'error',
                'error': f"Query failed with status code {response.status_code}: {response.text}"
            }

        data = response.json()

        if not data.get('pairs'):
            return {
                'status': 'no_data',
                'error': 'No pairs found for this query'
            }

        # Limit the number of pairs
        limit = 30
        pairs = data['pairs'][:limit] if len(data['pairs']) > limit else data['pairs']

        return {
            'status': 'success',
            'pairs': pairs
        }

    except Exception as e:
        return {
            'status': 'error',
            'error': f'Failed to fetch pairs: {str(e)}'
        }

def fetch_pair_info(chain: str, pair_address: str) -> Dict:
    """
    Fetches detailed information for a specific trading pair from DexScreener API.

    Args:
        chain (str): Chain identifier (e.g., 'solana、bsc、base、ethereum...')
        pair_address (str): Pair contract address

    Returns:
        Dict: Pair information with status and data
    """
    try:
        response = requests.get(
            f"https://api.dexscreener.com/latest/dex/pairs/{chain}/{pair_address}",
            headers={},
        )

        if response.status_code != 200:
            return {
                'status': 'error',
                'error': f"Query failed with status code {response.status_code}: {response.text}"
            }

        data = response.json()
        pairs = data.get('pairs', [])

        # Get first matching pair
        matching_pair = next(
            (pair for pair in pairs if pair.get('pairAddress', '').lower() == pair_address.lower()),
            None
        )

        if matching_pair:
            return {
                'status': 'success',
                'pair': matching_pair
            }

        return {
            'status': 'success',
            'message': 'No matching pair found'
        }

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

def fetch_token_pairs(chain: str, token_address: str) -> Dict:
    """
    Fetches all trading pairs for a specific token on a given chain from DexScreener API.

    Args:
        chain (str): Chain identifier (e.g., solana, bsc, ethereum)
        token_address (str): Token contract address

    Returns:
        Dict: Trading pairs information for the token
    """
    try:
        response = requests.get(
            f"https://api.dexscreener.com/tokens/v1/{chain}/{token_address}",
            headers={},
        )

        if response.status_code != 200:
            return {
                'status': 'error',
                'error': f"Query failed with status code {response.status_code}: {response.text}"
            }

        pairs = response.json()

        if not pairs:
            return {
                'status': 'no_data',
                'error': 'No pairs found for this token'
            }

        # Limit the number of pairs
        limit = 30
        limited_pairs = pairs[:limit] if len(pairs) > limit else pairs

        return {
            'status': 'success',
            'pairs': limited_pairs
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Failed to fetch token pairs: {str(e)}'
        }

def fetch_token_profiles() -> Dict:
    """
    Fetches the latest token profiles from DexScreener API.

    Returns:
        Dict: Token profiles information with status
    """
    try:
        response = requests.get(
            "https://api.dexscreener.com/token-profiles/latest/v1",
            headers={},
        )

        if response.status_code != 200:
            return {
                'status': 'error',
                'error': f"Query failed with status code {response.status_code}: {response.text}"
            }

        data = response.json()

        limit = 30
        # Limit the number of profiles
        profiles = data[:limit] if len(data) > limit else data

        return {
            'status': 'success',
            'profiles': profiles
        }

    except Exception as e:
        return {
            'status': 'error',
            'error': f'Failed to fetch token profiles: {str(e)}'
        }