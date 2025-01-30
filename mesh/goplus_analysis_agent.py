from typing import Dict, Any, Optional
from .mesh_agent import MeshAgent, monitor_execution, with_retry, with_cache
from core.llm import call_llm_with_tools_async, call_llm_async
import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

class TokenContractSecurityAgent(MeshAgent):
    def __init__(self):
        super().__init__()
        self.metadata.update({
            'name': 'Token Contract Security Agent',
            'version': '1.0.0',
            'author': 'Heurist Team',
            'description': 'Fetch and analyze security details of blockchain token contracts using GoPlus API. Analyze the token holder count, liquidity distribution, admin rights, mintable / pausable status, and other security metrics. This agents helps you understand whether the token is safe to interact with and catch any potential scams or honeypots.',
            'inputs': [
                {
                    'name': 'query',
                    'description': 'The query containing token contract address and chain ID or chain name',
                    'type': 'str'
                }
            ],
            'outputs': [
                {
                    'name': 'response',
                    'description': 'Security analysis and explanation',
                    'type': 'str'
                },
                {
                    'name': 'data',
                    'description': 'The security details of the token contract. Raw data from GoPlus API',
                    'type': 'dict'
                }
            ],
            'external_apis': ['goplus'],
            'tags': ['Security']
        })
        
        self.supported_blockchains = {
            "1": "Ethereum", "10": "Optimism", "25": "Cronos", "56": "BSC", 
            "100": "Gnosis", "128": "HECO", "137": "Polygon", "250": "Fantom", 
            "321": "KCC", "324": "zkSync Era", "10001": "ETHW", "201022": "FON", 
            "42161": "Arbitrum", "43114": "Avalanche", "59144": "Linea Mainnet", 
            "8453": "Base", "tron": "Tron", "534352": "Scroll", "204": "opBNB", 
            "5000": "Mantle", "42766": "ZKFair", "81457": "Blast", 
            "169": "Manta Pacific", "80085": "Berachain Artio Testnet", 
            "4200": "Merlin", "200901": "Bitlayer Mainnet", 
            "810180": "zkLink Nova", "196": "X Layer Mainnet"
        }

    @monitor_execution()
    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def fetch_security_details(self, contract_address: str, chain_id: int = 8453) -> Optional[Dict]:
        """
        Fetch security details from GoPlus API with retry and caching
        """
        base_url = f"https://api.gopluslabs.io/api/v1/token_security/{chain_id}"
        params = {
            "contract_addresses": contract_address
        }
        headers = {
            "accept": "*/*"
        }
        
        try:
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching security details: {e}")
            return None

    def get_system_prompt(self) -> str:
        return f"""You are a helpful assistant that can fetch and analyze security details of blockchain token contracts.
        You should identify the contract address and chain ID from the user's query and use the provided tool to fetch security information.
        Supported chains: {', '.join([f"{name} (Chain ID: {id})" for id, name in self.supported_blockchains.items()])}
        If the blockchain is not supported, return a brief error message explaining why.
        If you obtain data from the tool, provide a clear and concise analysis of the security details. Focus on important metrics and risks."""

    def get_tool_schema(self) -> Dict:
        return {
            'type': 'function',
            'function': {
                'name': 'fetch_security_details',
                'description': 'Fetch security details of a blockchain contract',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'contract_address': {
                            'type': 'string',
                            'description': 'The blockchain contract address'
                        },
                        'chain_id': {
                            'type': 'integer',
                            'description': 'The blockchain chain ID',
                            'default': 1
                        }
                    },
                    'required': ['contract_address'],
                },
            }
        }

    @monitor_execution()
    @with_retry(max_retries=3)
    async def handle_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get('query')
        if not query:
            raise ValueError("Query parameter is required")

        response = await call_llm_with_tools_async(
            base_url=self.heurist_base_url,
            api_key=self.heurist_api_key,
            model_id=self.large_model_id,
            system_prompt=self.get_system_prompt(),
            user_prompt=query,
            temperature=0.1,
            tools=[self.get_tool_schema()]
        )

        if not response or not response.get('tool_calls'):
            return {"response": response.get('content')}

        tool_call = response['tool_calls']
        function_args = json.loads(tool_call.function.arguments)
        tool_result = await self.fetch_security_details(
            function_args['contract_address'],
            function_args.get('chain_id', 1)
        )

        explanation = await call_llm_async(
            base_url=self.heurist_base_url,
            api_key=self.heurist_api_key,
            model_id=self.large_model_id,
            messages=[
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": query},
                {"role": "tool", "content": str(tool_result), "tool_call_id": tool_call.id}
            ],
            temperature=0.7
        )

        token_data = tool_result.get('result', {}).get(function_args['contract_address'].lower(), {})
        
        essential_security_info = {
            "token_info": {
                "name": token_data.get('token_name'),
                "symbol": token_data.get('token_symbol'),
                "total_supply": token_data.get('total_supply'),
                "holder_count": token_data.get('holder_count')
            },
            "security_metrics": {
                "is_honeypot": bool(int(token_data.get('is_honeypot', '0'))),
                "is_blacklisted": bool(int(token_data.get('is_blacklisted', '0'))),
                "is_open_source": bool(int(token_data.get('is_open_source', '0'))),
                "buy_tax": token_data.get('buy_tax', '0'),
                "sell_tax": token_data.get('sell_tax', '0'),
                "can_take_back_ownership": bool(int(token_data.get('can_take_back_ownership', '0'))),
                "is_proxy": bool(int(token_data.get('is_proxy', '0'))),
                "is_mintable": bool(int(token_data.get('is_mintable', '0')))
            },
            "liquidity_info": {
                "is_in_dex": bool(int(token_data.get('is_in_dex', '0'))),
                "dex": token_data.get('dex', []),
                "lp_holder_count": token_data.get('lp_holder_count')
            },
            "ownership": {
                "creator_address": token_data.get('creator_address'),
                "owner_address": token_data.get('owner_address'),
                "top_holders": token_data.get('holders', [])[:3]  # Only include top 3 holders
            }
        }
        
        return {
            "response": explanation,
            "data": essential_security_info
        }