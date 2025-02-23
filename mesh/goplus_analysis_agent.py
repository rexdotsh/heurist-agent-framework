from typing import Dict, Any, Optional, List
from .mesh_agent import MeshAgent, monitor_execution, with_retry, with_cache
from core.llm import call_llm_with_tools_async, call_llm_async
import os
import requests
from dotenv import load_dotenv
import json
import logging

logger = logging.getLogger(__name__)
load_dotenv()

class TokenContractSecurityAgent(MeshAgent):
    def __init__(self):
        super().__init__()
        self.metadata.update({
            'name': 'Token Contract Security Agent',
            'version': '1.0.0',
            'author': 'Heurist Team',
            'description': 'This agent can fetch and analyze security details of blockchain token contracts using GoPlus API.',
            'inputs': [
                {
                    'name': 'query',
                    'description': 'The query containing token contract address and chain ID or chain name',
                    'type': 'str',
                    'required': False
                },
                {
                    'name': 'tool',
                    'description': 'Direct tool name to call',
                    'type': 'str',
                    'required': False
                },
                {
                    'name': 'tool_arguments',
                    'description': 'Arguments for direct tool call',
                    'type': 'dict',
                    'required': False
                },
                {
                    'name': 'raw_data_only',
                    'description': 'If true, returns only raw data without analysis',
                    'type': 'bool',
                    'required': False,
                    'default': False
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
                    'description': 'The security details of the token contract',
                    'type': 'dict'
                }
            ],
            'external_apis': ['GoPlus'],
            'tags': ['Security', 'Data']
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

    def get_system_prompt(self) -> str:
        return f"""You are a blockchain security analyst that provides factual analysis of token contracts based on GoPlus API data.
        1. Extract the contract address and chain ID from the user's query
        2. Use the fetch_security_details tool to get the security data
        3. Present the findings in this structured format:
            - Basic Info: Token name, symbol, total supply, holder count
            - Contract Properties: Open source status, proxy status, mintable status
            - Ownership Analysis: Creator address, owner address, ownership takeback capability
            - Trading Properties: Buy/sell taxes, honeypot status, blacklist status
            - Liquidity: DEX presence, LP holder count, top LP holders
            - Holder Distribution: Top holders and their percentage of the total supply
            - Other Metrics: Any other relevant metrics or information
        4. Risk Assessment: Provide a risk assessment based on the data
        
        Supported chains: {', '.join([f"{name} (Chain ID: {id})" for id, name in self.supported_blockchains.items()])}"""

    def get_tool_schemas(self) -> List[Dict]:
        return [{
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
        }]

    async def _respond_with_llm(
        self, query: str, tool_call_id: str, data: dict, temperature: float
    ) -> str:
        return await call_llm_async(
            base_url=self.heurist_base_url,
            api_key=self.heurist_api_key,
            model_id=self.metadata['large_model_id'],
            messages=[
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": query},
                {"role": "tool", "content": str(data), "tool_call_id": tool_call_id}
            ],
            temperature=temperature
        )

    def _handle_error(self, maybe_error: dict) -> dict:
        if 'error' in maybe_error:
            return {"error": maybe_error['error']}
        return {}

    @monitor_execution()
    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def fetch_security_details(self, contract_address: str, chain_id: int = 1) -> Dict:
        """Fetch security details from GoPlus API"""
        try:
            base_url = f"https://api.gopluslabs.io/api/v1/token_security/{chain_id}"
            params = {"contract_addresses": contract_address}
            headers = {"accept": "*/*"}
            
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Process the response data
            token_data = data.get('result', {}).get(contract_address.lower(), {})
            
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
                    "top_holders": token_data.get('holders', [])[:3]
                }
            }
            return essential_security_info
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching security details: {e}")
            return {"error": f"Failed to fetch security details: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {"error": f"Unexpected error: {str(e)}"}

    async def _handle_tool_logic(
        self, tool_name: str, function_args: dict, query: str, tool_call_id: str, raw_data_only: bool
    ) -> Dict[str, Any]:
        """Handle direct tool calls with proper error handling and response formatting"""
        
        if tool_name != 'fetch_security_details':
            return {"error": f"Unsupported tool '{tool_name}'"}

        contract_address = function_args.get('contract_address')
        chain_id = function_args.get('chain_id', 1)
        
        if not contract_address:
            return {"error": "Missing 'contract_address' in tool_arguments"}

        if str(chain_id) not in self.supported_blockchains:
            return {"error": f"Unsupported chain ID: {chain_id}"}
            
        logger.info(f"Fetching security details for {contract_address} on chain {chain_id}")
        result = await self.fetch_security_details(contract_address, chain_id)
        
        errors = self._handle_error(result)
        if errors:
            return errors

        if raw_data_only:
            return {"response": "", "data": result}

        explanation = await self._respond_with_llm(
            query=query,
            tool_call_id=tool_call_id,
            data=result,
            temperature=0.3
        )
        return {"response": explanation, "data": result}

    @monitor_execution()
    @with_retry(max_retries=3)
    async def handle_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle both direct tool calls and natural language queries.
        Either 'query' or 'tool' must be provided in params.
        """
        query = params.get('query')
        tool_name = params.get('tool')
        tool_args = params.get('tool_arguments', {})
        raw_data_only = params.get('raw_data_only', False)

        # ---------------------
        # 1) DIRECT TOOL CALL
        # ---------------------
        if tool_name:
            return await self._handle_tool_logic(
                tool_name=tool_name,
                function_args=tool_args,
                query=query or "Direct tool call without LLM.",
                tool_call_id="direct_tool",
                raw_data_only=raw_data_only
            )

        # ---------------------
        # 2) NATURAL LANGUAGE QUERY (LLM decides the tool)
        # ---------------------
        if query:
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
            tool_call_name = tool_call.function.name
            tool_call_args = json.loads(tool_call.function.arguments)

            return await self._handle_tool_logic(
                tool_name=tool_call_name,
                function_args=tool_call_args,
                query=query,
                tool_call_id=tool_call.id,
                raw_data_only=raw_data_only
            )

        # ---------------------
        # 3) NEITHER query NOR tool
        # ---------------------
        return {"error": "Either 'query' or 'tool' must be provided in the parameters."}