from typing import Dict, Any
from .mesh_agent import MeshAgent, monitor_execution, with_retry, with_cache
from core.llm import call_llm_with_tools_async, call_llm_async
import os
import aiohttp
from dotenv import load_dotenv
import json
from datetime import datetime, timezone, timedelta

load_dotenv()

class PumpFunTokenAgent(MeshAgent):
    def __init__(self):
        super().__init__()
        self.session = None
        self.metadata.update({
            'name': 'PumpFun Token Analysis Agent',
            'version': '1.0.0',
            'author': 'Heurist Team',
            'description': 'Analyzes Solana token creation and metrics using Bitquery API',
            'inputs': [
                {
                    'name': 'query_type',
                    'description': 'Type of query to execute',
                    'type': 'str',
                    'required': True
                },
                {
                    'name': 'parameters',
                    'description': 'Query-specific parameters',
                    'type': 'dict',
                    'required': False
                }
            ],
            'outputs': [
                {
                    'name': 'response',
                    'description': 'Token analysis results',
                    'type': 'str'
                }
            ],
            'external_apis': ['Bitquery'],
            'tags': ['Solana', 'Token Analysis']
        })
        
        self.VALID_INTERVALS = {'hours', 'days'}

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            self.session = None

    @monitor_execution()
    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def run_creation_query(self, interval: str = 'hours', offset: int = 1) -> Dict:
        if interval not in self.VALID_INTERVALS:
            raise ValueError(f"Invalid interval. Must be one of: {', '.join(self.VALID_INTERVALS)}")
        
        query = """
        query {
          Solana {
            TokenSupplyUpdates(
              where: {
                Instruction: {
                  Program: {
                    Address: {is: "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"}, 
                    Method: {is: "create"}
                  }
                }
              }
              limit: {count: 10}
            ) {
              Block {
                Time(interval: {in: INTERVAL_PLACEHOLDER, offset: OFFSET_PLACEHOLDER})
              }
              TokenSupplyUpdate {
                Amount
                Currency {
                  Symbol
                  Name
                  MintAddress
                  ProgramAddress
                  Decimals
                }
                PostBalance
              }
              Transaction {
                Signer
              }
            }
          }
        }
        """.replace('INTERVAL_PLACEHOLDER', interval).replace('OFFSET_PLACEHOLDER', str(offset))

        result = await self._execute_query(query)
        
        if 'data' in result and 'Solana' in result['data']:
            tokens = result['data']['Solana']['TokenSupplyUpdates']
            filtered_tokens = []
            
            for token in tokens:
                if 'TokenSupplyUpdate' not in token or 'Currency' not in token['TokenSupplyUpdate']:
                    continue
                    
                currency = token['TokenSupplyUpdate']['Currency']
                filtered_token = {
                    'block_time': token['Block']['Time'],
                    'token_info': {
                        'name': currency.get('Name', 'Unknown'),
                        'symbol': currency.get('Symbol', 'Unknown'),
                        'mint_address': currency.get('MintAddress', ''),
                        'program_address': currency.get('ProgramAddress', ''),
                        'decimals': currency.get('Decimals', 0)
                    },
                    'amount': token['TokenSupplyUpdate']['Amount'],
                    'signer': token['Transaction']['Signer']
                }
                filtered_tokens.append(filtered_token)
            
            return {'tokens': filtered_tokens[:10]}
        
        return {'tokens': []}

    @monitor_execution()
    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def run_metrics_query(self, token_address: str, usdc_address: str) -> Dict:
        time_1h_ago = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        query = """
        query ($time_1h_ago: DateTime, $token: String, $side: String) {
          Solana {
            volume: DEXTradeByTokens(
              where: {
                Trade: {
                  Currency: { MintAddress: { is: $token } }
                  Side: { Currency: { MintAddress: { is: $side } } }
                }
                Block: { Time: { since: $time_1h_ago } }
              }
              limit: {count: 10}
            ) {
              sum(of: Trade_Side_AmountInUSD)
            }
            liquidity: DEXPools(
              where: {
                Pool: {
                  Market: {
                    BaseCurrency: { MintAddress: { is: $token } }
                    QuoteCurrency: { MintAddress: { is: $side } }
                  }
                }
                Block: { Time: { till: $time_1h_ago } }
              }
              limit: { count: 10 }
              orderBy: { descending: Block_Time }
            ) {
              Pool {
                Base {
                  PostAmountInUSD
                }
              }
            }
            marketcap: TokenSupplyUpdates(
              where: {
                TokenSupplyUpdate: { Currency: { MintAddress: { is: $token } } }
                Block: { Time: { till: $time_1h_ago } }
              }
              limitBy: { by: TokenSupplyUpdate_Currency_MintAddress, count: 10 }
              orderBy: { descending: Block_Time }
            ) {
              TokenSupplyUpdate {
                PostBalanceInUSD
                Currency {
                  Name
                  MintAddress
                  Symbol
                }
              }
            }
          }
        }
        """
        
        variables = {
            "time_1h_ago": time_1h_ago,
            "token": token_address,
            "side": usdc_address
        }
        
        result = await self._execute_query(query, variables)
        return result

    @monitor_execution()
    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def run_holders_query(self, token_address: str) -> Dict:
        """
        Query top token holders for a specific token.
        
        Args:
            token_address (str): The mint address of the token
            
        Returns:
            Dict: Dictionary containing holder information
        """
        query = """
        query ($token: String!) {
          Solana(dataset: realtime) {
            BalanceUpdates(
              limit: { count: 10 }
              orderBy: { descendingByField: "BalanceUpdate_Holding_maximum" }
              where: {
                BalanceUpdate: {
                  Currency: {
                    MintAddress: { is: $token }
                  }
                }
                Transaction: { Result: { Success: true } }
              }
            ) {
              BalanceUpdate {
                Currency {
                  Name
                  MintAddress
                  Symbol
                }
                Account {
                  Address
                }
                Holding: PostBalance(maximum: Block_Slot)
              }
            }
          }
        }
        """
        
        variables = {
            "token": token_address
        }
        
        result = await self._execute_query(query, variables)
        
        if 'data' in result and 'Solana' in result['data']:
            holders = result['data']['Solana']['BalanceUpdates']
            formatted_holders = []
            
            for holder in holders:
                if 'BalanceUpdate' not in holder:
                    continue
                    
                balance_update = holder['BalanceUpdate']
                formatted_holder = {
                    'address': balance_update['Account']['Address'],
                    'holding': balance_update['Holding'],
                    'token_info': {
                        'name': balance_update['Currency']['Name'],
                        'symbol': balance_update['Currency']['Symbol'],
                        'mint_address': balance_update['Currency']['MintAddress']
                    }
                }
                formatted_holders.append(formatted_holder)
            
            return {'holders': formatted_holders}
        
        return {'holders': []}

    @monitor_execution()
    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def run_buyers_query(self, token_address: str) -> Dict:
        """
        Query first 100 buyers of a specific token.
        
        Args:
            token_address (str): The mint address of the token
            
        Returns:
            Dict: Dictionary containing buyer information
        """
        query = """
        query ($token: String!) {
          Solana {
            DEXTrades(
              where: {
                Trade: {
                  Buy: {
                    Currency: {
                      MintAddress: {
                        is: $token
                      }
                    }
                  }
                }
              }
              limit: { count: 100 }
              orderBy: { ascending: Block_Time }
            ) {
              Trade {
                Buy {
                  Amount
                  Account {
                    Token {
                      Owner
                    }
                  }
                }
              }
              Block {
                Time
              }
            }
          }
        }
        """
        
        variables = {
            "token": token_address
        }
        
        result = await self._execute_query(query, variables)
        
        if 'data' in result and 'Solana' in result['data']:
            trades = result['data']['Solana']['DEXTrades']
            formatted_buyers = []
            
            for trade in trades:
                if 'Trade' not in trade or 'Buy' not in trade['Trade']:
                    continue
                    
                buy = trade['Trade']['Buy']
                formatted_buyer = {
                    'owner': buy['Account']['Token']['Owner'],
                    'amount': buy['Amount'],
                    'time': trade['Block']['Time']
                }
                formatted_buyers.append(formatted_buyer)
            
            return {'buyers': formatted_buyers}
        
        return {'buyers': []}


    @monitor_execution()
    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def run_holder_status_query(self, token_address: str, buyer_addresses: list[str]) -> Dict:
      """
      Query holder status for specific addresses for a token.
      
      Args:
          token_address (str): The mint address of the token
          buyer_addresses (list[str]): List of buyer addresses to check
          
      Returns:
          Dict: Dictionary containing holder status information
      """
      query = """
      query ($token: String!, $addresses: [String!]) {
        Solana {
          BalanceUpdates(
            where: {
              BalanceUpdate: {
                Account: {
                  Token: {
                    Owner: {
                      in: $addresses
                    }
                  }
                }
                Currency: {
                  MintAddress: { is: $token }
                }
              }
            }
          ) {
            BalanceUpdate {
              Account {
                Token {
                  Owner
                }
              }
              balance: PostBalance(maximum: Block_Slot)
            }
          }
        }
      }
      """
      
      variables = {
          "token": token_address,
          "addresses": buyer_addresses
      }
      
      result = await self._execute_query(query, variables)
      
      if 'data' in result and 'Solana' in result['data']:
          updates = result['data']['Solana']['BalanceUpdates']
          holder_statuses = []
          
          for update in updates:
              if 'BalanceUpdate' not in update:
                  continue
                  
              balance_update = update['BalanceUpdate']
              holder_status = {
                  'owner': balance_update['Account']['Token']['Owner'],
                  'current_balance': balance_update['balance']
              }
              holder_statuses.append(holder_status)
          
          return {'holder_statuses': holder_statuses}
      
      return {'holder_statuses': []}

    @monitor_execution()
    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def run_top_traders_query(self, token_address: str) -> Dict:
      """
      Query top traders for a specific token on Pump Fun DEX.
      
      Args:
          token_address (str): The mint address of the token
          
      Returns:
          Dict: Dictionary containing top trader information
      """
      query = """
      query ($token: String!) {
        Solana {
          DEXTradeByTokens(
            orderBy: {descendingByField: "volumeUsd"}
            limit: {count: 100}
            where: {
              Trade: {
                Currency: {
                  MintAddress: {is: $token}
                }
              }, 
              Transaction: {
                Result: {Success: true}
              }
            }
          ) {
            Trade {
              Account {
                Owner
              }
              Side {
                Account {
                  Address
                }
                Type
              }
            }
            bought: sum(of: Trade_Amount, if: {Trade: {Side: {Type: {is: buy}}}})
            sold: sum(of: Trade_Amount, if: {Trade: {Side: {Type: {is: sell}}}})
            volume: sum(of: Trade_Amount)
            volumeUsd: sum(of: Trade_Side_AmountInUSD)
          }
        }
      }
      """
      
      variables = {
          "token": token_address
      }
      
      result = await self._execute_query(query, variables)
      
      if 'data' in result and 'Solana' in result['data']:
          trades = result['data']['Solana']['DEXTradeByTokens']
          formatted_traders = []
          
          for trade in trades:
              formatted_trader = {
                  'owner': trade['Trade']['Account']['Owner'],
                  'bought': trade['bought'],
                  'sold': trade['sold'],
                  'total_volume': trade['volume'],
                  'volume_usd': trade['volumeUsd']
              }
              formatted_traders.append(formatted_trader)
          
          return {'traders': formatted_traders}
      
      return {'traders': []}

    async def _execute_query(self, query: str, variables: Dict = None) -> Dict:
        if not self.session:
            self.session = aiohttp.ClientSession()

        url = 'https://streaming.bitquery.io/eap'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {os.getenv("BITQUERY_API_KEY")}'
        }
        
        payload = {'query': query}
        if variables:
            payload['variables'] = variables

        async with self.session.post(url, json=payload, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()
            
            if 'errors' in data:
                raise Exception(f"GraphQL errors: {data['errors']}")
                
            return data

    def get_system_prompt(self, query_type: str) -> str:
        prompts = {
            'creation': """You are a Solana token analyst. Analyze new token creation events with these points:
                1. Basic token info (name, symbol, mint address)
                2. Initial supply
                3. Creator details
                Present findings concisely in 2-3 sentences max.""",
            'metrics': """You are a Solana token analyst. Analyze token metrics with these points:
                1. Market cap
                2. Liquidity
                3. Trade volume
                Present the analysis in a clear, concise manner focusing on key metrics.""",
            'holders': """You are a Solana token analyst. Analyze token holder distribution with these points:
                1. Token information
                2. Top holder addresses
                3. Distribution patterns
                Present a concise summary of the token's holder distribution focusing on concentration and notable patterns.""",
            'buyers': """You are a Solana token analyst. Analyze the first buyers of a token with these points:
                1. Number of unique buyers
                2. Total purchase amounts
                3. Notable patterns in buying behavior
                Present a concise summary of the token's early buyer activity.""",
            'holder_status': """You are a Solana token analyst. Analyze the current status of early token buyers with these points:
                1. Number of holders still holding
                2. Number of complete sellers
                3. Number of buyers who increased their position
                Present a concise summary of how early buyers have managed their positions.""",
            'top_traders': """You are a Solana token analyst. Analyze the top traders of a token with these points:
                1. Trading volume patterns
                2. Buy/sell ratios
                3. Notable trading behaviors
                Present a concise summary of the token's most active traders."""
        }
        return prompts.get(query_type, prompts['creation'])

    def get_tool_schema(self, query_type: str) -> Dict:
        schemas = {
            'creation': {
                'type': 'function',
                'function': {
                    'name': 'run_creation_query',
                    'description': 'Fetch Solana token creation data',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'interval': {
                                'type': 'string',
                                'enum': list(self.VALID_INTERVALS),
                                'default': 'hours'
                            },
                            'offset': {
                                'type': 'integer',
                                'minimum': 1,
                                'maximum': 99,
                                'default': 1
                            }
                        }
                    }
                }
            },
            'metrics': {
                'type': 'function',
                'function': {
                    'name': 'run_metrics_query',
                    'description': 'Fetch Solana token metrics data',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'token_address': {
                                'type': 'string',
                                'description': 'Token mint address'
                            },
                            'usdc_address': {
                                'type': 'string',
                                'description': 'USDC token address'
                            }
                        },
                        'required': ['token_address', 'usdc_address']
                    }
                }
            },
            'holders': {
                'type': 'function',
                'function': {
                    'name': 'run_holders_query',
                    'description': 'Fetch top token holders data',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'token_address': {
                                'type': 'string',
                                'description': 'Token mint address'
                            }
                        },
                        'required': ['token_address']
                    }
                }
            },
        'buyers': {
                'type': 'function',
                'function': {
                    'name': 'run_buyers_query',
                    'description': 'Fetch first 100 buyers data',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'token_address': {
                                'type': 'string',
                                'description': 'Token mint address'
                            }
                        },
                        'required': ['token_address']
                    }
                }
            },
        'holder_status': {
            'type': 'function',
            'function': {
                'name': 'run_holder_status_query',
                'description': 'Fetch holder status for specific addresses',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'token_address': {
                            'type': 'string',
                            'description': 'Token mint address'
                        },
                        'buyer_addresses': {
                            'type': 'array',
                            'items': {
                                'type': 'string'
                            },
                            'description': 'List of buyer addresses to check'
                        }
                    },
                    'required': ['token_address', 'buyer_addresses']
                }
            }
        },
        'top_traders': {
            'type': 'function',
            'function': {
                'name': 'run_top_traders_query',
                'description': 'Fetch top traders data',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'token_address': {
                            'type': 'string',
                            'description': 'Token mint address'
                        }
                    },
                    'required': ['token_address']
                }
            }
        }}
        
        return schemas.get(query_type, schemas['creation'])

    @monitor_execution()
    @with_retry(max_retries=3)
    async def handle_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
      query_type = params.get('query_type', 'creation')
      query = params.get('query')
      parameters = params.get('parameters', {})

      if not query:
          raise ValueError("Query parameter is required")

      response = await call_llm_with_tools_async(
          base_url=self.heurist_base_url,
          api_key=self.heurist_api_key,
          model_id=self.metadata['large_model_id'],
          system_prompt=self.get_system_prompt(query_type),
          user_prompt=query,
          temperature=0.1,
          tools=[self.get_tool_schema(query_type)]
      )

      if not response:
          return {"error": "Failed to call LLM"}

      if not response.get('tool_calls'):
          return {"response": response.get('content')}

      tool_call = response['tool_calls']
      function_args = json.loads(tool_call.function.arguments)
      
      try:
          if query_type == 'creation':
              result = await self.run_creation_query(
                  interval=function_args.get('interval', 'hours'),
                  offset=function_args.get('offset', 1)
              )
          elif query_type == 'metrics':
              result = await self.run_metrics_query(
                  token_address=function_args.get('token_address'),
                  usdc_address=function_args.get('usdc_address')
              )
          elif query_type == 'buyers':
              result = await self.run_buyers_query(
                  token_address=function_args.get('token_address')
              )
          elif query_type == 'holders':
              result = await self.run_holders_query(
                  token_address=function_args.get('token_address')
              )
          elif query_type == 'holder_status':
              result = await self.run_holder_status_query(
                  token_address=function_args.get('token_address'),
                  buyer_addresses=function_args.get('buyer_addresses', [])
              )
          elif query_type == 'top_traders':
              result = await self.run_top_traders_query(
                  token_address=function_args.get('token_address')
              )
          else:
              return {"error": f"Invalid query type: {query_type}"}
              
          analysis = await call_llm_async(
              base_url=self.heurist_base_url,
              api_key=self.heurist_api_key,
              model_id=self.metadata['large_model_id'],
              messages=[
                  {"role": "system", "content": self.get_system_prompt(query_type)},
                  {"role": "user", "content": query},
                  {"role": "tool", "content": json.dumps(result, indent=2), "tool_call_id": tool_call.id}
              ],
              temperature=0.7
          )

          return {
              "response": analysis,
              "data": result
          }
      except Exception as e:
          return {"error": str(e)}