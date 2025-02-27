from typing import Dict, Any
from .mesh_agent import MeshAgent, monitor_execution, with_retry, with_cache
from core.llm import call_llm_with_tools_async, call_llm_async
import os
import aiohttp
from dotenv import load_dotenv
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Type, List, Any

load_dotenv()

class PumpFunTokenAgent(MeshAgent):
    # Token address constants
    USDC_ADDRESS = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    SOL_ADDRESS = "So11111111111111111111111111111111111111112"
    VIRTUAL_ADDRESS = "3iQL8BFS2vE7mww4ehAqQHAsbmRNCrPxizWAT2Zfyr9y"
    
    # Supported quote tokens
    SUPPORTED_QUOTE_TOKENS = {
        "usdc": USDC_ADDRESS,
        "sol": SOL_ADDRESS,
        "virtual": VIRTUAL_ADDRESS
    }
    
    def __init__(self):
        super().__init__()
        self.session = None
        self.metadata.update({
            'name': 'PumpFun Token Analysis Agent',
            'version': '1.0.0',
            'author': 'Heurist Team',
            'description': 'This agent analyzes Pump.fun token on Solana using Bitquery API. It has access to token creation, market cap, liquidity, holders, buyers, and top traders data.',
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
    async def query_recent_token_creation(self, interval: str = 'hours', offset: int = 1) -> Dict:
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
    async def query_token_metrics(self, token_address: str, quote_token: str = "usdc") -> Dict:
        """
        Query token metrics including volume, liquidity and market cap.
        
        Args:
            token_address (str): The mint address of the token
            quote_token (str): The quote token to use (usdc, sol, virtual)
            
        Returns:
            Dict: Dictionary containing token metrics
        """
        # Get the quote token address
        if quote_token.lower() in self.SUPPORTED_QUOTE_TOKENS:
            quote_token_address = self.SUPPORTED_QUOTE_TOKENS[quote_token.lower()]
        else:
            quote_token_address = quote_token  # Assume it's a direct address if not a key
        
        time_1h_ago = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        query = """
        query ($time_1h_ago: DateTime, $token: String, $quote_token: String) {
          Solana {
            volume: DEXTradeByTokens(
              where: {
                Trade: {
                  Currency: { MintAddress: { is: $token } }
                  Side: { Currency: { MintAddress: { is: $quote_token } } }
                }
                Block: { Time: { since: $time_1h_ago } }
              }
              limit: {count: 10}
            ) {
              sum(of: Trade_Side_AmountInUSD)
            }
            buyVolume: DEXTradeByTokens(
              where: {
                Trade: {
                  Currency: { MintAddress: { is: $token } }
                  Side: { 
                    Currency: { MintAddress: { is: $quote_token } }
                  }
                }
                Block: { Time: { since: $time_1h_ago } }
              }
              limit: {count: 10}
            ) {
              sum(of: Trade_Side_AmountInUSD)
            }
            sellVolume: DEXTradeByTokens(
              where: {
                Trade: {
                  Currency: { MintAddress: { is: $token } }
                  Side: { 
                    Currency: { MintAddress: { is: $quote_token } }
                  }
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
                    QuoteCurrency: { MintAddress: { is: $quote_token } }
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
                Quote {
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
            # Add token price data
            tokenPrice: DEXTrades(
              limit: {count: 10}
              orderBy: {descending: Block_Time}
              where: {
                Trade: {
                  Buy: {Currency: {MintAddress: {is: $token}}}
                  Sell: {Currency: {MintAddress: {is: $quote_token}}}
                }
              }
            ) {
              Trade {
                Buy {
                  Currency {
                    Symbol
                  }
                }
                Sell {
                  Currency {
                    Symbol
                  }
                }
              }
            }
          }
        }
        """
        
        variables = {
            "time_1h_ago": time_1h_ago,
            "token": token_address,
            "quote_token": quote_token_address
        }
        
        result = await self._execute_query(query, variables)
        
        # If no data found with primary quote token, try alternatives
        if (not result.get('data', {}).get('Solana', {}).get('liquidity') and 
            quote_token.lower() != "sol" and 
            quote_token != self.SOL_ADDRESS):
            # Try with SOL as fallback
            sol_variables = {
                "time_1h_ago": time_1h_ago,
                "token": token_address,
                "quote_token": self.SOL_ADDRESS
            }
            result = await self._execute_query(query, sol_variables)
            
            # Add info about fallback
            if 'data' in result:
                result['data']['fallback_used'] = "Used SOL as fallback quote token"
                
        return result

    @monitor_execution()
    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def query_token_holders(self, token_address: str) -> Dict:
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
                  Decimals
                }
                Account {
                  Address
                }
                Holding: PostBalance(maximum: Block_Slot)
              }
            }
            
            # Add total supply information
            TotalSupply: TokenSupplyUpdates(
              limit: {count: 10}
              orderBy: {descending: Block_Time}
              where: {
                TokenSupplyUpdate: {
                  Currency: {
                    MintAddress: {is: $token}
                  }
                }
              }
            ) {
              TokenSupplyUpdate {
                PostBalance
                Currency {
                  Decimals
                }
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
            
            # Get total supply if available
            total_supply = 0
            total_supply_data = result['data']['Solana'].get('TotalSupply', [])
            if total_supply_data and len(total_supply_data) > 0:
                total_supply_update = total_supply_data[0].get('TokenSupplyUpdate', {})
                if 'PostBalance' in total_supply_update:
                    total_supply = total_supply_update['PostBalance']
            
            for holder in holders:
                if 'BalanceUpdate' not in holder:
                    continue
                    
                balance_update = holder['BalanceUpdate']
                currency = balance_update['Currency']
                holding = balance_update['Holding']
                
                # Calculate percentage of total supply if total supply is available
                percentage = 0
                if total_supply > 0:
                    percentage = (holding / total_supply) * 100
                
                formatted_holder = {
                    'address': balance_update['Account']['Address'],
                    'holding': holding,
                    'percentage_of_supply': round(percentage, 2),
                    'token_info': {
                        'name': currency.get('Name', 'Unknown'),
                        'symbol': currency.get('Symbol', 'Unknown'),
                        'mint_address': currency.get('MintAddress', ''),
                        'decimals': currency.get('Decimals', 0)
                    }
                }
                formatted_holders.append(formatted_holder)
            
            return {
                'holders': formatted_holders,
                'total_supply': total_supply
            }
        
        return {'holders': [], 'total_supply': 0}

    @monitor_execution()
    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def query_token_buyers(self, token_address: str, limit: int = 100) -> Dict:
        """
        Query first buyers of a specific token.
        
        Args:
            token_address (str): The mint address of the token
            limit (int): Number of buyers to return
            
        Returns:
            Dict: Dictionary containing buyer information
        """
        query = """
        query ($token: String!, $limit: Int!) {
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
              limit: { count: $limit }
              orderBy: { ascending: Block_Time }
            ) {
              Trade {
                Buy {
                  Amount
                  AmountInUSD
                  Account {
                    Token {
                      Owner
                    }
                  }
                  Currency {
                    Symbol
                    Decimals
                  }
                }
                Sell {
                  Currency {
                    Symbol
                    MintAddress
                  }
                }
              }
              Block {
                Time
              }
              Transaction {
                Index
              }
            }
          }
        }
        """
        
        variables = {
            "token": token_address,
            "limit": limit
        }
        
        result = await self._execute_query(query, variables)
        
        if 'data' in result and 'Solana' in result['data']:
            trades = result['data']['Solana']['DEXTrades']
            formatted_buyers = []
            unique_buyers = set()  # Track unique buyers
            
            for trade in trades:
                if 'Trade' not in trade or 'Buy' not in trade['Trade']:
                    continue
                    
                buy = trade['Trade']['Buy']
                sell_currency = trade['Trade']['Sell']['Currency']
                owner = buy['Account']['Token']['Owner']
                
                # Only add unique buyers
                if owner not in unique_buyers:
                    unique_buyers.add(owner)
                    
                    formatted_buyer = {
                        'owner': owner,
                        'amount': buy['Amount'],
                        'amount_usd': buy['AmountInUSD'],
                        'currency_pair': f"{buy['Currency']['Symbol']}/{sell_currency['Symbol']}",
                        'time': trade['Block']['Time'],
                        'transaction_index': trade['Transaction']['Index']
                    }
                    formatted_buyers.append(formatted_buyer)
            
            return {
                'buyers': formatted_buyers,
                'unique_buyer_count': len(unique_buyers)
            }
        
        return {'buyers': [], 'unique_buyer_count': 0}


    @monitor_execution()
    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def query_holder_status(self, token_address: str, buyer_addresses: List[str]) -> Dict:
        """
        Query holder status for specific addresses for a token.
        
        Args:
            token_address (str): The mint address of the token
            buyer_addresses (list[str]): List of buyer addresses to check
            
        Returns:
            Dict: Dictionary containing holder status information
        """
        # Split addresses into chunks of 50 to avoid query limitations
        max_addresses_per_query = 50
        address_chunks = [buyer_addresses[i:i + max_addresses_per_query] 
                          for i in range(0, len(buyer_addresses), max_addresses_per_query)]
        
        all_holder_statuses = []
        
        for address_chunk in address_chunks:
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
                  limit: {count: 10}
                ) {
                  BalanceUpdate {
                    Account {
                      Token {
                        Owner
                      }
                    }
                    balance: PostBalance(maximum: Block_Slot)
                    Currency {
                      Decimals
                    }
                  }
                  Transaction {
                    Index
                  }
                }
              }
            }
            """
            
            variables = {
                "token": token_address,
                "addresses": address_chunk
            }
            
            result = await self._execute_query(query, variables)
            
            if 'data' in result and 'Solana' in result['data']:
                updates = result['data']['Solana']['BalanceUpdates']
                
                # Process each balance update
                for update in updates:
                    if 'BalanceUpdate' not in update:
                        continue
                        
                    balance_update = update['BalanceUpdate']
                    current_balance = balance_update.get('balance', 0)
                    # Since we can't get initial_balance via sum, we'll use a placeholder
                    initial_balance = 0  # This would need to be obtained another way
                    decimals = balance_update.get('Currency', {}).get('Decimals', 0)
                    
                    # Determine status (modified since we can't truly compare with initial balance)
                    status = "holding"  # Default to holding
                    if current_balance == 0:
                        status = "sold_all"
                    
                    holder_status = {
                        'owner': balance_update['Account']['Token']['Owner'],
                        'current_balance': current_balance,
                        'initial_balance': initial_balance,
                        'status': status,
                        'last_tx_index': update.get('Transaction', {}).get('Index', 0),
                        'decimals': decimals
                    }
                    all_holder_statuses.append(holder_status)
        
        # Find addresses that weren't found in the query results
        found_addresses = {status['owner'] for status in all_holder_statuses}
        missing_addresses = [addr for addr in buyer_addresses if addr not in found_addresses]
        
        # Add placeholders for missing addresses
        for addr in missing_addresses:
            all_holder_statuses.append({
                'owner': addr,
                'current_balance': 0,
                'initial_balance': 0,
                'status': 'no_data',
                'last_tx_index': 0,
                'decimals': 0
            })
        
        # Generate summary statistics
        status_counts = {
            'holding': 0,
            'sold_all': 0,
            'no_data': 0
        }
        
        for status in all_holder_statuses:
            if status['status'] in status_counts:
                status_counts[status['status']] += 1
        
        return {
            'holder_statuses': all_holder_statuses,
            'summary': status_counts,
            'total_addresses_checked': len(buyer_addresses)
        }


    @monitor_execution()
    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def query_top_traders(self, token_address: str, limit: int = 100) -> Dict:
        """
        Query top traders for a specific token on Pump Fun DEX.
        
        Args:
            token_address (str): The mint address of the token
            limit (int): Number of traders to return
            
        Returns:
            Dict: Dictionary containing top trader information
        """
        # Directly query for top traders, without relying on Markets query
        query = """
        query ($token: String!, $limit: Int!) {
          Solana {
            DEXTradeByTokens(
              orderBy: {descendingByField: "volumeUsd"}
              limit: {count: $limit}
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
                  Currency {
                    Symbol
                  }
                }
              }
              bought: sum(of: Trade_Amount)
              sold: sum(of: Trade_Amount)
              volume: sum(of: Trade_Amount)
              volumeUsd: sum(of: Trade_Side_AmountInUSD)
              count: count
            }
          }
        }
        """
        
        variables = {
            "token": token_address,
            "limit": limit
        }
        
        result = await self._execute_query(query, variables)
        
        if 'data' in result and 'Solana' in result['data']:
            trades = result['data']['Solana']['DEXTradeByTokens']
            formatted_traders = []
            
            for trade in trades:
                buy_sell_ratio = 0
                if trade['sold'] > 0:
                    buy_sell_ratio = trade['bought'] / trade['sold']
                
                formatted_trader = {
                    'owner': trade['Trade']['Account']['Owner'],
                    'bought': trade['bought'],
                    'sold': trade['sold'],
                    'buy_sell_ratio': buy_sell_ratio,
                    'total_volume': trade['volume'],
                    'volume_usd': trade['volumeUsd'],
                    'transaction_count': trade['count'],
                    'side_currency_symbol': trade['Trade']['Side']['Currency']['Symbol'] if 'Currency' in trade['Trade']['Side'] else 'Unknown'
                }
                formatted_traders.append(formatted_trader)
            
            # We can't get market information since the Markets query is failing
            result_with_info = {
                'traders': formatted_traders,
                'markets': []  # Empty list since we can't query markets
            }
            
            return result_with_info
        
        return {'traders': [], 'markets': []}



    async def _execute_query(self, query: str, variables: Dict = None) -> Dict:
        """
        Execute a GraphQL query against the Bitquery API with improved error handling.
        
        Args:
            query (str): GraphQL query to execute
            variables (Dict, optional): Variables for the query
            
        Returns:
            Dict: Query results
        """
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

        try:
            async with self.session.post(url, json=payload, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
                
                if 'errors' in data:
                    error_messages = [error.get('message', 'Unknown error') for error in data['errors']]
                    raise Exception(f"GraphQL errors: {', '.join(error_messages)}")
                    
                return data
        except aiohttp.ClientResponseError as e:
            if e.status == 429:
                # Rate limit error
                raise Exception(f"Rate limit exceeded: {str(e)}")
            elif e.status >= 500:
                # Server-side error
                raise Exception(f"Bitquery server error: {str(e)}")
            else:
                raise Exception(f"API request failed: {str(e)}")
        except aiohttp.ClientError as e:
            # Network-related errors
            raise Exception(f"Network error when calling Bitquery: {str(e)}")
        except Exception as e:
            # Any other unexpected errors
            raise Exception(f"Unexpected error during query execution: {str(e)}")

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
                3. Trade volume (including buy/sell ratio)
                4. Current price
                Present the analysis in a clear, concise manner focusing on key metrics.""",
            'holders': """You are a Solana token analyst. Analyze token holder distribution with these points:
                1. Token information
                2. Top holder addresses and percentages
                3. Distribution patterns and concentration
                Present a concise summary of the token's holder distribution focusing on concentration and notable patterns.""",
            'buyers': """You are a Solana token analyst. Analyze the first buyers of a token with these points:
                1. Number of unique buyers
                2. Total purchase amounts
                3. Currency pairs used
                4. Notable patterns in buying behavior
                Present a concise summary of the token's early buyer activity.""",
            'holder_status': """You are a Solana token analyst. Analyze the current status of early token buyers with these points:
                1. Number of holders still holding
                2. Number of complete sellers
                3. Number of buyers who increased their position
                4. Number of partial sellers
                Present a concise summary of how early buyers have managed their positions.""",
            'top_traders': """You are a Solana token analyst. Analyze the top traders of a token with these points:
                1. Trading volume patterns
                2. Buy/sell ratios
                3. Notable trading behaviors
                4. Active market pairs
                Present a concise summary of the token's most active traders."""
        }
        return prompts.get(query_type, prompts['creation'])

    def get_tool_schema(self, query_type: str = None) -> List[Dict]:
        """
        Get the tool schema based on query type.
        
        Args:
            query_type (str, optional): Type of query to get schema for. Defaults to None.
            
        Returns:
            List[Dict]: List of tool schemas
        """
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'query_recent_token_creation',
                    'description': 'Fetch the data of tokens recently created on Pump.fun',
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
            {
                'type': 'function',
                'function': {
                    'name': 'query_token_metrics',
                    'description': 'Fetch Solana token metrics data',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'token_address': {
                                'type': 'string',
                                'description': 'Token mint address'
                            },
                            'quote_token': {
                                'type': 'string',
                                'description': 'Quote token to use (usdc, sol, virtual, or direct address)',
                                'default': 'usdc'
                            }
                        },
                        'required': ['token_address']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'query_token_holders',
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
            {
                'type': 'function',
                'function': {
                    'name': 'query_token_buyers',
                    'description': 'Fetch first buyers data',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'token_address': {
                                'type': 'string',
                                'description': 'Token mint address'
                            },
                            'limit': {
                                'type': 'integer',
                                'description': 'Number of buyers to fetch',
                                'default': 100
                            }
                        },
                        'required': ['token_address']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'query_holder_status',
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
            {
                'type': 'function',
                'function': {
                    'name': 'query_top_traders',
                    'description': 'Fetch top traders data',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'token_address': {
                                'type': 'string',
                                'description': 'Token mint address'
                            },
                            'limit': {
                                'type': 'integer',
                                'description': 'Number of traders to fetch',
                                'default': 100
                            }
                        },
                        'required': ['token_address']
                    }
                }
            }
        ]

    @monitor_execution()
    @with_retry(max_retries=3)
    async def handle_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query_type = params.get('query_type', 'creation')
        query = params.get('query')
        parameters = params.get('parameters', {})

        if not query:
            raise ValueError("Query parameter is required")

        # System prompt might vary based on query type and parameters
        system_prompt = self.get_system_prompt(query_type)
        
        # Add specific information about quote token if relevant
        if query_type == 'metrics' and 'quote_token' in parameters:
            quote_token = parameters.get('quote_token')
            system_prompt += f"\nYou are analyzing metrics using {quote_token.upper()} as the quote token."

        response = await call_llm_with_tools_async(
            base_url=self.heurist_base_url,
            api_key=self.heurist_api_key,
            model_id=self.metadata['large_model_id'],
            system_prompt=system_prompt,
            user_prompt=query,
            temperature=0.1,
            tools=self.get_tool_schema(query_type)
        )

        if not response:
            return {"error": "Failed to call LLM"}

        if not response.get('tool_calls'):
            return {"response": response.get('content')}

        tool_call = response['tool_calls']
        function_args = json.loads(tool_call.function.arguments)
        
        try:
            if query_type == 'creation':
                result = await self.query_recent_token_creation(
                    interval=function_args.get('interval', parameters.get('interval', 'hours')),
                    offset=function_args.get('offset', parameters.get('offset', 1))
                )
            elif query_type == 'metrics':
                result = await self.query_token_metrics(
                    token_address=function_args.get('token_address', parameters.get('token_address')),
                    quote_token=function_args.get('quote_token', parameters.get('quote_token', 'usdc'))
                )
            elif query_type == 'buyers':
                result = await self.query_token_buyers(
                    token_address=function_args.get('token_address', parameters.get('token_address')),
                    limit=function_args.get('limit', parameters.get('limit', 100))
                )
            elif query_type == 'holders':
                result = await self.query_token_holders(
                    token_address=function_args.get('token_address', parameters.get('token_address'))
                )
            elif query_type == 'holder_status':
                result = await self.query_holder_status(
                    token_address=function_args.get('token_address', parameters.get('token_address')),
                    buyer_addresses=function_args.get('buyer_addresses', parameters.get('buyer_addresses', []))
                )
            elif query_type == 'top_traders':
                result = await self.query_top_traders(
                    token_address=function_args.get('token_address', parameters.get('token_address')),
                    limit=function_args.get('limit', parameters.get('limit', 100))
                )
            else:
                return {"error": f"Invalid query type: {query_type}"}
                
            # Include quote token info in the result for metrics queries
            if query_type == 'metrics' and 'quote_token' in parameters:
                result['quote_token'] = parameters.get('quote_token')
                
            analysis = await call_llm_async(
                base_url=self.heurist_base_url,
                api_key=self.heurist_api_key,
                model_id=self.metadata['large_model_id'],
                messages=[
                    {"role": "system", "content": system_prompt},
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