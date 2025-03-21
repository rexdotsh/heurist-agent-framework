import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import aiohttp
from dotenv import load_dotenv

from core.llm import call_llm_async, call_llm_with_tools_async
from decorators import monitor_execution, with_cache, with_retry

from .mesh_agent import MeshAgent

logger = logging.getLogger(__name__)

load_dotenv()


class PumpFunTokenAgent(MeshAgent):
    # Token address constants
    USDC_ADDRESS = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    SOL_ADDRESS = "So11111111111111111111111111111111111111112"
    VIRTUAL_ADDRESS = "3iQL8BFS2vE7mww4ehAqQHAsbmRNCrPxizWAT2Zfyr9y"

    # Supported quote tokens
    SUPPORTED_QUOTE_TOKENS = {"usdc": USDC_ADDRESS, "sol": SOL_ADDRESS, "virtual": VIRTUAL_ADDRESS}

    def __init__(self):
        super().__init__()
        self.session = None
        self.metadata.update(
            {
                "name": "PumpFun Agent",
                "version": "1.0.0",
                "author": "Heurist Team",
                "description": "This agent analyzes Pump.fun token on Solana using Bitquery API. It has access to token creation, market cap, liquidity, holders, buyers, and top traders data.",
                "inputs": [
                    {
                        "name": "query",
                        "description": "Natural language query about a token (must include a token address), or a request for trending coins. ",
                        "type": "str",
                        "required": False,
                    },
                    {
                        "name": "raw_data_only",
                        "description": "If true, the agent will only return the raw or base structured data without additional LLM explanation.",
                        "type": "bool",
                        "required": False,
                        "default": False,
                    },
                ],
                "outputs": [
                    {
                        "name": "response",
                        "description": "Natural language explanation of the token information (empty if a direct tool call).",
                        "type": "str",
                    },
                    {"name": "data", "description": "Structured token analysis data.", "type": "dict"},
                ],
                "external_apis": ["Bitquery"],
                "tags": ["Solana", "Trading"],
                "image_url": "",  # use the logo of pumpfun
            }
        )

        self.VALID_INTERVALS = {"hours", "days"}

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            self.session = None

    def get_system_prompt(self) -> str:
        return """
IDENTITY:
You are a Solana blockchain analyst specializing in Pump.fun token data analysis.

CAPABILITIES:
- Retrieve recent token creations on Pump.fun
- Analyze token metrics including market cap, liquidity, and volume
- Identify token holder distributions and wallet patterns
- Track buyer activity and trading patterns

RESPONSE GUIDELINES:
- Keep responses focused on what was specifically asked
- Format numbers in a human-readable way (e.g., "150.4M SOL")
- Provide only relevant metrics for the query context
- Use bullet points for complex metrics when appropriate

DOMAIN-SPECIFIC RULES:
For token specific queries, identify whether the user has provided a token address or needs information about recent tokens.
- Token addresses on Solana are base58-encoded strings typically ending with 'pump'
- For token metrics, use SOL as the default quote token unless specified otherwise
- When analyzing token holders, focus on concentration patterns and whale activity
- For trading analysis, highlight unusual volume patterns and top trader behaviors

IMPORTANT:
- Never invent token addresses or data
- Keep responses concise and relevant
- Focus on on-chain data rather than speculation
- When information is incomplete, clearly state limitations
"""

    def get_tool_schemas(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "query_recent_token_creation",
                    "description": "Fetch data of tokens recently created on Pump.fun",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "interval": {
                                "type": "string",
                                "enum": list(self.VALID_INTERVALS),
                                "default": "hours",
                                "description": "Time interval (hours/days)",
                            },
                            "offset": {
                                "type": "number",
                                "minimum": 1,
                                "maximum": 99,
                                "default": 1,
                                "description": "Time offset for interval",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "query_token_metrics",
                    "description": "Fetch token metrics including market cap, liquidity, and volume",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "token_address": {"type": "string", "description": "Token mint address on Solana"},
                            "quote_token": {
                                "type": "string",
                                "description": "Quote token to use ('usdc', 'sol', 'virtual', or the token address)",
                                "default": "sol",
                            },
                        },
                        "required": ["token_address"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "query_token_holders",
                    "description": "Fetch top token holders data and distribution",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "token_address": {"type": "string", "description": "Token mint address on Solana"}
                        },
                        "required": ["token_address"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "query_token_buyers",
                    "description": "Fetch first buyers of a token",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "token_address": {"type": "string", "description": "Token mint address on Solana"},
                            "limit": {"type": "number", "description": "Number of buyers to fetch", "default": 100},
                        },
                        "required": ["token_address"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "query_holder_status",
                    "description": "Check if buyers are still holding, sold, or bought more",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "token_address": {"type": "string", "description": "Token mint address on Solana"},
                            "buyer_addresses": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of buyer wallet addresses to check",
                            },
                        },
                        "required": ["token_address", "buyer_addresses"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "query_top_traders",
                    "description": "Fetch top traders for a specific token",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "token_address": {"type": "string", "description": "Token mint address on Solana"},
                            "limit": {"type": "number", "description": "Number of traders to fetch", "default": 100},
                        },
                        "required": ["token_address"],
                    },
                },
            },
        ]

    @monitor_execution()
    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def query_recent_token_creation(self, interval: str = "hours", offset: int = 1) -> Dict:
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
        """.replace("INTERVAL_PLACEHOLDER", interval).replace("OFFSET_PLACEHOLDER", str(offset))

        result = await self._execute_query(query)

        if "data" in result and "Solana" in result["data"]:
            tokens = result["data"]["Solana"]["TokenSupplyUpdates"]
            filtered_tokens = []

            for token in tokens:
                if "TokenSupplyUpdate" not in token or "Currency" not in token["TokenSupplyUpdate"]:
                    continue

                currency = token["TokenSupplyUpdate"]["Currency"]
                filtered_token = {
                    "block_time": token["Block"]["Time"],
                    "token_info": {
                        "name": currency.get("Name", "Unknown"),
                        "symbol": currency.get("Symbol", "Unknown"),
                        "mint_address": currency.get("MintAddress", ""),
                        "program_address": currency.get("ProgramAddress", ""),
                        "decimals": currency.get("Decimals", 0),
                    },
                    "amount": token["TokenSupplyUpdate"]["Amount"],
                    "signer": token["Transaction"]["Signer"],
                }
                filtered_tokens.append(filtered_token)

            return {"tokens": filtered_tokens[:10]}

        return {"tokens": []}

    @monitor_execution()
    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def query_token_metrics(self, token_address: str, quote_token: str = "sol") -> Dict:
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

        variables = {"time_1h_ago": time_1h_ago, "token": token_address, "quote_token": quote_token_address}

        result = await self._execute_query(query, variables)

        # If no data found with primary quote token, try alternatives
        if (
            not result.get("data", {}).get("Solana", {}).get("liquidity")
            and quote_token.lower() != "sol"
            and quote_token != self.SOL_ADDRESS
        ):
            # Try with SOL as fallback
            sol_variables = {"time_1h_ago": time_1h_ago, "token": token_address, "quote_token": self.SOL_ADDRESS}
            result = await self._execute_query(query, sol_variables)

            # Add info about fallback
            if "data" in result:
                result["data"]["fallback_used"] = "Used SOL as fallback quote token"

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

        variables = {"token": token_address}

        result = await self._execute_query(query, variables)

        if "data" in result and "Solana" in result["data"]:
            holders = result["data"]["Solana"]["BalanceUpdates"]
            formatted_holders = []

            # Get total supply if available
            total_supply = 0
            total_supply_data = result["data"]["Solana"].get("TotalSupply", [])
            if total_supply_data and len(total_supply_data) > 0:
                total_supply_update = total_supply_data[0].get("TokenSupplyUpdate", {})
                if "PostBalance" in total_supply_update:
                    total_supply = total_supply_update["PostBalance"]

            for holder in holders:
                if "BalanceUpdate" not in holder:
                    continue

                balance_update = holder["BalanceUpdate"]
                currency = balance_update["Currency"]
                holding = balance_update["Holding"]

                # Calculate percentage of total supply if total supply is available
                percentage = 0
                if total_supply > 0:
                    percentage = (holding / total_supply) * 100

                formatted_holder = {
                    "address": balance_update["Account"]["Address"],
                    "holding": holding,
                    "percentage_of_supply": round(percentage, 2),
                    "token_info": {
                        "name": currency.get("Name", "Unknown"),
                        "symbol": currency.get("Symbol", "Unknown"),
                        "mint_address": currency.get("MintAddress", ""),
                        "decimals": currency.get("Decimals", 0),
                    },
                }
                formatted_holders.append(formatted_holder)

            return {"holders": formatted_holders, "total_supply": total_supply}

        return {"holders": [], "total_supply": 0}

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

        variables = {"token": token_address, "limit": limit}

        result = await self._execute_query(query, variables)

        if "data" in result and "Solana" in result["data"]:
            trades = result["data"]["Solana"]["DEXTrades"]
            formatted_buyers = []
            unique_buyers = set()  # Track unique buyers

            for trade in trades:
                if "Trade" not in trade or "Buy" not in trade["Trade"]:
                    continue

                buy = trade["Trade"]["Buy"]
                sell_currency = trade["Trade"]["Sell"]["Currency"]
                owner = buy["Account"]["Token"]["Owner"]

                # Only add unique buyers
                if owner not in unique_buyers:
                    unique_buyers.add(owner)

                    formatted_buyer = {
                        "owner": owner,
                        "amount": buy["Amount"],
                        "amount_usd": buy["AmountInUSD"],
                        "currency_pair": f"{buy['Currency']['Symbol']}/{sell_currency['Symbol']}",
                        "time": trade["Block"]["Time"],
                        "transaction_index": trade["Transaction"]["Index"],
                    }
                    formatted_buyers.append(formatted_buyer)

            return {"buyers": formatted_buyers, "unique_buyer_count": len(unique_buyers)}

        return {"buyers": [], "unique_buyer_count": 0}

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
        address_chunks = [
            buyer_addresses[i : i + max_addresses_per_query]
            for i in range(0, len(buyer_addresses), max_addresses_per_query)
        ]

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

            variables = {"token": token_address, "addresses": address_chunk}

            result = await self._execute_query(query, variables)

            if "data" in result and "Solana" in result["data"]:
                updates = result["data"]["Solana"]["BalanceUpdates"]

                # Process each balance update
                for update in updates:
                    if "BalanceUpdate" not in update:
                        continue

                    balance_update = update["BalanceUpdate"]
                    current_balance = balance_update.get("balance", 0)
                    # Since we can't get initial_balance via sum, we'll use a placeholder
                    initial_balance = 0  # This would need to be obtained another way
                    decimals = balance_update.get("Currency", {}).get("Decimals", 0)

                    # Determine status (modified since we can't truly compare with initial balance)
                    status = "holding"  # Default to holding
                    if current_balance == 0:
                        status = "sold_all"

                    holder_status = {
                        "owner": balance_update["Account"]["Token"]["Owner"],
                        "current_balance": current_balance,
                        "initial_balance": initial_balance,
                        "status": status,
                        "last_tx_index": update.get("Transaction", {}).get("Index", 0),
                        "decimals": decimals,
                    }
                    all_holder_statuses.append(holder_status)

        # Find addresses that weren't found in the query results
        found_addresses = {status["owner"] for status in all_holder_statuses}
        missing_addresses = [addr for addr in buyer_addresses if addr not in found_addresses]

        # Add placeholders for missing addresses
        for addr in missing_addresses:
            all_holder_statuses.append(
                {
                    "owner": addr,
                    "current_balance": 0,
                    "initial_balance": 0,
                    "status": "no_data",
                    "last_tx_index": 0,
                    "decimals": 0,
                }
            )

        # Generate summary statistics
        status_counts = {"holding": 0, "sold_all": 0, "no_data": 0}

        for status in all_holder_statuses:
            if status["status"] in status_counts:
                status_counts[status["status"]] += 1

        return {
            "holder_statuses": all_holder_statuses,
            "summary": status_counts,
            "total_addresses_checked": len(buyer_addresses),
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

        variables = {"token": token_address, "limit": limit}

        result = await self._execute_query(query, variables)

        if "data" in result and "Solana" in result["data"]:
            trades = result["data"]["Solana"]["DEXTradeByTokens"]
            formatted_traders = []

            for trade in trades:
                buy_sell_ratio = 0
                if float(trade["sold"]) > 0:
                    buy_sell_ratio = float(trade["bought"]) / float(trade["sold"])

                formatted_trader = {
                    "owner": trade["Trade"]["Account"]["Owner"],
                    "bought": trade["bought"],
                    "sold": trade["sold"],
                    "buy_sell_ratio": buy_sell_ratio,
                    "total_volume": trade["volume"],
                    "volume_usd": trade["volumeUsd"],
                    "transaction_count": trade["count"],
                    "side_currency_symbol": trade["Trade"]["Side"]["Currency"]["Symbol"]
                    if "Currency" in trade["Trade"]["Side"]
                    else "Unknown",
                }
                formatted_traders.append(formatted_trader)

            # We can't get market information since the Markets query is failing
            result_with_info = {
                "traders": formatted_traders,
                "markets": [],  # Empty list since we can't query markets
            }

            return result_with_info

        return {"traders": [], "markets": []}

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

        url = "https://streaming.bitquery.io/eap"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.getenv('BITQUERY_API_KEY')}"}

        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        try:
            async with self.session.post(url, json=payload, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()

                if "errors" in data:
                    error_messages = [error.get("message", "Unknown error") for error in data["errors"]]
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

    def _handle_error(self, maybe_error: dict) -> dict:
        """
        Helper to return the error if present in a dictionary with the 'error' key.
        """
        if "error" in maybe_error:
            return {"error": maybe_error["error"]}
        return {}

    async def _respond_with_llm(self, query: str, tool_call_id: str, data: dict, temperature: float) -> str:
        """
        Reusable helper to ask the LLM to generate a user-friendly explanation
        given a piece of data from a tool call.
        """
        return await call_llm_async(
            base_url=self.heurist_base_url,
            api_key=self.heurist_api_key,
            model_id=self.metadata["large_model_id"],
            messages=[
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": query},
                {"role": "tool", "content": str(data), "tool_call_id": tool_call_id},
            ],
            temperature=temperature,
        )

    async def _handle_tool_logic(
        self, tool_name: str, function_args: dict, query: str, tool_call_id: str, raw_data_only: bool
    ) -> Dict[str, Any]:
        """
        Process tool calls, handle errors/formatting, and optionally call the LLM to explain the result.
        """
        # Set temperature for explanation
        temp_for_explanation = 0.7

        if tool_name == "query_recent_token_creation":
            interval = function_args.get("interval", "hours")
            offset = function_args.get("offset", 1)

            result = await self.query_recent_token_creation(interval=interval, offset=offset)
            errors = self._handle_error(result)
            if errors:
                return errors

            if raw_data_only:
                return {"response": "", "data": result}

            explanation = await self._respond_with_llm(
                query=query, tool_call_id=tool_call_id, data=result, temperature=temp_for_explanation
            )
            return {"response": explanation, "data": result}

        elif tool_name == "query_token_metrics":
            token_address = function_args.get("token_address")
            quote_token = function_args.get("quote_token", "sol")

            if not token_address:
                return {"error": "Missing 'token_address' in tool_arguments"}

            result = await self.query_token_metrics(token_address=token_address, quote_token=quote_token)
            errors = self._handle_error(result)
            if errors:
                return errors

            if raw_data_only:
                return {"response": "", "data": result}

            explanation = await self._respond_with_llm(
                query=query, tool_call_id=tool_call_id, data=result, temperature=temp_for_explanation
            )
            return {"response": explanation, "data": result}

        elif tool_name == "query_token_holders":
            token_address = function_args.get("token_address")

            if not token_address:
                return {"error": "Missing 'token_address' in tool_arguments"}

            result = await self.query_token_holders(token_address=token_address)
            errors = self._handle_error(result)
            if errors:
                return errors

            if raw_data_only:
                return {"response": "", "data": result}

            explanation = await self._respond_with_llm(
                query=query, tool_call_id=tool_call_id, data=result, temperature=temp_for_explanation
            )
            return {"response": explanation, "data": result}

        elif tool_name == "query_token_buyers":
            token_address = function_args.get("token_address")
            limit = function_args.get("limit", 100)

            if not token_address:
                return {"error": "Missing 'token_address' in tool_arguments"}

            result = await self.query_token_buyers(token_address=token_address, limit=limit)
            errors = self._handle_error(result)
            if errors:
                return errors

            if raw_data_only:
                return {"response": "", "data": result}

            explanation = await self._respond_with_llm(
                query=query, tool_call_id=tool_call_id, data=result, temperature=temp_for_explanation
            )
            return {"response": explanation, "data": result}

        elif tool_name == "query_holder_status":
            token_address = function_args.get("token_address")
            buyer_addresses = function_args.get("buyer_addresses", [])

            if not token_address:
                return {"error": "Missing 'token_address' in tool_arguments"}
            if not buyer_addresses:
                return {"error": "Missing 'buyer_addresses' in tool_arguments"}

            result = await self.query_holder_status(token_address=token_address, buyer_addresses=buyer_addresses)
            errors = self._handle_error(result)
            if errors:
                return errors

            if raw_data_only:
                return {"response": "", "data": result}

            explanation = await self._respond_with_llm(
                query=query, tool_call_id=tool_call_id, data=result, temperature=temp_for_explanation
            )
            return {"response": explanation, "data": result}

        elif tool_name == "query_top_traders":
            token_address = function_args.get("token_address")
            limit = function_args.get("limit", 100)

            if not token_address:
                return {"error": "Missing 'token_address' in tool_arguments"}

            result = await self.query_top_traders(token_address=token_address, limit=limit)
            errors = self._handle_error(result)
            if errors:
                return errors

            if raw_data_only:
                return {"response": "", "data": result}

            explanation = await self._respond_with_llm(
                query=query, tool_call_id=tool_call_id, data=result, temperature=temp_for_explanation
            )
            return {"response": explanation, "data": result}

        else:
            return {"error": f"Unsupported tool '{tool_name}'"}

    @monitor_execution()
    @with_retry(max_retries=3)
    async def handle_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Either 'query' or 'tool' is required in params.
          - If 'tool' is provided, call that tool directly with 'tool_arguments' (bypassing the LLM).
          - If 'query' is provided, route via LLM for dynamic tool selection.
        """
        query = params.get("query")
        tool_name = params.get("tool")
        tool_args = params.get("tool_arguments", {})
        raw_data_only = params.get("raw_data_only", False)

        # ---------------------
        # 1) DIRECT TOOL CALL
        # ---------------------
        if tool_name:
            return await self._handle_tool_logic(
                tool_name=tool_name,
                function_args=tool_args,
                query=query or "Direct tool call without LLM.",
                tool_call_id="direct_tool",
                raw_data_only=raw_data_only,
            )

        # ---------------------
        # 2) NATURAL LANGUAGE QUERY (LLM decides the tool)
        # ---------------------
        if query:
            response = await call_llm_with_tools_async(
                base_url=self.heurist_base_url,
                api_key=self.heurist_api_key,
                model_id=self.metadata["large_model_id"],
                system_prompt=self.get_system_prompt(),
                user_prompt=query,
                temperature=0.1,
                tools=self.get_tool_schemas(),
            )

            if not response:
                return {"error": "Failed to process query"}
            if not response.get("tool_calls"):
                # No tool calls => the LLM just answered
                return {"response": response["content"], "data": {}}

            # LLM provided a tool call
            tool_call = response["tool_calls"]
            tool_call_name = tool_call.function.name
            tool_call_args = json.loads(tool_call.function.arguments)

            return await self._handle_tool_logic(
                tool_name=tool_call_name,
                function_args=tool_call_args,
                query=query,
                tool_call_id=tool_call.id,
                raw_data_only=raw_data_only,
            )

        # ---------------------
        # 3) NEITHER query NOR tool
        # ---------------------
        return {"error": "Either 'query' or 'tool' must be provided in the parameters."}
