import datetime
import json
import os
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from web3 import Web3

from core.llm import call_llm_async, call_llm_with_tools_async
from decorators import monitor_execution, with_cache, with_retry

from .mesh_agent import MeshAgent

load_dotenv()

# -----------------------------------------------------------------------------
# Helper functions and dummy implementations for Aave V3 reserve API
# -----------------------------------------------------------------------------


def named_tree(struct_definition, data):
    """
    Placeholder function that converts raw tuple data into a dictionary
    based on a given structure definition. In a full implementation, this
    would decode the anonymous tuple into named fields.
    """
    return data


# In a real implementation, you would import these from the Aave V3 library:
# from eth_defi.aave_v3.reserve import fetch_reserve_data, get_helper_contracts
#
# For this example, we define minimal dummy implementations.


class DummyFunctionCall:
    def __init__(self):
        self.abi = {
            "outputs": [
                {"components": []},  # AggregatedReserveData structure
                {"components": []},  # BaseCurrencyInfo structure
            ]
        }

    def call(self, block_identifier=None):
        # Return empty dummy data
        return ([], {})


class DummyFunctions:
    def getReservesData(self, address):
        return DummyFunctionCall()


class DummyProvider:
    def __init__(self):
        self.address = "0x0000000000000000000000000000000000000000"
        self.functions = DummyFunctions()


class DummyContracts:
    def __init__(self):
        self.ui_pool_data_provider = DummyProvider()
        self.pool_addresses_provider = DummyProvider()


def get_helper_contracts(web3):
    """
    In a real implementation, this would initialize and return the Aave
    helper contracts using the provided Web3 instance.
    """
    return DummyContracts()


def fetch_reserve_data(
    contracts,
    block_identifier=None,
) -> Tuple[List[Dict], Dict]:
    """
    Fetches data for all Aave V3 reserves using the helper contracts.

    :param contracts: Helper contracts needed to pull the data.
    :param block_identifier: (Optional) Block number or identifier.
    :return: A tuple of (list of aggregated reserve data, base currency info).
    """
    func = contracts.ui_pool_data_provider.functions.getReservesData(contracts.pool_addresses_provider.address)
    aggregated_reserve_data, base_currency_info = func.call(block_identifier=block_identifier)

    outputs = func.abi["outputs"]
    AggregatedReserveData = outputs[0]["components"]
    BaseCurrencyInfo = outputs[1]["components"]

    aggregated_reserve_data_decoded = [named_tree(AggregatedReserveData, a) for a in aggregated_reserve_data]
    base_currency_info_decoded = named_tree(BaseCurrencyInfo, base_currency_info)
    return aggregated_reserve_data_decoded, base_currency_info_decoded


# -----------------------------------------------------------------------------
# Aave Agent Implementation
# -----------------------------------------------------------------------------


class Aaveagent(MeshAgent):
    def __init__(self):
        super().__init__()
        self.metadata.update(
            {
                "name": "Aaveagent",
                "version": "1.0.0",
                "author": "Heurist team",
                "author_address": "0x7d9d1821d15B9e0b8Ab98A058361233E255E405D",
                "description": "This agent fetches Aave V3 reserve data.",
                "inputs": [
                    {
                        "name": "query",
                        "description": "Natural language query about Aave reserves data. Optionally include a block identifier to fetch data at a specific block.",
                        "type": "str",
                        "required": True,
                    },
                    {
                        "name": "raw_data_only",
                        "description": "If true, the agent will only return the raw data without explanation.",
                        "type": "bool",
                        "required": False,
                        "default": False,
                    },
                ],
                "outputs": [
                    {
                        "name": "response",
                        "description": "Natural language explanation of the Aave reserve data",
                        "type": "str",
                    },
                    {"name": "data", "description": "Structured Aave reserve data", "type": "dict"},
                ],
                "external_apis": ["Aave V3 Reserve Data"],
                "tags": ["Data", "Aave", "DeFi", "Reserves"],
            }
        )
        # Initialize a Web3 provider using an endpoint from environment variables.
        self.web3 = Web3(Web3.HTTPProvider(os.getenv("WEB3_PROVIDER")))

    def get_system_prompt(self) -> str:
        return (
            "You are a specialized assistant that analyzes Aave V3 reserve data. Your responses should be clear, concise, and data-driven. "
            "Answer the user's query based on the Aave reserve data. If some data is missing, simply say you don't know."
        )

    def get_tool_schemas(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_reserve_data",
                    "description": "Get Aave V3 reserve data, optionally at a specific block identifier.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "block_identifier": {
                                "type": "string",
                                "description": "Optional block number or identifier to fetch data at a specific time.",
                            }
                        },
                        "required": [],
                    },
                },
            }
        ]

    async def _respond_with_llm(self, query: str, tool_call_id: str, data: dict, temperature: float) -> str:
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

    def _handle_error(self, maybe_error: dict) -> dict:
        if "error" in maybe_error:
            return {"error": maybe_error["error"]}
        return {}

    @with_cache(ttl_seconds=300)  # Cache results for 5 minutes
    async def get_reserve_data(self, block_identifier: str = None) -> Dict:
        try:
            contracts = get_helper_contracts(self.web3)
            aggregated_reserve_data, base_currency_info = fetch_reserve_data(contracts, block_identifier)
            summary = {
                "total_reserves": len(aggregated_reserve_data),
                "base_currency_info": base_currency_info,
                "last_updated": datetime.datetime.utcnow().isoformat(),
            }
            return {
                "summary": summary,
                "detailed_data": {
                    "aggregated_reserve_data": aggregated_reserve_data,
                    "base_currency_info": base_currency_info,
                },
            }
        except Exception as e:
            return {"error": f"Failed to fetch reserve data: {str(e)}"}

    async def _handle_tool_logic(
        self, tool_name: str, function_args: dict, query: str, tool_call_id: str, raw_data_only: bool
    ) -> Dict[str, Any]:
        temp = 0.7

        if tool_name == "get_reserve_data":
            block_identifier = function_args.get("block_identifier")
            result = await self.get_reserve_data(block_identifier)
        else:
            return {"error": f"Unsupported tool: {tool_name}"}

        errors = self._handle_error(result)
        if errors:
            return errors

        if raw_data_only:
            return {"response": "", "data": result}

        explanation = await self._respond_with_llm(
            query=query, tool_call_id=tool_call_id, data=result, temperature=temp
        )

        return {"response": explanation, "data": result}

    @monitor_execution()
    @with_retry(max_retries=3)
    async def handle_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
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
                return {"response": response["content"], "data": {}}

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
