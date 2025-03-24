import asyncio
import sys
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent.parent))
from mesh.pumpfun_token_agent import PumpFunTokenAgent  # noqa: E402

load_dotenv()


QUERIES = {
    "creation": {"query": "Show me the latest Solana token creations in the last hour"},
    "metrics_usdc": {
        "query": "Get market cap, liquidity and trade volume for 98mb39tPFKQJ4Bif8iVg9mYb9wsfPZgpgN1sxoVTpump using USDC pair"
    },
    "metrics_sol": {
        "query": "Get market cap, liquidity and trade volume for 98mb39tPFKQJ4Bif8iVg9mYb9wsfPZgpgN1sxoVTpump using SOL pair"
    },
    "metrics_virtual": {
        "query": "Get market cap, liquidity and trade volume for 98mb39tPFKQJ4Bif8iVg9mYb9wsfPZgpgN1sxoVTpump using Virtual pair"
    },
    "holders": {"query": "Show me the top token holders of EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"},
    "buyers": {"query": "Show me the first 100 buyers of 2Z4FzKBcw48KBD2PaR4wtxo4sYGbS7QqTQCLoQnUpump"},
    "holder_status": {
        "query": "Check if address 'Z9y8X7w6V5u4T3s2R1q0P9o8N7m6L5k4J3i2H1g0F9e8' still holds token 'A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0U1v2'"
    },
    "top_traders": {
        "query": "List the top traders of token 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v' on Pump Fun DEX"
    },
}


async def format_query_result(query_name: str, agent_output: Dict[str, Any]) -> Dict[str, Any]:
    """Format the query results based on query type."""
    base_result = {"input": QUERIES[query_name], "output": {"response": agent_output.get("response", ""), "data": {}}}

    if "error" in agent_output:
        return {"input": QUERIES[query_name], "error": str(agent_output["error"])}

    if query_name == "creation":
        base_result["output"]["data"] = {
            "tokens": [
                {
                    "name": token["token_info"]["name"],
                    "symbol": token["token_info"]["symbol"],
                    "mint_address": token["token_info"]["mint_address"],
                    "amount": token["amount"],
                    "signer": token["signer"],
                }
                for token in agent_output.get("data", {}).get("tokens", [])[:10]
            ]
        }
    elif query_name == "holders":
        base_result["output"]["data"] = {
            "holders": [
                {
                    "address": holder["address"],
                    "holding": holder["holding"],
                    "percentage_of_supply": holder.get("percentage_of_supply", 0),
                    "token_info": holder["token_info"],
                }
                for holder in agent_output.get("data", {}).get("holders", [])[:10]
            ],
            "total_supply": agent_output.get("data", {}).get("total_supply", 0),
        }
    elif query_name == "buyers":
        base_result["output"]["data"] = {
            "buyers": [
                {
                    "owner": buyer["owner"],
                    "amount": buyer["amount"],
                    "amount_usd": buyer.get("amount_usd", 0),
                    "time": buyer["time"],
                    "currency_pair": buyer.get("currency_pair", ""),
                }
                for buyer in agent_output.get("data", {}).get("buyers", [])[:100]
            ],
            "unique_buyer_count": agent_output.get("data", {}).get("unique_buyer_count", 0),
        }
    elif query_name == "holder_status":
        # Updated to match the expected format in your example
        base_result["output"]["data"] = {
            "holder_statuses": [
                {
                    "owner": status["owner"],
                    "current_balance": status["current_balance"],
                    "initial_balance": status.get("initial_balance", 0),
                    "status": status.get("status", "unknown"),
                }
                for status in agent_output.get("data", {}).get("holder_statuses", [])
            ],
            "summary": agent_output.get("data", {}).get("summary", {}),
        }
    elif query_name == "top_traders":
        # Updated to match the expected format in your example
        base_result["output"]["data"] = {
            "traders": agent_output.get("data", {}).get("traders", []),
            "markets": agent_output.get("data", {}).get("markets", []),
        }
    elif query_name.startswith("metrics"):
        # Handle all metrics query types
        base_result["output"]["data"] = agent_output.get("data", {})
        # Check if fallback was used
        if "fallback_used" in agent_output.get("data", {}):
            base_result["output"]["fallback_used"] = agent_output["data"]["fallback_used"]

    return base_result


async def run_single_query(agent: PumpFunTokenAgent, query_name: str) -> Dict[str, Any]:
    """Run a single query using the PumpFunTokenAgent."""
    try:
        agent_input = QUERIES[query_name]
        agent_output = await agent.handle_message(agent_input)
        return await format_query_result(query_name, agent_output)
    except Exception as e:
        return {"input": QUERIES[query_name], "error": str(e)}


async def save_results(results: Dict[str, Any], output_file: Path):
    """Save results to a YAML file."""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(results, f, allow_unicode=True, sort_keys=False)
    except Exception as e:
        print(f"Error saving results: {str(e)}")


async def run_queries(query_type: str = "all", query_params: Dict = None):
    """Run queries based on the specified type with optional custom parameters."""
    results = {}
    script_dir = Path(__file__).parent
    current_file = Path(__file__).stem
    base_filename = f"{current_file}_example"
    output_file = script_dir / f"{base_filename}.yaml"

    query_params = query_params or {}

    async with PumpFunTokenAgent() as agent:
        try:
            if query_type.lower() == "all":
                for query_name in QUERIES.keys():
                    print(f"Running query: {query_name}")
                    if query_name in query_params:
                        # Update query if needed
                        if "query" in query_params[query_name]:
                            QUERIES[query_name]["query"] = query_params[query_name]["query"]
                    results[query_name] = await run_single_query(agent, query_name)
            elif query_type in QUERIES:
                if query_type in query_params:
                    # Update query if needed
                    if "query" in query_params[query_type]:
                        QUERIES[query_type]["query"] = query_params[query_type]["query"]
                results[query_type] = await run_single_query(agent, query_type)
            else:
                raise ValueError(f"Invalid query type. Must be one of: {', '.join(QUERIES.keys())} or 'all'")

            await save_results(results, output_file)
            print(f"Results saved to {output_file}")

        except Exception as e:
            print(f"Error executing queries: {str(e)}")


def main():
    """Main entry point for the script."""
    agent_query = "all"
    # Available agent_query options:
    # - creation (recent token creations)
    # - metrics_usdc (market metrics with USDC pair)
    # - metrics_sol (market metrics with SOL pair)
    # - metrics_virtual (market metrics with Virtual pair)
    # - holders (token holders analysis)
    # - buyers (first buyers analysis)
    # - holder_status (check if buyers are still holding)
    # - top_traders (top traders analysis)
    # - all (runs all queries)
    #
    print(f"Running query type: {agent_query}")
    asyncio.run(run_queries(agent_query))


if __name__ == "__main__":
    main()
