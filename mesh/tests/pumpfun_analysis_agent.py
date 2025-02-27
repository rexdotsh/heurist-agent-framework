import sys
from pathlib import Path
import yaml
import os
from dotenv import load_dotenv
import asyncio
import aiohttp
from typing import Dict, Any

sys.path.append(str(Path(__file__).parent.parent.parent))

from mesh.pumpfun_analysis_agent import PumpFunTokenAgent

load_dotenv()


QUERIES = {
    'creation': {
        'query_type': 'creation',
        'query': 'Show me the latest Solana token creations in the last hour',
        'parameters': {
            'interval': 'hours',
            'offset': 1
        }
    },
    'metrics_usdc': {
        'query_type': 'metrics',
        'query': 'Get market cap, liquidity and trade volume for the specified token using USDC pair',
        'parameters': {
            'token_address': '98mb39tPFKQJ4Bif8iVg9mYb9wsfPZgpgN1sxoVTpump',
            'quote_token': 'usdc'
        }
    },
    'metrics_sol': {
        'query_type': 'metrics',
        'query': 'Get market cap, liquidity and trade volume for the specified token using SOL pair',
        'parameters': {
            'token_address': '98mb39tPFKQJ4Bif8iVg9mYb9wsfPZgpgN1sxoVTpump',
            'quote_token': 'sol'
        }
    },
    'metrics_virtual': {
        'query_type': 'metrics',
        'query': 'Get market cap, liquidity and trade volume for the specified token using Virtual pair',
        'parameters': {
            'token_address': '2GxdEZQ5d9PsUqyGy43qv4fmNJWrnLp6qY4dTyNepump',
            'quote_token': 'virtual'
        }
    },
    'holders': {
        'query_type': 'holders',
        'query': 'Show me the top token holders and their balances',
        'parameters': {
            'token_address': '2GxdEZQ5d9PsUqyGy43qv4fmNJWrnLp6qY4dTyNepump'
        }
    },
    'buyers': {
        'query_type': 'buyers',
        'query': 'Show me the first 100 buyers of this token',
        'parameters': {
            'token_address': '2Z4FzKBcw48KBD2PaR4wtxo4sYGbS7QqTQCLoQnUpump',
            'limit': 100
        }
    },
    'holder_status': {
        'query_type': 'holder_status',
        'query': 'Check if the first 100 buyers are still holding, sold all, or bought more',
        'parameters': {
            'token_address': '2Z4FzKBcw48KBD2PaR4wtxo4sYGbS7QqTQCLoQnUpump',
            'buyer_addresses': [
                "ApRJBQEKfmcrViQkH94BkzRFUGWtA8uC71DXu6USdd3n",
                "9nG4zw1jVJFpEtSLmbGQpTnpG2TiKfLXWkkTyyRvxTt6"
            ]
        }
    },
    'top_traders': {
        'query_type': 'top_traders',
        'query': 'Show me the top traders for this token on Pump Fun DEX',
        'parameters': {
            'token_address': 'FbhypAF9LL93bCZy9atRRfbdBMyJAwBarULfCK3roP93',
            'limit': 100
        }
    }
}

async def format_query_result(query_name: str, agent_output: Dict[str, Any]) -> Dict[str, Any]:
    """Format the query results based on query type."""
    base_result = {
        'input': QUERIES[query_name],
        'output': {
            'response': agent_output.get('response', ''),
            'data': {}
        }
    }

    if 'error' in agent_output:
        return {
            'input': QUERIES[query_name],
            'error': str(agent_output['error'])
        }

    if query_name == 'creation':
        base_result['output']['data'] = {
            'tokens': [
                {
                    'name': token['token_info']['name'],
                    'symbol': token['token_info']['symbol'],
                    'mint_address': token['token_info']['mint_address'],
                    'amount': token['amount'],
                    'signer': token['signer']
                }
                for token in agent_output.get('data', {}).get('tokens', [])[:10]
            ]
        }
    elif query_name == 'holders':
        base_result['output']['data'] = {
            'holders': [
                {
                    'address': holder['address'],
                    'holding': holder['holding'],
                    'percentage_of_supply': holder.get('percentage_of_supply', 0),
                    'token_info': holder['token_info']
                }
                for holder in agent_output.get('data', {}).get('holders', [])[:10]
            ],
            'total_supply': agent_output.get('data', {}).get('total_supply', 0)
        }
    elif query_name == 'buyers':
        base_result['output']['data'] = {
            'buyers': [
                {
                    'owner': buyer['owner'],
                    'amount': buyer['amount'],
                    'amount_usd': buyer.get('amount_usd', 0),
                    'time': buyer['time'],
                    'currency_pair': buyer.get('currency_pair', '')
                }
                for buyer in agent_output.get('data', {}).get('buyers', [])[:100]
            ],
            'unique_buyer_count': agent_output.get('data', {}).get('unique_buyer_count', 0)
        }
    elif query_name == 'holder_status':
        base_result['output']['data'] = {
            'holder_statuses': [
                {
                    'owner': status['owner'],
                    'current_balance': status['current_balance'],
                    'initial_balance': status.get('initial_balance', 0),
                    'status': status.get('status', 'unknown')
                }
                for status in agent_output.get('data', {}).get('holder_statuses', [])
            ],
            'summary': agent_output.get('data', {}).get('summary', {})
        }
    elif query_name == 'top_traders':
        base_result['output']['data'] = {
            'traders': [
                {
                    'owner': trader['owner'],
                    'bought': trader['bought'],
                    'sold': trader['sold'],
                    'buy_sell_ratio': trader.get('buy_sell_ratio', 0),
                    'total_volume': trader['total_volume'],
                    'volume_usd': trader['volume_usd'],
                    'transaction_count': trader.get('transaction_count', 0)
                }
                for trader in agent_output.get('data', {}).get('traders', [])[:100]
            ],
            'markets': agent_output.get('data', {}).get('markets', [])
        }
    elif query_name.startswith('metrics'):
        # Handle all metrics query types
        base_result['output']['data'] = agent_output.get('data', {})
        # Check if fallback was used
        if 'fallback_used' in agent_output.get('data', {}):
            base_result['output']['fallback_used'] = agent_output['data']['fallback_used']

    return base_result

async def run_single_query(agent: PumpFunTokenAgent, query_name: str) -> Dict[str, Any]:
    """Run a single query using the PumpFunTokenAgent."""
    try:
        agent_input = QUERIES[query_name]
        agent_output = await agent.handle_message(agent_input)
        return await format_query_result(query_name, agent_output)
    except Exception as e:
        return {
            'input': QUERIES[query_name],
            'error': str(e)
        }

async def save_results(results: Dict[str, Any], output_file: Path):
    """Save results to a YAML file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(results, f, allow_unicode=True, sort_keys=False)
    except Exception as e:
        print(f"Error saving results: {str(e)}")

async def run_queries(query_type: str = 'all', query_params: Dict = None):
    """Run queries based on the specified type with optional custom parameters."""
    results = {}
    script_dir = Path(__file__).parent
    current_file = Path(__file__).stem
    base_filename = f"{current_file}_example"
    output_file = script_dir / f"{base_filename}.yaml"
    
    query_params = query_params or {}

    async with PumpFunTokenAgent() as agent:
        try:
            if query_type.lower() == 'all':
                for query_name in QUERIES.keys():
                    print(f"Running query: {query_name}")
                    if query_name in query_params:
                        QUERIES[query_name]['parameters'].update(query_params[query_name])
                    results[query_name] = await run_single_query(agent, query_name)
            elif query_type in QUERIES:
                if query_type in query_params:
                    QUERIES[query_type]['parameters'].update(query_params[query_type])
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
    # - creation (interval=hours|days, offset=1-99)
    # - metrics_usdc (token_address=address, quote_token=usdc)
    # - metrics_sol (token_address=address, quote_token=sol)
    # - metrics_virtual (token_address=address, quote_token=virtual)
    # - holders (token_address=address)
    # - buyers (token_address=address, limit=1-100)
    # - holder_status (token_address=address, buyer_addresses=[addr1,addr2,...])
    # - top_traders (token_address=address, limit=1-100)
    # - all (runs all queries with default parameters)
    # 
    print(f"Running query type: {agent_query}")
    asyncio.run(run_queries(agent_query))

if __name__ == "__main__":
    main()