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
    'metrics': {
        'query_type': 'metrics',
        'query': 'Get market cap, liquidity and trade volume for the specified token',
        'parameters': {
            'token_address': '98mb39tPFKQJ4Bif8iVg9mYb9wsfPZgpgN1sxoVTpump',
            'usdc_address': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v'
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
            'token_address': '2Z4FzKBcw48KBD2PaR4wtxo4sYGbS7QqTQCLoQnUpump'
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
            'token_address': 'FbhypAF9LL93bCZy9atRRfbdBMyJAwBarULfCK3roP93'
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
                    'token_info': holder['token_info']
                }
                for holder in agent_output.get('data', {}).get('holders', [])[:10]
            ]
        }
    elif query_name == 'buyers':
        base_result['output']['data'] = {
            'buyers': [
                {
                    'owner': buyer['owner'],
                    'amount': buyer['amount'],
                    'time': buyer['time']
                }
                for buyer in agent_output.get('data', {}).get('buyers', [])[:100]
            ]
        }
    elif query_name == 'holder_status':
        base_result['output']['data'] = {
            'holder_statuses': [
                {
                    'owner': status['owner'],
                    'current_balance': status['current_balance']
                }
                for status in agent_output.get('data', {}).get('holder_statuses', [])
            ]
        }
    elif query_name == 'top_traders':
        base_result['output']['data'] = {
            'traders': [
                {
                    'owner': trader['owner'],
                    'bought': trader['bought'],
                    'sold': trader['sold'],
                    'total_volume': trader['total_volume'],
                    'volume_usd': trader['volume_usd']
                }
                for trader in agent_output.get('data', {}).get('traders', [])[:100]
            ]
        }
    else:  # metrics query
        base_result['output']['data'] = agent_output.get('data', {})

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

async def run_queries(query_type: str = 'all'):
    """Run queries based on the specified type."""
    results = {}
    script_dir = Path(__file__).parent
    current_file = Path(__file__).stem
    base_filename = f"{current_file}_example"
    output_file = script_dir / f"{base_filename}.yaml"

    async with PumpFunTokenAgent() as agent:
        try:
            if query_type.lower() == 'all':
                for query_name in QUERIES.keys():
                    results[query_name] = await run_single_query(agent, query_name)
            elif query_type in QUERIES:
                results[query_type] = await run_single_query(agent, query_type)
            else:
                raise ValueError(f"Invalid query type. Must be one of: {', '.join(QUERIES.keys())} or 'all'")

            await save_results(results, output_file)

        except Exception as e:
            print(f"Error executing queries: {str(e)}")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Run PumpFun token analysis queries')
    parser.add_argument('--query-type', type=str, default='all',
                      help='Type of query to run. Options: creation, metrics, holders, buyers, holder_status, top_traders, all')
    
    args = parser.parse_args()
    
    asyncio.run(run_queries(args.query_type))

if __name__ == "__main__":
    import argparse
    main()