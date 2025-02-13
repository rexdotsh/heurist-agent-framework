import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, Any

load_dotenv()

from mesh.elfa_twitter_intelligence_agent import ElfaAPI

def save_results(results: Dict[str, Any], filename: str):
    """Save test results in a structured YAML file."""
    script_dir = Path(__file__).parent
    current_file = Path(__file__).stem
    base_filename = f"{current_file}_example"
    output_file = script_dir / f"{base_filename}.yaml"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(results, f, allow_unicode=True, sort_keys=False)
    
    print(f"Test results saved to {output_file}")

def format_results(test_name: str, description: str, input_data: Any, output_data: Any):
    """Structure results in a human-readable format."""
    return {
        "test_name": test_name,
        "description": description,
        "input": input_data,
        "output": output_data,
        "timestamp": datetime.utcnow().isoformat()
    }

def run_tests():
    """Execute API tests and store results in a structured format."""
    client = ElfaAPI()
    test_results = {}
    
    try:
        print("Running API tests...")
        
        # Test 1: API Connectivity Check
        # test_results['ping_test'] = format_results(
        #     "Ping Test",
        #     "Checks if the API is reachable and responding correctly.",
        #     None,
        #     client.ping()
        # )
        
        # # Test 2: API Key Status Verification
        # test_results['key_status_test'] = format_results(
        #     "API Key Status",
        #     "Retrieves current API key status, usage, and request limits.",
        #     None,
        #     client.get_key_status()
        # )
        
        # Test 3: Search Mentions
        search_input = {
            'keywords': ['Heurist', 'HEU', 'gheurist', '$HEU', '@heurist_ai'],
            'days_ago': 30,
            'limit': 20
        }
        test_results['search_mentions_test'] = format_results(
            "Search Mentions",
            "Fetches mentions of specific keywords within a given timeframe.",
            search_input,
            client.search_mentions(**search_input)
        )
        
        # Test 4: Retrieve Mentions
        mentions_input = { 'limit': 100, 'offset': 0 }
        test_results['get_mentions_test'] = format_results(
            "Get Mentions",
            "Retrieves recent mentions from the database.",
            mentions_input,
            client.get_mentions(**mentions_input)
        )
        
        # Test 5: Fetch Top Mentions
        top_mentions_input = {
            'ticker': '$HEU', 'time_window': '7d', 'page': 1, 'page_size': 50, 'include_account_details': False
        }
        test_results['top_mentions_test'] = format_results(
            "Top Mentions",
            "Finds the most discussed mentions for a specific ticker.",
            top_mentions_input,
            client.get_top_mentions(**top_mentions_input)
        )
        
        # Test 6: Fetch Trending Tokens
        trending_input = {
            'time_window': '24h', 'page': 1, 'page_size': 50, 'min_mentions': 5
        }
        test_results['trending_tokens_test'] = format_results(
            "Trending Tokens",
            "Lists tokens trending within the last 24 hours based on mentions.",
            trending_input,
            client.get_trending_tokens(**trending_input)
        )
        
        # Test 7: Fetch Account Smart Stats
        stats_input = { 'username': 'heurist_ai' }
        test_results['account_stats_test'] = format_results(
            "Account Smart Stats",
            "Retrieves engagement and analytics for a specific account.",
            stats_input,
            client.get_account_smart_stats(**stats_input)
        )
        
        # Save results
        save_results(test_results, 'elfa_api_test_results')
        print("All tests executed successfully.")
    
    except Exception as e:
        print(f"Error encountered: {str(e)}")
        test_results['error'] = str(e)
        save_results(test_results, 'elfa_api_test_results_error')

if __name__ == "__main__":
    run_tests()
