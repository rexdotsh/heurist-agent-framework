import os
import json
import dotenv
from interfaces.twitter_reply import TwitterSearchMonitor, QueueManager

def main():
    dotenv.load_dotenv()
    
    queue = QueueManager()
    monitor = TwitterSearchMonitor(
        api_key=os.getenv("TWITTER_SEARCH_API_KEY"),
        queue_manager=queue
    )
    
    # Set search terms
    monitor.set_search_terms(["@heurist_ai"])
    
    # Test fetch_tweets
    print("\n=== Raw API Response ===")
    response = monitor.fetch_tweets()
    print(json.dumps(response, indent=2))
    
    # Test process_mentions
    print("\n=== Filtered Candidates ===")
    candidates = monitor.process_mentions()
    print(json.dumps(candidates, indent=2))

if __name__ == "__main__":
    main()
