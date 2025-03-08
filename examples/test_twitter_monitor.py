import os
import sys

import dotenv

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from interfaces.twitter_reply import QueueManager, TwitterSearchMonitor


def main():
    dotenv.load_dotenv()

    queue = QueueManager()
    monitor = TwitterSearchMonitor(api_key=os.getenv("TWITTER_SEARCH_API_KEY"), queue_manager=queue)

    # Set search terms
    monitor.set_search_terms(["@heurist_ai"])

    monitor.process_mentions()


if __name__ == "__main__":
    main()
