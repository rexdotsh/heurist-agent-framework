import logging
from interfaces.twitter_post import TweetGenerator
import dotenv
import os
import random
import time
from datetime import datetime, timedelta
from platforms.twitter_api import tweet_with_image, tweet_text_only

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main entry point for the Heuman Agent Framework.
    Runs the Twitter agent for automated tweeting.
    """
    try:
        # Load environment variables
        dotenv.load_dotenv()
        
        # Initialize and run Twitter agent
        logger.info("Starting Twitter agent...")
        tweet_generator = TweetGenerator()
        DRYRUN = os.getenv("DRYRUN")
        while DRYRUN:
            try:
                tweet, image_url, tweet_data = tweet_generator.generate_tweet()
                
                if tweet:
                    if not DRYRUN:
                        if image_url:
                            tweet_id = tweet_with_image(tweet, image_url)
                            logger.info("Successfully posted tweet with image: %s", tweet)
                        else:
                            tweet_id = tweet_text_only(tweet)
                            logger.info("Successfully posted tweet: %s", tweet)
                        tweet_data['metadata']['tweet_id'] = tweet_id
                    else:
                        logger.info("DRYRUN - Generated tweet: %s", tweet)
                    
                    tweet_generator.history_manager.add_tweet(tweet_data)
                    wait_time = random.uniform(3600, 7200)  # 1-2 hours
                else:
                    logger.error("Failed to generate tweet")
                    wait_time = 10
                
                next_time = datetime.now() + timedelta(seconds=wait_time)
                logger.info("Next tweet scheduled for: %s", next_time.strftime('%H:%M:%S'))
                time.sleep(wait_time)
                
            except Exception as e:
                logger.error("Error in tweet generation loop: %s", str(e))
                time.sleep(10)
                continue
                
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
