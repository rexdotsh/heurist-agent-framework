import json
import logging
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any
import dotenv
import yaml
from core.llm import call_llm, LLMError
from platforms.twitter_api import tweet_with_image, tweet_text_only
from core.imgen import generate_image_with_retry, generate_image_prompt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

# Constants
HEURIST_BASE_URL = "https://llm-gateway.heurist.xyz"
HEURIST_API_KEY = os.getenv("HEURIST_API_KEY")
LARGE_MODEL_ID = os.getenv("LARGE_MODEL_ID")
SMALL_MODEL_ID = os.getenv("SMALL_MODEL_ID")
TWEET_WORD_LIMITS = [15, 20, 30, 35]
IMAGE_GENERATION_PROBABILITY = 0.3
TWEET_HISTORY_FILE = "tweet_history.json"
DRYRUN = os.getenv("DRYRUN")

if DRYRUN:
    print("DRYRUN MODE: Not posting real tweets")
else:
    print("LIVE MODE: Will post real tweets")

class PromptConfig:
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Get the project root directory (2 levels up from the current file)
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "prompts.yaml"
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> dict:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise

    def get_system_prompt(self) -> str:
        return self.config['system']['base']

    def get_basic_settings(self) -> list:
        return self.config['character']['basic_settings']

    def get_interaction_styles(self) -> list:
        return self.config['character']['interaction_styles']

    def get_basic_prompt_template(self) -> str:
        return self.config['templates']['basic_prompt']

    def get_tweet_instruction_template(self) -> str:
        return self.config['templates']['tweet_instruction']

    def get_context_twitter_template(self) -> str:
        return self.config['templates']['context_twitter']

    def get_tweet_ideas(self) -> list:
        return self.config['tweet_ideas']['options']

    def get_twitter_rules(self) -> str:
        return self.config['rules']['twitter']

class TweetHistoryManager:
    def __init__(self, history_file=TWEET_HISTORY_FILE):
        self.history_file = history_file
        self.history = self.load_history()

    def load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Error reading {self.history_file}, starting fresh")
                return []
        return []

    def add_tweet(self, tweet, metadata=None):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'tweet': tweet
        }
        if metadata:
            entry.update(metadata)
        
        entry = json.loads(json.dumps(entry, ensure_ascii=False))
        self.history.append(entry)
        self.save_history()

    def save_history(self):
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def get_recent_tweets(self, n=6):
        return [entry['tweet']['tweet'] for entry in self.history[-n:]]

class TweetGenerator:
    def __init__(self):
        self.prompt_config = PromptConfig()
        self.history_manager = TweetHistoryManager()

    def fill_basic_prompt(self, basic_options, style_options):
        return self.prompt_config.get_basic_prompt_template().format(
            basic_option_1=basic_options[0],
            basic_option_2=basic_options[1],
            style_option_1=style_options[0],
            style_option_2=style_options[1]
        )

    def format_tweet_instruction(self, basic_options, style_options, ideas=None):
        decoration_ideas = f"Ideas: {ideas}" if ideas else "\n"
        num_words = random.choice(TWEET_WORD_LIMITS)
        
        return self.prompt_config.get_tweet_instruction_template().format(
            basic_option_1=basic_options[0],
            basic_option_2=basic_options[1],
            style_option_1=style_options[0],
            style_option_2=style_options[1],
            decoration_ideas=decoration_ideas,
            num_words=num_words,
            rules=self.prompt_config.get_twitter_rules()
        )

    def format_context(self, tweets):
        if tweets is None:
            tweets = []
        return self.prompt_config.get_context_twitter_template().format(tweets=tweets)

    def generate_tweet(self) -> tuple[str | None, str | None, dict | None]:
        """Generate a tweet with improved error handling"""
        tweet_data: Dict[str, Any] = {'metadata': {}}
        
        try:
            # Get recent tweets for context
            past_tweets = self.history_manager.get_recent_tweets()
            
            # Generate randomized prompt
            basic_options = random.sample(self.prompt_config.get_basic_settings(), 2)
            style_options = random.sample(self.prompt_config.get_interaction_styles(), 2)
            tweet_data['metadata'].update({
                'basic_options': basic_options,
                'style_options': style_options
            })
            
            prompt = self.fill_basic_prompt(basic_options, style_options)
            
            # Generate ideas
            instruction_tweet_idea = random.choice(self.prompt_config.get_tweet_ideas())
            user_prompt = (prompt + self.prompt_config.get_twitter_rules() + 
                          self.format_context(past_tweets) + instruction_tweet_idea)
            
            try:
                ideas = call_llm(
                    HEURIST_BASE_URL, 
                    HEURIST_API_KEY, 
                    SMALL_MODEL_ID, 
                    self.prompt_config.get_system_prompt(), 
                    user_prompt, 
                    0.9
                )
                tweet_data['metadata']['ideas_instruction'] = instruction_tweet_idea
                tweet_data['metadata']['ideas'] = ideas
            except LLMError as e:
                logger.warning(f"Failed to generate ideas: {str(e)}")
                ideas = None
            
            # Generate final tweet
            user_prompt = (prompt + self.prompt_config.get_twitter_rules() + 
                          self.format_context(past_tweets) + 
                          self.format_tweet_instruction(basic_options, style_options, ideas))
            
            tweet = call_llm(
                HEURIST_BASE_URL, 
                HEURIST_API_KEY, 
                LARGE_MODEL_ID, 
                self.prompt_config.get_system_prompt(), 
                user_prompt, 
                0.9
            )
            if not tweet:
                raise LLMError("Empty tweet generated")
            
            # Clean and store tweet
            tweet = tweet.replace('"', '')
            tweet_data['tweet'] = tweet

            # Image generation
            image_url = None
            if random.random() < IMAGE_GENERATION_PROBABILITY:
                try:
                    image_prompt = generate_image_prompt(tweet)
                    image_url = generate_image_with_retry(image_prompt)
                    tweet_data['metadata']['image_prompt'] = image_prompt
                    tweet_data['metadata']['image_url'] = image_url
                except Exception as e:
                    logger.warning(f"Failed to generate image: {str(e)}")
            
            return tweet, image_url, tweet_data
            
        except LLMError as e:
            logger.error(f"Failed to generate tweet due to LLMError: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in tweet generation: {str(e)}")
        
        return None, None, None

def random_interval():
    """Generate a random interval between 1 and 2 hours in seconds"""
    return random.uniform(3600, 7200)

def main():
    generator = TweetGenerator()
    while True:
        try:
            tweet, image_url, tweet_data = generator.generate_tweet()
            
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
                    logger.info("Generated tweet: %s", tweet)
                
                generator.history_manager.add_tweet(tweet_data)
                wait_time = random_interval()
            else:
                logger.error("Failed to generate tweet")
                wait_time = 10
            
            next_time = datetime.now() + timedelta(seconds=wait_time)
            logger.info("Next tweet will be posted at: %s", next_time.strftime('%H:%M:%S'))
            time.sleep(wait_time)
            
        except Exception as e:
            logger.error("Error occurred: %s", str(e))
            time.sleep(10)
            continue

if __name__ == "__main__":
    try:
        logger.info("Starting tweet automation...")
        main()
    except KeyboardInterrupt:
        logger.info("\nTweet automation stopped by user")
    except Exception as e:
        logger.error("Fatal error: %s", str(e))