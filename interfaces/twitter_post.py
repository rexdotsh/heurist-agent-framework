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
from agents.core_agent import CoreAgent
from platforms.twitter_api import tweet_with_image, tweet_text_only
import asyncio

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
IMAGE_GENERATION_PROBABILITY = 1
TWEET_HISTORY_FILE = "tweet_history.json"
DRYRUN = False#os.getenv("DRYRUN")

if DRYRUN:
    print("DRYRUN MODE: Not posting real tweets")
else:
    print("LIVE MODE: Will post real tweets")

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

class TwitterAgent(CoreAgent):
    def __init__(self, core_agent=None):
        if core_agent:
            super().__setattr__('_parent', core_agent)
        else:
            super().__setattr__('_parent', self)
            super().__init__()
        
        # Initialize twitter specific stuff
        self.history_manager = TweetHistoryManager()
        self.register_interface('twitter', self)

    def __getattr__(self, name):
        return getattr(self._parent, name)
        
    def __setattr__(self, name, value):
        if not hasattr(self, '_parent'):
            super().__setattr__(name, value)
        elif name == "_parent" or self is self._parent or name in self.__dict__:
            super().__setattr__(name, value)
        else:
            setattr(self._parent, name, value)
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
    
    async def generate_tweet(self) -> tuple[str | None, str | None, dict | None]:
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
            
            # try:
            #     ideas = call_llm(
            #         HEURIST_BASE_URL, 
            #         HEURIST_API_KEY, 
            #         SMALL_MODEL_ID, 
            #         self.prompt_config.get_system_prompt(), 
            #         user_prompt, 
            #         0.9
            #     )
            #     tweet_data['metadata']['ideas_instruction'] = instruction_tweet_idea
            #     tweet_data['metadata']['ideas'] = ideas
            # except LLMError as e:
            #     logger.warning(f"Failed to generate ideas: {str(e)}")
            #     ideas = None
            ideas = None
            tweet_data['metadata']['ideas_instruction'] = instruction_tweet_idea
            ideas, _ = await self.handle_message(instruction_tweet_idea, source_interface='twitter')
            tweet_data['metadata']['ideas'] = ideas
            
            # Generate final tweet
            user_prompt = (prompt + self.prompt_config.get_twitter_rules() + 
                          self.format_context(past_tweets) + 
                          self.format_tweet_instruction(basic_options, style_options, ideas))
            

            tweet, _ = await self.handle_message(user_prompt, source_interface='twitter')
            # tweet = call_llm(
            #     HEURIST_BASE_URL, 
            #     HEURIST_API_KEY, 
            #     LARGE_MODEL_ID, 
            #     self.prompt_config.get_system_prompt(), 
            #     user_prompt, 
            #     0.9
            # )
            # if not tweet:
            #     raise LLMError("Empty tweet generated")
            
            # Clean and store tweet
            
            #tweet = tweet.replace('"', '')
            tweet = tweet.replace('"', '')
            tweet_data['tweet'] = tweet

            # Image generation
            image_url = None
            if random.random() < IMAGE_GENERATION_PROBABILITY:
                logger.info("Generating image")
                try:
                    image_prompt = await self.generate_image_prompt(tweet)
                    image_url = await self.handle_image_generation(image_prompt)
                    tweet_data['metadata']['image_prompt'] = image_prompt
                    tweet_data['metadata']['image_url'] = image_url
                except Exception as e:
                    logger.warning(f"Failed to generate image: {str(e)}")
            
            return tweet, image_url, tweet_data

        except Exception as e:
            logger.error(f"Unexpected error in tweet generation: {str(e)}")
        
        return None, None, None

    def run(self):
        """Start the Twitter bot"""
        logger.info("Starting Twitter bot...")
        asyncio.run(self._run())

    async def _run(self):
        while True:
            try:
                tweet, image_url, tweet_data = await self.generate_tweet()
                
                if tweet:
                    if not DRYRUN:
                        if image_url:
                            tweet_id = tweet_with_image(tweet, image_url)
                            logger.info("Successfully posted tweet with image: %s", tweet)
                        else:
                            tweet_id = tweet_text_only(tweet)
                            logger.info("Successfully posted tweet: %s", tweet)
                        tweet_data['metadata']['tweet_id'] = tweet_id
                        for interface_name, interface in self.interfaces.items():
                            if interface_name == 'telegram':
                                await self.send_to_interface(interface_name, {
                                    'type': 'message',
                                    'content': "Just posted a tweet: " + tweet_id,
                                    'image_url': None,
                                    'source': 'twitter',
                                    'chat_id': "0"
                                })
                    else:
                        logger.info("Generated tweet: %s", tweet)
                    
                    self.history_manager.add_tweet(tweet_data)
                    wait_time = random_interval()
                else:
                    logger.error("Failed to generate tweet")
                    wait_time = 10
                
                next_time = datetime.now() + timedelta(seconds=wait_time)
                logger.info("Next tweet will be posted at: %s", next_time.strftime('%H:%M:%S'))
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                logger.error("Error occurred: %s", str(e))
                await asyncio.sleep(10)
                continue

def random_interval():
    """Generate a random interval between 1 and 2 hours in seconds"""
    return random.uniform(3600, 7200)

def main():
    agent = TwitterAgent()
    agent.run()

if __name__ == "__main__":
    try:
        logger.info("Starting Twitter agent...")
        main()
    except KeyboardInterrupt:
        logger.info("\nTwitter agent stopped by user")
    except Exception as e:
        logger.error("Fatal error: %s", str(e))