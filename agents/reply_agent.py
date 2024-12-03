import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict
import dotenv
import yaml
import platforms.twitter_api as twitter_api
from core.llm import LLMError, call_llm
from core.imgen import generate_image_convo_prompt, generate_image_with_retry

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

# Constants
HEURIST_BASE_URL = "https://llm-gateway.heurist.xyz"
HEURIST_API_KEY = os.getenv("HEURIST_API_KEY")
LARGE_MODEL_ID = os.getenv("LARGE_MODEL_ID")
SMALL_MODEL_ID = os.getenv("SMALL_MODEL_ID")
IMAGE_GENERATION_PROBABILITY = 0.5
REPLY_CHECK_INTERVAL = 60
RATE_LIMIT_SLEEP = 120

DRYRUN = os.getenv("DRYRUN")
if DRYRUN:
    print("DRYRUN MODE: Not posting real tweets")
else:
    print("LIVE MODE: Will post real tweets")

class PromptConfig:
    def __init__(self, config_path: str = "prompts.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Load YAML config file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise

    def get_system_prompt(self) -> str:
        return self.config['system']['base']

    def get_basic_prompt(self) -> str:
        return self.config['reply']['basic_prompt']

    def get_heurist_knowledge(self) -> str:
        return self.config['reply']['heurist_knowledge']

    def get_reply_to_comment_template(self) -> str:
        return self.config['reply']['reply_to_comment_template']

    def get_reply_to_tweet_template(self) -> str:
        return self.config['reply']['reply_to_tweet_template']

class ReplyDatabase:
    """Handles persistent storage of replies and responses"""
    def __init__(self, db_file: str = "reply_history.json"):
        self.db_file = Path(db_file)
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        if not self.db_file.exists():
            self._write_data({
                "processed_replies": {},
                "pending_replies": {}
            })
    
    def _read_data(self) -> Dict:
        try:
            with self.db_file.open('r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Error reading database. Creating new one.")
            return {"processed_replies": {}, "pending_replies": {}}
    
    def _write_data(self, data: Dict):
        try:
            with self.db_file.open('w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error writing to database: {str(e)}")

    def mark_as_processed(self, reply_id, ai_response, add_metadata):
        """Move reply from pending to processed and store AI response"""
        data = self._read_data()
        if reply_id in data["pending_replies"]:
            reply_data = data["pending_replies"].pop(reply_id)
            reply_data["ai_response"] = ai_response
            reply_data["processed_timestamp"] = datetime.now().isoformat()
            reply_data.update(add_metadata)
            data["processed_replies"][reply_id] = reply_data
            self._write_data(data)
            logger.info(f"Marked reply as processed: {reply_id}")

    def mark_as_failed(self, reply_id: str, failure: str):
        data = self._read_data()
        if reply_id in data["pending_replies"]:
            reply_data = data["pending_replies"].pop(reply_id)
            reply_data["failure"] = failure
            reply_data["processed_timestamp"] = datetime.now().isoformat()
            data["processed_replies"][reply_id] = reply_data
            self._write_data(data)
            logger.info(f"Marked reply as failed: {reply_id}")

    def get_pending_replies(self) -> Dict[str, Dict]:
        return self._read_data()["pending_replies"]
    
    def is_processed(self, reply_id: str) -> bool:
        data = self._read_data()
        return reply_id in data["processed_replies"]

class ReplyGenerator:
    def __init__(self):
        self.prompt_config = PromptConfig()

    def format_instruction_tweet(self, original_tweet=None, user_reply=None, author_name=None) -> str:
        """Format the appropriate instruction template based on input"""
        if user_reply:
            return self.prompt_config.get_reply_to_comment_template().format(
                original_tweet=original_tweet,
                user_reply=user_reply
            )
        else:
            return self.prompt_config.get_reply_to_tweet_template().format(
                original_tweet=original_tweet,
                author_name=author_name
            )

    def generate_reply(self, original_tweet, user_reply, author_name):
        """Generate a reply with possible image"""
        try:
            user_prompt = self.format_instruction_tweet(original_tweet, user_reply, author_name)
            
            tweet = call_llm(
                HEURIST_BASE_URL, 
                HEURIST_API_KEY, 
                LARGE_MODEL_ID, 
                self.prompt_config.get_system_prompt(), 
                user_prompt, 
                0.8
            )
            if not tweet:
                raise LLMError("Empty tweet generated")
            
            # Clean tweet and prepare metadata
            tweet = tweet.replace('"', '')
            add_metadata = {}

            # Generate image for original tweets (not replies)
            if user_reply is None and author_name is not None:
                if random.random() < IMAGE_GENERATION_PROBABILITY:
                    try:
                        image_prompt = generate_image_convo_prompt(original_tweet, tweet)
                        image_url = generate_image_with_retry(image_prompt)
                        add_metadata['image_prompt'] = image_prompt
                        add_metadata['image_url'] = image_url
                    except Exception as e:
                        logger.warning(f"Failed to generate image: {str(e)}")

            return tweet, add_metadata
            
        except LLMError as e:
            logger.error(f"Failed to generate tweet due to LLMError: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in tweet generation: {str(e)}")
        
        return None, None

def post_ai_replies(db: ReplyDatabase, generator: ReplyGenerator):
    """Generate and post AI replies to pending replies"""
    pending_replies = db.get_pending_replies()
    
    for reply_id, reply_data in pending_replies.items():
        try:
            if reply_data.get("is_quest_reply"):
                continue

            # Generate AI response
            response, add_metadata = generator.generate_reply(
                original_tweet=reply_data.get("original_tweet"),
                user_reply=reply_data.get("content"),
                author_name=reply_data.get("author_name")
            )

            if response and "ignore" in response.lower():
                logger.info(f'Ignoring reply to {reply_id}')
                db.mark_as_processed(reply_id, "IGNORE", add_metadata)
                continue
            
            if response:
                if not DRYRUN:
                    if add_metadata.get("image_url"):
                        twitter_api.reply_with_image(response, add_metadata["image_url"], reply_id)
                    else:
                        twitter_api.reply(response, reply_id)
                    logger.info(f'Posted reply to {reply_id}. AI response: {response}')
                else:
                    logger.info(f"DRYRUN: Would reply to {reply_id} with: {response}")
                
                db.mark_as_processed(reply_id, response, add_metadata)
                time.sleep(RATE_LIMIT_SLEEP)
                
        except Exception as e:
            logger.error(f"Error posting reply to {reply_id}: {str(e)}")
            db.mark_as_failed(reply_id, str(e))

def main():
    db = ReplyDatabase()
    generator = ReplyGenerator()
    
    while True:
        try:
            post_ai_replies(db, generator)
            logger.info(f"Waiting {REPLY_CHECK_INTERVAL} seconds before next check...")
            time.sleep(REPLY_CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            logger.info("Stopping reply processor...")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {str(e)}")
            time.sleep(REPLY_CHECK_INTERVAL)

if __name__ == "__main__":
    try:
        logger.info("Starting reply automation...")
        main()
    except KeyboardInterrupt:
        logger.info("\nReply automation stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")