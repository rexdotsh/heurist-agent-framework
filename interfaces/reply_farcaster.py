import time
import json
import logging
import os
import base64
import yaml
import uuid
import random
import dotenv
import requests
from pathlib import Path
from urllib.request import urlopen
from typing import Dict, Optional, List
from datetime import datetime, timezone
from core.llm import LLMError, call_llm
from core.imgen import generate_image_prompt, generate_image
from tenacity import retry, stop_after_attempt, wait_exponential

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

# Farcaster Constants
FARCASTER_API_KEY = os.getenv("FARCASTER_API_KEY")
FARCASTER_SIGNER_UUID = os.getenv("FARCASTER_SIGNER_UUID")
FARCASTER_FID = int(os.getenv("FARCASTER_FID"))

if DRYRUN:
    print("DRYRUN MODE: Not posting real casts")
else:
    print("LIVE MODE: Will post real casts")

def upload_to_imgbb(image_url: str) -> Optional[str]:
    """Upload an image to IMGBB and return the direct image URL"""
    try:
        api_key = os.getenv('IMGBB_API_KEY')
        if not api_key:
            raise ValueError("IMGBB_API_KEY not found in environment variables")
            
        image_data = urlopen(image_url).read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        response = requests.post(
            "https://api.imgbb.com/1/upload",
            data={
                "key": api_key,
                "image": base64_image,
            }
        )
        
        if response.status_code == 200:
            return response.json()['data']['url']
        return None
            
    except Exception as e:
        logger.error(f"Error uploading to IMGBB: {str(e)}")
        return None

class FarcasterAPI:
    def __init__(self, api_key: str, signer_uuid: str):
        self.api_key = api_key
        self.signer_uuid = signer_uuid
        self.base_url = 'https://api.neynar.com/v2/farcaster'
        self.headers = {
            'accept': 'application/json',
            'api_key': self.api_key,
            'Content-Type': 'application/json'
        }

    def send_cast(self, message: str, parent_hash: Optional[str] = None, image_url: Optional[str] = None) -> Optional[Dict]:
        """Send a cast with optional parent and image"""
        try:
            data = {
                "signer_uuid": self.signer_uuid,
                "text": message
            }
            
            if parent_hash:
                data["parent"] = parent_hash
                
            if image_url:
                imgbb_url = upload_to_imgbb(image_url)
                if imgbb_url:
                    data["embeds"] = [{"url": imgbb_url}]

            response = requests.post(
                f"{self.base_url}/cast",
                headers=self.headers,
                json=data
            )
            
            if response.status_code == 200:
                logger.info("Cast sent successfully!")
                return response.json()
                
            logger.error(f"Failed to send cast. Status: {response.status_code}")
            logger.error(f"Error: {response.text}")
            return None
                
        except Exception as e:
            logger.error(f"Error sending cast: {str(e)}")
            return None

    def get_mentions(self, fid: int, limit: int = 25) -> List[Dict]:
        """Get mentions for a specific FID"""
        try:
            response = requests.get(
                f"{self.base_url}/notifications",
                headers=self.headers,
                params={
                    'fid': fid,
                    'type': 'mentions',
                    'priority_mode': 'false'
                }
            )
            
            if response.status_code == 200:
                return response.json().get('notifications', [])
                
            logger.error(f"Failed to get mentions. Status: {response.status_code}")
            return []

        except Exception as e:
            logger.error(f"Error getting mentions: {str(e)}")
            return []

    def get_cast(self, cast_hash: str) -> Optional[Dict]:
        """Get a specific cast by its hash"""
        try:
            response = requests.get(
                f"{self.base_url}/cast",
                headers=self.headers,
                params={'hash': cast_hash}
            )
            
            if response.status_code == 200:
                return response.json().get('cast')
                
            logger.error(f"Failed to get cast. Status: {response.status_code}")
            return None

        except Exception as e:
            logger.error(f"Error getting cast: {str(e)}")
            return None

def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """Convert ISO 8601 timestamp to datetime object"""
    try:
        return datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.000Z').replace(tzinfo=timezone.utc)
    except Exception as e:
        logger.error(f"Error parsing timestamp {timestamp_str}: {str(e)}")
        return None


class ReplyDatabase:
    def __init__(self, db_file: str = "farcaster_reply_history.json"):
        self.db_file = Path(db_file)
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        if not self.db_file.exists():
            self._write_data({
                "processed_replies": {},
                "pending_replies": {},
                "conversation_threads": {}
            })
    
    def _read_data(self) -> Dict:
        try:
            with self.db_file.open('r') as f:
                data = json.load(f)
                if "conversation_threads" not in data:
                    data["conversation_threads"] = {}
                return data
        except json.JSONDecodeError:
            logger.error(f"Error reading database. Creating new one.")
            return {
                "processed_replies": {}, 
                "pending_replies": {},
                "conversation_threads": {}
            }
    
    def _write_data(self, data: Dict):
        try:
            with self.db_file.open('w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error writing to database: {str(e)}")

    def add_to_conversation_thread(self, root_hash: str, cast_hash: str, cast_data: Dict):
        data = self._read_data()
        if root_hash not in data["conversation_threads"]:
            data["conversation_threads"][root_hash] = []
        
        # Add the cast to the conversation thread
        thread_entry = {
            "cast_hash": cast_hash,
            "timestamp": cast_data.get("cast", {}).get("timestamp"),
            "text": cast_data.get("cast", {}).get("text", ""),
            "author": cast_data.get("cast", {}).get("author", {}).get("username", "anonymous"),
            "parent_hash": cast_data.get("cast", {}).get("parent_hash")
        }
        data["conversation_threads"][root_hash].append(thread_entry)
        
        # Sort by timestamp to maintain chronological order
        data["conversation_threads"][root_hash].sort(
            key=lambda x: parse_timestamp(x["timestamp"]) if x["timestamp"] else datetime.min
        )
        
        self._write_data(data)

    def get_conversation_thread(self, root_hash: str) -> List[Dict]:
        data = self._read_data()
        return data["conversation_threads"].get(root_hash, [])

    def mark_as_processed(self, cast_hash: str, ai_response: str, add_metadata: Optional[Dict] = None):
        data = self._read_data()
        if cast_hash in data["pending_replies"]:
            reply_data = data["pending_replies"].pop(cast_hash)
            reply_data["ai_response"] = ai_response
            reply_data["processed_timestamp"] = datetime.now().isoformat()
            if add_metadata:
                reply_data.update(add_metadata)
            data["processed_replies"][cast_hash] = reply_data
            self._write_data(data)
            logger.info(f"Marked cast as processed: {cast_hash}")

    def mark_as_failed(self, cast_hash: str, failure: str):
        data = self._read_data()
        if cast_hash in data["pending_replies"]:
            reply_data = data["pending_replies"].pop(cast_hash)
            reply_data["failure"] = failure
            reply_data["processed_timestamp"] = datetime.now().isoformat()
            data["processed_replies"][cast_hash] = reply_data
            self._write_data(data)
            logger.info(f"Marked cast as failed: {cast_hash}")

    def add_pending_reply(self, cast_hash: str, cast_data: Dict):
        data = self._read_data()
        if cast_hash not in data["processed_replies"] and cast_hash not in data["pending_replies"]:
            data["pending_replies"][cast_hash] = cast_data
            self._write_data(data)
            logger.info(f"Added pending cast: {cast_hash}")

    def is_processed(self, cast_hash: str) -> bool:
        data = self._read_data()
        return cast_hash in data["processed_replies"]

class PromptConfig:
    def __init__(self, config_path: str = "config/prompts.yaml"):
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

    def get_reply_template(self, reply_type: str = "comment") -> str:
        if reply_type == "comment":
            return self.config['reply']['reply_to_comment_template']
        return self.config['reply']['reply_to_tweet_template']

class FarcasterReplyGenerator:
    def __init__(self):
        self.prompt_config = PromptConfig()

    def get_conversation_context(self, thread: List[Dict]) -> str:
        """Format the conversation thread into a readable context"""
        return "\n".join(f"@{msg['author']}: {msg['text']}" for msg in thread)

    def format_instruction(self, notification: Dict, conversation_context: Optional[List[Dict]] = None) -> str:
        """Format instruction for LLM based on notification and context"""
        cast = notification.get('cast', {})
        username = cast.get('author', {}).get('username', 'anonymous')
        content = cast.get('text', '')
        
        if conversation_context:
            context_str = self.get_conversation_context(conversation_context)
            return f"""This is a conversation thread:

{context_str}

The latest reply is from @{username}: "{content}"

Please generate a contextually relevant reply that takes into account the entire conversation history."""
        
        return self.prompt_config.get_reply_template("tweet").format(
            author_name=username,
            original_tweet=content
        )

    def generate_reply(self, notification: Dict, conversation_context: Optional[List[Dict]] = None) -> tuple[Optional[str], Optional[Dict]]:
        """Generate a reply based on the notification and conversation context"""
        try:
            user_prompt = self.format_instruction(notification, conversation_context)
            system_prompt = "\n\n".join([
                self.prompt_config.get_system_prompt(),
                self.prompt_config.get_basic_prompt(),
                self.prompt_config.get_heurist_knowledge()
            ])
            
            reply = call_llm(
                HEURIST_BASE_URL, 
                HEURIST_API_KEY, 
                LARGE_MODEL_ID, 
                system_prompt, 
                user_prompt, 
                0.8
            )
            
            if not reply:
                raise LLMError("Empty reply generated")
            
            return reply.replace('"', ''), {}
            
        except Exception as e:
            logger.error(f"Error generating reply: {str(e)}")
            return None, None

def build_conversation_tree(notification: Dict, farcaster_api: FarcasterAPI) -> List[Dict]:
    """Build a tree of conversation from a notification"""
    conversation = []
    current_cast = notification.get('cast', {})
    visited_hashes = set()
    
    while current_cast and current_cast.get('hash') not in visited_hashes:
        visited_hashes.add(current_cast.get('hash'))
        conversation.append({
            'hash': current_cast.get('hash'),
            'text': current_cast.get('text', ''),
            'author': current_cast.get('author', {}).get('username', 'anonymous'),
            'timestamp': current_cast.get('timestamp'),
            'parent_hash': current_cast.get('parent_hash')
        })
        
        if current_cast.get('parent_hash'):
            parent_cast = farcaster_api.get_cast(current_cast['parent_hash'])
            current_cast = parent_cast if parent_cast else None
        else:
            break
    
    return list(reversed(conversation))

def is_recent_mention(timestamp_str: str, threshold_seconds: int = 300) -> bool:
    """Check if a mention is within the recent threshold"""
    try:
        mention_time = datetime.strptime(
            timestamp_str, 
            '%Y-%m-%dT%H:%M:%S.000Z'
        ).replace(tzinfo=timezone.utc)
        
        return (datetime.now(timezone.utc) - mention_time).total_seconds() <= threshold_seconds
    except Exception as e:
        logger.error(f"Error parsing timestamp: {str(e)}")
        return False

def process_mentions(farcaster_api: FarcasterAPI, db: ReplyDatabase, generator: FarcasterReplyGenerator):
    """Process new mentions and generate contextual replies"""
    mentions = farcaster_api.get_mentions(FARCASTER_FID)
    
    if not mentions:
        logger.info("No mentions found.")
        return
    
    for notification in mentions:
        try:
            cast = notification.get('cast', {})
            cast_hash = cast.get('hash')
            parent_hash = cast.get('parent_hash')
            timestamp = cast.get('timestamp')
            
            if db.is_processed(cast_hash):
                continue
                
            db.add_pending_reply(cast_hash, notification)
            
            if parent_hash:
                conversation_tree = build_conversation_tree(notification, farcaster_api)
                root_hash = conversation_tree[0]['hash'] if conversation_tree else parent_hash
                
                for cast_entry in conversation_tree:
                    db.add_to_conversation_thread(root_hash, cast_entry['hash'], {
                        'cast': cast_entry
                    })
            
            if is_recent_mention(timestamp):
                logger.info(f"Processing mention from @{cast.get('author', {}).get('username')}")
                
                conversation_context = None
                if parent_hash:
                    root_hash = conversation_tree[0]['hash'] if conversation_tree else parent_hash
                    conversation_context = db.get_conversation_thread(root_hash)
                
                reply_text, metadata = generator.generate_reply(
                    notification,
                    conversation_context=conversation_context
                )
                
                if reply_text:
                    if not DRYRUN:
                        image_url = None
                        if random.random() < IMAGE_GENERATION_PROBABILITY:
                            image_prompt = generate_image_prompt(reply_text)
                            if image_prompt:
                                image_url = generate_image(image_prompt, HEURIST_API_KEY)
                                if image_url:
                                    metadata = metadata or {}
                                    metadata.update({
                                        'image_prompt': image_prompt,
                                        'image_url': image_url
                                    })
                        
                        response = farcaster_api.send_cast(
                            reply_text,
                            parent_hash=cast_hash,
                            image_url=image_url or (metadata or {}).get('image_url')
                        )
                        
                        if response:
                            if parent_hash:
                                db.add_to_conversation_thread(
                                    root_hash,
                                    response['cast']['hash'],
                                    {'cast': response['cast']}
                                )
                            
                            db.mark_as_processed(cast_hash, reply_text, metadata)
                            logger.info(f"Successfully sent reply to cast {cast_hash}")
                            time.sleep(RATE_LIMIT_SLEEP)
                    else:
                        logger.info(f"DRYRUN: Would reply to {cast_hash} with: {reply_text}")
                        if metadata and metadata.get('image_url'):
                            logger.info(f"DRYRUN: Would include image: {metadata['image_url']}")
                        db.mark_as_processed(cast_hash, reply_text, metadata)
                else:
                    logger.warning(f"Failed to generate reply for cast {cast_hash}")
                    db.mark_as_failed(cast_hash, "Failed to generate reply")
                    
        except Exception as e:
            logger.error(f"Error processing mention {cast_hash}: {str(e)}")
            logger.exception("Full traceback:")
            db.mark_as_failed(cast_hash, str(e))

def main():
    """Main execution loop"""
    farcaster_api = FarcasterAPI(FARCASTER_API_KEY, FARCASTER_SIGNER_UUID)
    db = ReplyDatabase(db_file="farcaster_reply_history.json")
    generator = FarcasterReplyGenerator()
    
    logger.info(f"Starting Farcaster reply bot for FID: {FARCASTER_FID}")
    logger.info("Monitoring for mentions...")
    
    while True:
        try:
            process_mentions(farcaster_api, db, generator)
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
        main()
    except KeyboardInterrupt:
        logger.info("\nFarcaster reply automation stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")