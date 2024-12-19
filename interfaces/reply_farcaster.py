import time
from typing import Dict, Optional, List
import os
import base64
from urllib.request import urlopen
import uuid
import dotenv
import random
import json
from datetime import datetime, timezone
import logging
import yaml
from pathlib import Path
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

# Constants
HEURIST_BASE_URL = "https://llm-gateway.heurist.xyz"
HEURIST_API_KEY = os.getenv("HEURIST_API_KEY")
LARGE_MODEL_ID = "nvidia/llama-3.1-nemotron-70b-instruct"
SMALL_MODEL_ID = "mistralai/mixtral-8x7b-instruct"
IMAGE_GENERATION_PROBABILITY = 1  # 100% chance to generate an image
REPLY_CHECK_INTERVAL = 10  # Seconds between checks
RATE_LIMIT_SLEEP = 5  # Seconds between API calls

# Farcaster Constants
FARCASTER_API_KEY = os.getenv("FARCASTER_API_KEY")
FARCASTER_SIGNER_UUID = os.getenv("FARCASTER_SIGNER_UUID")
FARCASTER_FID = int(os.getenv("FARCASTER_FID"))

DRYRUN = os.getenv("DRYRUN")
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
        
        url = "https://api.imgbb.com/1/upload"
        payload = {
            "key": api_key,
            "image": base64_image,
        }
        
        response = requests.post(url, payload)
        
        if response.status_code == 200:
            json_data = response.json()
            return json_data['data']['url']
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error uploading to IMGBB: {str(e)}")
        return None

class LLMError(Exception):
    """Custom exception for LLM-related errors"""
    pass

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

    def get_cast_with_context(self, cast_hash: str) -> Optional[Dict]:
            """Get a cast and its full context by hash"""
            endpoint = f"{self.base_url}/cast"
            params = {
                'identifier': cast_hash,
                'type': 'hash'
            }

            try:
                response = requests.get(
                    endpoint,
                    headers=self.headers,
                    params=params
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to get cast context. Status: {response.status_code}")
                    return None

            except Exception as e:
                logger.error(f"Error getting cast context: {str(e)}")
                return None
        
    def send_cast(self, message: str, parent_hash: Optional[str] = None, image_url: Optional[str] = None) -> Optional[Dict]:
        endpoint = f"{self.base_url}/cast"
        
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

        try:
            logger.info(f"Sending cast: {message}")
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("Cast sent successfully!")
                return result
            else:
                logger.error(f"Failed to send cast. Status: {response.status_code}")
                logger.error(f"Error: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error sending cast: {str(e)}")
            return None

    def get_mentions(self, fid: int, limit: int = 25) -> List[Dict]:
        endpoint = f"{self.base_url}/notifications"
        params = {
            'fid': fid,
            'type': 'mentions',
            'priority_mode': 'false'
        }

        try:
            response = requests.get(
                endpoint,
                headers=self.headers,
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('notifications', [])
            else:
                logger.error(f"Failed to get mentions. Status: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error getting mentions: {str(e)}")
            return []

    def get_cast(self, cast_hash: str) -> Optional[Dict]:
        """Get a specific cast by its hash"""
        endpoint = f"{self.base_url}/cast"
        params = {
            'hash': cast_hash
        }

        try:
            response = requests.get(
                endpoint,
                headers=self.headers,
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('cast')
            else:
                return None

        except Exception as e:
            logger.error(f"Error getting cast: {str(e)}")
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
        
        # Sort by timestamp, ensuring all timestamps are UTC
        data["conversation_threads"][root_hash].sort(
            key=lambda x: parse_timestamp(x["timestamp"]) or datetime.min.replace(tzinfo=timezone.utc)
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

class ReplyGenerator:
    def __init__(self):
        self.prompt_config = PromptConfig()

    def get_conversation_context(self, thread: List[Dict]) -> str:
        """Format the conversation thread into a readable context"""
        context = []
        for msg in thread:
            context.append(f"@{msg['author']}: {msg['text']}")
        return "\n".join(context)

    def generate_image_prompt(self, text: str) -> Optional[str]:
        """Generate a prompt for image creation based on the text content"""
        try:
            system_prompt = "You are an AI that converts social media posts into image generation prompts. Create vivid, detailed prompts that capture the essence of the post."
            user_prompt = f"""Given this social media post: "{text}"
            Create a detailed prompt for generating an image that would complement this post.
            The prompt should be vivid and specific, but avoid any text or words in the image.
            Keep the prompt under 100 words."""
            
            image_prompt = call_llm(
                HEURIST_BASE_URL,
                HEURIST_API_KEY,
                SMALL_MODEL_ID,
                system_prompt,
                user_prompt,
                0.7
            )
            return image_prompt.strip()
        except Exception as e:
            logger.error(f"Error generating image prompt: {str(e)}")
            return None

    def generate_image(self, prompt: str) -> Optional[str]:
        """Generate an image using Heurist Sequencer API"""
        try:
            url = "http://sequencer.heurist.xyz/submit_job"
            
            random_uuid = str(uuid.uuid4())
            job_id = f"heuman-sdk-{random_uuid}"
            
            payload = {
                "job_id": job_id,
                "model_input": {
                    "SD": {
                        "prompt": prompt,
                        "neg_prompt": "",
                        "num_iterations": 20,
                        "width": 1024,
                        "height": 512,
                        "guidance_scale": 7.5,
                        "seed": -1
                    }
                },
                "model_id": "HeuristLogo",
                "deadline": 60,
                "priority": 1
            }
            
            headers = {
                "Authorization": f"Bearer {HEURIST_API_KEY}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            image_url = response.text.strip().strip('"')
            if image_url and image_url.startswith('http'):
                return image_url
                
            logger.error(f"Image generation failed: {response.text}")
            return None
            
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            return None

    def format_instruction(self, notification: Dict, conversation_context: Optional[List[Dict]] = None) -> str:
        cast = notification.get('cast', {})
        author = cast.get('author', {})
        username = author.get('username', 'anonymous')
        content = cast.get('text', '')
        
        if conversation_context:
            context_str = self.get_conversation_context(conversation_context)
            return f"""This is a conversation thread:

{context_str}

The latest reply is from @{username}: "{content}"

Please generate a contextually relevant reply that takes into account the entire conversation history. 
Keep the response casual and engaging while maintaining the context of the discussion."""
        else:
            return self.prompt_config.get_reply_template("tweet").format(
                author_name=username,
                original_tweet=content
            )

    def generate_reply(self, notification: Dict, db: ReplyDatabase, conversation_context: Optional[List[Dict]] = None) -> tuple[Optional[str], Optional[Dict]]:
        """Generate a reply based on the notification and conversation context"""
        try:
            user_prompt = self.format_instruction(notification, conversation_context)
            system_prompt = self.prompt_config.get_system_prompt()
            
            full_system_prompt = f"{system_prompt}\n\n{self.prompt_config.get_basic_prompt()}\n\n{self.prompt_config.get_heurist_knowledge()}"
            
            reply = call_llm(
                HEURIST_BASE_URL, 
                HEURIST_API_KEY, 
                LARGE_MODEL_ID, 
                full_system_prompt, 
                user_prompt, 
                0.8
            )
            
            if not reply:
                raise LLMError("Empty reply generated")
            
            # Clean reply
            reply = reply.replace('"', '')
            
            # Prepare metadata
            metadata = {}
            
            return reply, metadata
            
        except LLMError as e:
            logger.error(f"Failed to generate reply due to LLMError: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in reply generation: {str(e)}")
            logger.exception("Full traceback:")
        
        return None, None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_llm(url: str, api_key: str, model_id: str, 
             system_prompt: str, user_prompt: str, 
             temperature: float = 0.7) -> str:
    """Call the LLM API with retry logic"""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature
        }
        
        response = requests.post(f"{url}/v1/chat/completions", 
                               headers=headers, 
                               json=payload)
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise LLMError(f"LLM call failed: {str(e)}")

def build_conversation_tree(notification: Dict, farcaster_api: FarcasterAPI) -> List[Dict]:
    """Build a tree of conversation from a notification, fetching parent casts as needed"""
    conversation = []
    current_cast = notification.get('cast', {})
    visited_hashes = set()  # To prevent infinite loops
    
    while current_cast and current_cast.get('hash') not in visited_hashes:
        visited_hashes.add(current_cast.get('hash'))
        
        # Get full cast context
        full_cast_data = farcaster_api.get_cast_with_context(current_cast.get('hash'))
        if full_cast_data and 'cast' in full_cast_data:
            cast_details = full_cast_data['cast']
            conversation.append({
                'hash': cast_details.get('hash'),
                'text': cast_details.get('text', ''),
                'author': cast_details.get('author', {}).get('username', 'anonymous'),
                'timestamp': cast_details.get('timestamp'),
                'parent_hash': cast_details.get('parent_hash')
            })
        else:
            # Fallback to basic cast info if full context fetch fails
            conversation.append({
                'hash': current_cast.get('hash'),
                'text': current_cast.get('text', ''),
                'author': current_cast.get('author', {}).get('username', 'anonymous'),
                'timestamp': current_cast.get('timestamp'),
                'parent_hash': current_cast.get('parent_hash')
            })
        
        if current_cast.get('parent_hash'):
            parent_cast = farcaster_api.get_cast_with_context(current_cast['parent_hash'])
            if parent_cast and 'cast' in parent_cast:
                current_cast = parent_cast['cast']
            else:
                break
        else:
            break
    
    # Reverse to get chronological order
    return list(reversed(conversation))

def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """Convert ISO 8601 timestamp to datetime object"""
    try:
        # Always parse to UTC timezone
        dt = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.000Z')
        return dt.replace(tzinfo=timezone.utc)
    except Exception as e:
        logger.error(f"Error parsing timestamp {timestamp_str}: {str(e)}")
        return None

def is_recent_mention(timestamp_str: str, threshold_seconds: int = 300) -> bool:  # 5 minutes
    mention_time = parse_timestamp(timestamp_str)
    if not mention_time:
        return False
        
    current_time = datetime.now(timezone.utc)
    time_diff = (current_time - mention_time).total_seconds()
    
    return time_diff <= threshold_seconds

def process_mentions(farcaster_api: FarcasterAPI, db: ReplyDatabase, generator: ReplyGenerator):
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
            
            # Skip if already processed
            if db.is_processed(cast_hash):
                continue
            
            # Store the mention as pending
            db.add_pending_reply(cast_hash, notification)
            
            # Build and store the conversation thread
            if parent_hash:
                # Get the full conversation tree
                conversation_tree = build_conversation_tree(notification, farcaster_api)
                
                # Find the root hash (first message in the conversation)
                root_hash = conversation_tree[0]['hash'] if conversation_tree else parent_hash
                
                # Add each part of the conversation to the thread
                for cast_entry in conversation_tree:
                    db.add_to_conversation_thread(root_hash, cast_entry['hash'], {
                        'cast': {
                            'hash': cast_entry['hash'],
                            'text': cast_entry['text'],
                            'author': {'username': cast_entry['author']},
                            'timestamp': cast_entry['timestamp'],
                            'parent_hash': cast_entry['parent_hash']
                        }
                    })
            
            # Process all mentions within the time window
            if is_recent_mention(timestamp):
                author = cast.get('author', {})
                logger.info(f"Processing mention from @{author.get('username')}")
                
                # Get the conversation context
                conversation_context = None
                if parent_hash:
                    root_hash = conversation_tree[0]['hash'] if conversation_tree else parent_hash
                    conversation_context = db.get_conversation_thread(root_hash)
                
                # Generate reply with conversation context
                reply_text, metadata = generator.generate_reply(
                    notification,
                    db,
                    conversation_context=conversation_context
                )
                
                if reply_text:
                    if not DRYRUN:
                        # Check if we should generate an image
                        image_url = None
                        if random.random() < IMAGE_GENERATION_PROBABILITY:
                            image_prompt = generator.generate_image_prompt(reply_text)
                            if image_prompt:
                                image_url = generator.generate_image(image_prompt)
                                if image_url and metadata is None:
                                    metadata = {}
                                if image_url and metadata:
                                    metadata['image_prompt'] = image_prompt
                                    metadata['image_url'] = image_url
                        
                        response = farcaster_api.send_cast(
                            reply_text,
                            parent_hash=cast_hash,
                            image_url=image_url if image_url else metadata.get('image_url') if metadata else None
                        )
                        
                        if response:
                            # Add our reply to the conversation thread
                            if parent_hash:
                                db.add_to_conversation_thread(root_hash, response['cast']['hash'], {
                                    'cast': response['cast']
                                })
                            
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
    farcaster_api = FarcasterAPI(FARCASTER_API_KEY, FARCASTER_SIGNER_UUID)
    db = ReplyDatabase()
    generator = ReplyGenerator()
    
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