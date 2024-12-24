import time, os, base64, uuid, dotenv, random, json, logging, yaml
from typing import Dict, Optional, List
from urllib.request import urlopen
from datetime import datetime, timezone
from pathlib import Path
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

# Constants consolidated into a config dict
CONFIG = {
    'HEURIST_BASE_URL': "https://llm-gateway.heurist.xyz",
    'HEURIST_API_KEY': os.getenv("HEURIST_API_KEY"),
    'LARGE_MODEL_ID': os.getenv("LARGE_MODEL_ID"),
    'SMALL_MODEL_ID': os.getenv("SMALL_MODEL_ID"),
    'IMAGE_GENERATION_PROBABILITY': 1,
    'REPLY_CHECK_INTERVAL': 10,
    'RATE_LIMIT_SLEEP': 5,
    'FARCASTER_API_KEY': os.getenv("FARCASTER_API_KEY"),
    'FARCASTER_SIGNER_UUID': os.getenv("FARCASTER_SIGNER_UUID"),
    'FARCASTER_FID': int(os.getenv("FARCASTER_FID")),
    'DRYRUN': False
}

print(f"{'DRYRUN' if CONFIG['DRYRUN'] else 'LIVE'} MODE: {'Not posting' if CONFIG['DRYRUN'] else 'Will post'} real casts")

@lru_cache(maxsize=100)
def upload_to_imgbb(image_url: str) -> Optional[str]:
    """Upload an image to IMGBB with caching for repeated uploads"""
    try:
        api_key = os.getenv('IMGBB_API_KEY')
        if not api_key:
            raise ValueError("IMGBB_API_KEY not found")
            
        image_data = urlopen(image_url).read()
        response = requests.post(
            "https://api.imgbb.com/1/upload",
            data={
                "key": api_key,
                "image": base64.b64encode(image_data).decode('utf-8')
            }
        )
        
        return response.json()['data']['url'] if response.status_code == 200 else None
            
    except Exception as e:
        logger.error(f"IMGBB upload error: {str(e)}")
        return None

class LLMError(Exception):
    pass

class FarcasterAPI:
    def __init__(self, api_key: str, signer_uuid: str):
        self.base_url = 'https://api.neynar.com/v2/farcaster'
        self.headers = {
            'accept': 'application/json',
            'api_key': api_key,
            'Content-Type': 'application/json'
        }
        self.signer_uuid = signer_uuid

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        try:
            response = requests.request(
                method,
                f"{self.base_url}/{endpoint}",
                headers=self.headers,
                **kwargs
            )
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            logger.error(f"API request error: {str(e)}")
            return None

    def get_cast_with_context(self, cast_hash: str) -> Optional[Dict]:
        return self._make_request('GET', 'cast', params={'identifier': cast_hash, 'type': 'hash'})
        
    def send_cast(self, message: str, parent_hash: Optional[str] = None, image_url: Optional[str] = None) -> Optional[Dict]:
        data = {
            "signer_uuid": self.signer_uuid,
            "text": message,
            **({"parent": parent_hash} if parent_hash else {}),
            **({"embeds": [{"url": imgbb_url}]} if (imgbb_url := upload_to_imgbb(image_url)) else {})
        }
        
        logger.info(f"Sending cast: {message}")
        return self._make_request('POST', 'cast', json=data)

    def get_mentions(self, fid: int, limit: int = 25) -> List[Dict]:
        return self._make_request(
            'GET', 
            'notifications',
            params={'fid': fid, 'type': 'mentions', 'priority_mode': 'false'}
        ).get('notifications', [])

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
                data.setdefault("conversation_threads", {})
                return data
        except json.JSONDecodeError:
            return {"processed_replies": {}, "pending_replies": {}, "conversation_threads": {}}
    
    def _write_data(self, data: Dict):
        try:
            with self.db_file.open('w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Database write error: {str(e)}")

    def add_to_conversation_thread(self, root_hash: str, cast_hash: str, cast_data: Dict):
        data = self._read_data()
        thread = data["conversation_threads"].setdefault(root_hash, [])
        
        cast = cast_data.get("cast", {})
        thread.append({
            "cast_hash": cast_hash,
            "timestamp": cast.get("timestamp"),
            "text": cast.get("text", ""),
            "author": cast.get("author", {}).get("username", "anonymous"),
            "parent_hash": cast.get("parent_hash")
        })
        
        thread.sort(key=lambda x: parse_timestamp(x["timestamp"]) or datetime.min.replace(tzinfo=timezone.utc))
        self._write_data(data)

    def get_conversation_thread(self, root_hash: str) -> List[Dict]:
        return self._read_data()["conversation_threads"].get(root_hash, [])

    def mark_as_processed(self, cast_hash: str, ai_response: str, add_metadata: Optional[Dict] = None):
        data = self._read_data()
        if cast_hash in data["pending_replies"]:
            reply_data = data["pending_replies"].pop(cast_hash)
            reply_data.update({
                "ai_response": ai_response,
                "processed_timestamp": datetime.now().isoformat(),
                **(add_metadata or {})
            })
            data["processed_replies"][cast_hash] = reply_data
            self._write_data(data)
            logger.info(f"Marked cast as processed: {cast_hash}")

    def mark_as_failed(self, cast_hash: str, failure: str):
        data = self._read_data()
        if cast_hash in data["pending_replies"]:
            reply_data = data["pending_replies"].pop(cast_hash)
            reply_data.update({
                "failure": failure,
                "processed_timestamp": datetime.now().isoformat()
            })
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
        return cast_hash in self._read_data()["processed_replies"]

@lru_cache(maxsize=1)
class PromptConfig:
    def __init__(self, config_path: str = "config/prompts.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    def get_system_prompt(self) -> str:
        return self.config['system']['base']

    def get_basic_prompt(self) -> str:
        return self.config['reply']['basic_prompt']

    def get_heurist_knowledge(self) -> str:
        return self.config['reply']['heurist_knowledge']

    def get_reply_template(self, reply_type: str = "comment") -> str:
        return self.config['reply'][f'reply_to_{reply_type}_template']

class ReplyGenerator:
    def __init__(self):
        self.prompt_config = PromptConfig()

    @staticmethod
    def get_conversation_context(thread: List[Dict]) -> str:
        return "\n".join(f"@{msg['author']}: {msg['text']}" for msg in thread)

    def generate_image_prompt(self, text: str) -> Optional[str]:
        try:
            return call_llm(
                CONFIG['HEURIST_BASE_URL'],
                CONFIG['HEURIST_API_KEY'],
                CONFIG['SMALL_MODEL_ID'],
                "You are an AI that converts social media posts into image generation prompts. Create vivid, detailed prompts that capture the essence of the post.",
                f"""Given this social media post: "{text}"
                Create a detailed prompt for generating an image that would complement this post.
                The prompt should be vivid and specific, but avoid any text or words in the image.
                Keep the prompt under 100 words.""",
                0.7
            ).strip()
        except Exception as e:
            logger.error(f"Image prompt generation error: {str(e)}")
            return None

    def generate_image(self, prompt: str) -> Optional[str]:
        try:
            response = requests.post(
                "http://sequencer.heurist.xyz/submit_job",
                headers={
                    "Authorization": f"Bearer {CONFIG['HEURIST_API_KEY']}",
                    "Content-Type": "application/json"
                },
                json={
                    "job_id": f"heuman-sdk-{uuid.uuid4()}",
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
            )
            
            image_url = response.text.strip().strip('"')
            return image_url if image_url and image_url.startswith('http') else None
                
        except Exception as e:
            logger.error(f"Image generation error: {str(e)}")
            return None

    def format_instruction(self, notification: Dict, conversation_context: Optional[List[Dict]] = None) -> str:
        cast = notification.get('cast', {})
        username = cast.get('author', {}).get('username', 'anonymous')
        content = cast.get('text', '')
        
        if conversation_context:
            context_str = self.get_conversation_context(conversation_context)
            return f"""This is a conversation thread:

{context_str}

The latest reply is from @{username}: "{content}"

Please generate a contextually relevant reply that takes into account the entire conversation history. 
Keep the response casual and engaging while maintaining the context of the discussion."""
        
        return self.prompt_config.get_reply_template("tweet").format(
            author_name=username,
            original_tweet=content
        )

    def generate_reply(self, notification: Dict, db: ReplyDatabase, conversation_context: Optional[List[Dict]] = None) -> tuple[Optional[str], Optional[Dict]]:
        try:
            reply = call_llm(
                CONFIG['HEURIST_BASE_URL'],
                CONFIG['HEURIST_API_KEY'],
                CONFIG['LARGE_MODEL_ID'],
                f"{self.prompt_config.get_system_prompt()}\n\n{self.prompt_config.get_basic_prompt()}\n\n{self.prompt_config.get_heurist_knowledge()}",
                self.format_instruction(notification, conversation_context),
                0.8
            )
            
            return (reply.replace('"', ''), {}) if reply else (None, None)
            
        except Exception as e:
            logger.error(f"Reply generation error: {str(e)}")
            return None, None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_llm(url: str, api_key: str, model_id: str, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    try:
        response = requests.post(
            f"{url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise LLMError(f"LLM call failed: {str(e)}")

def build_conversation_tree(notification: Dict, farcaster_api: FarcasterAPI) -> List[Dict]:
    conversation = []
    current_cast = notification.get('cast', {})
    visited_hashes = set()
    
    while current_cast and current_cast.get('hash') not in visited_hashes:
        visited_hashes.add(current_cast.get('hash'))
        
        full_cast_data = farcaster_api.get_cast_with_context(current_cast.get('hash'))
        cast_details = full_cast_data.get('cast', current_cast) if full_cast_data else current_cast
        
        conversation.append({
            'hash': cast_details.get('hash'),
            'text': cast_details.get('text', ''),
            'author': cast_details.get('author', {}).get('username', 'anonymous'),
            'timestamp': cast_details.get('timestamp'),
            'parent_hash': cast_details.get('parent_hash')
        })
        
        if current_cast.get('parent_hash'):
            parent_cast = farcaster_api.get_cast_with_context(current_cast['parent_hash'])
            current_cast = parent_cast.get('cast') if parent_cast else None
        else:
            break
    
    return list(reversed(conversation))

def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    try:
        dt = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.000Z')
        return dt.replace(tzinfo=timezone.utc)
    except Exception as e:
        logger.error(f"Timestamp parsing error {timestamp_str}: {str(e)}")
        return None

def is_recent_mention(timestamp_str: str, threshold_seconds: int = 300) -> bool:
    if mention_time := parse_timestamp(timestamp_str):
        return (datetime.now(timezone.utc) - mention_time).total_seconds() <= threshold_seconds
    return False

def process_mentions(farcaster_api: FarcasterAPI, db: ReplyDatabase, generator: ReplyGenerator):
    for notification in farcaster_api.get_mentions(CONFIG['FARCASTER_FID']) or []:
        try:
            cast = notification.get('cast', {})
            cast_hash, parent_hash = cast.get('hash'), cast.get('parent_hash')
            
            if db.is_processed(cast_hash):
                continue
            
            db.add_pending_reply(cast_hash, notification)
            
            if parent_hash:
                conversation_tree = build_conversation_tree(notification, farcaster_api)
                root_hash = conversation_tree[0]['hash'] if conversation_tree else parent_hash
                
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
            
            if is_recent_mention(cast.get('timestamp')):
                logger.info(f"Processing mention from @{cast.get('author', {}).get('username')}")
                
                conversation_context = db.get_conversation_thread(root_hash) if parent_hash else None
                reply_text, metadata = generator.generate_reply(notification, db, conversation_context)
                
                if reply_text:
                    if not CONFIG['DRYRUN']:
                        image_url = None
                        if random.random() < CONFIG['IMAGE_GENERATION_PROBABILITY']:
                            if image_prompt := generator.generate_image_prompt(reply_text):
                                image_url = generator.generate_image(image_prompt)
                                if image_url:
                                    metadata = metadata or {}
                                    metadata.update({
                                        'image_prompt': image_prompt,
                                        'image_url': image_url
                                    })
                        
                        if response := farcaster_api.send_cast(reply_text, parent_hash=cast_hash, image_url=image_url):
                            if parent_hash:
                                db.add_to_conversation_thread(root_hash, response['cast']['hash'], {'cast': response['cast']})
                            
                            db.mark_as_processed(cast_hash, reply_text, metadata)
                            logger.info(f"Successfully replied to cast {cast_hash}")
                            time.sleep(CONFIG['RATE_LIMIT_SLEEP'])
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
    farcaster_api = FarcasterAPI(CONFIG['FARCASTER_API_KEY'], CONFIG['FARCASTER_SIGNER_UUID'])
    db = ReplyDatabase()
    generator = ReplyGenerator()
    
    logger.info(f"Starting Farcaster reply bot for FID: {CONFIG['FARCASTER_FID']}")
    logger.info("Monitoring for mentions...")
    
    while True:
        try:
            process_mentions(farcaster_api, db, generator)
            logger.info(f"Waiting {CONFIG['REPLY_CHECK_INTERVAL']} seconds before next check...")
            time.sleep(CONFIG['REPLY_CHECK_INTERVAL'])
            
        except KeyboardInterrupt:
            logger.info("Stopping reply processor...")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {str(e)}")
            time.sleep(CONFIG['REPLY_CHECK_INTERVAL'])

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nFarcaster reply automation stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")