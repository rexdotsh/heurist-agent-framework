import os
import time
import random
import base64
from urllib.request import urlopen
import uuid
import json
import logging
import yaml
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()

# Constants
HEURIST_BASE_URL = "https://llm-gateway.heurist.xyz"
HEURIST_API_KEY = os.getenv("HEURIST_API_KEY")
FARCASTER_API_KEY = os.getenv("FARCASTER_API_KEY")
FARCASTER_SIGNER_UUID = os.getenv("FARCASTER_SIGNER_UUID")
DALLE_API_KEY = os.getenv("OPENAI_API_KEY")  # For DALL-E image generation

# Model IDs
LARGE_MODEL_ID = "nvidia/llama-3.1-nemotron-70b-instruct"
SMALL_MODEL_ID = "mistralai/mixtral-8x7b-instruct"
CAST_WORD_LIMITS = [15, 20, 30, 35]

class LLMError(Exception):
    """Custom exception for LLM-related errors"""
    pass


def upload_to_imgbb(image_url):
    """
    Upload an image to IMGBB and return the direct image URL
    """
    try:
        # Get API key from environment variables
        api_key = os.getenv('IMGBB_API_KEY')
        if not api_key:
            raise ValueError("IMGBB_API_KEY not found in environment variables")
            
        # Download image and encode in base64
        image_data = urlopen(image_url).read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # IMGBB API endpoint
        url = "https://api.imgbb.com/1/upload"
        
        # Prepare payload
        payload = {
            "key": api_key,
            "image": base64_image,
        }
        
        # Make the request to IMGBB
        response = requests.post(url, payload)
        
        if response.status_code == 200:
            json_data = response.json()
            return json_data['data']['url']
        else:
            return None
            
    except Exception as e:
        print(f"Error uploading to IMGBB: {str(e)}")
        return None
    
class FarcasterBot:
    def __init__(self, api_key, signer_uuid):
        self.api_key = api_key
        self.signer_uuid = signer_uuid
        self.base_url = 'https://api.neynar.com/v2/farcaster'
        self.headers = {
            'api_key': self.api_key,
            'Content-Type': 'application/json'
        }


    def send_cast(self, message, parent_hash=None, parent_url=None, image_url=None):
        """Send a cast to Farcaster"""
        endpoint = f"{self.base_url}/cast"
        
        embeds = []
        
        if parent_url:
            embeds.append({"url": parent_url})
        
        if image_url:
            imgbb_url = upload_to_imgbb(image_url)
            embeds.append({"url": imgbb_url})
        
        data = {
            "signer_uuid": self.signer_uuid,
            "text": message,
        }
        
        if parent_hash:
            data["parent"] = {"hash": parent_hash}
        if embeds:
            data["embeds"] = embeds

        try:
            logger.info(f"Sending cast with payload: {json.dumps(data, indent=2)}")
            
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=data
            )
            
            logger.info(f"Response status code: {response.status_code}")
            logger.info(f"Response content: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Cast successfully sent! Hash: {result.get('cast', {}).get('hash')}")
                return result
            else:
                logger.error(f"Failed to send cast. Status Code: {response.status_code}")
                logger.error(f"Error: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"An error occurred while sending cast: {str(e)}")
            return None

    def get_user_casts(self, fid, limit=20):
        """Get recent casts from a user"""
        endpoint = f"{self.base_url}/user/casts"
        params = {
            "fid": fid,
            "limit": limit
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
                logger.error(f"Failed to get casts. Status Code: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            return None

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

class CastHistoryManager:
    def __init__(self, history_file: str = "cast_history.json"):
        self.history_file = history_file
        self.history = self._load_history()

    def _load_history(self) -> list:
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Error reading {self.history_file}, starting fresh")
                return []
        return []

    def add_cast(self, cast_data: Dict[str, Any]) -> None:
        entry = {
            'timestamp': datetime.now().isoformat(),
            'cast': cast_data
        }
        self.history.append(entry)
        self._save_history()

    def _save_history(self) -> None:
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def get_recent_casts(self, n: int = 6) -> list:
        try:
            recent = self.history[-n:] if len(self.history) > 0 else []
            return [entry['cast']['cast'] for entry in recent]
        except Exception as e:
            logger.warning(f"Error getting recent casts: {str(e)}")
            return []

class ContentGenerator:
    def __init__(self):
        self.prompt_config = self._load_config()
        self.history_manager = CastHistoryManager()
        self.farcaster_bot = FarcasterBot(FARCASTER_API_KEY, FARCASTER_SIGNER_UUID)
        self.image_probability = 1  # 50% chance of generating an image

    def _load_config(self) -> dict:
        try:
            current_dir = Path(__file__).parent
            config_path = current_dir.parent / 'config' / 'prompts.yaml'
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise

    def fill_basic_prompt(self, basic_options, style_options) -> str:
        return self.prompt_config['templates']['basic_prompt'].format(
            basic_option_1=basic_options[0],
            basic_option_2=basic_options[1],
            style_option_1=style_options[0],
            style_option_2=style_options[1]
        )

    def format_cast_instruction(self, basic_options, style_options, ideas=None):
        num_words = random.choice(CAST_WORD_LIMITS)
        return self.prompt_config['templates']['tweet_instruction'].format(
            basic_option_1=basic_options[0],
            basic_option_2=basic_options[1],
            style_option_1=style_options[0],
            style_option_2=style_options[1],
            decoration_ideas=f"Ideas: {ideas}" if ideas else "\n",
            num_words=num_words,
            rules=self.prompt_config['rules']['twitter']
        )

    def format_context(self, casts):
        if not casts:
            return ""
        return self.prompt_config['templates']['context_twitter'].format(tweets="\n".join(casts))

    def generate_image_prompt(self, cast: str) -> str:
        """Generate a prompt for image creation based on the cast content"""
        try:
            system_prompt = "You are an AI that converts social media posts into image generation prompts. Create vivid, detailed prompts that capture the essence of the post."
            user_prompt = f"""Given this social media post: "{cast}"
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
            
            # The URL is directly in the response text
            image_url = response.text.strip().strip('"')
            if image_url and image_url.startswith('http'):
                return image_url
                
            logger.error(f"Image generation failed: {response.text}")
            return None
            
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            return None

    def generate_content(self, dry_run: bool = False) -> Dict[str, Any]:
        """Generate and post content with potential image"""
        try:
            # Get recent posts for context
            past_casts = self.history_manager.get_recent_casts()
            
            # Generate prompt options
            basic_options = random.sample(self.prompt_config['character']['basic_settings'], 2)
            style_options = random.sample(self.prompt_config['character']['interaction_styles'], 2)
            
            # Build base prompt
            base_prompt = self.fill_basic_prompt(basic_options, style_options)
            
            # Generate ideas
            idea_instruction = random.choice(self.prompt_config['tweet_ideas']['options'])
            ideas = call_llm(
                HEURIST_BASE_URL,
                HEURIST_API_KEY,
                SMALL_MODEL_ID,
                self.prompt_config['system']['base'],
                base_prompt + self.prompt_config['rules']['twitter'] + 
                self.format_context(past_casts) + idea_instruction,
                0.9
            )
            
            # Generate final cast
            cast = call_llm(
                HEURIST_BASE_URL,
                HEURIST_API_KEY,
                LARGE_MODEL_ID,
                self.prompt_config['system']['base'],
                base_prompt + self.prompt_config['rules']['twitter'] + 
                self.format_context(past_casts) + 
                self.format_cast_instruction(basic_options, style_options, ideas),
                0.9
            )
            
            if not cast:
                raise LLMError("Empty cast generated")
            
            # Clean up cast text
            cast = cast.strip().replace('"', '')
            
            # Decide whether to generate an image
            should_generate_image = random.random() < self.image_probability
            image_url = None
            image_prompt = None
            
            if should_generate_image:
                # Generate image prompt
                image_prompt = self.generate_image_prompt(cast)
                if image_prompt:
                    # Generate image
                    image_url = self.generate_image(image_prompt)
            
            # Prepare metadata
            cast_data = {
                'cast': cast,
                'metadata': {
                    'basic_options': basic_options,
                    'style_options': style_options,
                    'ideas_instruction': idea_instruction,
                    'ideas': ideas,
                    'image_generated': should_generate_image,
                    'image_prompt': image_prompt,
                    'image_url': image_url
                }
            }
            
            # Handle posting
            if not dry_run:
                result = self.farcaster_bot.send_cast(cast, image_url=image_url)
                if result:
                    cast_data['metadata']['cast_hash'] = result.get('cast', {}).get('hash')
                    self.history_manager.add_cast(cast_data)
                    logger.info(f"Successfully posted cast: {cast}")
                    if image_url:
                        logger.info(f"With image: {image_url}")
                else:
                    logger.error("Failed to post cast")
            else:
                logger.info(f"Dry run - generated cast: {cast[:50]}...")
                if image_url:
                    logger.info(f"With image: {image_url}")
            
            return cast_data
            
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            return {'error': str(e)}

def main():
    try:
        generator = ContentGenerator()
        result = generator.generate_content(dry_run=False)  # Set to True for testing
        
        if 'error' not in result:
            print("\nGenerated Content:")
            print("Cast:", result.get('cast', ''))
            print("\nMetadata:", json.dumps(result.get('metadata', {}), indent=2))
            if result.get('metadata', {}).get('image_url'):
                print("\nImage URL:", result['metadata']['image_url'])
        else:
            print("Error:", result['error'])
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()