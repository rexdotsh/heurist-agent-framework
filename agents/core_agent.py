import json
import logging
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import dotenv
import yaml
from core.llm import call_llm_with_tools, call_llm, LLMError
from core.imgen import generate_image_with_retry, generate_image_prompt
from core.voice import transcribe_audio, speak_text

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
DRYRUN = os.getenv("DRYRUN")
BASE_IMAGE_PROMPT = ""


# Add new constants
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate an image based on a text prompt, any request to create should be handled by this tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The prompt to generate the image from"
                    }
                },
                "required": ["prompt"]
            }
        }
    },

]

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
        except FileNotFoundError:
            logger.error(f"Config file not found at {self.config_path}")
            raise
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
    
    def get_reply_to_comment_template(self) -> str:
        return self.config['reply']['reply_to_comment_template']

    def get_reply_to_tweet_template(self) -> str:
        return self.config['reply']['reply_to_tweet_template']

class CoreAgent:
    def __init__(self):
        self.prompt_config = PromptConfig()

    async def handle_image_generation(self, prompt: str, base_prompt: str = "") -> Optional[str]:
        """
        Handle image generation requests with retry logic
        
        Args:
            prompt: The image generation prompt
            base_prompt: Optional base prompt to prepend
            
        Returns:
            Generated image URL or None if failed
        """
        try:
            full_prompt = base_prompt + prompt if base_prompt else prompt
            result = generate_image_with_retry(prompt=full_prompt)
            print(result)
            return result
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            return None



    async def transcribe_audio(self, audio_file_path: Path) -> str:
        """
        Handle voice transcription requests
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            return transcribe_audio(audio_file_path)
        except Exception as e:
            logger.error(f"Voice transcription failed: {str(e)}")
            raise

    async def handle_text_to_speech(self, text: str) -> Path:
        """
        Handle text-to-speech conversion
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Path to generated audio file
        """
        try:
            return speak_text(text)
        except Exception as e:
            logger.error(f"Text-to-speech conversion failed: {str(e)}")
            raise

    async def handle_message(self, message: str) -> tuple[str, str | None]:
        """
        Handle incoming messages and return appropriate response
        
        Args:
            message: The user's message
            
        Returns:
            tuple containing (text_response, image_url)
        """
        try:
            response = call_llm_with_tools(
                HEURIST_BASE_URL,
                HEURIST_API_KEY, 
                LARGE_MODEL_ID,
                self.prompt_config.get_system_prompt(),
                message,
                temperature=0.01,
                tools=TOOLS
            )
            
            # Check if response is valid
            if not response:
                return "Sorry, I couldn't process your message.", None

            # Handle tool calls (image generation)
            if 'tool_calls' in response and response['tool_calls']:
                tool_call = response['tool_calls']
                if tool_call.function.name == 'generate_image':
                    args = json.loads(tool_call.function.arguments)
                    image_url = await self.handle_image_generation(args['prompt'])
                    return None, image_url
            
            # Handle regular text response
            if hasattr(response, 'content') and response.content:
                return response.content, None
                
            return "I received your message but couldn't generate a proper response.", None
                
        except LLMError as e:
            logger.error(f"LLM call failed: {str(e)}")
            return "Sorry, I encountered an error processing your message.", None
