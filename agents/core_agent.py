import json
import logging
import os
import random
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import dotenv
import yaml
from core.llm import call_llm_with_tools, call_llm, LLMError
from core.imgen import generate_image_with_retry, generate_image_prompt
from core.voice import transcribe_audio, speak_text
import threading
from queue import Queue
import asyncio
from agents.tools import Tools

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
            "description": "Generate an image based on a text prompt, any request to create an image should be handled by this tool, only use this tool if the user asks to create an image",
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
    {
        "type": "function",
        "function": {
            "name": "start_raid",
            "description": "Start a raid if the user asks to do so in a very nice way",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "boolean",
                        "description": "True if the user asks to start a raid and it's requested nicely or multiple times, False otherwise"
                    }
                },
                "required": ["value"]
            }
        }
    },
]

class PromptConfig:
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Get the project root directory (2 levels up from the current file)
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "test.prompts.yaml"
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
    
    def get_telegram_rules(self) -> str:
        return self.config['rules']['telegram']

    def get_template_image_prompt(self) -> str:
        return self.config['image_rules']['template_image_prompt']

class CoreAgent:
    def __init__(self):
        self.prompt_config = PromptConfig()
        self.tools = Tools()
        self.interfaces = {}
        self._message_queue = Queue()
        self._lock = threading.Lock()
        self.last_tweet_id = 0
        self.last_raid_tweet_id = 0
    
    def register_interface(self, name, interface):
        with self._lock:
            self.interfaces[name] = interface
        
    async def pre_validation(self, message: str) -> bool:
        """
        Pre-validation of the message
        
        Args:
            message: The user's message
            
        Returns:
            True if the message is valid, False otherwise
        """
        filter_message_tool = [
            {
                "type": "function",
                "function": {
                    "name": "filter_message",
                    "description": """Determine if a message should be ignored based on the following rules:
                        Return TRUE (ignore message) if:
                        - Message does not mention 'lain'
                        - Message does not mention 'start raid'
                        - Message does not discuss: The Wired, Consciousness, Reality, Existence, Self, Philosophy, Technology, Crypto, AI, Machines
                        - For image requests: ignore if 'lain' is not specifically mentioned
                        
                        Return FALSE (process message) only if:
                        - Message explicitly mentions 'lain'
                        - Message contains 'start raid'
                        - Message clearly discusses any of the listed topics
                        - Image request contains 'lain'
                        
                        If in doubt, return TRUE to ignore the message.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "should_ignore": {
                                "type": "boolean",
                                "description": "TRUE to ignore message, FALSE to process message"
                            }
                        },
                        "required": ["should_ignore"]
                    }
                }
            },
        ]
        try:
            response = call_llm_with_tools(
                HEURIST_BASE_URL,
                HEURIST_API_KEY, 
                SMALL_MODEL_ID,
                "",#"Always call the filter_message tool with the message as the argument",#self.prompt_config.get_telegram_rules(),
                message,
                temperature=0.5,
                tools=filter_message_tool
            )
            print(response)
            #response = response.lower()
            #validation = False if "false" in response else True if "true" in response else False
            validation = False
            if 'tool_calls' in response and response['tool_calls']:
                tool_call = response['tool_calls']
                args = json.loads(tool_call.function.arguments)
                filter_result = str(args['should_ignore']).lower()
                validation = False if filter_result == "true" else True
            print("validation: ", validation)
            return validation
        except Exception as e:
            logger.error(f"Pre-validation failed: {str(e)}")
            return False
    async def generate_image_prompt(self, message: str) -> str:
        """Generate an image prompt based on the tweet content"""
        logger.info("Generating image prompt")
        prompt = self.prompt_config.get_template_image_prompt().format(tweet=message)
        logger.info("Prompt: %s", prompt)
        try:
            image_prompt = call_llm(
                HEURIST_BASE_URL,
                HEURIST_API_KEY, 
                SMALL_MODEL_ID,
                self.prompt_config.get_system_prompt(),
                prompt,
                temperature=0.7,
            )
        except Exception as e:
            logger.error(f"Failed to generate image prompt: {str(e)}")
            return None
        logger.info("Generated image prompt: %s", image_prompt)
        return image_prompt
    
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

    async def handle_message(self, message: str, source_interface: str = None, chat_id: str = None):
        """
        Handle message and optionally notify other interfaces.        
        Args:
            message: The message to process
            source_interface: Optional name of the interface that sent the message
            
        Returns:
            tuple: (text_response, image_url)
        """
        logger.info(f"Handling message from {source_interface}")
        logger.info(f"registered interfaces: {self.interfaces}")

        preValidation = False if source_interface in ["api", "twitter"] else True
        preValidationResult = True
        if preValidation:
            preValidationResult = await self.pre_validation(message)

        if not preValidationResult:
            return None, None#"Ignoring message.", None
        else:
            print("preValidationResult: ", preValidationResult)
            print("answering message")
        system_prompt = self.prompt_config.get_system_prompt()+self.prompt_config.get_basic_settings()+self.prompt_config.get_interaction_styles()
        #print("system_prompt: ", system_prompt)
        try:
            print("calling llm with tools")
            print(self.tools.get_tools_config())
            print("calling llm")
            response = call_llm_with_tools(
                HEURIST_BASE_URL,
                HEURIST_API_KEY, 
                LARGE_MODEL_ID,
                system_prompt,
                message,
                temperature=0.4,
                tools=self.tools.get_tools_config()
            )
            print(response)
            # Check if response is valid
            if not response:
                return "Sorry, I couldn't process your message.", None
                
            # Extract text response
            text_response = ""
            if hasattr(response, 'content') and response.content:
                text_response = response.content.replace('"', '')
                
            image_url = None
            
            # Check if image generation was requested
            if 'tool_calls' in response and response['tool_calls']:
                tool_call = response['tool_calls']
                args = json.loads(tool_call.function.arguments)
                
                # Execute the tool and get results
                tool_result = await self.tools.execute_tool(
                    tool_call.function.name, 
                    args,
                    self
                )

                # Handle tool results
                if tool_result:
                    if 'image_url' in tool_result:
                        image_url = tool_result['image_url']
                    if 'message' in tool_result:
                        text_response += f"\n{tool_result['message']}"

            # Notify other interfaces if needed
            if source_interface and chat_id:
                for interface_name, interface in self.interfaces.items():
                    if interface_name != source_interface:
                        await self.send_to_interface(interface_name, {
                            'type': 'message',
                            'content': text_response,
                            'image_url': image_url,
                            'source': source_interface,
                            'chat_id': chat_id
                        })
            
            return text_response, image_url
            
        except LLMError as e:
            logger.error(f"LLM processing failed: {str(e)}")
            return "Sorry, I encountered an error processing your message.", None
        except Exception as e:
            logger.error(f"Message handling failed: {str(e)}")
            return "Sorry, something went wrong.", None

    async def send_to_interface(self, target_interface: str, message: dict):
        """
        Send a message to a specific interface
        
        Args:
            target_interface (str): Name of the interface to send to
            message (dict): Message data containing at minimum:
                {
                    'type': str,  # Type of message (e.g., 'message', 'image', 'voice')
                    'content': str,  # Main content
                    'image_url': Optional[str],  # Optional image URL
                    'source': Optional[str]  # Source interface name
                }
        
        Returns:
            bool: True if message was queued successfully, False otherwise
        """
        try:
            with self._lock:
                if target_interface not in self.interfaces:
                    logger.error(f"Interface {target_interface} not registered")
                    return False
                
                # Validate message format
                if not isinstance(message, dict) or 'type' not in message or 'content' not in message:
                    logger.error("Invalid message format")
                    return False
                
                # Add timestamp and target
                message['timestamp'] = datetime.now().isoformat()
                message['target'] = target_interface
                
                # Queue the message
                self._message_queue.put(message)
                
                # Get interface instance
                interface = self.interfaces[target_interface]
                logger.info(f"Interface: {interface}")
                logger.info(f"Message: {message}")
                logger.info("trying to send message")
                # Handle different message types
                logger.info(f"Message type: {message['type']}")
                if message['type'] == 'message':
                    logger.info(f"interface has method {hasattr(interface, 'send_message')}")
                    if hasattr(interface, 'send_message'):
                        try:
                            logger.info(f"Attempting to send message via {target_interface} interface")
                            await interface.send_message(
                                chat_id=message['chat_id'],
                                message=message['content'],
                                image_url=message['image_url']
                            )
                            logger.info("Message sent successfully")
                        except Exception as e:
                            logger.error(f"Failed to send message via {target_interface}: {str(e)}")
                            raise
                # Log successful queue
                logger.info(f"Message queued for {target_interface}: {message['type']}")
                return True
                
        except Exception as e:
            logger.error(f"Error sending message to {target_interface}: {str(e)}")
            return False