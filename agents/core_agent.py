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
from core.embedding import get_embedding, MessageStore, PostgresConfig, PostgresVectorStorage, EmbeddingError, SQLiteConfig, SQLiteVectorStorage, MessageData
import threading
from queue import Queue
import asyncio
from agents.tools import Tools

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# os.reload_environ()
# dotenv.load_dotenv(override=True)


os.environ.clear()
dotenv.load_dotenv(override=True)
logger.info("Environment variables reloaded")
# Constants
HEURIST_BASE_URL = os.getenv("HEURIST_BASE_URL")
HEURIST_API_KEY = os.getenv("HEURIST_API_KEY")
LARGE_MODEL_ID = os.getenv("LARGE_MODEL_ID")
SMALL_MODEL_ID = os.getenv("SMALL_MODEL_ID")
TWEET_WORD_LIMITS = [15, 20, 30, 35]
IMAGE_GENERATION_PROBABILITY = 0.3
BASE_IMAGE_PROMPT = ""
CONFIG_PROMPTS = os.getenv("CONFIG_PROMPTS", "prompts.yaml")


class PromptConfig:
    def __init__(self, config_path: str = None):
    
        if config_path is None:
            # Get the project root directory (2 levels up from the current file)
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / CONFIG_PROMPTS
            logger.info(f"Using config file: {config_path}")
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

        # Use PostgreSQL if configured, otherwise default to SQLite
        if all([os.getenv(env) for env in ["VECTOR_DB_NAME", "VECTOR_DB_USER", "VECTOR_DB_PASSWORD"]]):
            vdb_config = PostgresConfig(
                host=os.getenv("VECTOR_DB_HOST", "localhost"),
                port=int(os.getenv("VECTOR_DB_PORT", 5432)),
                database=os.getenv("VECTOR_DB_NAME"),
                user=os.getenv("VECTOR_DB_USER"),
                password=os.getenv("VECTOR_DB_PASSWORD"),
                table_name=os.getenv("VECTOR_DB_TABLE", "message_embeddings")
            )
            storage = PostgresVectorStorage(vdb_config)
        else:
            config = SQLiteConfig()
            storage = SQLiteVectorStorage(config)
        
        self.message_store = MessageStore(storage)
    
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
            chat_id: Optional chat ID for the conversation
            
        Returns:
            tuple: (text_response, image_url)
        """
        logger.info(f"Handling message from {source_interface}")
        logger.info(f"registered interfaces: {self.interfaces}")

        preValidation = False if source_interface in ["api", "twitter"] else True
        if preValidation and not await self.pre_validation(message):
            return None, None
        
        try:
            # Generate embedding for the incoming message
            embedding = get_embedding(message)
            logger.info(f"Generated embedding for message: {message[:50]}...")
            
            # Create MessageData for incoming message
            message_data = MessageData(
                message=message,
                embedding=embedding,
                timestamp=datetime.now().isoformat(),
                message_type="user_message",
                chat_id=chat_id,
                source_interface=source_interface,
                original_query=None,
                response_type=None,
                key_topics=None
            )
            
            # Store the incoming message
            self.message_store.add_message(message_data)
            logger.info("Stored message and embedding in database")
            
            # Find similar messages and their responses
            similar_messages = self.message_store.find_similar_messages(embedding, threshold=0.8)
            logger.info(f"Found {len(similar_messages)} similar messages")
            
            # Build context from similar conversations and responses
            system_prompt = (self.prompt_config.get_system_prompt() + 
                            self.prompt_config.get_basic_settings() + 
                            self.prompt_config.get_interaction_styles())
            
            if similar_messages:
                context = "\n\nRelated previous conversations and responses\nNOTE: Please provide a response that differs from these recent replies:\n"
                seen_responses = set()  # Track unique responses
                
                for msg in similar_messages:
                    if msg.get('message_type') == 'agent_response' and msg.get('original_query'):
                        response = msg['message']
                        # Skip if we've seen this exact response before
                        if response in seen_responses:
                            continue
                        seen_responses.add(response)
                        context += f"""
                            Previous similar question: {msg.get('original_query')}
                            My response: {response}
                            Similarity score: {msg.get('similarity', 0):.2f}

                            """
                
                context += "\nConsider the above responses for context, but provide a fresh perspective that adds value to the conversation.\n"
                system_prompt += context
                
                logger.info("Added context from similar conversations")
            
            # Call LLM with tools and enhanced context
            response = call_llm_with_tools(
                HEURIST_BASE_URL,
                HEURIST_API_KEY,
                LARGE_MODEL_ID,
                system_prompt,
                message,
                temperature=0.4,
                tools=self.tools.get_tools_config()
            )
            
            # Process response and handle tools
            text_response = ""
            image_url = None
            
            if not response:
                return "Sorry, I couldn't process your message.", None
            
            if hasattr(response, 'content') and response.content:
                text_response = response.content.replace('"', '')
            
            # Handle tool calls
            if 'tool_calls' in response and response['tool_calls']:
                tool_call = response['tool_calls']
                args = json.loads(tool_call.function.arguments)
                tool_result = await self.tools.execute_tool(
                    tool_call.function.name,
                    args,
                    self
                )
                if tool_result:
                    if 'image_url' in tool_result:
                        image_url = tool_result['image_url']
                    if 'message' in tool_result:
                        text_response += f"\n{tool_result['message']}"
            
            # Create and store MessageData for the response
            response_data = MessageData(
                message=text_response,
                embedding=get_embedding(text_response),
                timestamp=datetime.now().isoformat(),
                message_type="agent_response",
                chat_id=chat_id,
                source_interface=source_interface,
                original_query=message,
                response_type=await self._classify_response_type(text_response),
                key_topics=await self._extract_key_topics(text_response)
            )
            
            # Store the response
            self.message_store.add_message(response_data)
            
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

    async def _classify_response_type(self, response: str) -> str:
        """Classify the type of response (factual, opinion, question, etc.)"""
        classify_prompt = {
            "role": "system",
            "content": "Classify this response as one of: FACTUAL, OPINION, QUESTION, EMOTIONAL, ACTION. Response:"
        }
        try:
            classification = await call_llm(
                HEURIST_BASE_URL,
                HEURIST_API_KEY,
                SMALL_MODEL_ID,  # Use smaller model for classification
                classify_prompt["content"],
                response,
                temperature=0.3
            )
            return classification.strip().upper()
        except:
            return "general"

    async def _extract_key_topics(self, text: str) -> List[str]:
        """Extract key topics from the response for better similarity matching"""
        topic_prompt = {
            "role": "system",
            "content": "Extract 2-3 main topics from this text as comma-separated keywords:"
        }
        try:
            topics = await call_llm(
                HEURIST_BASE_URL,
                HEURIST_API_KEY,
                SMALL_MODEL_ID,
                topic_prompt["content"],
                text,
                temperature=0.3
            )
            return [t.strip() for t in topics.split(',')]
        except:
            return []

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