from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
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
BASE_IMAGE_PROMPT = ""#" long straight purple hair, blunt bangs, blue eyes, purple witch hat, white robe, best quality, masterpiece,"

# Constants
TELEGRAM_API_TOKEN = os.getenv("TELEGRAM_API_TOKEN")

if not TELEGRAM_API_TOKEN:
    raise ValueError("TELEGRAM_API_TOKEN not found in environment variables")

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
    
class TelegramAgent:
    def __init__(self):
        self.app = Application.builder().token(TELEGRAM_API_TOKEN).build()
        self._setup_handlers()
        self.prompt_config = PromptConfig()

    def _setup_handlers(self):
        # Register the /start command handler
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("image", self.image))
        self.app.add_handler(MessageHandler(filters.VOICE, self.handle_voice))
        # Register a handler for echoing messages
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.message))

    def fill_basic_prompt(self, basic_options, style_options):
        return self.prompt_config.get_basic_prompt_template().format(
            basic_option_1=basic_options[0],
            basic_option_2=basic_options[1],
            style_option_1=style_options[0],
            style_option_2=style_options[1]
        )
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("Hello World! I'm not a bot... I promise... ")

    async def image(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        # Get the text after the /image command
        prompt = ' '.join(context.args) if context.args else None
        
        if not prompt:
            await update.message.reply_text("Please provide a prompt after /image command")
            return
        prompt = BASE_IMAGE_PROMPT + prompt
        # Generate image using the prompt
        try:
            result = generate_image_with_retry(prompt=prompt)
            if result:
                # Send the generated image as a photo
                await update.message.reply_photo(photo=result)
                await update.message.reply_text("Image generated successfully.")
            else:
                await update.message.reply_text("Failed to generate image")
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            await update.message.reply_text("Sorry, there was an error generating the image")

    async def message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_message = update.message.text
        
        try:
            response = call_llm_with_tools(
                HEURIST_BASE_URL,
                HEURIST_API_KEY, 
                LARGE_MODEL_ID,
                self.prompt_config.get_system_prompt(),
                user_message,
                temperature=0.01,
                tools=TOOLS
            )
            
            # Check if response is valid
            if not response:
                await update.message.reply_text("Sorry, I couldn't process your message.")
                return

            if 'tool_calls' in response and response['tool_calls']:
                tool_call = response['tool_calls']  # Now accessing the single tool call
                if tool_call.function.name == 'generate_image':
                    args = json.loads(tool_call.function.arguments)
                    image_result = await self.handle_image_generation(args['prompt'])
                    if image_result:
                        await update.message.reply_photo(photo=image_result)
            
            else:
                # Handle regular text response
                if hasattr(response, 'content') and response.content:
                    await update.message.reply_text(response.content)
                else:
                    await update.message.reply_text("I received your message but couldn't generate a proper response.")
                
        except LLMError as e:
            logger.error(f"LLM call failed: {str(e)}")
            await update.message.reply_text("Sorry, I encountered an error processing your message.")

    async def handle_image_generation(self, prompt: str) -> Optional[str]:
        """Handle image generation tool calls"""
        try:
            result = generate_image_with_retry(prompt=BASE_IMAGE_PROMPT + prompt)
            return result
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            return None

    async def handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message.voice:
            # Get the file ID of the voice note
            file_id = update.message.voice.file_id

            # Get the file from Telegram's servers
            file = await context.bot.get_file(file_id)

            project_root = Path(__file__).parent.parent
            audio_dir = project_root / "audio"
            audio_dir.mkdir(exist_ok=True)
            
            # Define the file path where the audio will be saved
            file_path = audio_dir / f"{file_id}.ogg"

            # Download the file
            await file.download_to_drive(file_path)

            # Notify the user
            await update.message.reply_text("Voice note received. Processing...")
            text = transcribe_audio(file_path)
            print(text)
            # basic_options = random.sample(self.prompt_config.get_basic_settings(), 2)
            # style_options = random.sample(self.prompt_config.get_interaction_styles(), 2)
            
            # prompt = self.fill_basic_prompt(basic_options, style_options)
            
            user_prompt = (text)
            
            try:
            
                response = call_llm(
                HEURIST_BASE_URL,
                HEURIST_API_KEY, 
                SMALL_MODEL_ID,
                self.prompt_config.get_system_prompt(),
                user_prompt,
                temperature=0.7
                )
                
                # Convert LLM response to speech
                audio_path = speak_text(response)
                
                # Send audio response back to user
                with open(audio_path, 'rb') as audio:
                    await update.message.reply_voice(voice=audio)
                    
                # Also send text response
                await update.message.reply_text(response)
                
            except Exception as e:
                logger.error(f"Error processing voice message: {str(e)}")
                await update.message.reply_text("Sorry, there was an error processing your voice message")

    def run(self):
        """Start the bot"""
        logger.info("Starting Telegram bot...")
        self.app.run_polling()

def main():
    agent = TelegramAgent()
    agent.run()

if __name__ == "__main__":
    try:
        logger.info("Starting Telegram agent...")
        main()
    except KeyboardInterrupt:
        logger.info("\nTelegram agent stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")