from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
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
from core.llm import call_llm, LLMError
from core.imgen import generate_image_with_retry, generate_image_prompt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

# Constants
HEURIST_BASE_URL = "https://llm-gateway.heurist.xyz"
HEURIST_API_KEY = os.getenv("HEURIST_API_KEY")
LARGE_MODEL_ID = "nvidia/llama-3.1-nemotron-70b-instruct"
SMALL_MODEL_ID = "mistralai/mixtral-8x7b-instruct"
TWEET_WORD_LIMITS = [15, 20, 30, 35]
IMAGE_GENERATION_PROBABILITY = 0.3
DRYRUN = os.getenv("DRYRUN")
BASE_IMAGE_PROMPT = " long straight purple hair, blunt bangs, blue eyes, purple witch hat, white robe, best quality, masterpiece,"

# Constants
TELEGRAM_API_TOKEN = os.getenv("TELEGRAM_API_TOKEN")

if not TELEGRAM_API_TOKEN:
    raise ValueError("TELEGRAM_API_TOKEN not found in environment variables")

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
        # Register a handler for echoing messages
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.message))

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
        # Call LLM with user message
        try:
            system_prompt = "You are a helpful AI."
            response = call_llm(
                HEURIST_BASE_URL,
                HEURIST_API_KEY, 
                SMALL_MODEL_ID,
                system_prompt,
                user_message,
                temperature=0.7
            )
        except LLMError as e:
            logger.error(f"LLM call failed: {str(e)}")
        await update.message.reply_text(response)

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