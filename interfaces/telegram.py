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
from agents.core_agent import CoreAgent


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()


# Constants
TELEGRAM_API_TOKEN = os.getenv("TELEGRAM_API_TOKEN")

if not TELEGRAM_API_TOKEN:
    raise ValueError("TELEGRAM_API_TOKEN not found in environment variables")
 
class TelegramAgent(CoreAgent):
    def __init__(self):
        super().__init__()
        # Initialize telegram specific stuff
        self.app = Application.builder().token(TELEGRAM_API_TOKEN).build()
        self._setup_handlers()

    def _setup_handlers(self):
        # Register the /start command handler
        self.app.add_handler(CommandHandler("start", self.start))
        # Register the /image command handler
        self.app.add_handler(CommandHandler("image", self.image))
        # Register a handler for voice messages
        self.app.add_handler(MessageHandler(filters.VOICE, self.handle_voice))
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

        # Generate image using the prompt
        try:
            result = await self.handle_image_generation(prompt=prompt)
            if result:
                # Send the generated image as a photo using the URL
                await update.message.reply_photo(photo=result)
                await update.message.reply_text("Image generated successfully.")
            else:
                await update.message.reply_text("Failed to generate image")
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            await update.message.reply_text("Sorry, there was an error generating the image")

    async def message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_message = update.message.text
        
        text_response, image_url = await self.handle_message(user_message)
        
        if image_url:
            await update.message.reply_photo(photo=image_url)
        elif text_response:
            await update.message.reply_text(text_response)

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
            user_message = await self.transcribe_audio(file_path)
            text_response, image_url = await self.handle_message(user_message)
        
            if image_url:
                await update.message.reply_photo(photo=image_url)
            elif text_response:
                await update.message.reply_text(text_response)
            
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