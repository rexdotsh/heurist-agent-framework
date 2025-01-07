import discord
from discord.ext import commands
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

    
class DiscordAgent(CoreAgent):
    def __init__(self):
        super().__init__()
        # Define intents to allow the bot to read message content
        intents = discord.Intents.default()
        intents.messages = True
        intents.message_content = True

        self.bot = commands.Bot(command_prefix="!", intents=intents)
        self.token = os.getenv("DISCORD_TOKEN")
        
        self.setup_handlers()

    def setup_handlers(self):
        @self.bot.event
        async def on_ready():
            print(f"Logged in as {self.bot.user}")

        @self.bot.command()
        async def hello(ctx):
            await ctx.send("Hello! How can I help you?")

        @self.bot.event
        async def on_message(message):
            # Ignore bot's own messages
            if message.author == self.bot.user:
                return

            try:
                # Get user message
                user_message = message.content.lower()

                # Handle message using core agent functionality
                text_response, image_url, _ = await self.handle_message(user_message)

                if image_url:
                    embed = discord.Embed(title="Here you go!", color=discord.Color.blue())
                    embed.set_image(url=image_url)
                    await message.channel.send(embed=embed)
                elif text_response:
                    await message.channel.send(text_response)
                else:
                    await message.channel.send("Sorry, I couldn't process your message.")

            except Exception as e:
                logger.error(f"Message handling failed: {str(e)}")
                await message.channel.send("Sorry, I encountered an error processing your message.")

            # Ensure other commands still work
            await self.bot.process_commands(message)

        # Command: Simple echo function
        @self.bot.command()
        async def echo(ctx, *, message: str):
            await ctx.send(f"You said: {message}")

    def run(self):
        self.bot.run(self.token)
