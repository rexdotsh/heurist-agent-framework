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

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate an image based on a text prompt",
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
    }
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
    
class DiscordAgent:
    def __init__(self):
        # Define intents to allow the bot to read message content
        intents = discord.Intents.default()
        intents.messages = True
        intents.message_content = True

        self.bot = commands.Bot(command_prefix="!", intents=intents)
        self.token = os.getenv("DISCORD_TOKEN")
        self.prompt_config = PromptConfig()
        
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

                # Call LLM with tools
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
                    await message.channel.send("Sorry, I couldn't process your message.")
                    return

                if 'tool_calls' in response and response['tool_calls']:
                    tool_call = response['tool_calls']
                    if tool_call.function.name == 'generate_image':
                        args = json.loads(tool_call.function.arguments)
                        image_result = await self.handle_image_generation(args['prompt'])
                        if image_result:
                            embed = discord.Embed(title="Here you go!", color=discord.Color.blue())
                            embed.set_image(url=image_result)
                            await message.channel.send(embed=embed)
                else:
                    # Handle regular text response
                    if hasattr(response, 'content') and response.content:
                        await message.channel.send(response.content)
                    else:
                        await message.channel.send("I received your message but couldn't generate a proper response.")
                
            except LLMError as e:
                logger.error(f"LLM call failed: {str(e)}")
                await message.channel.send("Sorry, I encountered an error processing your message.")

            # Ensure other commands still work
            await self.bot.process_commands(message)

        # Command: Simple echo function
        @self.bot.command()
        async def echo(ctx, *, message: str):
            await ctx.send(f"You said: {message}")

    async def handle_image_generation(self, prompt: str) -> Optional[str]:
        """Handle image generation tool calls"""
        try:
            result = generate_image_with_retry(prompt=BASE_IMAGE_PROMPT + prompt)
            return result
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            return None

    def run(self):
        self.bot.run(self.token)
