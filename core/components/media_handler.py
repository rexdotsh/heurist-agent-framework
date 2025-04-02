import logging
import os
import random
from pathlib import Path
from typing import Optional

from ..imgen import generate_image_with_retry_smartgen
from ..voice import speak_text, transcribe_audio

logger = logging.getLogger(__name__)


class MediaHandler:
    """Handles media processing (images, audio, etc.)"""

    def __init__(self, llm_provider):
        self.llm_provider = llm_provider
        self.base_image_prompt = os.getenv("BASE_IMAGE_PROMPT", "")

    async def generate_image_prompt(self, message: str) -> str:
        """Generate an image prompt based on the user message"""
        try:
            system_prompt = """
            You are an image prompt creator. Generate a vivid, detailed image prompt based on the user's message.
            Focus on visual elements that would make a good image. Be specific about style, colors, composition, and mood.
            Do not include any explanations, just output the prompt directly.
            """

            response = await self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=f"Create an image prompt based on this: {message}",
                temperature=0.7,
                skip_tools=True,
            )

            image_prompt = response[0]
            logger.info(f"Generated image prompt: {image_prompt}")
            return image_prompt

        except Exception as e:
            logger.error(f"Error generating image prompt: {str(e)}")
            return message  # Fall back to original message

    async def generate_image(self, prompt: str, base_prompt: str = "") -> Optional[str]:
        """Generate an image from a prompt"""
        try:
            actual_base_prompt = base_prompt or self.base_image_prompt

            # Combine prompts if base prompt exists
            full_prompt = f"{actual_base_prompt} {prompt}" if actual_base_prompt else prompt

            # Generate image
            image_url = await generate_image_with_retry_smartgen(full_prompt)
            logger.info(f"Generated image with URL: {image_url}")
            return image_url

        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            return None

    async def transcribe_audio(self, audio_file_path: Path) -> str:
        """Transcribe audio to text"""
        try:
            text = transcribe_audio(audio_file_path)
            logger.info(f"Transcribed audio: {text[:100]}...")
            return text

        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return ""

    async def text_to_speech(self, text: str) -> Optional[Path]:
        """Convert text to speech"""
        try:
            audio_path = speak_text(text)
            logger.info(f"Generated speech audio at: {audio_path}")
            return audio_path

        except Exception as e:
            logger.error(f"Error converting text to speech: {str(e)}")
            return None

    def should_generate_image(self, message: str) -> bool:
        """Determine if we should generate an image for this message"""
        # Simple probability-based determination
        image_words = [
            "picture",
            "image",
            "photo",
            "artwork",
            "drawing",
            "visualize",
            "show me",
            "generate",
            "create",
            "make",
        ]

        message_lower = message.lower()

        # Higher probability if message explicitly mentions images
        if any(word in message_lower for word in image_words):
            return random.random() < 0.8

        # Base probability otherwise
        image_probability = float(os.getenv("IMAGE_GENERATION_PROBABILITY", 0.3))
        return random.random() < image_probability
