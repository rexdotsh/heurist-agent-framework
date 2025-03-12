import os
import json
import random
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class PersonalityProvider:
    """Manages agent personality and prompt configurations"""
    
    def __init__(self, config_path=None, default_personality="balanced"):
        self.personality_configs = {
            "balanced": {
                "name": "Assistant",
                "basic_settings": [
                    "You're a helpful AI assistant.",
                    "You give concise, accurate answers.",
                    "You're friendly but professional."
                ],
                "interaction_styles": [
                    "You communicate in a clear, straightforward manner.",
                    "You use examples to illustrate complex concepts.",
                    "You occasionally use thoughtful questions to clarify understanding."
                ],
                "system_prompts": {
                    "standard": "You are a helpful AI assistant that provides accurate and thoughtful responses.",
                    "cot_final": "You are a thoughtful assistant that breaks down complex questions into steps before answering."
                }
            }
        }
        
        self.current_personality = default_personality
        
        # Try to load custom configuration if path provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    custom_configs = json.load(f)
                    self.personality_configs.update(custom_configs)
                    logger.info(f"Loaded custom personality configurations from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load personality configurations: {str(e)}")
    
    def set_personality(self, personality_name: str) -> bool:
        """Set the current personality"""
        if personality_name in self.personality_configs:
            self.current_personality = personality_name
            return True
        return False
        
    def get_system_prompt(self, workflow_type: str = "standard") -> str:
        """Get the appropriate system prompt based on personality and workflow"""
        personality = self.personality_configs.get(self.current_personality, {})
        system_prompts = personality.get("system_prompts", {})
        
        # Return requested prompt type or fall back to standard
        if workflow_type in system_prompts:
            return system_prompts[workflow_type]
        
        # Return standard prompt or default if not found
        return system_prompts.get("standard", "You are a helpful AI assistant.")
        
    def get_name(self) -> str:
        """Get the agent's name from current personality"""
        personality = self.personality_configs.get(self.current_personality, {})
        return personality.get("name", "Assistant")
        
    def get_basic_settings(self) -> List[str]:
        """Get basic personality settings"""
        personality = self.personality_configs.get(self.current_personality, {})
        return personality.get("basic_settings", [])
        
    def get_interaction_styles(self) -> List[str]:
        """Get interaction style settings"""
        personality = self.personality_configs.get(self.current_personality, {})
        return personality.get("interaction_styles", [])
        
    def randomize_traits(self) -> None:
        """Randomize personality traits within the current personality"""
        personality = self.personality_configs.get(self.current_personality, {})
        
        # Nothing to randomize if personality not found
        if not personality:
            return
            
        # Shuffle basic settings and interaction styles if available
        if "basic_settings" in personality and len(personality["basic_settings"]) > 2:
            random.shuffle(personality["basic_settings"])
            
        if "interaction_styles" in personality and len(personality["interaction_styles"]) > 2:
            random.shuffle(personality["interaction_styles"])
    
    def get_formatted_personality(self) -> str:
        """Get a formatted string representing the current personality"""
        personality = self.personality_configs.get(self.current_personality, {})
        
        basic_options = random.sample(self.get_basic_settings(), min(2, len(self.get_basic_settings())))
        style_options = random.sample(self.get_interaction_styles(), min(2, len(self.get_interaction_styles())))
        
        system_prompt = "Use the following settings as part of your personality and voice: "
        system_prompt += " ".join(basic_options) + " " + " ".join(style_options)
        
        return system_prompt 