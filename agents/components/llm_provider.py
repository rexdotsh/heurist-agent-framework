import os
import logging
from typing import Dict, List, Optional, Any

# Import your LLM functions
from core.llm import call_llm, call_llm_with_tools, LLMError

logger = logging.getLogger(__name__)

class LLMProvider:
    """Handles interactions with LLM services"""
    
    def __init__(
        self, 
        base_url: str = None, 
        api_key: str = None,
        large_model_id: str = None,
        small_model_id: str = None
    ):
        self.base_url = base_url or os.getenv("HEURIST_BASE_URL")
        self.api_key = api_key or os.getenv("HEURIST_API_KEY")
        self.large_model_id = large_model_id or os.getenv("LARGE_MODEL_ID")
        self.small_model_id = small_model_id or os.getenv("SMALL_MODEL_ID")
        
    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = None,
        model_id: str = None,
        skip_tools: bool = True,
        tools: List[Dict] = None,
        tool_choice: str = None,
        **kwargs
    ) -> Dict:
        """Make a call to the LLM"""
        try:
            # Determine which model to use
            use_model = model_id or self.large_model_id
            
            # Choose appropriate call based on tools requirement
            if not skip_tools and tools:
                response = call_llm_with_tools(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    model_id=use_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    tool_choice=tool_choice
                )
            else:
                response = call_llm(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    model_id=use_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
            return response
                
        except LLMError as e:
            logger.error(f"LLM call failed: {str(e)}")
            raise
            
    async def call_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = None,
        tools: List[Dict] = None,
        tool_choice: str = "auto",
        **kwargs
    ) -> Dict:
        """Specialized method for tool-based calls"""
        return await self.call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            skip_tools=False,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs
        )
        
    async def classify_text(self, text: str, classification_prompt: str) -> str:
        """Classify text using the small model"""
        try:
            result = call_llm(
                base_url=self.base_url,
                api_key=self.api_key,
                model_id=self.small_model_id,
                system_prompt=classification_prompt,
                user_prompt=text,
                temperature=0.3
            )
            
            return result.get("content", "UNKNOWN").strip().upper()
            
        except Exception as e:
            logger.error(f"Text classification failed: {str(e)}")
            return "UNKNOWN" 