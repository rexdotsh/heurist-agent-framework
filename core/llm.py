import requests
import json
import time
import os
import logging
from openai import OpenAI
from typing import Dict, List, Union
import requests
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMError(Exception):
    """Custom exception for LLM-related errors"""
    pass

def call_llm(
    base_url: str,
    api_key: str,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int = 500,
    max_retries: int = 3,
    initial_retry_delay: int = 1
) -> str:
    """
    Call LLM with retry mechanism.

    Parameters:
        model_id (str): The model identifier for the LLM.
        system_prompt (str): The system prompt.
        user_prompt (str): The user input prompt.
        temperature (float): The temperature setting for response generation.
        max_tokens (int): Maximum number of tokens to generate.
        max_retries (int): Number of retry attempts on failure.
        initial_retry_delay (int): Initial delay between retries, with exponential backoff.

    Returns:
        str: Generated text from LLM.
    
    Raises:
        LLMError: If all retry attempts fail.
    """

    client = OpenAI(base_url=base_url, api_key=api_key)
    
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ]
    
    retry_delay = initial_retry_delay
    
    for attempt in range(max_retries):
        try:
            result = client.chat.completions.create(
                model=model_id,
                messages=messages,
                stream=False,
                temperature=temperature,
                max_tokens=max_tokens
            )

            return result.choices[0].message.content
        
        except requests.exceptions.RequestException as e:
            logger.warning(f"API request failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.warning(f"Response parsing failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
        except Exception as e:
            logger.warning(f"Unexpected error (attempt {attempt + 1}/{max_retries}): {str(e)}")
        
        # Wait before next retry if there are attempts left
        if attempt < max_retries - 1:
            logger.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
    
    # Raise error if all attempts fail
    raise LLMError("All retry attempts failed")


def call_llm_with_tools(
    base_url: str,
    api_key: str,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int = 500,
    max_retries: int = 3,
    tools: List[Dict] = None
) -> Union[str, Dict]:
    client = OpenAI(base_url=base_url, api_key=api_key)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            tools=tools,
            tool_choice="auto"# if tools else None
        )

        message = response.choices[0].message

        # If there are tool calls, return both tool calls and content
        if hasattr(message, 'tool_calls') and message.tool_calls:
            return {
                'tool_calls': message.tool_calls[0],
                'content': message.content
            }
        if hasattr(message, 'content') and message.content:
            text_response = message.content
            return {
                'content': text_response
            }
        # Otherwise return just the content
        return message

    except Exception as e:
        raise LLMError(f"LLM API call failed: {str(e)}")