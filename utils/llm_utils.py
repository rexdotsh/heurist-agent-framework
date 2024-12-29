import json
import logging
from core.llm import call_llm

logger = logging.getLogger(__name__)

def should_ignore_message(
    base_url: str,
    api_key: str,
    model_id: str,
    criteria: str,
    message: str,
    temperature: float = 0.0
) -> bool:
    """
    Check if a message should be ignored based on given criteria.
    
    Args:
        base_url: LLM API base URL
        api_key: LLM API key
        model_id: LLM model ID
        criteria: Criteria for ignoring messages
        message: Message to check
        temperature: LLM temperature (defaults to 0.0 for consistent results)
    
    Returns:
        bool: True if message should be ignored, False otherwise
    """
    system_prompt = f"Determine if user message should be ignored. Criteria: {criteria} Your output should be in a JSON code block like this ```json {{\"ignore\": true or false}}``` DO NOT explain. DO NOToutput anything else."
    
    try:
        response = call_llm(
            base_url=base_url,
            api_key=api_key,
            model_id=model_id,
            system_prompt=system_prompt,
            user_prompt=message,
            temperature=temperature,
            max_tokens=100
        )

        # Extract JSON from code block if present
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].strip()
        else:
            json_str = response.strip()

        # Parse JSON and get ignore value
        result = json.loads(json_str)
        return bool(result.get("ignore", False))

    except (json.JSONDecodeError, KeyError, IndexError, Exception) as e:
        logger.warning(f"Error parsing ignore check response: {e}")
        return False  # Default to not ignoring on error