from typing import Dict, Any, List, Optional
import logging
import os
from dotenv import load_dotenv
from smolagents import ToolCallingAgent, tool, Model, ChatMessage, Tool
from smolagents.models import parse_tool_args_if_needed

def smolagents_system_prompt() -> str:
    return '''
You are an expert assistant who can solve any task using tool calls. You will be given a task to solve as best you can.
To do so, you have been given access to some tools.

The tool call you write is an action: after the tool is executed, you will get the result of the tool call as an "observation".
This Action/Observation can repeat multiple times, you should take several steps when needed.

You can use the observation result of the previous action as input for the next action. The observation will always be a string.

You only have access to the tools provided to you, including the final_answer tool which is the ONLY way to get back to the user.

Here are the rules you should always follow to solve your task:
1. ALWAYS provide a tool call, else you will fail.
2. Always use the right arguments for the tools. Never use variable names as the action arguments, use the value instead.
3. Reflect on the previous conversation and decide if you need to call a different tool to find out more information or perform a different action.
4. Call a tool only when needed: for example, do not call the search tool if you do not need information, try to solve the task yourself. If no tool call is needed, MUST use the final_answer tool to return your answer.
5. NEVER re-do a tool call that you previously did with the exact same function name and exact same parameters.

Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.
'''

class OpenAIServerModel(Model):
    """This model connects to an OpenAI-compatible API server.

    Parameters:
        model_id (`str`):
            The model identifier to use on the server (e.g. "gpt-3.5-turbo").
        api_base (`str`, *optional*):
            The base URL of the OpenAI-compatible API server.
        api_key (`str`, *optional*):
            The API key to use for authentication.
        organization (`str`, *optional*):
            The organization to use for the API request.
        project (`str`, *optional*):
            The project to use for the API request.
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        **kwargs:
            Additional keyword arguments to pass to the OpenAI API.
    """

    def __init__(
        self,
        model_id: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] | None = None,
        project: Optional[str] | None = None,
        custom_role_conversions: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        try:
            import openai
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install 'openai' extra to use OpenAIServerModel: `pip install 'smolagents[openai]'`"
            ) from None

        super().__init__(**kwargs)
        self.model_id = model_id
        self.client = openai.OpenAI(
            base_url=api_base,
            api_key=api_key,
            organization=organization,
            project=project,
        )
        self.custom_role_conversions = custom_role_conversions

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        response = self.client.chat.completions.create(**completion_kwargs)
        #print(f"Response: {response}")
        # Some models don't return usage data
        self.last_input_token_count = 0 #response.usage.prompt_tokens
        self.last_output_token_count = 0 #response.usage.completion_tokens

        message = ChatMessage.from_dict(
            response.choices[0].message.model_dump(include={"role", "content", "tool_calls"})
        )
        #print(f"Message: {message}")
        message.raw = response
        if tools_to_call_from is not None:
            return parse_tool_args_if_needed(message)
        return message