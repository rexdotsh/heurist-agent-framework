import json
import logging
from typing import Tuple, Optional, Dict, List

logger = logging.getLogger(__name__)

class AugmentedLLMCall:
    """Standard RAG + tools pattern"""
    
    def __init__(self, knowledge_provider, conversation_provider, tool_manager, llm_provider):
        self.knowledge_provider = knowledge_provider
        self.conversation_provider = conversation_provider
        self.tool_manager = tool_manager
        self.llm_provider = llm_provider
    
    async def process(
        self, 
        message: str,
        personality_provider=None,
        chat_id: str = None,
        workflow_options: Dict = None,
        **kwargs
    ) -> Tuple[Optional[str], Optional[str], Optional[Dict]]:
        """Process message using RAG + tools"""
        
        # Set default workflow options
        options = {
            "use_knowledge": True,
            "use_conversation": True,
            "use_similar": True,
            "use_tools": True,
            "max_tokens": None,
            "temperature": 0.7,
        }
        
        # Override with provided options
        if workflow_options:
            options.update(workflow_options)
            
        # Get message embedding if not provided
        embedding = kwargs.get("embedding")
        
        # Build system prompt components
        system_prompt = ""
        
        # Add personality if provided
        if personality_provider:
            system_prompt = personality_provider.get_formatted_personality()
            
        # Get knowledge context if enabled
        if options["use_knowledge"] and embedding:
            knowledge_context = await self.knowledge_provider.get_knowledge_context(message, embedding)
            if knowledge_context:
                system_prompt += f"\n\n{knowledge_context}"
            
        # Get conversation context if enabled
        if options["use_conversation"] and chat_id:
            conversation_context = await self.conversation_provider.get_conversation_context(chat_id)
            if conversation_context:
                system_prompt += f"\n\n{conversation_context}"
                
        # Get similar messages if enabled  
        if options["use_similar"] and embedding:
            similar_context = await self.conversation_provider.get_similar_messages(embedding, chat_id)
            if similar_context:
                system_prompt += f"\n\n{similar_context}"
        
        # Make LLM call with tools if enabled
        try:
            response = await self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=message,
                temperature=options["temperature"],
                max_tokens=options["max_tokens"],
                tools=self.tool_manager.get_tools_config() if options["use_tools"] else None,
            )
            
            # Extract response elements
            text_response = response.get("content", "")
            image_url = None
            tool_result = None
            
            # Handle tool calls if present
            if options["use_tools"] and "tool_calls" in response and response["tool_calls"]:
                tool_call = response["tool_calls"]
                args = json.loads(tool_call.function.arguments)
                tool_name = tool_call.function.name
                
                # Execute the tool
                agent = kwargs.get("agent")
                if agent and tool_name:
                    tool_output = await self.tool_manager.execute_tool(tool_name, args, agent)
                    if tool_output:
                        if "image_url" in tool_output:
                            image_url = tool_output["image_url"]
                        if "result" in tool_output:
                            text_response += f"\n{tool_output['result']}"
                        if "tool_call" in tool_output:
                            tool_result = tool_output["tool_call"]
            
            # Store interaction if needed
            if not kwargs.get("skip_store", False) and chat_id:
                await self.conversation_provider.store_interaction(
                    message, 
                    text_response, 
                    chat_id, 
                    kwargs.get("metadata", {})
                )
                
            return text_response, image_url, tool_result
            
        except Exception as e:
            logger.error(f"LLM processing failed: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}", None, None 