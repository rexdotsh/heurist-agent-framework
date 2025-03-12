import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from core.embedding import get_embedding, MessageData

logger = logging.getLogger(__name__)

class ConversationManager:
    """Manages conversation context and history"""

    def __init__(self, message_store):
        self.message_store = message_store
        
    async def get_conversation_context(
        self,
        chat_id: str,
        limit: int = 10
    ) -> str:
        """Get conversation history context"""
        if not chat_id:
            return ""
            
        try:
            # Retrieve recent messages for this chat
            messages = self.message_store.get_recent_messages(
                chat_id=chat_id,
                limit=limit,
                message_types=["user_message", "agent_response"]
            )
            
            if not messages:
                return ""
                
            # Format conversation context
            conversation_context = "Here's our recent conversation history:\n\n"
            
            for msg in messages:
                msg_type = msg.get("metadata", {}).get("message_type", "unknown")
                if msg_type == "user_message":
                    conversation_context += f"User: {msg['message']}\n"
                else:
                    conversation_context += f"Assistant: {msg['message']}\n"
                    
            conversation_context += "\nContinue the conversation in a consistent and helpful manner.\n"
            
            return conversation_context
            
        except Exception as e:
            logger.error(f"Error retrieving conversation context: {str(e)}")
            return ""

    async def get_similar_messages(
        self,
        embedding: List[float],
        chat_id: Optional[str] = None,
        threshold: float = 0.7,
        limit: int = 5
    ) -> str:
        """Get context from similar previous conversations"""
        if not embedding:
            return ""
            
        try:
            # Find similar previous conversations
            similar_messages = self.message_store.find_similar_messages(
                embedding,
                threshold=threshold,
                message_types=["user_message"],
                limit=limit * 2,  # Get more to filter later
            )
            
            if not similar_messages:
                return ""
                
            # Get related responses
            context = "Here are some similar questions and answers that might be relevant:\n\n"
            seen_responses = set()
            message_count = 0
            
            for similar_msg in similar_messages:
                # Skip if we've already included enough messages
                if message_count >= limit:
                    break
                    
                # Skip if from the current chat (to avoid duplicating current context)
                if chat_id and similar_msg.get("metadata", {}).get("chat_id") == chat_id:
                    continue
                    
                # Get agent responses to this similar message
                agent_responses = self.message_store.get_responses_for_message(
                    similar_msg["id"],
                    message_type="agent_response"
                )
                
                for response in agent_responses:
                    if response["message"] in seen_responses:
                        continue
                        
                    seen_responses.add(response["message"])
                    context += f"""
                    Previous similar question: {similar_msg["message"]}
                    My response: {response["message"]}
                    Similarity score: {similar_msg.get("similarity", 0):.2f}
                    """
                    message_count += 1
                    if message_count >= limit:
                        break
                        
            if message_count > 0:
                context += "\nConsider the above responses for context, but provide a fresh perspective that adds value to the conversation.\n"
                return context
                
            return ""
            
        except Exception as e:
            logger.error(f"Error retrieving similar messages: {str(e)}")
            return ""

    async def store_interaction(
        self,
        message: str,
        response: str,
        chat_id: str,
        metadata: Dict = None
    ) -> None:
        """Store a conversation interaction"""
        if not message or not response:
            return
            
        try:
            # Prepare metadata
            msg_metadata = metadata.copy() if metadata else {}
            msg_metadata.update({
                "message_type": "user_message",
                "chat_id": chat_id,
                "timestamp": datetime.now().isoformat(),
            })
            
            # Store user message
            message_embedding = get_embedding(message)
            message_id = self.message_store.add_message(
                MessageData(
                    message=message,
                    embedding=message_embedding,
                    metadata=msg_metadata
                )
            )
            
            # Prepare response metadata
            resp_metadata = {
                "message_type": "agent_response",
                "chat_id": chat_id,
                "timestamp": datetime.now().isoformat(),
                "parent_message_id": message_id
            }
            
            # Store agent response
            response_embedding = get_embedding(response)
            self.message_store.add_message(
                MessageData(
                    message=response,
                    embedding=response_embedding,
                    metadata=resp_metadata
                )
            )
            
        except Exception as e:
            logger.error(f"Error storing interaction: {str(e)}")