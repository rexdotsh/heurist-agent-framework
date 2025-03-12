import json
import logging
import os
from typing import Dict, List, Optional, Any

from core.embedding import get_embedding, MessageData

logger = logging.getLogger(__name__)

class KnowledgeProvider:
    """Manages knowledge storage and retrieval"""
    
    def __init__(self, message_store):
        self.message_store = message_store
        
    async def update_knowledge_base(self, json_file_path: str = "data/data.json") -> None:
        """Updates the knowledge base by processing JSON data and storing embeddings"""
        logger.info(f"Updating knowledge base from {json_file_path}")

        try:
            # Read JSON file
            with open(json_file_path, "r") as f:
                data = json.load(f)

            # Handle both list and dict formats
            items = data if isinstance(data, list) else [data]

            # Process each item
            for item in items:
                if not isinstance(item, dict):
                    continue

                # Create message content by combining all key-value pairs
                message_parts = []
                for key, value in item.items():
                    if isinstance(value, (str, int, float, bool)):
                        message_parts.append(f"{key}: {value}")
                    elif isinstance(value, (list, dict)):
                        # Handle nested structures by converting to string
                        message_parts.append(f"{key}: {json.dumps(value)}")

                message = "\n\n".join(message_parts)

                # Generate embedding for the message
                message_embedding = get_embedding(message)

                # Check if this exact message already exists
                existing_entries = self.message_store.find_similar_messages(
                    message_embedding,
                    threshold=0.99,  # Very high threshold to match nearly identical content
                )

                if existing_entries:
                    logger.info(f"Knowledge item already exists in database, skipping: {message[:100]}...")
                    continue

                # Store the message with metadata
                self.message_store.add_message(
                    MessageData(
                        message=message,
                        embedding=message_embedding,
                        metadata={
                            "message_type": "knowledge",
                            "source": json_file_path,
                            "timestamp": None,
                        },
                    )
                )

            logger.info(f"Successfully updated knowledge base from {json_file_path}")
            
        except FileNotFoundError:
            logger.error(f"Knowledge base file not found: {json_file_path}")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in knowledge base file: {json_file_path}")
        except Exception as e:
            logger.error(f"Error updating knowledge base: {str(e)}")
            
    async def get_knowledge_context(self, message: str, embedding: List[float], threshold: float = 0.6) -> str:
        """Get relevant knowledge for a query"""
        try:
            if not embedding:
                return ""
                
            # Find similar messages in knowledge base
            similar_messages = self.message_store.find_similar_messages(
                embedding, 
                threshold=threshold,
                message_types=["knowledge"],
                limit=5,
            )
            
            if not similar_messages:
                return ""
                
            # Format knowledge context
            knowledge_context = "Here's some information that might be relevant to the query:\n\n"
            
            for item in similar_messages:
                similarity = item.get("similarity", 0)
                knowledge_context += f"--- Knowledge (similarity: {similarity:.2f}) ---\n{item['message']}\n\n"
                
            return knowledge_context
            
        except Exception as e:
            logger.error(f"Error retrieving knowledge context: {str(e)}")
            return "" 