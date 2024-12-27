import os
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import execute_values
from dataclasses import dataclass
import sqlite3
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingError(Exception):
    """Custom exception for embedding-related errors"""
    pass

@dataclass
class StorageConfig:
    """Base configuration for storage providers"""
    pass

@dataclass
class PostgresConfig(StorageConfig):
    """PostgreSQL specific configuration"""
    host: str
    port: int
    database: str
    user: str
    password: str
    table_name: str = "message_embeddings"

@dataclass
class SQLiteConfig(StorageConfig):
    """SQLite specific configuration"""
    db_path: str = "embeddings.db"
    table_name: str = "message_embeddings"

@dataclass
class MessageData:
    message: str
    embedding: List[float]
    timestamp: str
    message_type: str
    chat_id: Optional[str]
    source_interface: Optional[str]
    original_query: Optional[str]
    response_type: Optional[str]
    key_topics: Optional[List[str]]

class VectorStorageProvider(ABC):
    """Abstract base class for vector storage providers"""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the storage (create tables, indexes, etc.)"""
        pass
    
    @abstractmethod
    def store_embedding(self, message_data: MessageData) -> None:
        """Store a message and its metadata with embedding"""
        pass
    
    @abstractmethod
    def find_similar(self, embedding: List[float], threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find similar messages based on embedding similarity"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Clean up resources"""
        pass

class PostgresVectorStorage(VectorStorageProvider):
    def __init__(self, config: PostgresConfig):
        self.config = config
        self.conn = None
        
    def initialize(self) -> None:
        """Initialize PostgreSQL connection and create necessary tables"""
        try:
            self.conn = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password
            )
            
            with self.conn.cursor() as cur:
                # Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                
                # Create table with extended fields
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.config.table_name} (
                        id SERIAL PRIMARY KEY,
                        message TEXT NOT NULL,
                        embedding vector(1536) NOT NULL,
                        timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                        message_type VARCHAR(50) NOT NULL,
                        chat_id VARCHAR(100),
                        source_interface VARCHAR(50),
                        original_query TEXT,
                        response_type VARCHAR(50),
                        key_topics TEXT[],
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create vector similarity index
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS embedding_idx 
                    ON {self.config.table_name} 
                    USING ivfflat (embedding vector_cosine_ops)
                """)
                
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL storage: {str(e)}")
            raise

    def store_embedding(self, message_data: MessageData) -> None:
        """Store a message and its embedding in PostgreSQL"""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"""INSERT INTO {self.config.table_name} 
                    (message, embedding, timestamp, message_type, chat_id,
                    source_interface, original_query, response_type, key_topics)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (message_data.message, message_data.embedding,
                     message_data.timestamp, message_data.message_type,
                     message_data.chat_id, message_data.source_interface,
                     message_data.original_query, message_data.response_type,
                     message_data.key_topics)
                )
            self.conn.commit()
            logger.info("Successfully stored message with metadata in database")
        except Exception as e:
            logger.error(f"Failed to store message: {str(e)}")
            raise

    def find_similar(self, embedding: List[float], threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find similar messages using vector similarity search"""
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"""
                    SELECT message, 1 - (embedding <=> %s::vector) as similarity
                    FROM {self.config.table_name}
                    WHERE 1 - (embedding <=> %s::vector) >= %s
                    ORDER BY similarity DESC
                """, (embedding, embedding, threshold))
                
                results = []
                for message, similarity in cur.fetchall():
                    results.append({
                        'message': message,
                        'similarity': similarity
                    })
                return results
        except Exception as e:
            logger.error(f"Failed to find similar messages: {str(e)}")
            raise

    def close(self) -> None:
        """Close PostgreSQL connection"""
        if self.conn:
            self.conn.close()

class SQLiteVectorStorage(VectorStorageProvider):
    def __init__(self, config: SQLiteConfig):
        self.config = config
        self.conn = None
        
    def initialize(self) -> None:
        """Initialize SQLite connection and create necessary tables"""
        try:
            self.conn = sqlite3.connect(self.config.db_path)
            with self.conn:
                cur = self.conn.cursor()
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.config.table_name} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        message TEXT NOT NULL,
                        embedding TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        message_type TEXT NOT NULL,
                        chat_id TEXT,
                        source_interface TEXT,
                        original_query TEXT,
                        response_type TEXT,
                        key_topics TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            logger.info(f"Initialized SQLite storage at {self.config.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite storage: {str(e)}")
            raise

    def store_embedding(self, message_data: MessageData) -> None:
        """Store a message and its embedding in SQLite"""
        try:
            embedding_json = json.dumps(message_data.embedding)
            key_topics_json = json.dumps(message_data.key_topics) if message_data.key_topics else None
            
            with self.conn:
                self.conn.execute(
                    f"""INSERT INTO {self.config.table_name}
                    (message, embedding, timestamp, message_type, chat_id,
                    source_interface, original_query, response_type, key_topics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (message_data.message, embedding_json, message_data.timestamp,
                     message_data.message_type, message_data.chat_id,
                     message_data.source_interface, message_data.original_query,
                     message_data.response_type, key_topics_json)
                )
            logger.info("Successfully stored message with metadata in database")
        except Exception as e:
            logger.error(f"Failed to store message: {str(e)}")
            raise

    def find_similar(self, embedding: List[float], threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find similar messages using cosine similarity"""
        try:
            with self.conn:
                cur = self.conn.cursor()
                cur.execute(f"SELECT message, embedding FROM {self.config.table_name}")
                results = []
                for message, embedding_json in cur.fetchall():
                    stored_embedding = json.loads(embedding_json)
                    similarity = compute_similarity(embedding, stored_embedding)
                    if similarity >= threshold:
                        results.append({
                            'message': message,
                            'similarity': similarity
                        })
                results.sort(key=lambda x: x['similarity'], reverse=True)
                return results
        except Exception as e:
            logger.error(f"Failed to find similar messages: {str(e)}")
            raise

    def close(self) -> None:
        """Close SQLite connection"""
        if self.conn:
            self.conn.close()

def get_embedding(text: str, model: str = "text-embedding-ada-002") -> list:
    """
    Generate an embedding for the given text using OpenAI's API.
    
    Args:
        text (str): The text to generate an embedding for
        model (str): The model to use for embedding generation
        
    Returns:
        list: The embedding vector
        
    Raises:
        EmbeddingError: If embedding generation fails
    """
    try:
        response = openai.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {str(e)}")
        raise EmbeddingError(f"Embedding generation failed: {str(e)}")

def compute_similarity(embedding1: list, embedding2: list) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1 (list): First embedding vector
        embedding2 (list): Second embedding vector
        
    Returns:
        float: Cosine similarity score between 0 and 1
    """
    return cosine_similarity([embedding1], [embedding2])[0][0]

class MessageStore:
    def __init__(self, storage_provider: VectorStorageProvider):
        """Initialize the store with a storage provider."""
        self.storage_provider = storage_provider
        self.storage_provider.initialize()

    def add_message(self, message_data: MessageData) -> None:
        """
        Add a message and its embedding to the store.
        
        Args:
            message (str): The message text
            embedding (list): The embedding vector for the message
        """
        self.storage_provider.store_embedding(message_data)

    def find_similar_messages(self, embedding: List[float], threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Find messages similar to the given embedding.
        
        Args:
            new_embedding (list): The embedding vector to compare against
            threshold (float): Similarity threshold (0-1) to consider a message as similar
            
        Returns:
            list: List of dictionaries containing similar messages and their similarity scores
        """
        return self.storage_provider.find_similar(embedding, threshold)

    def __del__(self):
        """Cleanup resources when the store is destroyed"""
        self.storage_provider.close()
