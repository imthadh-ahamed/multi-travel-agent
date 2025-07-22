"""
Phase 2: Intermediate Integration - Memory System

This module implements various memory systems for agents to remember
user preferences, conversation history, and learned information.

Learning Objectives:
- Implement conversation memory
- Create persistent user preferences
- Use vector stores for long-term memory
- Handle memory summarization and retrieval
"""

import os
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import chromadb
from chromadb.utils import embedding_functions
import openai
from dotenv import load_dotenv

load_dotenv()


@dataclass
class UserPreference:
    """User preference with metadata"""
    key: str
    value: Any
    confidence: float  # 0-1, how confident we are about this preference
    source: str  # Where this preference came from
    last_updated: datetime
    usage_count: int = 0


@dataclass
class ConversationMemory:
    """Single conversation memory entry"""
    timestamp: datetime
    user_message: str
    assistant_response: str
    tools_used: List[str]
    context_tags: List[str]
    session_id: str


class MemorySystem(ABC):
    """Abstract base class for memory systems"""
    
    @abstractmethod
    def store(self, key: str, value: Any, metadata: Optional[Dict] = None):
        """Store information in memory"""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, limit: int = 5) -> List[Any]:
        """Retrieve information from memory"""
        pass
    
    @abstractmethod
    def update(self, key: str, value: Any, metadata: Optional[Dict] = None):
        """Update existing memory entry"""
        pass
    
    @abstractmethod
    def delete(self, key: str):
        """Delete memory entry"""
        pass


class ConversationSummaryMemory(MemorySystem):
    """
    Memory system that maintains conversation summaries and key information
    """
    
    def __init__(self, max_tokens: int = 4000):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.max_tokens = max_tokens
        self.conversation_history: List[ConversationMemory] = []
        self.summary = ""
        self.key_facts: Dict[str, Any] = {}
        
    def store(self, key: str, value: Any, metadata: Optional[Dict] = None):
        """Store conversation entry"""
        if key == "conversation":
            self.conversation_history.append(value)
            self._update_summary_if_needed()
        elif key == "fact":
            self.key_facts[metadata.get("fact_key", str(len(self.key_facts)))] = {
                "value": value,
                "timestamp": datetime.now(),
                "confidence": metadata.get("confidence", 0.8),
                "source": metadata.get("source", "conversation")
            }
    
    def retrieve(self, query: str, limit: int = 5) -> List[Any]:
        """Retrieve relevant conversation history and facts"""
        results = []
        
        # Add current summary
        if self.summary:
            results.append({
                "type": "summary",
                "content": self.summary,
                "relevance": 0.9
            })
        
        # Add recent conversations
        recent_conversations = self.conversation_history[-limit:]
        for conv in recent_conversations:
            results.append({
                "type": "conversation",
                "content": f"User: {conv.user_message}\nAssistant: {conv.assistant_response[:200]}...",
                "timestamp": conv.timestamp,
                "tools_used": conv.tools_used,
                "relevance": 0.7
            })
        
        # Add relevant facts
        for fact_key, fact_data in self.key_facts.items():
            if query.lower() in str(fact_data["value"]).lower():
                results.append({
                    "type": "fact",
                    "key": fact_key,
                    "content": fact_data["value"],
                    "confidence": fact_data["confidence"],
                    "timestamp": fact_data["timestamp"],
                    "relevance": 0.8
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:limit]
    
    def update(self, key: str, value: Any, metadata: Optional[Dict] = None):
        """Update memory entry"""
        if key in self.key_facts:
            self.key_facts[key]["value"] = value
            self.key_facts[key]["timestamp"] = datetime.now()
            if metadata:
                self.key_facts[key].update(metadata)
    
    def delete(self, key: str):
        """Delete memory entry"""
        if key in self.key_facts:
            del self.key_facts[key]
    
    def _update_summary_if_needed(self):
        """Update conversation summary when history gets too long"""
        total_tokens = self._estimate_tokens()
        
        if total_tokens > self.max_tokens:
            self._generate_summary()
            # Keep only recent conversations
            self.conversation_history = self.conversation_history[-10:]
    
    def _estimate_tokens(self) -> int:
        """Estimate token count of conversation history"""
        total_text = self.summary + " ".join([
            f"{conv.user_message} {conv.assistant_response}"
            for conv in self.conversation_history
        ])
        return len(total_text) // 4  # Rough token estimation
    
    def _generate_summary(self):
        """Generate conversation summary using OpenAI"""
        try:
            conversation_text = "\n".join([
                f"User: {conv.user_message}\nAssistant: {conv.assistant_response}"
                for conv in self.conversation_history[:-5]  # Don't summarize recent ones
            ])
            
            prompt = f"""
            Please create a concise summary of this travel conversation, focusing on:
            1. User's travel preferences and requirements
            2. Destinations discussed
            3. Important decisions made
            4. Key information provided
            
            Previous summary: {self.summary}
            
            New conversation content:
            {conversation_text}
            
            Updated summary:
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            self.summary = response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating summary: {e}")
    
    def get_summary(self) -> str:
        """Get current conversation summary"""
        return self.summary
    
    def get_key_facts(self) -> Dict[str, Any]:
        """Get all stored key facts"""
        return self.key_facts


class VectorMemory(MemorySystem):
    """
    Vector-based memory system using ChromaDB for semantic search
    """
    
    def __init__(self, collection_name: str = "travel_memory", persist_directory: str = "./data/memory"):
        """Initialize vector memory with ChromaDB"""
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Set up embedding function
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
    
    def store(self, key: str, value: Any, metadata: Optional[Dict] = None):
        """Store information with vector embedding"""
        try:
            # Prepare document
            if isinstance(value, dict):
                document = json.dumps(value, default=str)
            else:
                document = str(value)
            
            # Prepare metadata
            meta = {
                "timestamp": datetime.now().isoformat(),
                "type": metadata.get("type", "general") if metadata else "general",
                "source": metadata.get("source", "user") if metadata else "user"
            }
            if metadata:
                meta.update(metadata)
            
            # Store in collection
            self.collection.add(
                documents=[document],
                metadatas=[meta],
                ids=[key]
            )
            
        except Exception as e:
            print(f"Error storing in vector memory: {e}")
    
    def retrieve(self, query: str, limit: int = 5) -> List[Any]:
        """Retrieve similar information using vector search"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                include=["documents", "metadatas", "distances"]
            )
            
            retrieved_items = []
            for doc, meta, distance in zip(
                results["documents"][0],
                results["metadatas"][0], 
                results["distances"][0]
            ):
                # Parse document if it's JSON
                try:
                    content = json.loads(doc)
                except (json.JSONDecodeError, TypeError):
                    content = doc
                
                retrieved_items.append({
                    "content": content,
                    "metadata": meta,
                    "similarity": 1 - distance,  # Convert distance to similarity
                    "relevance": 1 - distance
                })
            
            return retrieved_items
            
        except Exception as e:
            print(f"Error retrieving from vector memory: {e}")
            return []
    
    def update(self, key: str, value: Any, metadata: Optional[Dict] = None):
        """Update existing vector memory entry"""
        try:
            # Delete existing entry
            self.collection.delete(ids=[key])
            
            # Store updated entry
            self.store(key, value, metadata)
            
        except Exception as e:
            print(f"Error updating vector memory: {e}")
    
    def delete(self, key: str):
        """Delete vector memory entry"""
        try:
            self.collection.delete(ids=[key])
        except Exception as e:
            print(f"Error deleting from vector memory: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory collection"""
        try:
            count = self.collection.count()
            return {
                "total_entries": count,
                "collection_name": self.collection.name
            }
        except Exception as e:
            return {"error": str(e)}


class UserPreferenceManager:
    """
    Manages user preferences with confidence tracking and updates
    """
    
    def __init__(self, storage_path: str = "./data/user_preferences.json"):
        self.storage_path = storage_path
        self.preferences: Dict[str, UserPreference] = {}
        self._load_preferences()
    
    def _load_preferences(self):
        """Load preferences from storage"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    
                for key, pref_data in data.items():
                    # Convert datetime string back to datetime object
                    pref_data["last_updated"] = datetime.fromisoformat(pref_data["last_updated"])
                    self.preferences[key] = UserPreference(**pref_data)
                    
        except Exception as e:
            print(f"Error loading preferences: {e}")
    
    def _save_preferences(self):
        """Save preferences to storage"""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            # Convert to serializable format
            data = {}
            for key, pref in self.preferences.items():
                pref_dict = asdict(pref)
                pref_dict["last_updated"] = pref_dict["last_updated"].isoformat()
                data[key] = pref_dict
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Error saving preferences: {e}")
    
    def set_preference(self, key: str, value: Any, confidence: float = 0.8, source: str = "user"):
        """Set or update a user preference"""
        if key in self.preferences:
            # Update existing preference
            pref = self.preferences[key]
            pref.value = value
            pref.confidence = max(pref.confidence, confidence)  # Increase confidence
            pref.last_updated = datetime.now()
            pref.usage_count += 1
            if confidence > pref.confidence:
                pref.source = source
        else:
            # Create new preference
            self.preferences[key] = UserPreference(
                key=key,
                value=value,
                confidence=confidence,
                source=source,
                last_updated=datetime.now(),
                usage_count=1
            )
        
        self._save_preferences()
    
    def get_preference(self, key: str) -> Optional[UserPreference]:
        """Get a user preference"""
        pref = self.preferences.get(key)
        if pref:
            pref.usage_count += 1
            self._save_preferences()
        return pref
    
    def get_all_preferences(self) -> Dict[str, UserPreference]:
        """Get all user preferences"""
        return self.preferences.copy()
    
    def infer_preference_from_text(self, text: str, context: str = ""):
        """Infer preferences from user text using simple heuristics"""
        text_lower = text.lower()
        
        # Budget preferences
        if any(word in text_lower for word in ["budget", "cheap", "affordable", "low cost"]):
            self.set_preference("budget_preference", "low", 0.6, f"inferred_from: {context}")
        elif any(word in text_lower for word in ["luxury", "expensive", "premium", "high-end"]):
            self.set_preference("budget_preference", "high", 0.6, f"inferred_from: {context}")
        
        # Climate preferences
        if any(word in text_lower for word in ["warm", "hot", "tropical", "beach"]):
            self.set_preference("climate_preference", "warm", 0.5, f"inferred_from: {context}")
        elif any(word in text_lower for word in ["cold", "snow", "winter", "skiing"]):
            self.set_preference("climate_preference", "cold", 0.5, f"inferred_from: {context}")
        
        # Activity preferences
        if any(word in text_lower for word in ["adventure", "hiking", "outdoor", "active"]):
            self.set_preference("activity_preference", "adventure", 0.6, f"inferred_from: {context}")
        elif any(word in text_lower for word in ["relaxing", "spa", "peaceful", "calm"]):
            self.set_preference("activity_preference", "relaxation", 0.6, f"inferred_from: {context}")
        
        # Accommodation preferences
        if any(word in text_lower for word in ["hotel", "luxury hotel", "resort"]):
            self.set_preference("accommodation_preference", "hotel", 0.7, f"inferred_from: {context}")
        elif any(word in text_lower for word in ["airbnb", "apartment", "local"]):
            self.set_preference("accommodation_preference", "apartment", 0.7, f"inferred_from: {context}")
    
    def get_preference_summary(self) -> str:
        """Get a summary of user preferences"""
        if not self.preferences:
            return "No preferences stored yet."
        
        summary_parts = []
        for key, pref in self.preferences.items():
            if pref.confidence > 0.8:
                confidence_label = "High"
            elif pref.confidence > 0.5:
                confidence_label = "Medium"
            else:
                confidence_label = "Low"
            
            summary_parts.append(
                f"• {key.replace('_', ' ').title()}: {pref.value} "
                f"(Confidence: {confidence_label}, Used: {pref.usage_count} times)"
            )
        
        return "User Preferences:\n" + "\n".join(summary_parts)


class HybridMemorySystem:
    """
    Combines multiple memory systems for comprehensive memory management
    """
    
    def __init__(self):
        self.conversation_memory = ConversationSummaryMemory()
        self.vector_memory = VectorMemory()
        self.preference_manager = UserPreferenceManager()
    
    def store_conversation(self, user_message: str, assistant_response: str, 
                          tools_used: List[str] = None, session_id: str = "default"):
        """Store conversation in memory systems"""
        # Store in conversation memory
        conv_memory = ConversationMemory(
            timestamp=datetime.now(),
            user_message=user_message,
            assistant_response=assistant_response,
            tools_used=tools_used or [],
            context_tags=[],
            session_id=session_id
        )
        
        self.conversation_memory.store("conversation", conv_memory)
        
        # Store in vector memory for semantic search
        conversation_text = f"User: {user_message}\nAssistant: {assistant_response}"
        self.vector_memory.store(
            key=f"conv_{datetime.now().isoformat()}",
            value=conversation_text,
            metadata={
                "type": "conversation",
                "tools_used": tools_used or [],
                "session_id": session_id
            }
        )
        
        # Infer preferences from user message
        self.preference_manager.infer_preference_from_text(user_message, "conversation")
    
    def retrieve_relevant_context(self, query: str, limit: int = 5) -> Dict[str, List[Any]]:
        """Retrieve relevant context from all memory systems"""
        return {
            "conversations": self.conversation_memory.retrieve(query, limit),
            "similar_content": self.vector_memory.retrieve(query, limit),
            "preferences": self.preference_manager.get_all_preferences()
        }
    
    def get_memory_summary(self) -> str:
        """Get a comprehensive memory summary"""
        conv_summary = self.conversation_memory.get_summary()
        pref_summary = self.preference_manager.get_preference_summary()
        vector_stats = self.vector_memory.get_collection_stats()
        
        return f"""
Memory Summary:
==============

Conversation Summary:
{conv_summary or "No conversation summary yet."}

{pref_summary}

Vector Memory: {vector_stats.get('total_entries', 0)} entries stored

Key Facts: {len(self.conversation_memory.get_key_facts())} facts remembered
        """.strip()


# Example usage and testing
def test_memory_systems():
    """Test the memory systems"""
    print("🧠 Testing Memory Systems")
    print("=" * 50)
    
    # Test hybrid memory system
    memory = HybridMemorySystem()
    
    # Store some conversations
    memory.store_conversation(
        "I want to visit Japan in spring for cherry blossoms",
        "Great choice! Spring is perfect for cherry blossoms in Japan. The best time is usually late March to early May.",
        ["get_weather"]
    )
    
    memory.store_conversation(
        "I prefer luxury hotels and have a high budget",
        "Perfect! I can recommend some luxury hotels in Japan with excellent cherry blossom views.",
        ["search_hotels"]
    )
    
    # Retrieve context
    context = memory.retrieve_relevant_context("Japan travel recommendations")
    
    print("Retrieved Context:")
    print(f"- Conversations: {len(context['conversations'])}")
    print(f"- Similar Content: {len(context['similar_content'])}")
    print(f"- Preferences: {len(context['preferences'])}")
    
    print("\nMemory Summary:")
    print(memory.get_memory_summary())


if __name__ == "__main__":
    test_memory_systems()
