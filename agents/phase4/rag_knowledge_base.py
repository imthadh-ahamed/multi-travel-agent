"""
Task 11: RAG Knowledge Base Implementation
Goal: Use RAG to allow agents to query documentation and travel rules
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import pickle
import openai
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import aiofiles

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KnowledgeDocument:
    """Represents a document in the knowledge base"""
    doc_id: str
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    embedding: Optional[List[float]] = None
    chunk_size: int = 1000
    
    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique document ID"""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()
        return f"doc_{content_hash[:12]}"

@dataclass
class QueryResult:
    """Result of a knowledge base query"""
    query: str
    documents: List[KnowledgeDocument]
    relevance_scores: List[float]
    total_results: int
    query_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for list of texts"""
        pass
    
    @abstractmethod
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for single text"""
        pass

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider"""
    
    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None):
        self.model = model
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.embedding_cache = {}
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for single text with caching"""
        # Check cache first
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embedding = response.data[0].embedding
            
            # Cache the result
            self.embedding_cache[cache_key] = embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            raise
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        try:
            # Check cache for each text
            embeddings = []
            texts_to_fetch = []
            indices_to_fetch = []
            
            for i, text in enumerate(texts):
                cache_key = hashlib.md5(text.encode()).hexdigest()
                if cache_key in self.embedding_cache:
                    embeddings.append(self.embedding_cache[cache_key])
                else:
                    embeddings.append(None)
                    texts_to_fetch.append(text)
                    indices_to_fetch.append(i)
            
            # Fetch missing embeddings
            if texts_to_fetch:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=texts_to_fetch
                )
                
                # Fill in the missing embeddings
                for i, embedding_data in enumerate(response.data):
                    embedding = embedding_data.embedding
                    original_index = indices_to_fetch[i]
                    embeddings[original_index] = embedding
                    
                    # Cache the result
                    cache_key = hashlib.md5(texts_to_fetch[i].encode()).hexdigest()
                    self.embedding_cache[cache_key] = embedding
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
            raise

class VectorStore:
    """Simple vector store implementation"""
    
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.documents: List[KnowledgeDocument] = []
        self.embeddings: List[List[float]] = []
        self.document_index: Dict[str, int] = {}
    
    def add_document(self, document: KnowledgeDocument, embedding: List[float]):
        """Add document with its embedding to the store"""
        if document.doc_id in self.document_index:
            # Update existing document
            index = self.document_index[document.doc_id]
            self.documents[index] = document
            self.embeddings[index] = embedding
        else:
            # Add new document
            self.documents.append(document)
            self.embeddings.append(embedding)
            self.document_index[document.doc_id] = len(self.documents) - 1
    
    def search(self, query_embedding: List[float], k: int = 5) -> Tuple[List[KnowledgeDocument], List[float]]:
        """Search for similar documents"""
        if not self.embeddings:
            return [], []
        
        # Calculate cosine similarity
        similarities = []
        for embedding in self.embeddings:
            similarity = self._cosine_similarity(query_embedding, embedding)
            similarities.append(similarity)
        
        # Get top-k results
        top_indices = sorted(range(len(similarities)), 
                           key=lambda i: similarities[i], 
                           reverse=True)[:k]
        
        top_documents = [self.documents[i] for i in top_indices]
        top_scores = [similarities[i] for i in top_indices]
        
        return top_documents, top_scores
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = sum(x * x for x in a) ** 0.5
        magnitude_b = sum(x * x for x in b) ** 0.5
        
        if magnitude_a == 0 or magnitude_b == 0:
            return 0
        
        return dot_product / (magnitude_a * magnitude_b)
    
    def save(self, filepath: Path):
        """Save vector store to file"""
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'document_index': self.document_index,
            'dimension': self.dimension
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: Path):
        """Load vector store from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.documents = data['documents']
        self.embeddings = data['embeddings']
        self.document_index = data['document_index']
        self.dimension = data['dimension']

class RAGKnowledgeBase:
    """
    RAG (Retrieval-Augmented Generation) Knowledge Base
    Stores travel documentation and rules for agent querying
    """
    
    def __init__(self, 
                 storage_path: Path = Path("./data/knowledge_base"),
                 embedding_provider: EmbeddingProvider = None,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.embedding_provider = embedding_provider or OpenAIEmbeddingProvider()
        self.vector_store = VectorStore()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Load existing knowledge base if available
        self._load_knowledge_base()
        
        # Initialize with default travel knowledge
        self.default_knowledge = self._get_default_travel_knowledge()
    
    async def add_document(self, 
                         title: str, 
                         content: str, 
                         category: str = "general",
                         source: str = "",
                         metadata: Dict = None) -> str:
        """Add a document to the knowledge base"""
        try:
            # Create document
            document = KnowledgeDocument(
                doc_id="",  # Will be auto-generated
                title=title,
                content=content,
                category=category,
                source=source,
                metadata=metadata or {},
                chunk_size=self.chunk_size
            )
            
            # Chunk the document if it's large
            chunks = self._chunk_text(content)
            
            for i, chunk in enumerate(chunks):
                chunk_doc = KnowledgeDocument(
                    doc_id=f"{document.doc_id}_chunk_{i}",
                    title=f"{title} (Part {i+1})",
                    content=chunk,
                    category=category,
                    source=source,
                    metadata={**metadata or {}, 'chunk_index': i, 'total_chunks': len(chunks)}
                )
                
                # Get embedding
                embedding = await self.embedding_provider.get_embedding(chunk)
                
                # Add to vector store
                self.vector_store.add_document(chunk_doc, embedding)
            
            # Save the updated knowledge base
            self._save_knowledge_base()
            
            logger.info(f"Added document '{title}' with {len(chunks)} chunks")
            return document.doc_id
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            raise
    
    async def query(self, 
                   query: str, 
                   k: int = 5, 
                   category_filter: str = None,
                   min_relevance: float = 0.7) -> QueryResult:
        """Query the knowledge base"""
        start_time = datetime.now()
        
        try:
            # Get query embedding
            query_embedding = await self.embedding_provider.get_embedding(query)
            
            # Search vector store
            documents, scores = self.vector_store.search(query_embedding, k * 2)  # Get more for filtering
            
            # Apply filters
            filtered_docs = []
            filtered_scores = []
            
            for doc, score in zip(documents, scores):
                # Category filter
                if category_filter and doc.category != category_filter:
                    continue
                
                # Relevance filter
                if score < min_relevance:
                    continue
                
                filtered_docs.append(doc)
                filtered_scores.append(score)
            
            # Limit to k results
            filtered_docs = filtered_docs[:k]
            filtered_scores = filtered_scores[:k]
            
            query_time = (datetime.now() - start_time).total_seconds()
            
            return QueryResult(
                query=query,
                documents=filtered_docs,
                relevance_scores=filtered_scores,
                total_results=len(filtered_docs),
                query_time=query_time,
                metadata={'category_filter': category_filter, 'min_relevance': min_relevance}
            )
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
    
    async def get_context_for_query(self, 
                                   query: str, 
                                   max_context_length: int = 4000) -> str:
        """Get relevant context for a query"""
        try:
            # Query the knowledge base
            result = await self.query(query, k=10)
            
            if not result.documents:
                return "No relevant information found in the knowledge base."
            
            # Build context from results
            context_parts = []
            current_length = 0
            
            for doc, score in zip(result.documents, result.relevance_scores):
                doc_text = f"[{doc.category.upper()}] {doc.title}\n{doc.content}\n"
                
                if current_length + len(doc_text) > max_context_length:
                    break
                
                context_parts.append(doc_text)
                current_length += len(doc_text)
            
            context = "\n---\n".join(context_parts)
            
            # Add query metadata
            context_header = f"Relevant information for: {query}\n"
            context_header += f"Found {len(context_parts)} relevant documents (scores: {[f'{s:.2f}' for s in result.relevance_scores[:len(context_parts)]]})\n\n"
            
            return context_header + context
            
        except Exception as e:
            logger.error(f"Failed to get context: {e}")
            return f"Error retrieving context: {str(e)}"
    
    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text into smaller pieces"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                for i in range(min(self.chunk_overlap, end - start)):
                    if text[end - i - 1] in '.!?':
                        end = end - i
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _save_knowledge_base(self):
        """Save knowledge base to disk"""
        try:
            kb_file = self.storage_path / "knowledge_base.pkl"
            self.vector_store.save(kb_file)
            
            # Also save metadata
            metadata_file = self.storage_path / "metadata.json"
            metadata = {
                'last_updated': datetime.now().isoformat(),
                'total_documents': len(self.vector_store.documents),
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")
    
    def _load_knowledge_base(self):
        """Load knowledge base from disk"""
        try:
            kb_file = self.storage_path / "knowledge_base.pkl"
            if kb_file.exists():
                self.vector_store.load(kb_file)
                logger.info(f"Loaded knowledge base with {len(self.vector_store.documents)} documents")
            else:
                logger.info("No existing knowledge base found, starting fresh")
                
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
    
    def _get_default_travel_knowledge(self) -> List[Dict[str, str]]:
        """Get default travel knowledge to populate the knowledge base"""
        return [
            {
                'title': 'Passport and Visa Requirements',
                'content': '''
                Passport and Visa Requirements for International Travel:
                
                1. Passport Validity: Most countries require passports to be valid for at least 6 months beyond your planned departure date.
                
                2. Visa Requirements:
                   - EU Citizens: Can travel freely within EU/Schengen area
                   - US Citizens: Visa-free to 185+ countries for tourism (90-day limits common)
                   - Check specific country requirements on government websites
                
                3. Transit Visas: Required when changing planes in certain countries, even without leaving the airport
                
                4. Documentation for Minors: Children traveling alone or with one parent may need additional documentation
                
                5. Return/Onward Tickets: Many countries require proof of departure
                ''',
                'category': 'documentation',
                'source': 'travel_regulations'
            },
            {
                'title': 'Flight Booking Best Practices',
                'content': '''
                Flight Booking Tips and Best Practices:
                
                1. Best Booking Times:
                   - Domestic flights: 1-3 months in advance
                   - International flights: 2-8 months in advance
                   - Tuesday/Wednesday typically cheapest days to fly
                
                2. Price Comparison:
                   - Use multiple search engines (Google Flights, Kayak, Skyscanner)
                   - Check airline websites directly
                   - Clear browser cookies between searches
                
                3. Flexible Dates: Use date range searches to find cheaper options
                
                4. Connecting Flights: Consider layovers for savings, but allow adequate connection time
                
                5. Seat Selection: Pay attention to seat maps and reviews for comfort
                
                6. Baggage Policies: Check weight limits and fees for carry-on and checked bags
                ''',
                'category': 'flights',
                'source': 'booking_guide'
            },
            {
                'title': 'Hotel Accommodation Guidelines',
                'content': '''
                Hotel Booking and Accommodation Guidelines:
                
                1. Location Considerations:
                   - Proximity to attractions, public transport
                   - Safety of the neighborhood
                   - Noise levels (avoid main roads if sensitive)
                
                2. Amenities to Consider:
                   - Free WiFi, breakfast, parking
                   - Gym, pool, spa facilities
                   - Room service, concierge
                
                3. Booking Platforms:
                   - Compare prices across multiple sites
                   - Read recent reviews carefully
                   - Check cancellation policies
                
                4. Room Types:
                   - Standard, superior, deluxe classifications
                   - City view vs. quiet courtyard
                   - Bed configurations for groups
                
                5. Payment and Policies:
                   - Understand cancellation terms
                   - Check for resort fees or city taxes
                   - Consider travel insurance
                ''',
                'category': 'accommodation',
                'source': 'hotel_guide'
            },
            {
                'title': 'Budget Planning and Money Tips',
                'content': '''
                Travel Budget Planning and Money Management:
                
                1. Budget Categories:
                   - Transportation (flights, local transport): 30-40%
                   - Accommodation: 25-35%
                   - Food and drinks: 20-25%
                   - Activities and attractions: 10-15%
                   - Shopping and miscellaneous: 5-10%
                
                2. Money-Saving Tips:
                   - Travel during shoulder seasons
                   - Use public transportation
                   - Mix of street food and restaurants
                   - Free walking tours and museums
                
                3. Currency and Payments:
                   - Notify banks of travel plans
                   - Use cards with no foreign transaction fees
                   - Have some local cash for small vendors
                   - Research local tipping customs
                
                4. Emergency Fund: Keep 20% buffer for unexpected expenses
                
                5. Travel Insurance: Consider comprehensive coverage for expensive trips
                ''',
                'category': 'budget',
                'source': 'financial_guide'
            },
            {
                'title': 'Health and Safety Guidelines',
                'content': '''
                Travel Health and Safety Guidelines:
                
                1. Health Preparations:
                   - Check vaccination requirements 4-6 weeks before travel
                   - Consult travel medicine clinic for destination-specific advice
                   - Pack essential medications with prescriptions
                   - Consider travel health insurance
                
                2. Safety Precautions:
                   - Research destination safety conditions
                   - Register with embassy/consulate for extended stays
                   - Share itinerary with family/friends
                   - Keep copies of important documents
                
                3. Food and Water Safety:
                   - Bottled water in developing countries
                   - Avoid raw vegetables and fruits you can't peel
                   - Choose busy restaurants with high turnover
                
                4. Emergency Contacts:
                   - Local emergency numbers
                   - Embassy/consulate information
                   - Travel insurance emergency line
                   - Local emergency services: 112 (Europe), 911 (US)
                ''',
                'category': 'health_safety',
                'source': 'safety_guide'
            }
        ]
    
    async def initialize_default_knowledge(self):
        """Initialize the knowledge base with default travel information"""
        if len(self.vector_store.documents) > 0:
            logger.info("Knowledge base already contains documents, skipping initialization")
            return
        
        logger.info("Initializing knowledge base with default travel information...")
        
        for knowledge_item in self.default_knowledge:
            try:
                await self.add_document(
                    title=knowledge_item['title'],
                    content=knowledge_item['content'],
                    category=knowledge_item['category'],
                    source=knowledge_item['source']
                )
            except Exception as e:
                logger.error(f"Failed to add default knowledge item '{knowledge_item['title']}': {e}")
        
        logger.info(f"Initialized knowledge base with {len(self.default_knowledge)} documents")

class TravelKnowledgeAgent:
    """
    Agent that uses RAG knowledge base to provide informed responses
    """
    
    def __init__(self, 
                 agent_id: str = "travel_knowledge_agent",
                 model: str = "gpt-4",
                 knowledge_base: RAGKnowledgeBase = None):
        self.agent_id = agent_id
        self.model = model
        self.knowledge_base = knowledge_base or RAGKnowledgeBase()
        
        # Initialize OpenAI client
        self.client = openai.AsyncOpenAI()
        
        # Performance tracking
        self.query_stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'average_response_time': 0.0,
            'knowledge_usage_rate': 0.0
        }
    
    async def answer_with_knowledge(self, 
                                  question: str, 
                                  category_filter: str = None,
                                  include_sources: bool = True) -> Dict[str, Any]:
        """Answer question using RAG knowledge base"""
        start_time = datetime.now()
        
        try:
            self.query_stats['total_queries'] += 1
            
            # Get relevant context from knowledge base
            context = await self.knowledge_base.get_context_for_query(
                question, max_context_length=3000
            )
            
            # Build prompt with context
            system_prompt = f"""
            You are {self.agent_id}, a knowledgeable travel assistant with access to comprehensive travel information.
            
            Use the provided context to answer the user's question accurately and helpfully.
            If the context doesn't contain relevant information, clearly state that and provide general guidance.
            
            Guidelines:
            - Prioritize information from the context
            - Be specific and actionable
            - Include relevant details like costs, timing, requirements
            - Organize information clearly
            - Cite sources when available
            
            Context Information:
            {context}
            """
            
            user_prompt = f"""
            Question: {question}
            
            Please provide a comprehensive answer based on the available information.
            """
            
            # Generate response
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            answer = response.choices[0].message.content
            
            # Get source documents for transparency
            source_query = await self.knowledge_base.query(question, k=3)
            sources = []
            
            if include_sources and source_query.documents:
                for doc, score in zip(source_query.documents, source_query.relevance_scores):
                    sources.append({
                        'title': doc.title,
                        'category': doc.category,
                        'relevance_score': round(score, 3),
                        'source': doc.source
                    })
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Update stats
            self.query_stats['successful_queries'] += 1
            self.query_stats['average_response_time'] = (
                (self.query_stats['average_response_time'] * (self.query_stats['successful_queries'] - 1) + response_time) / 
                self.query_stats['successful_queries']
            )
            
            # Calculate knowledge usage rate
            knowledge_used = len(context.strip()) > 100  # Simple heuristic
            if knowledge_used:
                self.query_stats['knowledge_usage_rate'] = (
                    (self.query_stats['knowledge_usage_rate'] * (self.query_stats['successful_queries'] - 1) + 1) / 
                    self.query_stats['successful_queries']
                )
            else:
                self.query_stats['knowledge_usage_rate'] = (
                    self.query_stats['knowledge_usage_rate'] * (self.query_stats['successful_queries'] - 1) / 
                    self.query_stats['successful_queries']
                )
            
            return {
                'question': question,
                'answer': answer,
                'sources': sources,
                'response_time': response_time,
                'knowledge_used': knowledge_used,
                'agent_id': self.agent_id,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            return {
                'question': question,
                'answer': f"I'm sorry, I encountered an error while processing your question: {str(e)}",
                'sources': [],
                'response_time': (datetime.now() - start_time).total_seconds(),
                'knowledge_used': False,
                'error': str(e)
            }
    
    async def add_knowledge_from_url(self, url: str, title: str = None, category: str = "web_content") -> str:
        """Add knowledge from a web URL"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Simple content extraction (in production, use proper HTML parsing)
                        # Remove HTML tags (basic approach)
                        import re
                        text_content = re.sub(r'<[^>]+>', '', content)
                        text_content = re.sub(r'\s+', ' ', text_content).strip()
                        
                        doc_title = title or f"Web content from {url}"
                        
                        doc_id = await self.knowledge_base.add_document(
                            title=doc_title,
                            content=text_content,
                            category=category,
                            source=url,
                            metadata={'url': url, 'fetch_date': datetime.now().isoformat()}
                        )
                        
                        logger.info(f"Added knowledge from URL: {url}")
                        return doc_id
                    else:
                        raise Exception(f"Failed to fetch URL: {response.status}")
                        
        except Exception as e:
            logger.error(f"Failed to add knowledge from URL {url}: {e}")
            raise
    
    async def add_knowledge_from_file(self, filepath: Path, title: str = None, category: str = "document") -> str:
        """Add knowledge from a text file"""
        try:
            async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            doc_title = title or filepath.stem
            
            doc_id = await self.knowledge_base.add_document(
                title=doc_title,
                content=content,
                category=category,
                source=str(filepath),
                metadata={'filepath': str(filepath), 'file_size': len(content)}
            )
            
            logger.info(f"Added knowledge from file: {filepath}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to add knowledge from file {filepath}: {e}")
            raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics"""
        return {
            'agent_id': self.agent_id,
            'query_stats': self.query_stats.copy(),
            'knowledge_base_stats': {
                'total_documents': len(self.knowledge_base.vector_store.documents),
                'storage_path': str(self.knowledge_base.storage_path)
            }
        }

# Example usage and testing functions
async def test_rag_knowledge_base():
    """Test the RAG knowledge base functionality"""
    print("🧠 Testing RAG Knowledge Base System...")
    
    # Create knowledge base and agent
    kb = RAGKnowledgeBase()
    agent = TravelKnowledgeAgent(knowledge_base=kb)
    
    # Initialize with default knowledge
    await kb.initialize_default_knowledge()
    
    # Test queries
    test_questions = [
        "What are the passport requirements for international travel?",
        "How can I save money when booking flights?",
        "What should I consider when choosing a hotel?",
        "How should I plan my travel budget?",
        "What health precautions should I take when traveling?"
    ]
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        
        result = await agent.answer_with_knowledge(question)
        
        print(f"✅ Answer: {result['answer'][:200]}...")
        print(f"📚 Sources: {len(result['sources'])} documents")
        print(f"⏱️ Response time: {result['response_time']:.2f}s")
    
    # Get performance stats
    stats = agent.get_performance_stats()
    print(f"\n📊 Performance Stats:")
    print(f"Total queries: {stats['query_stats']['total_queries']}")
    print(f"Success rate: {stats['query_stats']['successful_queries']/stats['query_stats']['total_queries']*100:.1f}%")
    print(f"Average response time: {stats['query_stats']['average_response_time']:.2f}s")
    print(f"Knowledge usage rate: {stats['query_stats']['knowledge_usage_rate']*100:.1f}%")
    
    return stats

if __name__ == "__main__":
    asyncio.run(test_rag_knowledge_base())
