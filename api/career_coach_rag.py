"""
Career Coach RAG System with Confidence Scoring
Based on the original notebook implementation and medical assistant pattern
"""

import os
import time
from typing import Dict, List, Optional, Any
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

class CareerCoachRAG:
    """RAG system for career coach with confidence scoring"""
    
    def __init__(self, data_path: str = "data/job_dataset.csv", collection_name: str = "job_market_data"):
        self.data_path = data_path
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.vector_store = None
        self.retriever = None
        self._initialize_vector_store()
        
    def _initialize_vector_store(self):
        """Initialize Qdrant vector store with job data"""
        # Create in-memory Qdrant client
        self.client = QdrantClient(":memory:")
        
        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        
        # Initialize vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
        
        # Load and process documents
        self._load_documents()
        
    def _load_documents(self, max_docs: int = 5000):
        """Load job data from CSV and add to vector store"""
        try:
            # Load CSV data
            loader = CSVLoader(file_path=self.data_path)
            docs = loader.load()
            
            # Limit documents for performance
            if len(docs) > max_docs:
                docs = docs[:max_docs]
                
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            split_documents = text_splitter.split_documents(docs)
            
            # Add to vector store in batches
            logger.info(f"Adding {len(split_documents)} documents to vector store...")
            batch_size = 50
            
            for i in range(0, len(split_documents), batch_size):
                batch = split_documents[i:i+batch_size]
                self.vector_store.add_documents(documents=batch)
                
            # Create retriever
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            # Create empty retriever if loading fails
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
    
    def retrieve_and_generate(self, query: str, chat_history: str = "") -> Dict[str, Any]:
        """
        Retrieve relevant documents and generate response with confidence scoring
        Following the pattern from the medical assistant
        """
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retriever.invoke(query) if self.retriever else []
            
            # Calculate confidence based on retrieval quality
            confidence = self._calculate_confidence(query, retrieved_docs)
            
            # Format context
            docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
            
            # Create prompt following notebook pattern
            rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant who answers questions based on provided job market context. 
You must only use the provided context about jobs, salaries, and skills.

If the context does not contain sufficient information to answer the question, respond with:
"I don't have enough information in the job market data to answer this question."

### Question
{question}

### Chat History
{chat_history}

### Job Market Context
{context}
""")
            
            # Generate response
            messages = rag_prompt.format_messages(
                question=query,
                chat_history=chat_history,
                context=docs_content
            )
            response = self.llm.invoke(messages)
            
            # Check if response indicates insufficient information
            response_text = response.content
            insufficient_info = self._check_insufficient_info(response_text)
            
            return {
                "response": response_text,
                "confidence": confidence,
                "insufficient_info": insufficient_info,
                "sources": retrieved_docs,
                "tool_used": "Job Database"
            }
            
        except Exception as e:
            logger.error(f"RAG error: {str(e)}")
            return {
                "response": "Error processing query with job database.",
                "confidence": 0.0,
                "insufficient_info": True,
                "sources": [],
                "tool_used": "Job Database"
            }
    
    def _calculate_confidence(self, query: str, retrieved_docs: List[Document]) -> float:
        """
        Calculate confidence score based on retrieval quality
        Following medical assistant pattern
        """
        if not retrieved_docs:
            return 0.0
            
        # Basic confidence calculation
        # More sophisticated scoring can be implemented
        base_confidence = min(len(retrieved_docs) / 5.0, 1.0)  # Max 1.0 for 5 docs
        
        # Check if documents are relevant to query terms
        query_terms = set(query.lower().split())
        relevance_scores = []
        
        for doc in retrieved_docs:
            doc_terms = set(doc.page_content.lower().split())
            overlap = len(query_terms.intersection(doc_terms)) / len(query_terms)
            relevance_scores.append(overlap)
            
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        # Combine base confidence with relevance
        final_confidence = (base_confidence * 0.4) + (avg_relevance * 0.6)
        
        return round(final_confidence, 2)
    
    def _check_insufficient_info(self, response_text: str) -> bool:
        """
        Check if response indicates insufficient information
        Following medical assistant pattern
        """
        insufficient_phrases = [
            "don't have enough information",
            "not enough information",
            "insufficient information",
            "cannot answer",
            "unable to answer",
            "no information available",
            "not found in the context"
        ]
        
        response_lower = response_text.lower()
        return any(phrase in response_lower for phrase in insufficient_phrases)