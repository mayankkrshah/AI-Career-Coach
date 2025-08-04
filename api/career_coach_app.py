#!/usr/bin/env python3
"""
AI Career Coach - Improved Version with Proper RAG + Fallback
Following the notebook implementation and medical assistant pattern
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import asyncio

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Import our RAG system
from career_coach_rag import CareerCoachRAG

# Import tools from notebook pattern
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Career Coach API",
    description="AI Career Coach with RAG + Intelligent Fallback",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4002", "http://127.0.0.1:4002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
rag_system = None
tavily_tool = None
arxiv_tool = None
llm = None

# Configuration
MIN_CONFIDENCE_THRESHOLD = 0.6  # Following medical assistant pattern

# Models
class CareerCoachRequest(BaseModel):
    user_message: str
    api_key: str
    tavily_api_key: Optional[str] = None

class CareerCoachResponse(BaseModel):
    response: str
    tool_calls: List[Dict[str, Any]] = []
    context_used: List[Dict[str, Any]] = []
    processing_time: float
    tools_used: List[str] = []
    primary_source: str = "AI Assistant"
    confidence_score: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, str]

# Helper functions
async def initialize_services():
    """Initialize all services on startup"""
    global rag_system, tavily_tool, arxiv_tool, llm
    
    try:
        # Initialize RAG system
        logger.info("Initializing RAG system...")
        rag_system = CareerCoachRAG(data_path="../data/job_dataset.csv")
        
        # Initialize tools
        if os.getenv("TAVILY_API_KEY"):
            tavily_tool = TavilySearchResults(max_results=5)
            logger.info("Tavily search initialized")
        
        arxiv_tool = ArxivQueryRun()
        logger.info("ArXiv search initialized")
        
        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        logger.info("LLM initialized")
        
    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}")

def determine_query_type(query: str) -> str:
    """
    Determine the type of query to route appropriately
    Following the notebook pattern
    """
    query_lower = query.lower()
    
    # Research paper indicators
    research_keywords = [
        'research', 'paper', 'papers', 'study', 'studies', 'publication',
        'arxiv', 'academic', 'journal', 'article', 'literature',
        'research paper', 'research papers', 'academic paper'
    ]
    
    # Job/career indicators  
    job_keywords = [
        'salary', 'pay', 'wage', 'compensation', 'benefits',
        'job', 'position', 'role', 'opportunity', 'opening',
        'company', 'employer', 'hire', 'hiring', 'career',
        'experience level', 'requirements', 'skills needed',
        'ai engineer', 'data scientist', 'machine learning'
    ]
    
    # General advice indicators
    advice_keywords = [
        'how to', 'tips', 'advice', 'guide', 'recommendation',
        'course', 'training', 'learn', 'study', 'certification',
        'interview', 'resume', 'prepare', 'become'
    ]
    
    # Count keyword matches
    research_score = sum(1 for keyword in research_keywords if keyword in query_lower)
    job_score = sum(1 for keyword in job_keywords if keyword in query_lower)
    advice_score = sum(1 for keyword in advice_keywords if keyword in query_lower)
    
    # Determine primary type
    if research_score > job_score and research_score > advice_score:
        return "research"
    elif job_score >= research_score and job_score > 0:
        return "job"
    else:
        return "general"

async def handle_research_query(query: str, api_key: str) -> Dict[str, Any]:
    """Handle research paper queries using ArXiv"""
    try:
        if arxiv_tool:
            # Search ArXiv
            results = arxiv_tool.invoke({"query": query})
            
            # Format results
            if results:
                response = f"Here are relevant research papers on '{query}':\\n\\n{results}"
                return {
                    "response": response,
                    "tool_used": "ArXiv",
                    "confidence": 1.0
                }
        
        # Fallback if ArXiv not available
        return await handle_general_query(query, api_key)
        
    except Exception as e:
        logger.error(f"ArXiv search error: {str(e)}")
        return await handle_general_query(query, api_key)

async def handle_job_query(query: str, api_key: str) -> Dict[str, Any]:
    """Handle job-related queries using RAG with fallback"""
    try:
        # First try RAG
        rag_result = rag_system.retrieve_and_generate(query)
        
        # Check confidence and insufficient info
        confidence = rag_result.get("confidence", 0.0)
        insufficient_info = rag_result.get("insufficient_info", False)
        
        logger.info(f"RAG confidence: {confidence}, Insufficient info: {insufficient_info}")
        
        # If confidence is high and info is sufficient, return RAG result
        if confidence >= MIN_CONFIDENCE_THRESHOLD and not insufficient_info:
            return {
                "response": rag_result["response"],
                "tool_used": "Job Database",
                "confidence": confidence,
                "sources": len(rag_result.get("sources", []))
            }
        
        # Otherwise, fallback to web search
        logger.info("RAG confidence low or insufficient info, falling back to web search")
        
        if tavily_tool and api_key.startswith('sk-'):
            # Search web for current information
            search_query = f"AI job market {query} 2024 2025"
            search_results = tavily_tool.invoke({"query": search_query})
            
            # Process search results with LLM
            prompt = f"""Based on these web search results about the AI job market, answer the user's question:
            
Question: {query}

Web Search Results:
{search_results}

Provide a comprehensive answer with current information."""
            
            response = llm.invoke(prompt)
            return {
                "response": response.content,
                "tool_used": "Tavily Web Search",
                "confidence": 0.8
            }
        else:
            # No API key or Tavily not available
            return {
                "response": rag_result["response"] + "\\n\\n*Note: For more current information, please provide an OpenAI API key.*",
                "tool_used": "Job Database (Limited)",
                "confidence": confidence
            }
            
    except Exception as e:
        logger.error(f"Job query error: {str(e)}")
        return {
            "response": "I encountered an error while searching for job information. Please try again.",
            "tool_used": "Error",
            "confidence": 0.0
        }

async def handle_general_query(query: str, api_key: str) -> Dict[str, Any]:
    """Handle general career advice queries"""
    try:
        if not api_key.startswith('sk-'):
            return {
                "response": "Please provide a valid OpenAI API key for general career advice.",
                "tool_used": "None",
                "confidence": 0.0
            }
        
        prompt = f"""You are an AI Career Coach. Answer this career-related question:

{query}

Provide helpful, actionable advice based on current industry best practices."""
        
        response = llm.invoke(prompt)
        return {
            "response": response.content,
            "tool_used": "OpenAI GPT-4",
            "confidence": 0.9
        }
        
    except Exception as e:
        logger.error(f"General query error: {str(e)}")
        return {
            "response": "I encountered an error. Please try again.",
            "tool_used": "Error",
            "confidence": 0.0
        }

# Routes
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    services = {
        "api": "healthy",
        "rag_system": "initialized" if rag_system else "not_initialized",
        "tavily": "available" if tavily_tool else "not_available",
        "arxiv": "available" if arxiv_tool else "not_available",
    }
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        services=services
    )

@app.post("/api/career-coach")
async def career_coach(request: CareerCoachRequest):
    """
    Main career coach endpoint with intelligent routing
    Following the notebook and medical assistant patterns
    """
    try:
        start_time = datetime.now()
        
        # Update Tavily API key if provided
        if request.tavily_api_key:
            os.environ["TAVILY_API_KEY"] = request.tavily_api_key
            global tavily_tool
            tavily_tool = TavilySearchResults(max_results=5)
        
        # Determine query type
        query_type = determine_query_type(request.user_message)
        logger.info(f"Query type: {query_type} for query: {request.user_message}")
        
        # Route to appropriate handler
        if query_type == "research":
            result = await handle_research_query(request.user_message, request.api_key)
        elif query_type == "job":
            result = await handle_job_query(request.user_message, request.api_key)
        else:
            result = await handle_general_query(request.user_message, request.api_key)
        
        # Prepare response
        processing_time = (datetime.now() - start_time).total_seconds()
        
        tools_used = [result["tool_used"]] if result.get("tool_used") else []
        
        # Map tool names to user-friendly names
        tool_display_map = {
            "Job Database": "Job Database (15,000+ positions)",
            "ArXiv": "ArXiv Research Papers",
            "Tavily Web Search": "Tavily Web Search",
            "OpenAI GPT-4": "AI Assistant (GPT-4)",
            "Job Database (Limited)": "Job Database",
            "None": "No tools available",
            "Error": "Error occurred"
        }
        
        primary_source = tool_display_map.get(result.get("tool_used", "AI Assistant"), result.get("tool_used", "AI Assistant"))
        
        return CareerCoachResponse(
            response=result["response"],
            processing_time=processing_time,
            tools_used=tools_used,
            primary_source=primary_source,
            confidence_score=result.get("confidence"),
            context_used=[{"source": result.get("tool_used", "Unknown"), "confidence": result.get("confidence", 0.0)}]
        )
        
    except Exception as e:
        logger.error(f"Career coach error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Startup
@app.on_event("startup")
async def startup():
    """Initialize services on startup"""
    logger.info("Starting AI Career Coach API...")
    await initialize_services()
    logger.info("Startup complete")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 9000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "career_coach_app:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )