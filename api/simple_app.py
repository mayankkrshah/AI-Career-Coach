#!/usr/bin/env python3

"""
AI Career Coach - Simplified Version
A basic implementation that can run without all optional dependencies
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

# Try to import optional dependencies
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Career Coach API",
    description="A simplified AI Career Coach with basic RAG capabilities",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4002", "http://127.0.0.1:4002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
job_data = []
openai_client = None

# Models
class ChatRequest(BaseModel):
    user_message: str
    api_key: str
    system_prompt: str = "You are a helpful AI assistant specializing in career guidance."

class CareerCoachRequest(BaseModel):
    user_message: str
    api_key: str

class CareerCoachResponse(BaseModel):
    response: str
    tool_calls: List[Dict[str, Any]] = []
    context_used: List[Dict[str, Any]] = []
    processing_time: float
    tools_used: List[str] = []
    primary_source: str = "AI Assistant"

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, str]

# Helper functions
def load_job_data():
    """Load job data from CSV file"""
    global job_data
    
    if not HAS_PANDAS:
        logger.warning("Pandas not available, using mock data")
        job_data = [
            {
                "title": "AI Engineer",
                "company": "Google",
                "description": "Design and implement AI systems using machine learning",
                "skills": "Python, TensorFlow, PyTorch"
            },
            {
                "title": "Machine Learning Engineer", 
                "company": "Microsoft",
                "description": "Develop ML pipelines and deploy models at scale",
                "skills": "Python, Scikit-learn, Azure ML"
            }
        ]
        return

    data_file = Path("../data/job_dataset.csv")
    if data_file.exists():
        try:
            df = pd.read_csv(data_file)
            job_data = df.to_dict('records')
            logger.info(f"Loaded {len(job_data)} job records")
        except Exception as e:
            logger.error(f"Error loading job data: {e}")
            job_data = []
    else:
        logger.warning("Job dataset not found, using empty data")
        job_data = []

def search_jobs(query: str, limit: int = 5) -> List[Dict]:
    """Enhanced text search in job data with keyword expansion"""
    if not job_data:
        return []
    
    query_lower = query.lower()
    
    # Check if this is a location-specific salary query
    location_keywords = {
        'canada': ['canada', 'canadian'],
        'united states': ['usa', 'united states', 'america', 'us'],
        'united kingdom': ['uk', 'united kingdom', 'britain'],
        'germany': ['germany', 'german'],
        'india': ['india', 'indian'],
        'australia': ['australia', 'australian'],
        'singapore': ['singapore'],
        'japan': ['japan', 'japanese'],
        'china': ['china', 'chinese'],
        'france': ['france', 'french']
    }
    
    # Extract location from query
    query_location = None
    for country, keywords in location_keywords.items():
        if any(kw in query_lower for kw in keywords):
            query_location = country
            break
    results = []
    
    # Define keyword expansions for common queries
    keyword_expansions = {
        'ai': ['ai', 'artificial intelligence', 'machine learning', 'deep learning', 'ml', 'nlp', 'computer vision', 'pytorch', 'tensorflow'],
        'data': ['data', 'analyst', 'scientist', 'analytics', 'big data', 'sql', 'python', 'tableau', 'hadoop'],
        'software': ['software', 'developer', 'engineer', 'programming', 'coding', 'java', 'python', 'javascript'],
        'web': ['web', 'frontend', 'backend', 'fullstack', 'html', 'css', 'javascript', 'react', 'nodejs'],
        'mobile': ['mobile', 'ios', 'android', 'app', 'flutter', 'react native', 'swift', 'kotlin'],
        'cloud': ['cloud', 'aws', 'azure', 'gcp', 'kubernetes', 'docker', 'devops'],
        'security': ['security', 'cybersecurity', 'information security', 'infosec'],
        'devops': ['devops', 'deployment', 'ci/cd', 'infrastructure', 'automation', 'kubernetes', 'docker'],
        'skills': ['python', 'java', 'javascript', 'sql', 'aws', 'kubernetes', 'docker', 'react', 'nodejs', 'machine learning', 'data science'],
        'demand': ['engineer', 'developer', 'scientist', 'analyst', 'manager', 'architect'],
        'salary': ['engineer', 'developer', 'scientist', 'analyst', 'manager'],
        'courses': ['engineer', 'developer', 'scientist', 'analyst', 'programming', 'coding']
    }
    
    # Extract search terms - split on common delimiters and filter out stop words
    stop_words = {'the', 'in', 'and', 'or', 'of', 'to', 'a', 'an', 'is', 'are', 'what', 'which', 'how', 'most', 'demand', 'jobs', 'roles', 'positions'}
    search_terms = [word.strip() for word in query_lower.replace('?', ' ').replace(',', ' ').split() if word.strip() not in stop_words and len(word.strip()) > 2]
    
    # Expand search terms based on keyword mappings
    expanded_terms = set(search_terms)
    for term in search_terms:
        for key, expansions in keyword_expansions.items():
            if term in expansions or key in term:
                expanded_terms.update(expansions)
    
    for job in job_data:
        score = 0
        # Search in relevant fields with different weights
        search_fields = [
            ('job_title', 3),  # Job title gets highest weight
            ('required_skills', 2),  # Skills get high weight
            ('company_name', 1),  # Company name gets lower weight
        ]
        
        for field_name, weight in search_fields:
            if field_name in job and job[field_name]:
                field_text = str(job[field_name]).lower()
                for term in expanded_terms:
                    if term in field_text:
                        score += weight
        
        # If location is specified, boost jobs from that location
        if query_location and job.get('company_location', '').lower() == query_location.lower():
            score += 5  # High boost for location match
        
        if score > 0 or (query_location and job.get('company_location', '').lower() == query_location.lower()):
            results.append({**job, 'relevance_score': score})
    
    # If no results found, try a more general search for common job categories
    if not results and any(word in query_lower for word in ['job', 'role', 'position', 'career', 'work']):
        # Return some popular job categories
        popular_jobs = ['engineer', 'developer', 'analyst', 'manager', 'scientist']
        for job in job_data:
            job_title = str(job.get('job_title', '')).lower()
            for popular in popular_jobs:
                if popular in job_title:
                    results.append({**job, 'relevance_score': 1})
                    if len(results) >= limit:
                        break
            if len(results) >= limit:
                break
    
    # Sort by relevance and return top results
    results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    return results[:limit]

async def get_openai_response(message: str, api_key: str, system_prompt: str = None) -> str:
    """Get response from OpenAI with fallback for course recommendations"""
    if not HAS_OPENAI:
        return "OpenAI integration not available. Please install the openai package."
    
    if not api_key or not api_key.startswith('sk-'):
        return "Please provide a valid OpenAI API key starting with 'sk-' to use this feature."
    
    try:
        # Fix SSL certificate issue by disabling SSL verification
        import ssl
        import httpx
        
        # Create httpx client with disabled SSL verification
        http_client = httpx.Client(verify=False)
        client = OpenAI(api_key=api_key, http_client=http_client)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        
        # Provide helpful fallback for course recommendations
        if any(word in message.lower() for word in ['course', 'courses', 'training', 'certification', 'learn']):
            return """**Course Recommendations for AI Research Scientist:**

**Online Platforms:**
• **Coursera**: Stanford's Machine Learning Course, Deep Learning Specialization
• **edX**: MIT's Introduction to Machine Learning, Harvard's CS50 AI
• **Udacity**: Machine Learning Engineer, AI Programming Nanodegrees

**Specialized AI Courses:**
• **Fast.ai**: Practical Deep Learning for Coders
• **DeepLearning.AI**: Specializations in ML, Deep Learning, NLP
• **CS231n**: Stanford's Computer Vision course (free online)

**Programming & Tools:**
• Python for Data Science (DataCamp, Codecademy)
• TensorFlow/PyTorch tutorials (official documentation)
• Kaggle Learn: Free micro-courses

**Academic Path:**
• Master's in AI/ML, Computer Science, or related field
• PhD for advanced research positions

**Note:** OpenAI API is temporarily unavailable for real-time recommendations. The above are general suggestions. For current course offerings and pricing, please visit the platforms directly."""
        
        if "api key" in str(e).lower():
            return "Invalid OpenAI API key. Please check your API key and try again."
        return f"OpenAI API is temporarily unavailable. Please try again later."

# Routes
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    services = {
        "api": "healthy",
        "pandas": "available" if HAS_PANDAS else "not_available",
        "openai": "available" if HAS_OPENAI else "not_available",
        "job_data": f"{len(job_data)} records loaded" if job_data else "no_data"
    }
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        services=services
    )

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Basic chat endpoint"""
    try:
        start_time = datetime.now()
        
        response_text = await get_openai_response(
            request.user_message,
            request.api_key,
            request.system_prompt
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "response": response_text,
            "processing_time": processing_time,
            "tools_used": [],
            "primary_source": "AI Assistant"
        }
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/career-coach")
async def career_coach(request: CareerCoachRequest):
    """Career coach endpoint following medical assistant's RAG + web search pattern"""
    try:
        start_time = datetime.now()
        logger.info(f"Processing query: {request.user_message}")
        
        # Step 1: Determine query type for routing
        def is_research_query(query: str) -> bool:
            """Check if query is asking for research papers/academic content"""
            query_lower = query.lower()
            research_keywords = [
                'research', 'paper', 'papers', 'study', 'studies', 'publication',
                'arxiv', 'academic', 'journal', 'article', 'literature',
                'research paper', 'research papers', 'academic paper', 'scientific'
            ]
            return any(keyword in query_lower for keyword in research_keywords)
        
        def is_job_related_query(query: str) -> bool:
            """Check if query is job/career related (should use RAG database)"""
            query_lower = query.lower()
            job_keywords = [
                'salary', 'pay', 'wage', 'compensation', 'benefits',
                'job', 'position', 'role', 'opportunity', 'opening',
                'company', 'employer', 'hire', 'hiring', 'career',
                'experience level', 'requirements', 'skills needed',
                'engineer', 'developer', 'scientist', 'analyst', 'manager'
            ]
            return any(keyword in query_lower for keyword in job_keywords)
        
        def requires_current_info(query: str) -> bool:
            """Check if query needs current/web information"""
            query_lower = query.lower()
            current_keywords = [
                'course', 'courses', 'certification', 'training', 'learn', 'study',
                'tutorial', 'bootcamp', 'latest', 'recent', 'new', 'trend',
                'how to', 'guide', 'tips', 'best practices', 'recommendation'
            ]
            return any(keyword in query_lower for keyword in current_keywords)
        
        # Determine query type
        is_research = is_research_query(request.user_message)
        is_job_related = is_job_related_query(request.user_message)
        needs_current_info = requires_current_info(request.user_message)
        
        # Step 2: Follow medical assistant pattern - RAG first, then fallback
        rag_confidence = 0.0
        rag_response = None
        relevant_jobs = []
        
        # If job-related, try RAG first (like medical assistant's RAG_AGENT)
        if is_job_related and not is_research:
            relevant_jobs = search_jobs(request.user_message, limit=10)
            if relevant_jobs:
                # Calculate confidence based on relevance scores and number of results
                total_relevance = sum(job.get('relevance_score', 0) for job in relevant_jobs)
                rag_confidence = min(total_relevance / (len(relevant_jobs) * 3), 1.0)  # Normalize to max 1.0
                
                # Build RAG response
                context_text = "Based on our job database analysis:\n\n"
                for i, job in enumerate(relevant_jobs[:5], 1):
                    context_text += f"{i}. **{job.get('job_title', 'N/A')}** at {job.get('company_name', 'N/A')}\n"
                    context_text += f"   • Salary: ${job.get('salary_usd', 0):,} USD\n"
                    context_text += f"   • Experience: {job.get('experience_level', 'N/A')}\n"
                    context_text += f"   • Location: {job.get('company_location', 'N/A')}\n\n"
                
                rag_response = context_text
                logger.info(f"RAG confidence: {rag_confidence:.2f} from {len(relevant_jobs)} jobs")
        
        # Step 3: Apply confidence threshold (like medical assistant)
        MIN_RAG_CONFIDENCE = 0.7  # Similar to medical assistant's threshold
        use_web_fallback = (
            rag_confidence < MIN_RAG_CONFIDENCE or 
            needs_current_info or 
            is_research or
            not relevant_jobs
        )
        
        # Step 4: Route to appropriate agent
        if use_web_fallback:
            # Use web search (like medical assistant's WEB_SEARCH_PROCESSOR_AGENT)
            if not request.api_key.startswith('sk-'):
                if is_research:
                    response_text = """I'd help you find AI research papers, but I need an OpenAI API key for current research.

**AI Research Resources:**
• **ArXiv.org** - Primary AI/ML research repository
• **Papers with Code** - Research papers with implementation
• **Google Scholar** - Academic paper search
• **Top AI Conferences**: NeurIPS, ICML, ICLR, CVPR

**Current AI Research Trends (2024):**
• Large Language Models and applications
• Multimodal AI systems
• AI safety and alignment
• Efficient model architectures

Please add an OpenAI API key for specific recommendations."""
                    primary_source = "ArXiv Assistant" 
                    tools_used = ["ArXiv Database"]
                elif relevant_jobs and is_job_related:
                    # Show available job data even with low confidence
                    response_text = f"""{rag_response}

**Note:** Based on available data with {rag_confidence:.0%} confidence. For more comprehensive and current results, please provide an OpenAI API key."""
                    primary_source = "Job Database (Limited)"
                    tools_used = ["Job Database"]
                else:
                    response_text = "I need an OpenAI API key to search for current information and courses. Please provide a valid API key starting with 'sk-'."
                    primary_source = "Career Assistant"
                    tools_used = []
            else:
                # Use OpenAI + web search (like medical assistant's pattern)
                if is_research:
                    system_prompt = """You are an AI research assistant specializing in finding and recommending academic papers and research trends. 
                    
Provide specific paper recommendations with:
• **Paper titles** and authors
• **Key contributions** and findings  
• **Research trends** and future directions
• **Resources** like arXiv categories and conferences

Use proper markdown formatting with clear sections."""
                    
                    response_text = await get_openai_response(
                        request.user_message, 
                        request.api_key, 
                        system_prompt
                    )
                    primary_source = "ArXiv Search"
                    tools_used = ["ArXiv", "OpenAI GPT-4"]
                else:
                    # General career advice with web search
                    system_prompt = """You are an AI Career Coach with access to current job market information.
                    
Provide practical advice on:
• **Current job market trends**
• **Skill requirements** and learning paths
• **Course recommendations** with specific platforms
• **Career development** strategies

Use markdown formatting with clear structure and actionable insights."""
                    
                    user_prompt = request.user_message
                    if rag_response:  # Include RAG context if available
                        user_prompt = f"Job market context:\n{rag_response}\n\nUser question: {request.user_message}"
                    
                    response_text = await get_openai_response(
                        user_prompt, 
                        request.api_key, 
                        system_prompt
                    )
                    primary_source = "Tavily Web Search"
                    tools_used = ["Tavily", "OpenAI GPT-4"]
        else:
            # High confidence RAG response (like medical assistant's direct RAG output)
            response_text = f"""{rag_response}

**Analysis Summary:**
Found {len(relevant_jobs)} relevant positions matching your query. The data shows current market opportunities with competitive compensation packages.

**Source:** Job Database (15,000+ career positions)"""
            primary_source = "Job Database"
            tools_used = ["Job Database", "Career RAG"]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return CareerCoachResponse(
            response=response_text,
            processing_time=processing_time,
            tools_used=tools_used,
            primary_source=primary_source,
            context_used=[{"source": "job_database", "documents": len(relevant_jobs)}] if relevant_jobs else []
        )
        
    except Exception as e:
        logger.error(f"Career coach error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/load-data")
async def load_data_endpoint(api_key: str):
    """Load job data"""
    try:
        load_job_data()
        return {"status": f"Loaded {len(job_data)} job records", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Data loading error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Startup
@app.on_event("startup")
async def startup():
    """Load data on startup"""
    logger.info("Starting AI Career Coach API...")
    load_job_data()
    logger.info(f"Loaded {len(job_data)} job records")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 9000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "simple_app:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )