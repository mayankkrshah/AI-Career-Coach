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
    """Career coach endpoint with basic RAG"""
    try:
        start_time = datetime.now()
        
        # First determine the type of query before searching
        logger.info(f"Processing query: {request.user_message}")
        logger.info(f"API key starts with sk-: {request.api_key.startswith('sk-')}, API key: '{request.api_key[:10]}...'")
        
        # Intelligent routing: determine if query should use RAG data, ArXiv, or internet search
        def should_use_arxiv(query: str) -> bool:
            """Determine if query is asking for research papers"""
            query_lower = query.lower()
            arxiv_keywords = [
                'research', 'paper', 'papers', 'study', 'studies', 'publication',
                'arxiv', 'academic', 'journal', 'article', 'literature',
                'research paper', 'research papers', 'academic paper'
            ]
            return any(keyword in query_lower for keyword in arxiv_keywords)
        
        def should_use_internet_search(query: str) -> bool:
            """Determine if query requires internet search vs RAG database"""
            query_lower = query.lower()
            
            # Internet search indicators
            internet_keywords = [
                'course', 'courses', 'certification', 'training', 'learn', 'study',
                'tutorial', 'bootcamp', 'degree', 'university', 'college',
                'latest', 'recent', 'new', 'trend', 'future', 'prediction',
                'news', 'update', 'current events', 'market outlook',
                'how to', 'step by step', 'guide', 'tips', 'advice',
                'best practices', 'recommendation', 'suggest'
            ]
            
            # RAG database indicators (job-related queries)
            rag_keywords = [
                'salary', 'pay', 'wage', 'compensation', 'benefits',
                'job', 'position', 'role', 'opportunity', 'opening',
                'company', 'employer', 'hire', 'hiring',
                'experience level', 'requirements', 'skills needed'
            ]
            
            # Check for internet search indicators
            internet_score = sum(1 for keyword in internet_keywords if keyword in query_lower)
            rag_score = sum(1 for keyword in rag_keywords if keyword in query_lower)
            
            # If asking about courses/training, definitely use internet
            if any(word in query_lower for word in ['course', 'courses', 'training', 'learn', 'certification']):
                return True
                
            # If more internet indicators than RAG indicators, use internet
            return internet_score > rag_score
        
        use_arxiv = should_use_arxiv(request.user_message)
        use_internet = should_use_internet_search(request.user_message)
        
        # Check if this is a data analysis question that we can answer from our database
        def can_analyze_from_database(query: str) -> bool:
            """Check if we can analyze this question using our job database"""
            analysis_keywords = [
                'which country', 'highest salary', 'maximum salary', 'best paying country',
                'country comparison', 'salary by country', 'top paying countries',
                'where pay most', 'highest paid', 'best location for salary',
                'salary in', 'salary of', 'how much', 'average salary', 'pay in',
                'earn in', 'income in', 'compensation in'
            ]
            query_lower = query.lower()
            return any(keyword in query_lower for keyword in analysis_keywords)
        
        def analyze_salary_by_country(jobs: List[Dict], specific_country: str = None) -> str:
            """Analyze salary data by country from job listings"""
            if not jobs:
                return "No salary data available for analysis."
            
            # Extract specific country from query if not provided
            if not specific_country:
                query_lower = request.user_message.lower()
                country_map = {
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
                
                for country, keywords in country_map.items():
                    if any(kw in query_lower for kw in keywords):
                        specific_country = country
                        break
            
            country_salaries = {}
            for job in jobs:
                country = job.get('company_location', 'Unknown')
                salary = job.get('salary_usd', 0)
                if country and salary and salary > 0:
                    if country not in country_salaries:
                        country_salaries[country] = []
                    country_salaries[country].append(salary)
            
            if not country_salaries:
                return "No salary data available by country for AI Research positions."
            
            # Calculate average salaries by country
            country_averages = {}
            for country, salaries in country_salaries.items():
                country_averages[country] = {
                    'avg_salary': sum(salaries) / len(salaries),
                    'job_count': len(salaries),
                    'min_salary': min(salaries),
                    'max_salary': max(salaries)
                }
            
            # Sort by average salary
            sorted_countries = sorted(country_averages.items(), 
                                    key=lambda x: x[1]['avg_salary'], reverse=True)
            
            # If looking for specific country
            if specific_country:
                country_data = None
                for country, data in country_averages.items():
                    if country.lower() == specific_country.lower():
                        country_data = data
                        specific_country = country  # Use proper case
                        break
                
                if country_data:
                    result = f"**AI Engineer Salary in {specific_country}:**\n\n"
                    result += f"• **Average Salary:** ${country_data['avg_salary']:,.0f} USD\n"
                    result += f"• **Salary Range:** ${country_data['min_salary']:,.0f} - ${country_data['max_salary']:,.0f} USD\n"
                    result += f"• **Number of Positions:** {country_data['job_count']}\n\n"
                    
                    # Add some specific job examples
                    result += f"**Sample AI positions in {specific_country}:**\n\n"
                    country_jobs = [job for job in jobs if job.get('company_location', '').lower() == specific_country.lower()]
                    for job in country_jobs[:3]:
                        result += f"• {job.get('job_title', 'N/A')} at {job.get('company_name', 'N/A')}\n"
                        result += f"  Salary: ${job.get('salary_usd', 0):,} USD\n"
                        result += f"  Experience Level: {job.get('experience_level', 'N/A')}\n\n"
                    
                    result += "\n**Note:** Salaries vary based on experience level, company size, and specific skills required."
                else:
                    result = f"Sorry, I don't have salary data for AI engineers in {specific_country} in our current database."
                
                return result
            
            # Otherwise show top countries
            result = "**AI Engineer Salary Analysis by Country (Based on our job database):**\n\n"
            
            for i, (country, data) in enumerate(sorted_countries[:5], 1):
                result += f"**{i}. {country}**\n"
                result += f"   • Average Salary: ${data['avg_salary']:,.0f} USD\n"
                result += f"   • Salary Range: ${data['min_salary']:,.0f} - ${data['max_salary']:,.0f} USD\n"
                result += f"   • Available Positions: {data['job_count']}\n\n"
            
            if len(sorted_countries) > 5:
                result += f"*Analysis includes {len(sorted_countries)} countries from our database.*\n\n"
            
            result += "**Note:** This analysis is based on current job postings in our database. "
            result += "Actual salaries may vary based on experience, company size, and specific role requirements."
            
            return result
        
        can_analyze = can_analyze_from_database(request.user_message)
        
        # If looking for research papers, provide curated response
        if use_arxiv:
            # For now, provide a helpful response about research papers
            # In a full implementation, this would use the ArXiv API
            if not request.api_key.startswith('sk-'):
                response_text = """I'd love to help you find the latest AI research papers, but I need an OpenAI API key to search for current research.

**In the meantime, here are some top AI research resources:**

• **arXiv.org** - The primary repository for AI/ML research papers
• **Papers with Code** - Research papers with implementation code
• **Google Scholar** - Academic paper search engine
• **AI conferences**: NeurIPS, ICML, ICLR, CVPR, ACL

**Recent AI research trends (2024):**
• Large Language Models (LLMs) and their applications
• Multimodal AI and vision-language models
• AI safety and alignment research
• Efficient model architectures and quantization
• Reinforcement learning from human feedback (RLHF)

Please add an OpenAI API key to get specific paper recommendations."""
                primary_source = "AI Assistant"
                tools_used = []
            else:
                # Use OpenAI to provide research paper information
                research_prompt = f"""You are an AI research assistant. The user is asking: {request.user_message}

Please provide:
1. **Recent Papers** - List 3-5 relevant recent AI research papers with titles, authors, and brief descriptions
2. **Key Findings** - Summarize the main contributions
3. **Research Trends** - Highlight current trends in this area
4. **Where to Find More** - Suggest resources like arXiv categories, conferences, or researchers to follow

Format your response with clear sections and use markdown formatting."""
                
                response_text = await get_openai_response(
                    research_prompt,
                    request.api_key,
                    "You are an expert AI research assistant with deep knowledge of recent papers and trends in artificial intelligence."
                )
                primary_source = "Research Assistant"
                tools_used = ["OpenAI GPT-4"]
        # If we can analyze from database, do the analysis
        elif can_analyze:
            # For salary analysis, get ALL relevant jobs, not just 5
            all_relevant_jobs = search_jobs(request.user_message, limit=1000)  # Get more data for analysis
            response_text = analyze_salary_by_country(all_relevant_jobs)
            primary_source = "Job Database"
            tools_used = ["Job Database", "Data Analysis"]
        # If we should use internet search OR no relevant jobs found, use OpenAI + internet context
        elif use_internet or not relevant_jobs:
            if not request.api_key.startswith('sk-'):
                response_text = "I'd love to help with that question, but it requires internet search capabilities. Please provide a valid OpenAI API key starting with 'sk-' to access current information from the web."
                primary_source = "AI Assistant"
                tools_used = []
            else:
                # Build prompt for OpenAI with job context if available
                system_prompt = """You are an AI Career Coach specializing in helping people with their career goals, job search, and professional development.

IMPORTANT FORMATTING RULES:
• Use proper markdown formatting for ALL responses
• Use **bold** for headings and important terms
• Use bullet points (•) or numbered lists for better readability
• Add line breaks between sections for clarity
• Format course recommendations with clear structure:
  - Course name in **bold**
  - Platform on a new line
  - Description on a new line
  - Add spacing between courses

Provide helpful, actionable advice based on current job market trends and best practices. If you have been provided with specific job data, reference it in your response. For course recommendations, training suggestions, or learning paths, provide current and relevant options with proper formatting."""
                
                user_prompt = request.user_message
                if context_text and relevant_jobs:
                    user_prompt = f"Context from job database:\n{context_text}\n\nUser question: {request.user_message}"
                
                # Get AI response
                response_text = await get_openai_response(user_prompt, request.api_key, system_prompt)
                primary_source = "Tavily Web Search"
                tools_used = ["Tavily", "OpenAI GPT-4"]
        else:
            # For job-related queries, search the database
            relevant_jobs = search_jobs(request.user_message)
            logger.info(f"Found {len(relevant_jobs)} relevant jobs for query: {request.user_message}")
            
            if relevant_jobs:
                # Build context from job results
                context_text = "Here are some relevant job opportunities:\n\n"
                for job in relevant_jobs[:3]:
                    context_text += f"- {job.get('job_title', 'N/A')} at {job.get('company_name', 'N/A')}\n"
                    context_text += f"  Salary: ${job.get('salary_usd', 'N/A'):,} USD\n"
                    context_text += f"  Experience: {job.get('experience_level', 'N/A')} level\n"
                    context_text += f"  Skills: {job.get('required_skills', 'N/A')}\n"
                    context_text += f"  Location: {job.get('company_location', 'N/A')} (Remote: {job.get('remote_ratio', 0)}%)\n\n"
                
                primary_source = "Job Database"
                tools_used = ["Job Database"]
                
        if relevant_jobs and not use_arxiv and not use_internet and not can_analyze:
            response_text = f"""Based on the job market data in our database, I found {len(relevant_jobs)} relevant opportunities for your query:

{context_text}

These positions represent current opportunities in the job market. Each role requires specific technical skills and offers competitive compensation packages.

**Source: Job Database (15,000+ AI positions)**"""
            else:
                # No relevant jobs found, use general AI response
            system_prompt = """You are an AI Career Coach specializing in helping people with their career goals, job search, and professional development.

IMPORTANT FORMATTING RULES:
• Use proper markdown formatting for ALL responses
• Use **bold** for headings and important terms
• Use bullet points (•) or numbered lists for better readability
• Add line breaks between sections for clarity
• Keep responses well-structured and easy to scan

Provide helpful, actionable advice based on current job market trends and best practices. If you have been provided with specific job data, reference it in your response."""
                
                user_prompt = request.user_message
                if context_text:
                    user_prompt = f"Context from job database:\n{context_text}\n\nUser question: {request.user_message}"
                
                # Get AI response
                response_text = await get_openai_response(user_prompt, request.api_key, system_prompt)
                primary_source = "AI Assistant"
                tools_used = ["OpenAI GPT-4"]
        
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