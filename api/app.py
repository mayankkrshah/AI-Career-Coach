#!/usr/bin/env python3

"""
AI Career Coach - Medical Assistant Style Structure
Context-first with internet fallback pattern
"""

import pandas as pd
import os
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
import uvicorn
from dotenv import load_dotenv

# Import agent system
from agents.agent_decision import (
    ai_job_rag_tool, 
    career_advisor_tool, 
    resume_optimizer_tool,
    interview_coach_tool,
    skill_assessor_tool,
    networking_tool,
    tavily_tool,
    create_agent_graph
)

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Career Coach",
    description="Career coaching with context-first, internet-fallback approach",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load job dataset for context checking
try:
    df = pd.read_csv("../data/job_dataset.csv", nrows=100)
    job_docs = []
    for _, row in df.iterrows():
        doc_text = f"Job: {row['job_title']}, Salary: ${row['salary_usd']}, Skills: {row['required_skills']}, Location: {row['company_location']}, Company: {row['company_name']}"
        job_docs.append({
            "content": doc_text,
            "title": row['job_title'],
            "salary": row['salary_usd'],
            "skills": row['required_skills'],
            "location": row['company_location'],
            "company": row['company_name']
        })
    print(f"✅ Loaded {len(job_docs)} jobs from dataset")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    job_docs = []

class CareerCoachRequest(BaseModel):
    user_message: str

class CareerCoachResponse(BaseModel):
    response: str
    agent_name: str
    processing_time: float
    source: str  # "context" or "internet" or "general"
    jobs_found: int

class HealthResponse(BaseModel):
    status: str
    timestamp: str

def check_context_relevance(query: str) -> Dict[str, Any]:
    """Check if query can be answered from context (job dataset)"""
    query_lower = query.lower()
    relevant_jobs = []
    
    # Search for relevant jobs in dataset
    for job in job_docs:
        if any(keyword in job["content"].lower() for keyword in query_lower.split()):
            relevant_jobs.append(job)
        if len(relevant_jobs) >= 5:
            break
    
    # Determine if context is sufficient
    context_sufficient = len(relevant_jobs) > 0
    
    # Check for specific context-answerable queries
    context_keywords = [
        "salary", "skills", "roles", "positions", "job", "career", 
        "ai engineer", "data scientist", "machine learning", "python", 
        "requirements", "qualifications", "companies"
    ]
    
    has_context_keywords = any(keyword in query_lower for keyword in context_keywords)
    
    return {
        "relevant_jobs": relevant_jobs,
        "context_sufficient": context_sufficient and has_context_keywords,
        "job_count": len(relevant_jobs)
    }

def need_internet_search(query: str) -> bool:
    """Determine if query needs internet search"""
    internet_keywords = [
        "latest", "recent", "current", "new", "today", "trends", 
        "2024", "2025", "market trends", "industry news", 
        "hiring trends", "breaking", "update"
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in internet_keywords)

def format_context_response(query: str, context_data: Dict[str, Any]) -> str:
    """Format response based on context data"""
    relevant_jobs = context_data["relevant_jobs"]
    query_lower = query.lower()
    
    if not relevant_jobs:
        return "I couldn't find relevant information in our job database for your query."
    
    # Handle different types of queries
    if any(word in query_lower for word in ["roles", "positions", "types", "different", "what are"]):
        # Extract unique job titles/roles
        unique_roles = list(set([job["title"] for job in relevant_jobs[:10]]))
        if len(unique_roles) > 5:
            unique_roles = unique_roles[:5]
        
        response = f"Based on our job database, here are the different AI roles available:\n\n"
        for i, role in enumerate(unique_roles, 1):
            response += f"{i}. {role}\n"
        
        response += f"\nThese roles represent {len(unique_roles)} different types of AI positions from our database."
        return response
    
    elif "salary" in query_lower:
        salaries = [job["salary"] for job in relevant_jobs[:5]]
        avg_salary = sum(salaries) / len(salaries)
        job_info = "\n".join([job["content"] for job in relevant_jobs[:3]])
        return f"Based on {len(relevant_jobs)} relevant positions, the average salary is ${avg_salary:,.0f}. Here are some examples:\n\n{job_info}"
    
    elif "skill" in query_lower:
        all_skills = []
        for job in relevant_jobs[:5]:
            all_skills.extend([skill.strip() for skill in job["skills"].split(",")])
        common_skills = list(set(all_skills))[:10]
        return f"Based on {len(relevant_jobs)} relevant job postings, the most common skills are: {', '.join(common_skills)}"
    
    else:
        # General job information
        job_info = "\n".join([job["content"] for job in relevant_jobs[:3]])
        return f"I found {len(relevant_jobs)} relevant job opportunities in our database:\n\n{job_info}"

# Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=str(datetime.now())
    )

@app.post("/career-coach", response_model=CareerCoachResponse)
async def career_coach_chat(request: CareerCoachRequest):
    """Main career coaching endpoint with context-first, internet-fallback pattern"""
    start_time = time.time()
    
    try:
        query = request.user_message
        logger.info(f"Processing query: {query}")
        
        # Step 1: Check if query can be answered from context
        context_data = check_context_relevance(query)
        needs_internet = need_internet_search(query)
        
        logger.info(f"Context sufficient: {context_data['context_sufficient']}, Needs internet: {needs_internet}")
        
        # Step 2: Decision tree - Context first, then internet
        if context_data["context_sufficient"] and not needs_internet:
            # Answer from context
            response = format_context_response(query, context_data)
            agent_name = "Context RAG Agent"
            source = "context"
            jobs_found = context_data["job_count"]
            logger.info("Using context-based response")
            
        elif needs_internet or not context_data["context_sufficient"]:
            # Use internet search via agent system
            try:
                logger.info("Using internet search via agent system")
                agent_graph = create_agent_graph()
                inputs = {"messages": [HumanMessage(content=query)]}
                result = agent_graph.invoke(inputs)
                
                # Extract response from agent result
                final_message = result["messages"][-1]
                if hasattr(final_message, 'content'):
                    response = final_message.content
                else:
                    response = str(final_message)
                
                agent_name = "Multi-Agent with Internet"
                source = "internet"
                jobs_found = 0
                
            except Exception as e:
                logger.error(f"Agent system error: {e}")
                # Fallback to specific tool based on query type
                if "resume" in query.lower():
                    response = resume_optimizer_tool.invoke({"question": query})
                    agent_name = "Resume Optimizer"
                elif "interview" in query.lower():
                    response = interview_coach_tool.invoke({"question": query})
                    agent_name = "Interview Coach"
                elif "skill" in query.lower():
                    response = skill_assessor_tool.invoke({"question": query})
                    agent_name = "Skill Assessor"
                elif "network" in query.lower():
                    response = networking_tool.invoke({"question": query})
                    agent_name = "Networking Expert"
                else:
                    response = career_advisor_tool.invoke({"question": query})
                    agent_name = "Career Advisor"
                
                source = "general"
                jobs_found = 0
        
        else:
            # General career advice
            response = career_advisor_tool.invoke({"question": query})
            agent_name = "General Career Advisor"
            source = "general"
            jobs_found = 0
            logger.info("Using general career advice")
        
        processing_time = time.time() - start_time
        logger.info(f"Response generated in {processing_time:.2f}s using {source} source")
        
        return CareerCoachResponse(
            response=response,
            agent_name=agent_name,
            processing_time=processing_time,
            source=source,
            jobs_found=jobs_found
        )
        
    except Exception as e:
        logger.error(f"Error in career coach endpoint: {e}")
        processing_time = time.time() - start_time
        
        return CareerCoachResponse(
            response="I apologize, but I'm experiencing some technical difficulties. Please try again later.",
            agent_name="Error Handler",
            processing_time=processing_time,
            source="error",
            jobs_found=0
        )

# Run the app
if __name__ == "__main__":
    import sys
    
    # Get port from command line argument or default to 8000
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Invalid port number, using default 8000")
    
    print(f"🚀 Starting AI Career Coach on port {port}")
    print("📋 Context-first, internet-fallback pattern enabled")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )