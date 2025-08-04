#!/usr/bin/env python3

"""
AI Career Coach - Simple Multi-Agent Implementation
Using tool-based approach similar to the notebook
"""

import os
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Import our simple multi-agent system
from agents.agent_decision import process_query

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Career Coach",
    description="A simple career coaching assistant using multi-agent system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CareerCoachRequest(BaseModel):
    user_message: str

class CareerCoachResponse(BaseModel):
    response: str
    agent_name: str
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    timestamp: str

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
    """Main career coaching endpoint"""
    start_time = time.time()
    
    try:
        # Process query through our simple multi-agent system
        result = process_query(request.user_message)
        
        # Extract response
        final_message = result["messages"][-1]
        response_text = final_message.content
        
        processing_time = time.time() - start_time
        
        return CareerCoachResponse(
            response=response_text,
            agent_name="Multi-Agent Career Coach",
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in career coach endpoint: {e}")
        processing_time = time.time() - start_time
        
        return CareerCoachResponse(
            response="I apologize, but I'm experiencing some technical difficulties. Please try again later.",
            agent_name="Error Handler",
            processing_time=processing_time
        )

# Run the app
if __name__ == "__main__":
    uvicorn.run(
        "simple_app_new:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )