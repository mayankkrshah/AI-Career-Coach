"""
Configuration file for the Career Coach Multi-Agent System

This file contains all the configuration parameters for the project.
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Load environment variables from .env file
load_dotenv()

class AgentDecisionConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1  # Deterministic for routing decisions
        )

class ConversationConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7  # Creative but factual
        )

class CareerAdvisorConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.3  # Slightly creative but factual
        )
        self.context_limit = 20  # include last 20 messages in history

class ResumeOptimizerConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.2  # More deterministic for resume optimization
        )

class InterviewCoachConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.4  # Balanced for interview coaching
        )

class SkillAssessorConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.2  # Deterministic for skill assessment
        )

class SalaryNegotiatorConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.3  # Factual for salary information
        )

class NetworkingConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.5  # Creative for networking strategies
        )

class APIConfig:
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 8000
        self.debug = True
        self.rate_limit = 10
        self.max_upload_size = 5  # max upload size in MB

class UIConfig:
    def __init__(self):
        self.theme = "light"
        self.enable_file_upload = True
        self.max_file_size = 10  # MB

class Config:
    def __init__(self):
        self.agent_decision = AgentDecisionConfig()
        self.conversation = ConversationConfig()
        self.career_advisor = CareerAdvisorConfig()
        self.resume_optimizer = ResumeOptimizerConfig()
        self.interview_coach = InterviewCoachConfig()
        self.skill_assessor = SkillAssessorConfig()
        self.salary_negotiator = SalaryNegotiatorConfig()
        self.networking = NetworkingConfig()
        self.api = APIConfig()
        self.ui = UIConfig()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.max_conversation_history = 20  # Include last 20 messages in history