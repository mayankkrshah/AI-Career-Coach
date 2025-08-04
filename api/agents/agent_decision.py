"""
Career Coach Multi-Agent System with intelligent routing between RAG and web search

This system uses a sophisticated routing mechanism inspired by the medical assistant:
- Primary: Attempts RAG (document retrieval) for job market data
- Fallback: Uses web search when RAG confidence is low or insufficient
- Smart: Automatically routes to the most appropriate data source
"""

from typing import Dict, List, Optional, Any, TypedDict, Union, Annotated
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
# Temporarily use simple in-memory storage instead of Qdrant for testing
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import os
import pandas as pd
import time
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define State for RAG system (like in notebook)
class State(TypedDict):
    question: str
    context: List[Document]
    response: str
    confidence: float  # Added confidence score for routing decisions

load_dotenv()

from config import Config

# Load configuration
config = Config()

# Create tools with enhanced configuration
tavily_tool = TavilySearchResults(max_results=5, description="Search the web for current job market trends, salary data, and career advice")
arxiv_tool = ArxivQueryRun(description="Search for academic papers and research on AI, machine learning, and career-related topics")

# Simplified RAG system for testing without Qdrant dependencies
def setup_rag_system():
    """Setup simple RAG system with job dataset for testing"""
    try:
        # Load job dataset
        df = pd.read_csv("../data/job_dataset.csv", nrows=100)  # Smaller subset for testing
        
        # Convert to simple text documents
        job_docs = []
        for _, row in df.iterrows():
            doc_text = f"Job: {row['job_title']}, Salary: ${row['salary_usd']}, Skills: {row['required_skills']}, Location: {row['company_location']}, Company: {row['company_name']}"
            job_docs.append(Document(page_content=doc_text, metadata={"source": "job_dataset"}))
        
        print(f"Loaded {len(job_docs)} job documents")
        
        # Simple retrieval function (without vector search for now)
        def simple_retrieve(question):
            # Simple keyword matching for demo
            relevant_docs = []
            question_lower = question.lower()
            
            for doc in job_docs:
                if any(keyword in doc.page_content.lower() for keyword in question_lower.split()):
                    relevant_docs.append(doc)
                if len(relevant_docs) >= 5:  # Limit to 5 docs
                    break
            
            return relevant_docs[:5] if relevant_docs else job_docs[:5]  # Return first 5 if no matches
        
        # Create RAG functions
        def retrieve(state):
            """Simple retrieve function."""
            retrieved_docs = simple_retrieve(state["question"])
            return {"context": retrieved_docs}
        
        def generate(state):
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            
            # Calculate confidence based on document relevance
            confidence = 0.8 if len(state["context"]) >= 3 else 0.4
            
            # Check if we have sufficient information
            insufficient_phrases = [
                "no information", "cannot find", "no data", "not available",
                "insufficient", "unable to answer"
            ]
            
            rag_prompt = ChatPromptTemplate.from_template("""
            You are a helpful assistant who answers questions based on provided job market context. You must only use the provided context about jobs, salaries, and skills.
            
            If the context does not contain sufficient information to answer the question, respond with:
            "I don't have enough information in the job market data to answer this question."

            ### Question
            {question}

            ### Job Market Context
            {context}
            """)
            messages = rag_prompt.format_messages(question=state["question"], context=docs_content)
            response = config.career_advisor.llm.invoke(messages)
            
            # Check response for insufficient information indicators
            response_text = response.content.lower()
            if any(phrase in response_text for phrase in insufficient_phrases):
                confidence = 0.2
                
            return {"response": response.content, "confidence": confidence}
        
        # Create graph
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()
        
        return graph
        
    except Exception as e:
        print(f"Error setting up RAG system: {e}")
        return None

# Initialize RAG system
print("Setting up RAG system...")
rag_graph = setup_rag_system()

@tool
def ai_job_rag_tool(question: str) -> str:
    """Primary tool for AI job market questions. Searches our Job Database of 15,000+ AI positions, salaries, and skills. Use this first for job-related queries."""
    if not rag_graph:
        return "Job market data is currently unavailable. Please try using general career advice instead."
    
    try:
        logger.info(f"RAG tool processing: {question}")
        response = rag_graph.invoke({"question": question})
        
        # Get confidence score
        confidence = response.get("confidence", 0.5)
        logger.info(f"RAG confidence: {confidence}")
        
        # If low confidence, indicate need for web search
        if confidence < 0.6:
            return f"RAG_LOW_CONFIDENCE: {response['response']}"
        
        return response["response"]
    except Exception as e:
        logger.error(f"RAG tool error: {str(e)}")
        return f"Error accessing Job Database: {str(e)}"

@tool
def web_search_fallback(question: str) -> str:
    """Fallback tool that uses Tavily web search with career expertise. Use when Job Database is insufficient or for current trends."""
    try:
        logger.info(f"Web search fallback for: {question}")
        
        # First, try Tavily search
        search_results = tavily_tool.invoke({"query": question})
        
        # Process search results with career context
        prompt = f"""
        You are a career advisor. Based on these web search results, answer the user's question:
        
        Question: {question}
        
        Web Search Results:
        {search_results}
        
        Provide a comprehensive answer focusing on:
        - Current job market trends
        - Practical career advice
        - Actionable recommendations
        """
        
        response = config.career_advisor.llm.invoke(prompt)
        return response.content
        
    except Exception as e:
        logger.error(f"Web search fallback error: {str(e)}")
        return career_advisor_tool(question)  # Fall back to general advice

@tool
def career_advisor_tool(question: str) -> str:
    """Useful for general career guidance, job search advice, and professional development questions when specific job market data is not needed. Input should be a fully formed question."""
    prompt = f"""You are a career advisor. Answer this question: {question}
    
    Provide helpful career guidance including:
    - Actionable advice
    - Industry insights
    - Professional development tips
    - Job search strategies
    """
    
    response = config.career_advisor.llm.invoke(prompt)
    return response.content

@tool  
def resume_optimizer_tool(question: str) -> str:
    """Useful for resume review, optimization, and formatting suggestions. Input should be a fully formed question."""
    prompt = f"""You are a resume optimization expert. Answer this question: {question}
    
    Provide guidance on:
    - Resume formatting and structure
    - Content optimization
    - Keyword suggestions
    - ATS-friendly formatting
    - Industry-specific resume tips
    """
    
    response = config.resume_optimizer.llm.invoke(prompt)
    return response.content

@tool
def interview_coach_tool(question: str) -> str:
    """Useful for interview preparation, mock interviews, and interview strategy. Input should be a fully formed question."""
    prompt = f"""You are an interview coach. Answer this question: {question}
    
    Provide guidance on:
    - Common interview questions and answers
    - Interview preparation strategies
    - Body language and presentation tips
    - Follow-up best practices
    - Salary negotiation in interviews
    """
    
    response = config.interview_coach.llm.invoke(prompt)
    return response.content

@tool
def skill_assessor_tool(question: str) -> str:
    """Useful for skill gap analysis and learning path recommendations. Input should be a fully formed question."""
    prompt = f"""You are a skill assessment expert. Answer this question: {question}
    
    Provide guidance on:
    - In-demand skills for different roles
    - Learning paths and resources
    - Skill gap analysis
    - Certification recommendations
    - Online learning platforms
    """
    
    response = config.skill_assessor.llm.invoke(prompt)
    return response.content

@tool
def networking_tool(question: str) -> str:
    """Useful for networking strategies and professional relationship building. Input should be a fully formed question."""
    prompt = f"""You are a networking expert. Answer this question: {question}
    
    Provide guidance on:
    - Networking strategies and tips
    - Professional relationship building
    - LinkedIn optimization
    - Industry events and conferences
    - Building professional connections
    """
    
    response = config.networking.llm.invoke(prompt)
    return response.content

# Tool belt
tool_belt = [
    ai_job_rag_tool,  # Primary tool for job market data - always try first
    arxiv_tool,  # For research papers and academic content
    web_search_fallback,  # Secondary tool for when RAG is insufficient
    tavily_tool,  # Direct web search when needed
    career_advisor_tool,
    resume_optimizer_tool,
    interview_coach_tool,
    skill_assessor_tool,
    networking_tool,
]

# Agent model
model = config.agent_decision.llm.bind_tools(tool_belt)

# Agent state
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# Enhanced agent functions with intelligent routing
def call_model(state):
    messages = state["messages"]
    
    # Add system prompt for better tool selection
    system_prompt = """
    You are a career coach assistant with intelligent tool routing:
    
    TOOL USAGE PRIORITY:
    1. **ai_job_rag_tool** - ALWAYS use FIRST for job market questions (salaries, skills, positions)
    2. **arxiv** - Use for research papers, academic studies, or when user asks for "research" or "papers"
    3. **web_search_fallback** - Use if RAG returns "RAG_LOW_CONFIDENCE" or for general web search
    4. **Other tools** - Use for specific career services (resume, interview, networking)
    
    ROUTING RULES:
    - If you see "RAG_LOW_CONFIDENCE" in a response, immediately use web_search_fallback
    - For current events or trends ("latest", "2024", "recent"), try RAG first, then web search
    - Never give up after one tool - always try alternatives if the first doesn't work
    """
    
    # Prepend system message if not already present
    if not any(isinstance(msg, AIMessage) and "career coach assistant" in msg.content for msg in messages):
        messages = [AIMessage(content=system_prompt)] + messages
    
    response = model.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tool_belt)

def should_continue(state):
    last_message = state["messages"][-1]
    
    # Check if we need to continue with tool calls
    if last_message.tool_calls:
        return "action"
    
    # Check if the last tool response indicates low confidence
    if len(state["messages"]) >= 2:
        prev_message = state["messages"][-2]
        if hasattr(prev_message, "content") and "RAG_LOW_CONFIDENCE" in str(prev_message.content):
            logger.info("Low RAG confidence detected, agent should use web search")
            # The agent should have already been prompted to use web search
            
    return END

# Create graph
def create_agent_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", tool_node)
    
    workflow.set_entry_point("agent")
    
    workflow.add_conditional_edges(
        "agent",
        should_continue
    )
    
    workflow.add_edge("action", "agent")
    
    return workflow.compile()

def process_query(query: str, max_iterations: int = 3) -> Dict:
    """Process a user query through the multi-agent system with intelligent routing.
    
    Args:
        query: User's question
        max_iterations: Maximum number of tool iterations to prevent infinite loops
        
    Returns:
        Dict containing the conversation messages and final response
    """
    graph = create_agent_graph()
    
    # Enhanced initial prompt to guide the agent
    enhanced_query = f"""
    User Question: {query}
    
    Please help answer this career-related question using the available tools.
    Start with the AI Job RAG tool for job market data.
    """
    
    inputs = {"messages": [HumanMessage(content=enhanced_query)]}
    
    # Add iteration limit to config
    config = {"recursion_limit": max_iterations * 2}  # Each iteration may have 2 steps
    
    try:
        result = graph.invoke(inputs, config=config)
        
        # Clean up the response to remove internal markers
        if result["messages"]:
            last_message = result["messages"][-1]
            if hasattr(last_message, "content"):
                # Remove RAG_LOW_CONFIDENCE markers from final response
                cleaned_content = last_message.content.replace("RAG_LOW_CONFIDENCE: ", "")
                result["messages"][-1] = AIMessage(content=cleaned_content)
                
        return result
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {
            "messages": [
                HumanMessage(content=query),
                AIMessage(content=f"I encountered an error while processing your question. Please try rephrasing or ask a different question. Error: {str(e)}")
            ]
        }