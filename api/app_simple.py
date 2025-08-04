#!/usr/bin/env python3

"""
Simple Career Coach API without SSL dependencies for testing
"""

import pandas as pd
import os
import json
import time
from datetime import datetime
from typing import Dict, Any
from collections import Counter
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="AI Career Coach - Simple Test",
    description="Testing dataset integration without complex dependencies",
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

# Load job dataset
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

class QueryRequest(BaseModel):
    user_message: str

class QueryResponse(BaseModel):
    response: str
    agent_used: str
    jobs_found: int
    processing_time: float

def search_jobs(query: str) -> list:
    """Improved job search function with better keyword matching"""
    query_lower = query.lower()
    relevant_jobs = []
    
    # Handle specific AI/tech terms that should match broadly
    ai_terms = ["ai", "artificial intelligence", "machine learning", "ml", "data science", "python", "engineer"]
    broad_match = any(term in query_lower for term in ai_terms)
    
    if broad_match:
        # For AI-related queries, use broader matching
        query_keywords = [word.strip() for word in query_lower.split() if len(word.strip()) > 2]
        
        for job in job_docs:
            job_content_lower = job["content"].lower()
            # For AI terms, match more liberally
            matches = sum(1 for keyword in query_keywords if keyword in job_content_lower)
            title_match = any(keyword in job["title"].lower() for keyword in query_keywords)
            
            if matches >= 1 or title_match:
                relevant_jobs.append(job)
            if len(relevant_jobs) >= 5:
                break
    else:
        # For specific queries (like "blockchain"), require exact matches
        query_keywords = [word.strip() for word in query_lower.split() if len(word.strip()) > 2]
        
        for job in job_docs:
            job_content_lower = job["content"].lower()
            # Require at least 2 keyword matches or exact job title match
            matches = sum(1 for keyword in query_keywords if keyword in job_content_lower)
            title_match = any(keyword in job["title"].lower() for keyword in query_keywords)
            
            if matches >= 2 or title_match:
                relevant_jobs.append(job)
            if len(relevant_jobs) >= 5:
                break
    
    return relevant_jobs

def classify_query_type(query: str) -> list:
    """Classify the query type to route to appropriate handler - can return multiple types"""
    query_lower = query.lower()
    query_types = []
    
    # Check for multiple patterns in the same query
    # Interview/advice queries
    if any(word in query_lower for word in ["interview", "negotiate", "negotiation", "behavioral", "prepare"]):
        query_types.append("interview_advice")
    
    # Resume queries
    if any(word in query_lower for word in ["resume", "cv", "portfolio", "format", "optimize"]):
        query_types.append("resume_advice")
    
    # Networking queries  
    if any(word in query_lower for word in ["network", "linkedin", "connections", "professional relationship"]):
        query_types.append("networking_advice")
    
    # Skill/learning queries
    if any(word in query_lower for word in ["learn", "course", "certification", "skill", "training"]):
        query_types.append("skill_advice")
    
    # Current trends/internet queries
    if any(word in query_lower for word in ["latest", "recent", "trends", "current", "2024", "2025", "market trends", "breaking"]):
        query_types.append("internet_search")
    
    # Job database queries - roles
    if any(word in query_lower for word in ["roles", "positions", "jobs", "what are", "different"]):
        query_types.append("job_roles")
    
    # Job database queries - salary
    if any(word in query_lower for word in ["salary", "salaries", "pay", "compensation", "wage", "earn"]):
        query_types.append("job_salary")
        
    # Job database queries - skills  
    if any(word in query_lower for word in ["skills required", "skills needed", "requirements", "qualifications"]):
        query_types.append("job_skills")
    
    # General career advice
    if any(word in query_lower for word in ["career", "advice", "guidance", "help", "should i", "how to"]) and not query_types:
        query_types.append("general_advice")
    
    # Default if nothing matches
    if not query_types:
        query_types.append("general_advice")
        
    return query_types

def extract_context_from_query(query: str) -> dict:
    """Extract context like company names, locations, specific roles"""
    context = {
        "companies": [],
        "locations": [],
        "specific_roles": [],
        "experience_level": None
    }
    
    query_lower = query.lower()
    
    # Extract company names
    companies = ["google", "amazon", "microsoft", "apple", "meta", "facebook", "netflix", "tesla", "openai", "anthropic"]
    for company in companies:
        if company in query_lower:
            context["companies"].append(company.title())
    
    # Extract locations
    locations = ["san francisco", "sf", "austin", "seattle", "new york", "nyc", "boston", "chicago", "los angeles", "la", "remote"]
    for location in locations:
        if location in query_lower:
            context["locations"].append(location.title())
    
    # Extract specific roles
    roles = ["machine learning engineer", "data scientist", "software engineer", "ai engineer", "backend engineer", "frontend engineer"]
    for role in roles:
        if role in query_lower:
            context["specific_roles"].append(role.title())
    
    # Extract experience level
    if any(word in query_lower for word in ["junior", "entry", "new grad", "fresh"]):
        context["experience_level"] = "junior"
    elif any(word in query_lower for word in ["senior", "lead", "principal", "staff"]):
        context["experience_level"] = "senior"
    
    return context

def search_jobs_with_filters(query: str, location: str = None, company: str = None) -> list:
    """Enhanced job search with location and company filtering"""
    query_lower = query.lower()
    relevant_jobs = []
    
    # Handle specific AI/tech terms that should match broadly
    ai_terms = ["ai", "artificial intelligence", "machine learning", "ml", "data science", "python", "engineer"]
    broad_match = any(term in query_lower for term in ai_terms)
    
    for job in job_docs:
        job_content_lower = job["content"].lower()
        
        # Apply location filter if specified
        if location and location.lower() not in job["location"].lower():
            continue
            
        # Apply company filter if specified  
        if company and company.lower() not in job["company"].lower():
            continue
        
        # Match based on query
        if broad_match:
            query_keywords = [word.strip() for word in query_lower.split() if len(word.strip()) > 2]
            matches = sum(1 for keyword in query_keywords if keyword in job_content_lower)
            title_match = any(keyword in job["title"].lower() for keyword in query_keywords)
            
            if matches >= 1 or title_match:
                relevant_jobs.append(job)
        else:
            # For specific queries, require exact matches
            query_keywords = [word.strip() for word in query_lower.split() if len(word.strip()) > 2]
            matches = sum(1 for keyword in query_keywords if keyword in job_content_lower)
            title_match = any(keyword in job["title"].lower() for keyword in query_keywords)
            
            if matches >= 2 or title_match:
                relevant_jobs.append(job)
                
        if len(relevant_jobs) >= 10:  # Get more results for filtering
            break
    
    return relevant_jobs[:5]

def generate_context_aware_response(query: str, query_types: list, context: dict) -> str:
    """Generate context-aware responses for multiple query types"""
    
    # Handle empty query
    if not query.strip():
        return """Hello! I'm your AI Career Coach. I can help you with:

• **Job Information** - AI roles, salaries, required skills
• **Interview Preparation** - Tips, practice questions, company research
• **Resume Optimization** - Formatting, content, ATS optimization  
• **Networking Strategies** - LinkedIn tips, professional connections
• **Skill Development** - Learning paths, certifications, courses
• **Career Guidance** - Planning, job search strategies

What would you like help with today?"""

    responses = []
    
    # Handle compound queries by processing each type
    for query_type in query_types:
        if query_type == "interview_advice":
            if context["companies"]:
                company = context["companies"][0]
                response = f"""**Interview Tips for {company}:**

1. **Company Research** - Study {company}'s mission, recent products, and engineering culture
2. **Technical Preparation** - Practice coding problems and system design
3. **Behavioral Questions** - Prepare STAR format examples showing impact
4. **Questions to Ask** - About team dynamics, growth opportunities, and technical challenges

**{company}-Specific Tips:**
- Review their engineering blog and recent technical talks
- Understand their technology stack and engineering practices
- Be ready to discuss how you'd contribute to their specific challenges

**Salary Negotiation:**
- Research {company}'s compensation bands on levels.fyi
- Negotiate total compensation including equity
- Be prepared with competing offers if available"""
            else:
                response = """**Interview Preparation Tips:**

1. **Research the company** - Know their mission, values, and recent news
2. **Practice common questions** - "Tell me about yourself", "Why do you want this role?"
3. **Prepare STAR examples** - Situation, Task, Action, Result stories
4. **Technical preparation** - Coding problems, system design, domain knowledge
5. **Questions to ask** - About the role, team, and company culture

**Salary Negotiation:**
- Research market rates beforehand (use levels.fyi, Glassdoor)
- Let them make the first offer
- Negotiate the total package, not just salary
- Be prepared to justify your ask with specific achievements"""
            responses.append(response)
            
        elif query_type == "resume_advice":
            response = """**Resume Optimization Tips:**

1. **Format & Structure:**
   - Keep to 1-2 pages maximum
   - Use clear headings and bullet points
   - Choose ATS-friendly fonts (Arial, Calibri, Times New Roman)

2. **Content Optimization:**
   - Start with a strong professional summary
   - Use action verbs and quantify achievements (increased X by Y%)
   - Tailor keywords to match job descriptions
   - Include relevant technical skills section

3. **For AI/Tech roles specifically:**
   - Highlight programming languages and frameworks
   - Show project outcomes with specific metrics
   - Include GitHub/portfolio links
   - Mention relevant certifications and publications

4. **Common Mistakes to Avoid:**
   - Generic objectives instead of targeted summaries
   - Missing quantified achievements
   - Poor formatting that breaks ATS parsing
   - Including irrelevant work experience"""
            responses.append(response)
            
        elif query_type == "networking_advice":
            response = """**Networking Strategies:**

1. **LinkedIn Optimization:**
   - Professional headshot and compelling headline
   - Detailed summary with industry keywords
   - Regular engagement with industry content
   - Join relevant professional groups

2. **Building Connections:**
   - Attend industry meetups and conferences
   - Join professional associations (IEEE, ACM, etc.)
   - Participate in online communities (Reddit, Discord, Slack)
   - Contribute to open source projects

3. **Relationship Building:**
   - Offer value before asking for help
   - Follow up consistently but not aggressively
   - Share relevant opportunities with your network
   - Send personalized connection requests

4. **For AI/Tech professionals:**
   - Engage with AI communities on Kaggle, Papers with Code
   - Attend ML conferences (NeurIPS, ICML, local meetups)
   - Contribute to technical discussions on Twitter/X
   - Write technical blog posts to establish expertise"""
            responses.append(response)
            
        elif query_type == "skill_advice":
            if context["specific_roles"]:
                role = context["specific_roles"][0]
                response = f"""**Skill Development for {role}:**

**Core Technical Skills:**
- Python, SQL, Git (foundational)
- Machine Learning: TensorFlow, PyTorch, scikit-learn
- Data manipulation: pandas, numpy
- Cloud platforms: AWS, GCP, or Azure
- Statistics and mathematics fundamentals

**Learning Path:**
1. **Foundations** - Statistics, linear algebra, Python programming
2. **ML Fundamentals** - Supervised/unsupervised learning, model evaluation
3. **Specialized Skills** - Deep learning, NLP, computer vision (based on interest)
4. **Production Skills** - MLOps, Docker, Kubernetes, monitoring
5. **Domain Knowledge** - Specific to your target industry

**Recommended Resources:**
- Coursera: Andrew Ng's ML Course, Deep Learning Specialization
- Books: "Hands-On Machine Learning" by Aurélien Géron
- Practice: Kaggle competitions, personal projects
- Certifications: AWS ML, Google Cloud ML, Azure AI"""
            else:
                response = """**AI/Tech Skill Development:**

**In-Demand Skills:**
- Python, R, SQL for data analysis
- Machine Learning frameworks (TensorFlow, PyTorch, scikit-learn)
- Cloud platforms (AWS, Azure, GCP)
- Version control (Git) and software engineering practices
- Statistics, linear algebra, and domain expertise

**Learning Strategy:**
1. **Start with fundamentals** - Programming, statistics, math
2. **Take structured courses** - Coursera, edX, Udacity specializations
3. **Build projects** - GitHub portfolio with real-world applications
4. **Get certified** - Cloud platforms, specific technologies
5. **Join communities** - Kaggle, Reddit ML, local meetups

**Skill Gap Analysis:**
- Compare your skills to job requirements
- Focus on high-impact skills first
- Practice through real projects, not just tutorials
- Get feedback from experienced professionals"""
            responses.append(response)
            
        elif query_type == "general_advice":
            response = """**Career Guidance:**

1. **Career Planning:**
   - Define clear short and long-term goals
   - Identify your strengths and interests through self-assessment
   - Research different career paths and growth opportunities
   - Create a development plan with specific milestones

2. **Job Search Strategy:**
   - Use multiple channels: job boards, networking, referrals
   - Customize applications for each role
   - Follow up professionally but not excessively
   - Track applications and responses for optimization

3. **Professional Development:**
   - Continuously update technical and soft skills
   - Seek feedback and mentorship from senior professionals
   - Build a strong professional brand and online presence
   - Stay current with industry trends and technologies

4. **Work-Life Balance:**
   - Set clear boundaries between work and personal time
   - Prioritize mental health and well-being
   - Develop stress management techniques
   - Build supportive relationships both professionally and personally"""
            responses.append(response)
    
    return "\n\n---\n\n".join(responses)

def generate_internet_search_response(query: str) -> str:
    """Generate response for queries requiring internet search"""
    # Since we don't have actual internet access, provide intelligent fallback
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["2024", "2025", "latest", "recent", "current"]):
        return f"""**Current Industry Information:**

I don't have access to real-time data, but here's how to find the latest information about "{query}":

**Best Sources for Current AI/Tech Trends:**
1. **Industry Reports:** McKinsey Global Institute, Deloitte Tech Trends, PwC AI Analysis
2. **Tech News:** TechCrunch, Ars Technica, Wired, IEEE Spectrum
3. **Research:** arXiv.org for latest research papers, Papers with Code
4. **Professional Networks:** LinkedIn industry updates, Twitter/X tech leaders
5. **Company Blogs:** Google AI Blog, OpenAI Blog, Meta AI Research
6. **Conferences:** NeurIPS, ICML, ICLR proceedings and presentations

**Key Areas to Monitor:**
- Generative AI developments and applications
- MLOps and AI infrastructure trends
- AI ethics and regulation updates
- New model architectures and capabilities
- Industry adoption patterns and use cases

**For Job Market Trends:**
- LinkedIn Economic Graph data
- Stack Overflow Developer Survey
- GitHub State of the Octoverse
- Company hiring announcements and layoffs news

Would you like me to help you analyze any specific aspect of AI trends based on general industry knowledge?"""
    
    elif any(word in query_lower for word in ["trends", "market", "industry"]):
        return """**AI Industry Trends (General Outlook):**

**Key Technology Trends:**
1. **Generative AI Expansion** - Beyond text to multimodal applications
2. **MLOps Maturation** - Better tools for model deployment and monitoring
3. **Edge AI** - Moving inference closer to data sources
4. **AI Democratization** - No-code/low-code AI platforms
5. **Responsible AI** - Focus on ethics, fairness, and explainability

**Job Market Patterns:**
- High demand for ML engineers and AI researchers
- Growth in AI product manager and AI safety roles
- Increasing need for domain experts with AI knowledge
- Emphasis on production ML skills vs pure research

**Skills in Demand:**
- MLOps and model deployment
- Large language model fine-tuning
- Multimodal AI applications
- AI system monitoring and maintenance
- Cross-functional collaboration skills

For the most current data, I recommend checking recent industry reports and company announcements."""
    
    else:
        return f"""**Information Search Guidance:**

I don't have access to real-time internet data for "{query}", but here's how you can find current information:

**Research Strategy:**
1. **Primary Sources** - Company websites, official announcements, research papers
2. **Industry Analysis** - Gartner, Forrester, McKinsey reports
3. **Professional Networks** - LinkedIn insights, industry experts
4. **News Aggregators** - Google News, AllSides, industry publications
5. **Academic Sources** - Google Scholar, arXiv, university research

**Search Tips:**
- Use specific keywords and date filters
- Cross-reference multiple sources
- Look for peer-reviewed or authoritative sources
- Check for recent updates or corrections

Would you like help formulating a more specific question I can answer with my knowledge base?"""

def process_job_database_query(query: str, query_types: list, context: dict) -> dict:
    """Process job database queries with enhanced filtering"""
    
    # Get location and company filters from context
    location_filter = context["locations"][0] if context["locations"] else None
    company_filter = context["companies"][0] if context["companies"] else None
    
    # Search with filters
    relevant_jobs = search_jobs_with_filters(query, location_filter, company_filter)
    
    responses = []
    
    if not relevant_jobs:
        if location_filter or company_filter:
            filters = []
            if location_filter:
                filters.append(f"location: {location_filter}")
            if company_filter:
                filters.append(f"company: {company_filter}")
            
            return {
                "response": f"I couldn't find jobs matching your criteria ({', '.join(filters)}) in our database. This might be because:\n\n1. Our dataset has limited coverage for specific locations/companies\n2. The search terms might need adjustment\n3. The role might be very specialized\n\nTry broadening your search or ask me for general advice about working in {location_filter or 'that company'}!",
                "jobs_found": 0
            }
        else:
            return {
                "response": f"I couldn't find specific jobs matching '{query}' in our database. Our dataset focuses on AI/ML roles. Try asking about 'AI roles', 'data science positions', or 'machine learning jobs' instead!",
                "jobs_found": 0
            }
    
    # Process different query types
    for query_type in query_types:
        if query_type == "job_roles":
            unique_roles = list(set([job["title"] for job in relevant_jobs[:10]]))
            if len(unique_roles) > 5:
                unique_roles = unique_roles[:5]
            
            filter_text = ""
            if location_filter or company_filter:
                filters = []
                if location_filter:
                    filters.append(f"in {location_filter}")
                if company_filter:
                    filters.append(f"at {company_filter}")
                filter_text = f" {' '.join(filters)}"
            
            response = f"Based on our job database, here are the different AI roles available{filter_text}:\n\n"
            for i, role in enumerate(unique_roles, 1):
                response += f"{i}. {role}\n"
            
            response += f"\nThese represent {len(unique_roles)} different types of AI positions from {len(relevant_jobs)} matching jobs in our database."
            responses.append(response)
            
        elif query_type == "job_salary":
            salaries = [job["salary"] for job in relevant_jobs[:10]]
            avg_salary = sum(salaries) / len(salaries)
            min_salary = min(salaries)
            max_salary = max(salaries)
            
            filter_text = ""
            if location_filter or company_filter:
                filters = []
                if location_filter:
                    filters.append(f"in {location_filter}")
                if company_filter:
                    filters.append(f"at {company_filter}")
                filter_text = f" {' '.join(filters)}"
            
            response = f"**Salary Information{filter_text}:**\n\n"
            response += f"📊 **Salary Range:** ${min_salary:,.0f} - ${max_salary:,.0f}\n"
            response += f"💰 **Average Salary:** ${avg_salary:,.0f}\n"
            response += f"📈 **Based on:** {len(relevant_jobs)} positions\n\n"
            
            response += "**Sample Positions:**\n"
            for job in relevant_jobs[:3]:
                response += f"• {job['title']}: ${job['salary']:,.0f} at {job['company']} ({job['location']})\n"
            
            responses.append(response)
            
        elif query_type == "job_skills":
            all_skills = []
            for job in relevant_jobs[:10]:
                skills_list = [skill.strip() for skill in job["skills"].split(",")]
                all_skills.extend(skills_list)
            
            # Count skill frequency
            skill_counts = Counter(all_skills)
            top_skills = skill_counts.most_common(10)
            
            response = f"**Skills Required for AI Roles** (based on {len(relevant_jobs)} positions):\n\n"
            response += "**Most In-Demand Skills:**\n"
            for i, (skill, count) in enumerate(top_skills, 1):
                percentage = (count / len(relevant_jobs)) * 100
                response += f"{i}. **{skill}** - {percentage:.0f}% of jobs\n"
            
            responses.append(response)
    
    return {
        "response": "\n\n---\n\n".join(responses),
        "jobs_found": len(relevant_jobs)
    }

def generate_advice_response(query: str, advice_type: str) -> str:
    """Legacy function - kept for compatibility"""
    context = extract_context_from_query(query)
    return generate_context_aware_response(query, [advice_type], context)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": str(datetime.now()),
        "jobs_loaded": len(job_docs)
    }

@app.post("/career-coach", response_model=QueryResponse)
async def career_coach_chat(request: QueryRequest):
    start_time = time.time()
    
    try:
        query = request.user_message
        
        # Handle empty query
        if not query.strip():
            return QueryResponse(
                response="Hello! I'm your AI Career Coach. I can help you with job information, interview preparation, resume optimization, networking strategies, skill development, and career guidance. What would you like help with today?",
                agent_used="welcome_agent",
                jobs_found=0,
                processing_time=time.time() - start_time
            )
        
        # Step 1: Classify the query types (can be multiple)
        query_types = classify_query_type(query)
        context = extract_context_from_query(query)
        
        # Step 2: Route to appropriate handlers
        advice_types = [qt for qt in query_types if qt in ["interview_advice", "resume_advice", "networking_advice", "skill_advice", "general_advice"]]
        job_db_types = [qt for qt in query_types if qt in ["job_roles", "job_salary", "job_skills"]]
        internet_types = [qt for qt in query_types if qt == "internet_search"]
        
        responses = []
        total_jobs_found = 0
        agents_used = []
        
        # Handle advice queries
        if advice_types:
            advice_response = generate_context_aware_response(query, advice_types, context)
            responses.append(advice_response)
            agents_used.extend([f"{at}_agent" for at in advice_types])
        
        # Handle job database queries
        if job_db_types:
            job_result = process_job_database_query(query, job_db_types, context)
            responses.append(job_result["response"])
            total_jobs_found = job_result["jobs_found"]
            agents_used.append("job_database_agent")
        
        # Handle internet search queries
        if internet_types:
            internet_response = generate_internet_search_response(query)
            responses.append(internet_response)
            agents_used.append("internet_search_agent")
        
        # If no specific type was matched, provide general help
        if not responses:
            response = generate_context_aware_response(query, ["general_advice"], context)
            responses.append(response)
            agents_used.append("general_advice_agent")
        
        # Combine responses
        final_response = "\n\n---\n\n".join(responses)
        agent_used = " + ".join(agents_used) if len(agents_used) > 1 else agents_used[0]
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            response=final_response,
            agent_used=agent_used,
            jobs_found=total_jobs_found,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        return QueryResponse(
            response=f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question or ask for help with a specific topic like job information, interview tips, or career advice.",
            agent_used="error_handler",
            jobs_found=0,
            processing_time=processing_time
        )

if __name__ == "__main__":
    import sys
    
    # Get port from command line argument or default to 8000
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Invalid port number, using default 8000")
    
    print(f"🚀 Starting Career Coach API on port {port}")
    uvicorn.run(
        "app_simple:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )