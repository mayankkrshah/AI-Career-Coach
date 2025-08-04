#!/usr/bin/env python3
"""
Comprehensive test script for Career Coach API
Tests all query types to ensure proper routing and no hallucination
"""

import requests
import json
import time

# Test configuration
API_URL = "http://localhost:9000/api/career-coach"
API_KEY = ""  # No API key for testing without OpenAI

# Test queries covering all scenarios
TEST_QUERIES = [
    # Research paper queries (should NOT return job listings)
    {
        "category": "Research Papers",
        "queries": [
            "What are the latest research papers in AI?",
            "Show me recent AI research papers",
            "Find research papers on machine learning",
            "What are the latest studies on neural networks?",
            "ArXiv papers on deep learning"
        ]
    },
    # Job-related queries (should use Job Database)
    {
        "category": "Job Queries",
        "queries": [
            "What are the highest paying AI jobs?",
            "Show me AI engineer positions",
            "What is the average salary for data scientists?",
            "List AI job roles",
            "What companies are hiring AI engineers?"
        ]
    },
    # Location-specific salary queries
    {
        "category": "Location-based Salary",
        "queries": [
            "What is the salary for AI engineers in Canada?",
            "How much do data scientists earn in the UK?",
            "AI engineer salary in Singapore",
            "Best paying countries for AI jobs"
        ]
    },
    # General career advice (should use general assistant)
    {
        "category": "Career Advice",
        "queries": [
            "How to prepare for AI interviews?",
            "What courses should I take to become an AI engineer?",
            "Tips for building a career in AI",
            "Best AI certifications for beginners"
        ]
    },
    # Edge cases
    {
        "category": "Edge Cases",
        "queries": [
            "France job?",
            "What about underwater basket weaving jobs?",
            "Show me jobs for quantum computing specialists",
            "Latest trends in AI job market"
        ]
    }
]

def test_query(query: str, api_key: str = "") -> dict:
    """Test a single query against the API"""
    try:
        response = requests.post(
            API_URL,
            json={
                "user_message": query,
                "api_key": api_key
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"HTTP {response.status_code}",
                "detail": response.text
            }
    except Exception as e:
        return {
            "error": str(e)
        }

def analyze_response(query: str, response: dict) -> dict:
    """Analyze response for correctness"""
    analysis = {
        "query": query,
        "primary_source": response.get("primary_source", "Unknown"),
        "tools_used": response.get("tools_used", []),
        "has_job_listings": False,
        "is_appropriate": True,
        "issues": []
    }
    
    # Check if response contains job listings
    response_text = response.get("response", "").lower()
    job_indicators = ["salary:", "experience:", "location:", "company:", "position at", "job title"]
    
    if any(indicator in response_text for indicator in job_indicators):
        analysis["has_job_listings"] = True
    
    # Check appropriateness based on query type
    query_lower = query.lower()
    
    # Research queries should NOT have job listings
    if any(word in query_lower for word in ["research", "paper", "arxiv", "study", "publication"]):
        if analysis["has_job_listings"]:
            analysis["is_appropriate"] = False
            analysis["issues"].append("Research query returned job listings instead of papers")
    
    # Job queries SHOULD have job listings or indicate no data
    elif any(word in query_lower for word in ["job", "salary", "position", "hire", "company"]):
        if not analysis["has_job_listings"] and "don't have" not in response_text:
            analysis["is_appropriate"] = False
            analysis["issues"].append("Job query didn't return job information")
    
    return analysis

def run_all_tests():
    """Run all test queries and generate report"""
    print("=" * 80)
    print("CAREER COACH API COMPREHENSIVE TEST")
    print("=" * 80)
    
    all_results = []
    
    for category in TEST_QUERIES:
        print(f"\n\n### {category['category']} ###")
        print("-" * 40)
        
        for query in category["queries"]:
            print(f"\nQuery: {query}")
            
            # Test the query
            response = test_query(query, API_KEY)
            
            if "error" in response:
                print(f"❌ Error: {response['error']}")
                continue
            
            # Analyze response
            analysis = analyze_response(query, response)
            all_results.append(analysis)
            
            # Print results
            print(f"Source: {analysis['primary_source']}")
            print(f"Tools: {', '.join(analysis['tools_used']) if analysis['tools_used'] else 'None'}")
            
            if analysis["is_appropriate"]:
                print("✅ Response appropriate")
            else:
                print("❌ Issues found:")
                for issue in analysis["issues"]:
                    print(f"   - {issue}")
            
            # Show snippet of response
            response_text = response.get("response", "")
            snippet = response_text[:200] + "..." if len(response_text) > 200 else response_text
            print(f"Response snippet: {snippet}")
            
            # Small delay between requests
            time.sleep(0.5)
    
    # Summary report
    print("\n\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    
    total_queries = len(all_results)
    appropriate_responses = sum(1 for r in all_results if r["is_appropriate"])
    
    print(f"\nTotal queries tested: {total_queries}")
    print(f"Appropriate responses: {appropriate_responses}")
    print(f"Issues found: {total_queries - appropriate_responses}")
    
    if total_queries > 0:
        success_rate = (appropriate_responses / total_queries) * 100
        print(f"Success rate: {success_rate:.1f}%")
    
    # List all issues
    if total_queries > appropriate_responses:
        print("\n### Issues Summary ###")
        for result in all_results:
            if not result["is_appropriate"]:
                print(f"\nQuery: {result['query']}")
                for issue in result["issues"]:
                    print(f"  - {issue}")
    
    return all_results

if __name__ == "__main__":
    run_all_tests()