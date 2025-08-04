#!/usr/bin/env python3
"""
Test script to verify the intelligent routing between RAG and web search
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.agent_decision import process_query
import logging

# Set up logging to see the routing decisions
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_routing():
    """Test different query types to see routing behavior"""
    
    test_queries = [
        # Should use RAG first
        "What are the top skills for AI engineers?",
        "What is the average salary for a data scientist?",
        
        # Should trigger web search fallback (obscure or recent)
        "What are the latest AI job trends in December 2024?",
        "How is the job market for quantum computing engineers?",
        
        # Should use specialized tools
        "How should I format my resume for an AI role?",
        "What interview questions should I prepare for a machine learning position?",
    ]
    
    print("=" * 80)
    print("TESTING INTELLIGENT ROUTING BETWEEN RAG AND WEB SEARCH")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n\nTest {i}: {query}")
        print("-" * 80)
        
        try:
            result = process_query(query)
            
            # Extract the final response
            if result.get("messages"):
                final_message = result["messages"][-1]
                print(f"\nFinal Response:\n{final_message.content[:500]}...")
                
                # Check which tools were used
                tool_calls = []
                for msg in result["messages"]:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tool_calls.append(tc.get("name", "Unknown"))
                
                if tool_calls:
                    print(f"\nTools Used: {' -> '.join(tool_calls)}")
            else:
                print("No response received")
                
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_routing()