#!/usr/bin/env python3
"""
Demo script showing the improved chatbot with intelligent RAG + Web Search routing
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.agent_decision import process_query
import logging

# Suppress most logs for cleaner demo
logging.basicConfig(level=logging.WARNING)

def print_header():
    """Print demo header"""
    print("\n" + "="*80)
    print("🚀 CAREER COACH CHATBOT - Intelligent RAG + Web Search Demo")
    print("="*80)
    print("\nThis chatbot intelligently routes between:")
    print("1. 📊 Local job database (RAG) - for job market data")
    print("2. 🌐 Web search - when local data is insufficient")
    print("3. 💼 Specialized tools - for career services")
    print("\nType 'quit' to exit\n")

def extract_tools_used(result):
    """Extract which tools were used from the result"""
    tools = []
    if result.get("messages"):
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc.get("name", "Unknown")
                    tools.append(tool_name)
    return tools

def chat_loop():
    """Main chat loop"""
    print_header()
    
    while True:
        # Get user input
        user_input = input("\n👤 You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\n👋 Goodbye! Thanks for using Career Coach.")
            break
        
        if not user_input:
            continue
        
        # Process query
        print("\n🤖 Career Coach: ", end="", flush=True)
        
        try:
            result = process_query(user_input)
            
            # Get the response
            if result.get("messages"):
                final_message = result["messages"][-1]
                response = final_message.content
                
                # Clean up any remaining internal markers
                response = response.replace("RAG_LOW_CONFIDENCE: ", "")
                response = response.replace("User Question:", "").strip()
                response = response.replace("Please help answer this career-related question using the available tools.", "").strip()
                response = response.replace("Start with the AI Job RAG tool for job market data.", "").strip()
                
                print(response)
                
                # Show which tools were used (optional - for demo purposes)
                tools = extract_tools_used(result)
                if tools:
                    print(f"\n📋 [Tools used: {' → '.join(tools)}]")
            else:
                print("I'm sorry, I couldn't process your question. Please try again.")
                
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Please try rephrasing your question.")

def main():
    """Main function with example queries"""
    chat_loop()

if __name__ == "__main__":
    # Show some example queries
    print("\n💡 Example queries to try:")
    print("- What are the top skills for AI engineers?")
    print("- What's the average salary for data scientists?")
    print("- What are the latest AI job trends in 2024?")
    print("- How should I prepare for a machine learning interview?")
    print("- What companies are hiring for AI roles right now?")
    
    main()