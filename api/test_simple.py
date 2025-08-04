#!/usr/bin/env python3

"""
Simple test to verify dataset integration
"""

import pandas as pd
import os
from pathlib import Path

# Test dataset loading
try:
    # Try to load the dataset
    data_path = "../data/job_dataset.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, nrows=5)
        print("✅ Dataset loaded successfully!")
        print(f"Found {len(df)} sample rows:")
        print(df[['job_title', 'salary_usd', 'required_skills', 'company_location']].to_string())
        
        # Test creating documents
        job_docs = []
        for _, row in df.iterrows():
            doc_text = f"Job: {row['job_title']}, Salary: ${row['salary_usd']}, Skills: {row['required_skills']}, Location: {row['company_location']}, Company: {row['company_name']}"
            job_docs.append(doc_text)
        
        print(f"\n✅ Created {len(job_docs)} document strings:")
        for i, doc in enumerate(job_docs):
            print(f"{i+1}. {doc}")
            
        # Test simple keyword search
        query = "AI engineer salary"
        print(f"\n🔍 Testing keyword search for: '{query}'")
        
        relevant_docs = []
        query_lower = query.lower()
        
        for doc in job_docs:
            if any(keyword in doc.lower() for keyword in query_lower.split()):
                relevant_docs.append(doc)
        
        print(f"✅ Found {len(relevant_docs)} relevant documents:")
        for doc in relevant_docs:
            print(f"- {doc}")
            
    else:
        print(f"❌ Dataset not found at {data_path}")
        print("Current working directory:", os.getcwd())
        print("Files in current directory:", os.listdir("."))
        
except Exception as e:
    print(f"❌ Error loading dataset: {e}")

# Test directory structure
print(f"\n📁 Directory structure:")
print(f"Current directory: {os.getcwd()}")
print(f"API directory exists: {os.path.exists('.')}")
print(f"Data directory exists: {os.path.exists('../data')}")
print(f"Agents directory exists: {os.path.exists('agents')}")

print(f"\n🎯 Test completed!")