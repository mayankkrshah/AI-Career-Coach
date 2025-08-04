#!/usr/bin/env python3
"""Test script to verify research paper queries are handled correctly"""

import requests
import json

# Test research paper query
test_query = "What are the latest research papers in AI?"

# Send request to the API
response = requests.post(
    "http://localhost:9000/api/career-coach",
    json={
        "user_message": test_query,
        "api_key": ""  # No API key
    }
)

print(f"Status Code: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Check if it's using the research assistant
data = response.json()
print(f"\nPrimary Source: {data.get('primary_source', 'N/A')}")
print(f"Tools Used: {data.get('tools_used', [])}")