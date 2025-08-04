"""
Local Guardrails System for Career Coach

This module provides input and output validation to ensure safe and appropriate responses.
"""

import re
from typing import Tuple, List, Any
from langchain_core.language_models.base import BaseLanguageModel


class LocalGuardrails:
    """Local guardrails for input and output validation."""
    
    def __init__(self, llm: BaseLanguageModel):
        """
        Initialize the guardrails system.
        
        Args:
            llm: Language model for content validation
        """
        self.llm = llm
        
        # Define blocked content patterns
        self.blocked_patterns = [
            r'\b(illegal|hack|exploit|breach|steal)\b',
            r'\b(violence|harm|hurt|kill|murder)\b',
            r'\b(drug|cocaine|heroin|meth)\b',
            r'\b(suicide|self-harm|depression medication)\b'
        ]
        
        # Define inappropriate career advice patterns
        self.inappropriate_career_patterns = [
            r'\b(lie on resume|fake experience|cheat)\b',
            r'\b(discriminat|bias|unfair hiring)\b',
            r'\b(illegal interview questions)\b'
        ]
    
    def check_input(self, user_input: str) -> Tuple[bool, str]:
        """
        Check if user input is appropriate and safe.
        
        Args:
            user_input: The user's input text
            
        Returns:
            Tuple of (is_allowed, message)
        """
        user_input_lower = user_input.lower()
        
        # Check for blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, user_input_lower, re.IGNORECASE):
                return False, "I can't help with that request. Let's focus on your career development instead."
        
        # Check for inappropriate career advice requests
        for pattern in self.inappropriate_career_patterns:
            if re.search(pattern, user_input_lower, re.IGNORECASE):
                return False, "I can't provide advice on unethical practices. Let me help you with legitimate career strategies instead."
        
        # Check for overly personal or sensitive information
        if any(keyword in user_input_lower for keyword in ['ssn', 'social security', 'credit card', 'password']):
            return False, "Please don't share sensitive personal information. I can help with career advice without those details."
        
        return True, ""
    
    def check_output(self, output: str, original_input: str) -> str:
        """
        Sanitize and validate output before sending to user.
        
        Args:
            output: The generated output text
            original_input: The original user input
            
        Returns:
            Sanitized output text
        """
        # Remove any potential sensitive information
        sanitized_output = output
        
        # Check for inappropriate advice in output
        if any(phrase in output.lower() for phrase in ['lie', 'fake', 'cheat', 'illegal']):
            validation_prompt = f"""
            Review this career advice for appropriateness and ethics:
            
            Original query: {original_input}
            Generated response: {output}
            
            If the response contains unethical advice, rewrite it to be ethical and helpful.
            If it's appropriate, return it as-is.
            
            Ethical response:
            """
            
            try:
                sanitized_response = self.llm.invoke(validation_prompt)
                if hasattr(sanitized_response, 'content'):
                    sanitized_output = sanitized_response.content
                else:
                    sanitized_output = str(sanitized_response)
            except Exception as e:
                # Fallback if LLM validation fails
                sanitized_output = "I want to make sure I provide ethical career advice. Could you rephrase your question so I can better assist you?"
        
        # Ensure response ends appropriately
        if not sanitized_output.endswith(('.', '!', '?')):
            sanitized_output += '.'
        
        # Add disclaimer for sensitive career topics
        sensitive_topics = ['salary negotiation', 'workplace conflict', 'discrimination', 'harassment']
        if any(topic in original_input.lower() for topic in sensitive_topics):
            sanitized_output += "\n\nNote: For legal or HR-related issues, consider consulting with appropriate professionals."
        
        return sanitized_output
    
    def validate_career_advice(self, advice: str) -> bool:
        """
        Validate that career advice is ethical and appropriate.
        
        Args:
            advice: The career advice text to validate
            
        Returns:
            True if advice is appropriate, False otherwise
        """
        advice_lower = advice.lower()
        
        # Check for unethical advice
        unethical_keywords = [
            'lie', 'fake', 'cheat', 'deceive', 'mislead',
            'discriminate', 'bias', 'unfair', 'illegal'
        ]
        
        for keyword in unethical_keywords:
            if keyword in advice_lower:
                return False
        
        return True
    
    def get_safety_guidelines(self) -> List[str]:
        """
        Get a list of safety guidelines for the career coach.
        
        Returns:
            List of safety guidelines
        """
        return [
            "Always provide ethical and legal career advice",
            "Respect privacy and confidentiality",
            "Avoid discriminatory or biased guidance",
            "Encourage honest and authentic professional practices",
            "Refer to appropriate professionals for legal or medical issues",
            "Maintain professional boundaries in conversations"
        ]