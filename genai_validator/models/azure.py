from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
from .base import BaseModel

class AzureOpenAIModel(BaseModel):
    def __init__(
        self,
        deployment_name: str,
        api_key: str,
        api_base: str,
        api_version: str = "2024-02-15-preview",
        max_tokens: int = 1000
    ):
        """
        Initialize Azure OpenAI model.
        
        Args:
            deployment_name: Name of the deployed model
            api_key: Azure OpenAI API key
            api_base: Azure OpenAI API base URL
            api_version: API version to use
            max_tokens: Maximum tokens for generation
        """
        self.deployment_name = deployment_name
        self.max_tokens = max_tokens
        
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=api_base
        )
    
    def _format_prompt(self, prompt: str, context: Optional[str] = None) -> List[Dict[str, str]]:
        """Format prompt as messages for chat completion."""
        messages = []
        
        if context:
            messages.append({
                "role": "system",
                "content": f"Use the following context to answer the question: {context}"
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        return messages
    
    def generate_response(self, prompt: str, context: str = None) -> str:
        """Generate a response using the Azure OpenAI model."""
        messages = self._format_prompt(prompt, context)
        
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def batch_generate(self, prompts: List[str], contexts: List[str] = None) -> List[str]:
        """Generate responses for multiple prompts."""
        if contexts is None:
            contexts = [None] * len(prompts)
        
        responses = []
        for prompt, context in zip(prompts, contexts):
            response = self.generate_response(prompt, context)
            responses.append(response)
        
        return responses
    
    def generate_test_data(self, development_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate test data based on development data."""
        test_data = []
        
        for item in development_data:
            context = item.get("context", "")
            original_question = item.get("question", "")
            
            # Generate new test question
            question_prompt = (
                "Based on the following context and original question, "
                "generate a new, related but different question that tests "
                "similar knowledge or understanding:\n\n"
                f"Context: {context}\n"
                f"Original Question: {original_question}\n\n"
                "Generate a new question:"
            )
            
            new_question = self.generate_response(question_prompt)
            
            # Generate ground truth answer
            answer_prompt = (
                f"Context: {context}\n"
                f"Question: {new_question}\n\n"
                "Provide a detailed and accurate answer:"
            )
            
            ground_truth = self.generate_response(answer_prompt)
            
            test_data.append({
                "context": context,
                "question": new_question,
                "ground_truth": ground_truth,
                "original_question": original_question,
                "original_answer": item.get("answer", "")
            })
        
        return test_data 