from typing import List, Dict, Any, Optional
import boto3
from .base import BaseModel

class BedrockModel(BaseModel):
    def __init__(
        self,
        model_id: str,
        region_name: str = "us-east-1",
        credentials: Optional[Dict[str, str]] = None,
        max_tokens: int = 1000
    ):
        """
        Initialize Bedrock model.
        
        Args:
            model_id: Bedrock model ID (e.g., "anthropic.claude-v2")
            region_name: AWS region name
            credentials: AWS credentials (access key, secret key)
            max_tokens: Maximum tokens for generation
        """
        self.model_id = model_id
        self.max_tokens = max_tokens
        
        session_kwargs = {"region_name": region_name}
        if credentials:
            session_kwargs.update({
                "aws_access_key_id": credentials.get("access_key"),
                "aws_secret_access_key": credentials.get("secret_key")
            })
        
        self.session = boto3.Session(**session_kwargs)
        self.client = self.session.client("bedrock-runtime")
        
    def _format_prompt(self, prompt: str, context: Optional[str] = None) -> str:
        """Format prompt based on model type."""
        if context:
            if "claude" in self.model_id.lower():
                return f"Context: {context}\n\nHuman: {prompt}\n\nAssistant:"
            else:
                return f"Context: {context}\nQuestion: {prompt}"
        return prompt
    
    def generate_response(self, prompt: str, context: str = None) -> str:
        """Generate a response using the Bedrock model."""
        formatted_prompt = self._format_prompt(prompt, context)
        
        response = self.client.invoke_model(
            modelId=self.model_id,
            body={
                "prompt": formatted_prompt,
                "max_tokens_to_sample": self.max_tokens,
                "temperature": 0.7
            }
        )
        
        return response["body"]["completion"]
    
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
            # Create a prompt for generating test questions/scenarios
            context = item.get("context", "")
            original_question = item.get("question", "")
            
            prompt = (
                f"Based on the following context and original question, "
                f"generate a new, related but different question that tests "
                f"similar knowledge or understanding:\n\n"
                f"Context: {context}\n"
                f"Original Question: {original_question}\n\n"
                f"Generate a new question:"
            )
            
            new_question = self.generate_response(prompt)
            
            # Generate ground truth answer
            answer_prompt = (
                f"Context: {context}\n"
                f"Question: {new_question}\n\n"
                f"Provide a detailed and accurate answer:"
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