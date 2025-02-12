from typing import Dict, Optional
from .models.base import BaseModel
from .models.azure import AzureOpenAIModel
from .models.bedrock import BedrockModel

class ChallengerSelector:
    """Selector for challenger models based on task type."""
    
    def __init__(self, task_type: str):
        """
        Initialize challenger selector.
        
        Args:
            task_type: Type of task ("qa", "summarization", "reasoning")
        """
        self.task_type = task_type.lower()
        
        # Mapping of task types to recommended models and their benchmarks
        self.task_models = {
            "qa": {
                "model": "gpt-4",
                "provider": "azure",
                "benchmark_score": 0.92,
                "benchmark_name": "SQuAD 2.0"
            },
            "summarization": {
                "model": "anthropic.claude-v2",
                "provider": "bedrock",
                "benchmark_score": 0.89,
                "benchmark_name": "ROUGE-L on CNN/DailyMail"
            },
            "reasoning": {
                "model": "gpt-4",
                "provider": "azure",
                "benchmark_score": 0.90,
                "benchmark_name": "GSM8K"
            }
        }
    
    def get_best_model(
        self,
        azure_credentials: Optional[Dict[str, str]] = None,
        bedrock_credentials: Optional[Dict[str, str]] = None
    ) -> BaseModel:
        """
        Get the best challenger model for the specified task.
        
        Args:
            azure_credentials: Credentials for Azure OpenAI (if needed)
            bedrock_credentials: Credentials for AWS Bedrock (if needed)
            
        Returns:
            Instance of the best model for the task
        """
        if self.task_type not in self.task_models:
            raise ValueError(
                f"Unknown task type: {self.task_type}. "
                f"Available tasks: {list(self.task_models.keys())}"
            )
        
        model_info = self.task_models[self.task_type]
        provider = model_info["provider"]
        
        if provider == "azure":
            if not azure_credentials:
                raise ValueError(
                    "Azure credentials required for the selected challenger model"
                )
            
            return AzureOpenAIModel(
                deployment_name=model_info["model"],
                api_key=azure_credentials["api_key"],
                api_base=azure_credentials["api_base"],
                api_version=azure_credentials.get("api_version", "2024-02-15-preview")
            )
        
        elif provider == "bedrock":
            if not bedrock_credentials:
                raise ValueError(
                    "AWS credentials required for the selected challenger model"
                )
            
            return BedrockModel(
                model_id=model_info["model"],
                credentials=bedrock_credentials
            )
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def get_benchmark_info(self) -> Dict[str, str]:
        """
        Get information about the benchmark used for the selected task.
        
        Returns:
            Dictionary containing benchmark information
        """
        if self.task_type not in self.task_models:
            raise ValueError(
                f"Unknown task type: {self.task_type}. "
                f"Available tasks: {list(self.task_models.keys())}"
            )
        
        model_info = self.task_models[self.task_type]
        return {
            "model": model_info["model"],
            "provider": model_info["provider"],
            "benchmark_name": model_info["benchmark_name"],
            "benchmark_score": str(model_info["benchmark_score"])
        } 