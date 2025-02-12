from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseModel(ABC):
    """Base class for all models (original and challenger)."""
    
    @abstractmethod
    def generate_response(self, prompt: str, context: str = None) -> str:
        """Generate a response for the given prompt and optional context."""
        pass
    
    @abstractmethod
    def generate_test_data(self, development_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate test data based on development data."""
        pass
    
    @abstractmethod
    def batch_generate(self, prompts: List[str], contexts: List[str] = None) -> List[str]:
        """Generate responses for multiple prompts in batch."""
        pass 