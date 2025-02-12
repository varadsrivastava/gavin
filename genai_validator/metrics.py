from typing import List, Dict, Any
import ragas
from ragas.metrics import (
    answer_relevancy,
    context_relevancy,
    faithfulness,
    context_recall
)
from .models.base import BaseModel

class MetricsCalculator:
    """Calculator for various evaluation metrics."""
    
    def __init__(self):
        """Initialize metrics calculator with available metrics."""
        self.available_metrics = {
            "faithfulness": self._calculate_faithfulness,
            "context_utilization": self._calculate_context_utilization,
            "answer_relevancy": self._calculate_answer_relevancy,
            "context_recall": self._calculate_context_recall
        }
    
    def _calculate_faithfulness(
        self,
        model: BaseModel,
        test_data: List[Dict[str, Any]]
    ) -> float:
        """Calculate faithfulness score."""
        evaluator = faithfulness.Faithfulness()
        scores = []
        
        for item in test_data:
            context = item["context"]
            question = item["question"]
            answer = model.generate_response(question, context)
            
            score = evaluator.score(
                question=question,
                answer=answer,
                context=context
            )
            scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _calculate_context_utilization(
        self,
        model: BaseModel,
        test_data: List[Dict[str, Any]]
    ) -> float:
        """Calculate context utilization score."""
        evaluator = context_relevancy.ContextRelevancy()
        scores = []
        
        for item in test_data:
            context = item["context"]
            question = item["question"]
            answer = model.generate_response(question, context)
            
            score = evaluator.score(
                question=question,
                answer=answer,
                context=context
            )
            scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _calculate_answer_relevancy(
        self,
        model: BaseModel,
        test_data: List[Dict[str, Any]]
    ) -> float:
        """Calculate answer relevancy score."""
        evaluator = answer_relevancy.AnswerRelevancy()
        scores = []
        
        for item in test_data:
            context = item["context"]
            question = item["question"]
            answer = model.generate_response(question, context)
            
            score = evaluator.score(
                question=question,
                answer=answer,
                context=context
            )
            scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _calculate_context_recall(
        self,
        model: BaseModel,
        test_data: List[Dict[str, Any]]
    ) -> float:
        """Calculate context recall score."""
        evaluator = context_recall.ContextRecall()
        scores = []
        
        for item in test_data:
            context = item["context"]
            question = item["question"]
            answer = model.generate_response(question, context)
            
            score = evaluator.score(
                question=question,
                answer=answer,
                context=context
            )
            scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def calculate_metrics(
        self,
        model: BaseModel,
        test_data: List[Dict[str, Any]],
        metrics: List[str] = None
    ) -> Dict[str, float]:
        """
        Calculate specified metrics for the model on test data.
        
        Args:
            model: Model to evaluate
            test_data: Test data to evaluate on
            metrics: List of metric names to calculate (defaults to all available metrics)
            
        Returns:
            Dictionary mapping metric names to scores
        """
        if metrics is None:
            metrics = list(self.available_metrics.keys())
        
        results = {}
        for metric in metrics:
            if metric not in self.available_metrics:
                print(f"Warning: Unknown metric '{metric}', skipping...")
                continue
            
            calculator = self.available_metrics[metric]
            score = calculator(model, test_data)
            results[metric] = score
        
        return results 