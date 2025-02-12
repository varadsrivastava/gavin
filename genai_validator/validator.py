from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import ragas
from ragas.metrics import (
    answer_relevancy,
    context_relevancy,
    faithfulness,
    context_recall
)
from .models.base import BaseModel as GenAIModel
from .metrics import MetricsCalculator
from .challenger import ChallengerSelector

class ValidationResults(BaseModel):
    original_metrics: Dict[str, float]
    challenger_metrics: Dict[str, float]
    comparison_metrics: Dict[str, Dict[str, float]]
    
    def generate_report(self) -> str:
        """Generate a formatted report comparing the metrics."""
        report = ["# Model Validation Report\n"]
        
        report.append("## Original Model Metrics")
        for metric, value in self.original_metrics.items():
            report.append(f"- {metric}: {value:.4f}")
            
        report.append("\n## Challenger Model Metrics")
        for metric, value in self.challenger_metrics.items():
            report.append(f"- {metric}: {value:.4f}")
            
        report.append("\n## Comparison Analysis")
        for metric, comparison in self.comparison_metrics.items():
            report.append(f"\n### {metric}")
            report.append(f"- Difference: {comparison['difference']:.4f}")
            report.append(f"- Relative Improvement: {comparison['relative_improvement']:.2%}")
            
        return "\n".join(report)

class ModelValidator:
    def __init__(
        self,
        original_model: GenAIModel,
        challenger_model: Optional[GenAIModel] = None,
        task_type: str = "qa"
    ):
        self.original_model = original_model
        self.challenger_model = challenger_model
        self.task_type = task_type
        self.metrics_calculator = MetricsCalculator()
        
        if challenger_model is None:
            self.challenger_selector = ChallengerSelector(task_type=task_type)
            self.challenger_model = self.challenger_selector.get_best_model()
    
    def validate(
        self,
        development_data: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None
    ) -> ValidationResults:
        """
        Validate models using the provided development data and metrics.
        
        Args:
            development_data: List of data points from the development dataset
            metrics: List of metrics to evaluate (defaults to all available metrics)
        
        Returns:
            ValidationResults object containing the evaluation results
        """
        if metrics is None:
            metrics = ["faithfulness", "context_utilization", "answer_relevancy"]
            
        # Generate test data using challenger model
        test_data = self.challenger_model.generate_test_data(development_data)
        
        # Evaluate original model
        original_results = self.metrics_calculator.calculate_metrics(
            model=self.original_model,
            test_data=test_data,
            metrics=metrics
        )
        
        # Evaluate challenger model
        challenger_results = self.metrics_calculator.calculate_metrics(
            model=self.challenger_model,
            test_data=test_data,
            metrics=metrics
        )
        
        # Compare results
        comparison_metrics = {}
        for metric in metrics:
            orig_value = original_results[metric]
            chall_value = challenger_results[metric]
            comparison_metrics[metric] = {
                "difference": chall_value - orig_value,
                "relative_improvement": (chall_value - orig_value) / orig_value
            }
        
        return ValidationResults(
            original_metrics=original_results,
            challenger_metrics=challenger_results,
            comparison_metrics=comparison_metrics
        ) 