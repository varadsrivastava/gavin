from genai_validator import ModelValidator, S3DataExtractor
from genai_validator.models import BedrockModel, AzureOpenAIModel

def main():
    # AWS credentials for S3 and Bedrock
    aws_credentials = {
        "access_key": "your_aws_access_key",
        "secret_key": "your_aws_secret_key"
    }
    
    # Azure OpenAI credentials
    azure_credentials = {
        "api_key": "your_azure_api_key",
        "api_base": "your_azure_endpoint",
        "api_version": "2024-02-15-preview"
    }
    
    # Initialize data extractor
    data_extractor = S3DataExtractor(
        bucket_name="your-data-bucket",
        prefix="development/data",
        credentials=aws_credentials
    )
    
    # Initialize your original model (e.g., using Bedrock)
    original_model = BedrockModel(
        model_id="anthropic.claude-v2",
        credentials=aws_credentials
    )
    
    # Create validator (it will automatically select the best challenger model)
    validator = ModelValidator(
        original_model=original_model,
        task_type="qa"  # or "summarization" or "reasoning"
    )
    
    # Extract development data
    development_data = data_extractor.extract()
    
    # Validate and get results
    results = validator.validate(
        development_data=development_data,
        metrics=["faithfulness", "context_utilization", "answer_relevancy"]
    )
    
    # Print the comparison report
    print(results.generate_report())

if __name__ == "__main__":
    main() 