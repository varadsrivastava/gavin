import click
import json
from typing import Dict
from .validator import ModelValidator
from .data import S3DataExtractor
from .models import BedrockModel, AzureOpenAIModel

@click.group()
def cli():
    """GenAI Validator CLI - Evaluate and compare GenAI models."""
    pass

@cli.command()
@click.option('--task-type', type=click.Choice(['qa', 'summarization', 'reasoning']), required=True,
              help='Type of task to evaluate')
@click.option('--original-model-provider', type=click.Choice(['bedrock', 'azure']), required=True,
              help='Provider of the original model')
@click.option('--original-model-id', required=True,
              help='Model ID or deployment name of the original model')
@click.option('--s3-bucket', required=True,
              help='S3 bucket containing development data')
@click.option('--s3-prefix', default='',
              help='Prefix (folder path) in the S3 bucket')
@click.option('--aws-credentials-file', required=True,
              help='Path to JSON file containing AWS credentials')
@click.option('--azure-credentials-file', 
              help='Path to JSON file containing Azure credentials (if needed)')
@click.option('--metrics', default='faithfulness,context_utilization,answer_relevancy',
              help='Comma-separated list of metrics to evaluate')
def validate(task_type: str, original_model_provider: str, original_model_id: str,
            s3_bucket: str, s3_prefix: str, aws_credentials_file: str,
            azure_credentials_file: str, metrics: str):
    """Run validation comparing original model against a challenger."""
    
    # Load credentials
    with open(aws_credentials_file) as f:
        aws_credentials = json.load(f)
    
    azure_credentials = None
    if azure_credentials_file:
        with open(azure_credentials_file) as f:
            azure_credentials = json.load(f)
    
    # Initialize data extractor
    data_extractor = S3DataExtractor(
        bucket_name=s3_bucket,
        prefix=s3_prefix,
        credentials=aws_credentials
    )
    
    # Initialize original model
    if original_model_provider == 'bedrock':
        original_model = BedrockModel(
            model_id=original_model_id,
            credentials=aws_credentials
        )
    else:  # azure
        if not azure_credentials:
            raise click.UsageError("Azure credentials required for Azure model")
        original_model = AzureOpenAIModel(
            deployment_name=original_model_id,
            **azure_credentials
        )
    
    # Create validator
    validator = ModelValidator(
        original_model=original_model,
        task_type=task_type
    )
    
    # Extract development data
    click.echo("Extracting development data...")
    development_data = data_extractor.extract()
    
    # Parse metrics
    metrics_list = [m.strip() for m in metrics.split(',')]
    
    # Run validation
    click.echo("Running validation...")
    results = validator.validate(
        development_data=development_data,
        metrics=metrics_list
    )
    
    # Print results
    click.echo("\nValidation Results:")
    click.echo(results.generate_report())

if __name__ == '__main__':
    cli() 