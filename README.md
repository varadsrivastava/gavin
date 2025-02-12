# GenAI Validator

[![PyPI version](https://badge.fury.io/py/genai-validator.svg)](https://badge.fury.io/py/genai-validator)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A powerful Python library for validating and comparing GenAI models using challenger models and development data. Built on top of [Ragas](https://docs.ragas.io) and [RAGChecker](https://github.com/amazon-science/RAGChecker), this library provides comprehensive tools to evaluate and improve your RAG applications and model performance.

## 🌟 Features

- 🤖 **Challenger Model Selection**: Automatically select the best challenger models based on public benchmarks
- 📊 **Data Management**: Extract and validate development data from AWS S3
- 🔄 **Test Generation**: Generate reference test data using state-of-the-art challenger models
- 📈 **Comprehensive Metrics**: Evaluate models on multiple dimensions:
  - Faithfulness
  - Context Utilization
  - Answer Relevancy
  - Context Recall
- 🔍 **Detailed Comparison**: Compare performance between original and challenger models
- ☁️ **Cloud Support**: Native support for both AWS Bedrock and Azure OpenAI models

## 🚀 Quick Start

### Installation

```bash
pip install genai-validator
```

### Basic Usage

```python
from genai_validator import ModelValidator, S3DataExtractor
from genai_validator.models import BedrockModel, AzureOpenAIModel

# Initialize data extractor
data_extractor = S3DataExtractor(
    bucket_name="your-bucket",
    prefix="your/prefix"
)

# Initialize your original model
original_model = BedrockModel(
    model_id="anthropic.claude-v2",
    credentials={...}
)

# Create validator (automatically selects best challenger)
validator = ModelValidator(
    original_model=original_model,
    task_type="qa"  # or "summarization" or "reasoning"
)

# Run validation
results = validator.validate(
    development_data=data_extractor.extract(),
    metrics=["faithfulness", "context_utilization"]
)

# Get comparison report
print(results.generate_report())
```

## 🛠️ CLI Usage

The library provides a powerful command-line interface. First, set up your credentials:

### 1. Credential Setup

Create `aws_credentials.json`:
```json
{
    "access_key": "your_aws_access_key",
    "secret_key": "your_aws_secret_key"
}
```

If using Azure OpenAI, create `azure_credentials.json`:
```json
{
    "api_key": "your_azure_api_key",
    "api_base": "your_azure_endpoint",
    "api_version": "2024-02-15-preview"
}
```

### 2. Run Validation

For AWS Bedrock model:
```bash
genai-validator validate \
    --task-type qa \
    --original-model-provider bedrock \
    --original-model-id anthropic.claude-v2 \
    --s3-bucket your-bucket \
    --s3-prefix development/data \
    --aws-credentials-file aws_credentials.json
```

For Azure OpenAI model:
```bash
genai-validator validate \
    --task-type qa \
    --original-model-provider azure \
    --original-model-id gpt-4 \
    --s3-bucket your-bucket \
    --s3-prefix development/data \
    --aws-credentials-file aws_credentials.json \
    --azure-credentials-file azure_credentials.json
```

### CLI Options

| Option | Description | Required | Default |
|--------|-------------|----------|---------|
| `--task-type` | Type of task (`qa`, `summarization`, `reasoning`) | ✅ | - |
| `--original-model-provider` | Provider (`bedrock`, `azure`) | ✅ | - |
| `--original-model-id` | Model ID or deployment name | ✅ | - |
| `--s3-bucket` | S3 bucket with development data | ✅ | - |
| `--s3-prefix` | Prefix in S3 bucket | ❌ | "" |
| `--aws-credentials-file` | Path to AWS credentials | ✅ | - |
| `--azure-credentials-file` | Path to Azure credentials | ❌ | - |
| `--metrics` | Metrics to evaluate | ❌ | All metrics |

## 📊 Supported Metrics

- **Faithfulness**: Measures how well the model's responses align with the provided context
- **Context Utilization**: Evaluates how effectively the model uses the given context
- **Answer Relevancy**: Assesses the relevance of responses to questions
- **Context Recall**: Measures the model's ability to recall and use context information

## 🔧 Supported Models

### AWS Bedrock
- Claude 2.1
- Claude Instant
- Titan
- And more...

### Azure OpenAI
- GPT-4
- GPT-3.5 Turbo
- And more...

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and contribute to the project.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏷️ Tags

`#genai` `#validation` `#rag` `#llm` `#machinelearning` `#nlp` `#aws` `#azure` `#evaluation` `#testing` `#qa` `#summarization` `#reasoning` `#python` `#datascience` `#ai` 