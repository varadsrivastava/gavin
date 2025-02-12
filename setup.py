from setuptools import setup, find_packages

setup(
    name="genai_validator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ragas>=0.0.22",
        "boto3>=1.34.0",
        "azure-openai>=1.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "langchain>=0.1.0",
        "ragchecker>=0.1.0",
        "pytest>=7.0.0",
        "pydantic>=2.0.0",
        "click>=8.0.0",
    ],
    entry_points={
        'console_scripts': [
            'genai-validator=genai_validator.cli:cli',
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A library for validating GenAI models using challenger models and development data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/genai_validator",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 