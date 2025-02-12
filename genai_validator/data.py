from typing import List, Dict, Any, Optional
import json
import boto3
from botocore.exceptions import ClientError

class S3DataExtractor:
    def __init__(
        self,
        bucket_name: str,
        prefix: str = "",
        region_name: str = "us-east-1",
        credentials: Optional[Dict[str, str]] = None
    ):
        """
        Initialize S3 data extractor.
        
        Args:
            bucket_name: Name of the S3 bucket
            prefix: Prefix for S3 objects (folder path)
            region_name: AWS region name
            credentials: AWS credentials (access key, secret key)
        """
        self.bucket_name = bucket_name
        self.prefix = prefix
        
        session_kwargs = {"region_name": region_name}
        if credentials:
            session_kwargs.update({
                "aws_access_key_id": credentials.get("access_key"),
                "aws_secret_access_key": credentials.get("secret_key")
            })
        
        self.session = boto3.Session(**session_kwargs)
        self.s3_client = self.session.client("s3")
    
    def _read_json_file(self, key: str) -> Dict[str, Any]:
        """Read and parse a JSON file from S3."""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=key
            )
            content = response["Body"].read().decode("utf-8")
            return json.loads(content)
        except ClientError as e:
            print(f"Error reading file {key}: {str(e)}")
            return {}
    
    def extract(self) -> List[Dict[str, Any]]:
        """
        Extract development data from S3 bucket.
        
        Returns:
            List of dictionaries containing the development data
        """
        development_data = []
        
        try:
            # List all objects in the bucket with the given prefix
            paginator = self.s3_client.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=self.prefix
            )
            
            for page in page_iterator:
                if "Contents" not in page:
                    continue
                
                for obj in page["Contents"]:
                    key = obj["Key"]
                    
                    # Only process JSON files
                    if not key.endswith(".json"):
                        continue
                    
                    data = self._read_json_file(key)
                    
                    # Handle both single items and lists of items
                    if isinstance(data, list):
                        development_data.extend(data)
                    else:
                        development_data.append(data)
        
        except ClientError as e:
            print(f"Error listing objects in bucket: {str(e)}")
        
        return development_data
    
    def validate_data_format(self, data: List[Dict[str, Any]]) -> bool:
        """
        Validate that the extracted data has the required format.
        
        Args:
            data: List of data items to validate
            
        Returns:
            True if data format is valid, False otherwise
        """
        required_fields = {"context", "question", "answer"}
        
        for item in data:
            if not all(field in item for field in required_fields):
                print(f"Missing required fields in item: {item}")
                return False
            
            if not isinstance(item["context"], str):
                print(f"Context must be a string in item: {item}")
                return False
                
            if not isinstance(item["question"], str):
                print(f"Question must be a string in item: {item}")
                return False
                
            if not isinstance(item["answer"], str):
                print(f"Answer must be a string in item: {item}")
                return False
        
        return True 