import boto3
import tempfile
import json
from botocore.exceptions import ClientError



class S3Handler:
    def __init__(self, bucket_name):
        self.s3 = boto3.client('s3')
        self.bucket_name = bucket_name


    def download_image(self, s3_key):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                self.s3.download_fileobj(self.bucket_name, s3_key, temp_file)
                temp_file_path = temp_file.name
            return temp_file_path
        except ClientError as e:
            print(f"Error downloading image from S3: {e}")
            return None

    def save_to_s3(self, message_id, message_data, bucket_name):
        
        # Convert message data to JSON string
        message_json = json.dumps(message_data)
        
        # Upload to S3
        self.s3.put_object(
            Bucket=bucket_name,
            Key=message_id,
            Body=message_json
        )
        
        return message_id

print("s3_handler.py execution completed")
