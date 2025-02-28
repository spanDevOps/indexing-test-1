from typing import List, Dict, Any, Union
import boto3
from botocore.exceptions import ClientError

class SQSHandler:
    def __init__(self, queue_url: str):
        self.queue_url = queue_url
        self.sqs = boto3.client('sqs')

    def send_message(self, message_body: str) -> Dict[str, Any]:
        try:
            return self.sqs.send_message(
                QueueUrl=self.queue_url,
                MessageBody=message_body
            )
        except ClientError as e:
            print(f"Error sending message: {str(e)}")
            return {}

    def receive_messages(self, **kwargs) -> List[Dict[str, Any]]:
        try:
            response = self.sqs.receive_message(
                QueueUrl=self.queue_url,
                **kwargs
            )
            return response.get('Messages', [])
        except ClientError as e:
            print(f"Error receiving messages: {str(e)}")
            return []

    def delete_message(self, receipt_handle: str) -> None:
        try:
            print(self.queue_url,"QUEUE_URL")
            self.sqs.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle
            )
        except ClientError as e:
            print(f"Error deleting message: {str(e)}")

    def change_message_visibility(self, receipt_handle: str, visibility_timeout: int) -> None:
        try:
            self.sqs.change_message_visibility(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle,
                VisibilityTimeout=visibility_timeout
            )
        except ClientError as e:
            print(f"Error changing message visibility: {str(e)}")

    def get_queue_attributes(self, attribute_names: List[str]) -> Dict[str, Any]:
        """
        Get queue attributes.
        
        Args:
            attribute_names: List of attribute names to retrieve.
                           Valid values: All, Policy, VisibilityTimeout, MaximumMessageSize,
                           MessageRetentionPeriod, ApproximateNumberOfMessages, etc.
        
        Returns:
            Dictionary containing the requested attributes and their values
        """
        try:
            response = self.sqs.get_queue_attributes(
                QueueUrl=self.queue_url,
                AttributeNames=attribute_names
            )
            return response.get('Attributes', {})
        except ClientError as e:
            print(f"Error getting queue attributes: {str(e)}")
            return {}