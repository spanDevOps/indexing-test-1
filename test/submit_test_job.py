import boto3
import json
import os
from datetime import datetime
from pathlib import Path

def upload_to_s3(s3_client, local_path, bucket, key):
    """Upload a file to S3 and return its version ID"""
    response = s3_client.upload_file(
        local_path,
        bucket,
        key,
        ExtraArgs={'ContentType': 'image/jpeg'}
    )
    # Get version ID
    version = s3_client.head_object(Bucket=bucket, Key=key)['VersionId']
    return version

def send_to_queue(sqs_client, queue_url, message_body):
    """Send message to SQS queue"""
    response = sqs_client.send_message(
        QueueUrl=queue_url,
        MessageBody=json.dumps(message_body)
    )
    return response['MessageId']

def submit_batch_job(batch_client, job_queue, job_definition, message):
    """Submit AWS Batch job"""
    response = batch_client.submit_job(
        jobName=f"face-recognition-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        jobQueue=job_queue,
        jobDefinition=job_definition,
        containerOverrides={
            'environment': [
                {
                    'name': 'MESSAGE',
                    'value': json.dumps(message)
                }
            ]
        }
    )
    return response['jobId']

def process_images(image_dir, bucket_name, queue_url, job_queue, job_definition):
    """Process a batch of images"""
    s3 = boto3.client('s3')
    sqs = boto3.client('sqs')
    batch = boto3.client('batch')
    
    # Get all jpg/jpeg files
    image_files = list(Path(image_dir).glob('*.jp*g'))
    batch_id = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    for img_path in image_files:
        # Upload to S3
        s3_key = f'uploads/{batch_id}/{img_path.name}'
        version_id = upload_to_s3(s3, str(img_path), bucket_name, s3_key)
        
        # Prepare message
        message = {
            "tenantId": "dev-tenant",
            "gallery_id": "test-gallery",
            "album_id": "test-album",
            "image_id": img_path.stem,
            "imageVersionId": version_id,
            "optimizedImageS3Key": s3_key,
            "uploadBatchId": batch_id
        }
        
        # Send to queue
        msg_id = send_to_queue(sqs, queue_url, message)
        print(f"Processed {img_path.name}: Message ID {msg_id}")
        
        # Submit batch job
        job_id = submit_batch_job(batch, job_queue, job_definition, message)
        print(f"Submitted batch job: {job_id}")

def main():
    # Load from environment
    bucket = os.getenv('DEV_GALLERY_BUCKET')
    queue_url = os.getenv('DEV_INDEXING_QUEUE_URL')
    image_dir = os.getenv('TEST_IMAGES_DIR', 'test-images')
    job_queue = os.getenv('PRIMARY_BATCH_QUEUE')
    job_definition = os.getenv('FACE_RECOGNITION_JOB_DEFINITION')
    
    if not all([bucket, queue_url, job_queue, job_definition]):
        raise ValueError("Missing required environment variables")
    
    if not os.path.exists(image_dir):
        raise ValueError(f"Test images directory not found: {image_dir}")
    
    print(f"Processing images from {image_dir}")
    process_images(image_dir, bucket, queue_url, job_queue, job_definition)
    print("Done! Check AWS Batch console for job status")

if __name__ == '__main__':
    main()
