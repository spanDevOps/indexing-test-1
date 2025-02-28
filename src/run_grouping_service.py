from typing import Dict, List, Optional
from data_handlers.mongodb_handler import MongoDBHandler
from data_handlers.sqs_handler import SQSHandler
from data_handlers.s3_handler import S3Handler
from face_processing.searcher import GPUMultiWorkerGrouper
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np
from bson import ObjectId
import config
import boto3
from pydantic import BaseModel
import json
import time
import signal

# Setup


# Initialize handlers
mongodb_handler = MongoDBHandler(
                config.GALLERY_IMAGES_MONGO_URI,
                config.HUEMN_MONGO_URI
            )
grouper = GPUMultiWorkerGrouper()
print("Grouping Queue URL:", config.GROUP_FACE_QUEUE_URL)


# Keep Pydantic model for validation
class GroupFaceRequest(BaseModel):
    uploadBatchId: str
    gallery_id: str
    tenant_id: str

# Make running variable global
running = True

def convert_objectid_to_str(obj):
    """Recursively convert ObjectId to string in nested dictionaries and lists"""
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: convert_objectid_to_str(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_objectid_to_str(item) for item in obj]
    return obj

def process_sqs_message(message_body: str, is_local: bool = False):
    try:
        # Parse message body
        print(f"Processing message body: {message_body}")
        request_data = json.loads(message_body)
        request = GroupFaceRequest(**request_data)
        
        print(f"Processing message for gallery: {request.gallery_id}, uploadBatchId: {request.uploadBatchId}")
        
        # Get faces from gallery
        result = mongodb_handler.get_faces_by_gallery_id(request.gallery_id)

        
        
        if not result:
            print(f"Failed to retrieve face data for gallery_id: {request.gallery_id}")
            return False

        try:
            all_faces = result['all_faces']
            start_time = time.time()            
            similar_faces = grouper.group_faces(all_faces)
            end_time = time.time()
        except Exception as e:
            print(f"[Grouping Service] Error in face_searcher.search_similar_faces: {str(e)}")
            return False
            
        print(f"[Grouping Service] Found {len(similar_faces)} similar face groups in {end_time - start_time} seconds")

        # Process each group
        start_time = time.time()
        s3_handler = S3Handler(config.FACE_GROUP_BUCKET_NAME)
        # Process groups in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            
            for i, group in enumerate(similar_faces):
                def process_group(group_idx, group_data):
                    try:
                        # Create message data with metadata
                        message_data = {
                            "group": group_data,
                            "gallery_id": str(request.gallery_id), 
                            "tenant_id": str(request.tenant_id),
                            "uploadBatchId": request.uploadBatchId
                        }

                        # Upload to S3
                        if is_local:
                            print(f"Local message data: {message_data}")
                        else:
                            message_id = f"{message_data['tenant_id']}/{message_data['gallery_id']}/{message_data['group']['center_face']['_id']}.json"
                            start_time = time.time()
                            s3_handler.save_to_s3(message_id, message_data, config.FACE_GROUP_BUCKET_NAME)
                            end_time = time.time()
                            print(f"Uploaded message {group_idx+1} to S3 in {end_time - start_time} seconds")
                        
                    except Exception as e:
                        print(f"Error processing group {group_idx+1}: {str(e)}")
                        print(f"Group data that caused error: {group_data}")

                futures.append(executor.submit(process_group, i, group))

            # Wait for all uploads to complete
            wait(futures)

        end_time = time.time()
        print(f"\n[Grouping Service] Successfully processed {len(similar_faces)} groups in {end_time - start_time} seconds")
        return True

    except Exception as e:
        print(f"Error in process_sqs_message: {str(e)}")
        return False

def handle_shutdown(signum, frame):
    """Handle shutdown signals"""
    global running
    print("Received shutdown signal, stopping processor...")
    running = False

def start_processing(sqs_handler):
    """
    Main loop to continuously process SQS messages
    """
    
    global running
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)
    print("[Grouping Service] Starting message processing loop...")
    poll_count = 0
    
    while running:
        try:
            poll_count += 1
            print(f"\n[POLL #{poll_count}] Polling Grouping SQS queue...")

            # Add a check for running status
            if not running:
                print("Shutdown signal received, exiting...")
                break

            messages = sqs_handler.receive_messages(
                MaxNumberOfMessages=1,
                WaitTimeSeconds=config.POLLING_WAIT_TIME,
                VisibilityTimeout=config.MESSAGE_VISIBILITY_TIMEOUT
            )

            if not messages:
                print(f"[POLL #{poll_count}] No messages received")
                continue

            print(f"[POLL #{poll_count}] Received {len(messages)} messages")

            for message in messages:
                try:
                    message_body = message.get('Body')
                    receipt_handle = message.get('ReceiptHandle')
                    
                    success = process_sqs_message(message_body)
                    print(f"[Grouping Service] Status of processed message {message['MessageId']}: {success}")
                    if success:
                        # Delete message only if processing was successful
                        sqs_handler.delete_message(receipt_handle)
                        print(f"[Grouping Service] Deleted message with receipt handle: {receipt_handle}")
                    else:
                        print("[Grouping Service] Failed to process message, keeping in queue for retry")

                except Exception as e:
                    print(f"[Grouping Service] Error processing message: {str(e)}")
                    continue
                    
            if not messages:
                # If no messages, wait briefly before polling again
                time.sleep(1)
                
        except Exception as e:
            print(f"[Grouping Service] Error in message processing loop: {str(e)}", exc_info=True)
            time.sleep(5)  # Wait before retrying on error

    print("[Grouping Service] Shutdown complete")

def start_local_processing():

    message_body = json.dumps({
        "uploadBatchId": "YbneAtWC9I",
        "gallery_id": "6790fa38be4eaa631bbd329b", 
        "tenant_id": "677928abf6c4e4f22db36b88"
    })

    success = process_sqs_message(message_body, is_local=True)
    print(f"[Grouping Service] Status of processed message : {success}")

def start_prod_processing():
    boto3.setup_default_session(region_name=config.AWS_REGION)
    sqs_handler = SQSHandler(config.GROUP_FACE_QUEUE_URL)
    start_processing(sqs_handler)

if __name__ == "__main__":
    print("[Grouping Service] Starting")
    start_prod_processing()

    #start_local_processing()

    
