import json
import os
from typing import Dict, Any, List
from bson.json_util import dumps
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
import signal
import sys
import traceback

from pydantic import BaseModel, ValidationError
from botocore.exceptions import ClientError

from data_handlers.mongodb_handler import MongoDBHandler
from data_handlers.s3_handler import S3Handler
from data_handlers.sqs_handler import SQSHandler
from face_processing.detector import FaceDetector
from utils.logging_config import get_logger
import config
import boto3
from bson import ObjectId
from exceptions import RetryableError, NonRetryableError, S3DownloadError

print(config.AWS_REGION)
# Add this line after imports and before any AWS service usage
boto3.setup_default_session(region_name=config.AWS_REGION)

#print boto3 credentials
print("boto3.Session().get_credentials() =>",boto3.Session().get_credentials())
print("boto3.Session().get_credentials().access_key =>",boto3.Session().get_credentials().access_key)
print("boto3.Session().get_credentials().secret_key =>",boto3.Session().get_credentials().secret_key)
print("boto3.Session().get_credentials().token =>",boto3.Session().get_credentials().token)
print("boto3.Session().get_credentials().method =>",boto3.Session().get_credentials().method)

#print boto3 region
print("boto3.Session().region_name =>",boto3.Session().region_name)

# Define message model (same as previous request model)
class ImageProcessingMessage(BaseModel):
    tenantId: str
    gallery_id: str
    album_id: str
    image_id: str
    imageVersionId: str
    optimizedImageS3Key: str
    uploadBatchId: str

logger = get_logger("face-recognition-processor")

def serialize_response(data: Dict) -> Dict:
    """Convert ObjectId to string in the response data"""
    if isinstance(data, dict):
        return {k: serialize_response(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [serialize_response(item) for item in data]
    elif isinstance(data, ObjectId):
        return str(data)
    return data

class FaceProcessor:
    def __init__(self):
        try:
           
            print("Initializing S3 Handler...")
            self.s3_handler = S3Handler(config.BUCKET_NAME)
            
            print("Initializing MongoDB Handler...")
            self.mongodb_handler = MongoDBHandler(
                config.GALLERY_IMAGES_MONGO_URI,
                config.HUEMN_MONGO_URI
            )
            
            print("Initializing SQS Handler...")
            print("QUEUE_URL =>", config.QUEUE_URL)
            self.sqs_handler = SQSHandler(config.QUEUE_URL)
            
            print("Creating models directory...")
            models_dir = os.path.join(os.getcwd(), 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            print("Initializing Face Detector...")
            self.face_detector = FaceDetector(config.FACE_DETECTION_CONFIG)
            print("Face Detector initialized")
            
            self.max_face_workers = config.MAX_FACE_WORKERS
            self.temp_dir = Path('/tmp/face_processing')
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            self.logger = get_logger(self.__class__.__name__)
            self.running = True
            print("FaceProcessor initialization completed")
        except Exception as e:
            print(f"Failed to initialize FaceProcessor: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise

    def process_single_face(self, face_data: Dict, body: Dict, idx: int, total: int) -> Dict:
        """Process a single face from the detected faces"""
        print(f"Processing face {idx}/{total}")
        face_start = datetime.datetime.now()
        
        try:
            # face_data should already be a dictionary from detector.py
            # No need to convert, just validate
            if not isinstance(face_data, dict):
                print(f"Unexpected face_data type: {type(face_data)}")
                print(f"Face data content: {face_data}")
                raise ValueError(f"Expected dictionary for face_data, got {type(face_data)}")
            
            # Create a new dictionary with the face data
            face_doc = face_data.copy()  # Make a copy to avoid modifying the original
            
            ec2_face_id = self.mongodb_handler.create_face_document(face_doc, body)
            if ec2_face_id:
                face_doc['ec2FaceId'] = str(ec2_face_id)
                print(f"Face {idx} processed successfully")
            else:
                print(f"Failed to create face document for face {idx}")
                
            return face_doc
            
        except Exception as e:
            print(f"Error processing face {idx}: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise
        finally:
            face_time = (datetime.datetime.now() - face_start).total_seconds()
            print(f"Face {idx} processing took {face_time:.2f}s")

    async def process_message(self, message_body: Dict) -> Dict[str, Any]:
        """Process a single SQS message"""
        start_time = datetime.datetime.now()
        temp_image_path = None
        
        try:
            message_data = ImageProcessingMessage(**message_body)
            
            print(f"Processing image: {message_data}")
            try:
                temp_image_path = self.s3_handler.download_image(message_data.optimizedImageS3Key)
            except ClientError as e:
                raise NonRetryableError(f"S3 error: {str(e)}")
            
            # File validation
            if not os.path.exists(temp_image_path):
                raise NonRetryableError("Downloaded file does not exist")
            
            if os.path.getsize(temp_image_path) == 0:
                raise NonRetryableError("Downloaded file is empty")

            try:
                faces_data = self.face_detector.process_image(temp_image_path, message_data.dict())
            except Exception as e:
                raise NonRetryableError(f"Face detection error: {str(e)}")

            face_count = len(faces_data)

            print(f"Detected {face_count} faces in image {message_data.image_id}")
            
            if face_count == 0:
                print(f"No valid faces detected in image {message_data.image_id}")
                # Update the counter even if no faces were detected
                self.mongodb_handler.update_gallery_upload_batch_counter(message_data.uploadBatchId)
                return serialize_response({
                    "status": "success",
                    "image_id": message_data.image_id,
                    "processed_faces": 0
                })

            processed_faces = []
            if faces_data:
                # Create a list to store futures
                futures = []
                
                # Create ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=min(len(faces_data), self.max_face_workers)) as executor:
                    # Submit all tasks
                    for idx, face_data in enumerate(faces_data):
                        future = executor.submit(
                            self.process_single_face,
                            face_data,
                            message_data.dict(),
                            idx + 1,
                            len(faces_data)
                        )
                        futures.append(future)
                    
                    # Wait for all futures to complete
                    for future in as_completed(futures):
                        try:
                            face_result = future.result()
                            if face_result:
                                processed_faces.append(face_result)
                        except Exception as e:
                            print(f"Face processing failed: {str(e)}", exc_info=True)

                if processed_faces:
                    success = self.mongodb_handler.add_faces_to_image(
                        message_data.image_id,
                        message_data.imageVersionId,
                        processed_faces
                    )
                    if not success:
                        raise Exception("Failed to update database")
                else:
                    print(f"No faces were successfully processed for image {message_data.image_id}")

            #update the counter in galleryuploabatch collection
            self.mongodb_handler.update_gallery_upload_batch_counter(message_data.uploadBatchId)

            return serialize_response({
                "status": "success",
                "image_id": message_data.image_id,
                "processed_faces": len(processed_faces)
            })

        except (RetryableError, NonRetryableError):
            raise
        except Exception as e:
            # Convert unexpected errors to RetryableError
            raise NonRetryableError(f"Unexpected error: {str(e)}")
        finally:
            if temp_image_path and os.path.exists(temp_image_path):
                try:
                    os.unlink(temp_image_path)
                except Exception as e:
                    print(f"Failed to clean up temp file: {str(e)}")
            
            total_time = (datetime.datetime.now() - start_time).total_seconds()
            print(f"Total processing time: {total_time:.2f}s")

    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        print("Received shutdown signal, stopping processor...")
        self.running = False
        
    async def process_messages(self):
        """Main loop to process messages from SQS"""
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        signal.signal(signal.SIGINT, self.handle_shutdown)

        print("Starting message processing loop...")
        poll_count = 0
        
        while self.running:
            try:
                poll_count += 1
                print(f"\n[POLL #{poll_count}] Polling Detection SQS queue...")
                
                messages = self.sqs_handler.receive_messages(
                    MaxNumberOfMessages=config.BATCH_SIZE,
                    WaitTimeSeconds=config.POLLING_WAIT_TIME,
                    VisibilityTimeout=config.MESSAGE_VISIBILITY_TIMEOUT
                )

                if not messages:
                    print(f"[POLL #{poll_count}] No messages received")
                    continue

                print(f"[POLL #{poll_count}] Received {len(messages)} messages")

                for message in messages:
                    try:
                        message_body = json.loads(message['Body'])
                        await self.process_message(message_body)
                        self.sqs_handler.delete_message(message['ReceiptHandle'])
                        print(f"[POLL #{poll_count}] Successfully processed message {message['MessageId']}")
                        
                    except RetryableError as e:
                        print(f"[POLL #{poll_count}] Retryable error for message {message['MessageId']}: {str(e)}")
                        retry_count = int(message.get('Attributes', {}).get('ApproximateReceiveCount', 1))
                        if retry_count <= config.MAX_RETRIES:
                            visibility_timeout = min(30 * (2 ** (retry_count - 1)), 900)
                            try:
                                self.sqs_handler.change_message_visibility(
                                    message['ReceiptHandle'],
                                    visibility_timeout
                                )
                                print(f"[POLL #{poll_count}] Message {message['MessageId']} will retry in {visibility_timeout}s (attempt {retry_count})")
                            except Exception as ve:
                                print(f"[POLL #{poll_count}] Failed to change message visibility: {str(ve)}")
                        else:
                            print(f"[POLL #{poll_count}] Max retries exceeded for message {message['MessageId']}")
                            await self.handle_failed_message(message, str(e))
                            self.sqs_handler.delete_message(message['ReceiptHandle'])
                            
                    except NonRetryableError as e:
                        print(f"[POLL #{poll_count}] Non-retryable error for message {message['MessageId']}: {str(e)}")
                        await self.handle_failed_message(message, str(e))
                        self.sqs_handler.delete_message(message['ReceiptHandle'])
                        
                    except Exception as e:
                        print(f"[POLL #{poll_count}] Unexpected error for message {message['MessageId']}: {str(e)}")
                        try:
                            self.sqs_handler.change_message_visibility(
                                message['ReceiptHandle'],
                                0
                            )
                        except Exception as ve:
                            print(f"[POLL #{poll_count}] Failed to change message visibility: {str(ve)}")
                            self.sqs_handler.delete_message(message['ReceiptHandle'])
                            self.mongodb_handler.record_failed_message(message, str(e))


            except Exception as e:
                print(f"[POLL #{poll_count}] Unexpected error: {str(e)}")
                self.sqs_handler.delete_message(message['ReceiptHandle'])
                self.mongodb_handler.record_failed_message(message, str(e))

    async def handle_failed_message(self, message: Dict, error: str):
        """Handle permanently failed messages"""
        try:
            failure_record = {
                'message_id': message['MessageId'],
                'error': error,
                'timestamp': datetime.datetime.utcnow(),
                'message_body': message['Body']
            }
            self.mongodb_handler.record_failed_message(failure_record)
        except Exception as e:
            print(f"Error handling failed message: {str(e)}")

def check_gpu():
    try:
        import torch
        print("PyTorch GPU available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU Device:", torch.cuda.get_device_name(0))
            print("GPU Memory:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
    except ImportError:
        print("PyTorch not installed")

    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        print("ONNX Runtime Providers:", providers)
        if 'CUDAExecutionProvider' in providers:
            print("CUDA is available for ONNX Runtime")
    except ImportError:
        print("ONNX Runtime not installed")

if __name__ == "__main__":
    print("\nStarting Face Recognition Service...")
    print("1. Checking imports...")
    import asyncio
    import torch
    import onnxruntime
    print("2. Imports completed")
    
    print("3. Checking GPU...")
    check_gpu()
    print("4. GPU check completed")
    
    print("5. Creating FaceProcessor...")
    processor = FaceProcessor()
    print("6. FaceProcessor created")
    
    try:
        print("7. Setting up event loop...")
        loop = asyncio.get_event_loop()
        loop.set_default_executor(
            ThreadPoolExecutor(max_workers=config.NUM_WORKERS)
        )
        print("8. Starting message processing...")
        loop.run_until_complete(processor.process_messages())
    except KeyboardInterrupt:
        print("Shutting down due to keyboard interrupt...")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
    finally:
        loop.close()