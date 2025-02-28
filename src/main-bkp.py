import json
import os
from typing import Dict, Any, List
from bson.json_util import dumps
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
from threading import Event, Lock
import signal
import traceback
from pathlib import Path
import time

from data_handlers.mongodb_handler import MongoDBHandler
from data_handlers.sqs_handler import SQSHandler
from data_handlers.s3_handler import S3Handler
from face_processing.detector import FaceDetector
from utils.logging_config import get_logger
import config
import boto3 


# Add this line after imports and before any AWS service usage
boto3.setup_default_session(region_name=config.AWS_REGION)  

class MessageProcessor:
    def __init__(self, s3_handler: S3Handler, mongodb_handler: MongoDBHandler, 
                 face_detector: FaceDetector, sqs_handler: SQSHandler):
        self.s3_handler = s3_handler
        self.mongodb_handler = mongodb_handler
        self.face_detector = face_detector
        self.sqs_handler = sqs_handler
        self.group_face_sqs_handler = SQSHandler(config.GROUP_FACE_QUEUE_URL)
        self.should_stop = Event()
        self.image_counter = 0
        self.counter_lock = Lock()
        self.logger = get_logger(self.__class__.__name__)
        
        # Use configuration values
        self.message_queue = queue.Queue(maxsize=config.MAX_QUEUE_SIZE)
        self.batch_size = config.BATCH_SIZE
        self.num_workers = config.NUM_WORKERS
        self.max_face_workers = config.MAX_FACE_WORKERS
        
        # Create temp directory if it doesn't exist
        self.temp_dir = Path('/tmp/face_processing')
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self._log_configuration()

    def _log_configuration(self):
        """Log the current configuration settings"""
        self.logger.info("=== Processor Configuration ===")
        self.logger.info(f"Workers:")
        self.logger.info(f"- Processing Workers: {self.num_workers}")
        self.logger.info(f"- Face Processing Threads: {self.max_face_workers}")
        self.logger.info(f"Queue Settings:")
        self.logger.info(f"- Batch Size: {self.batch_size}")
        self.logger.info(f"- Queue Buffer Size: {config.MAX_QUEUE_SIZE}")
        self.logger.info(f"- Message Visibility Timeout: {config.MESSAGE_VISIBILITY_TIMEOUT}s")
        self.logger.info(f"- Polling Wait Time: {config.POLLING_WAIT_TIME}s")

    def increment_counter(self) -> int:
        """Thread-safe counter incrementer"""
        with self.counter_lock:
            self.image_counter += 1
            return self.image_counter

    def push_to_group_face_queue(self, face_data: Dict[str, Any], body: Dict[str, Any]) -> None:
        """Push processed face data to the group face queue"""
        if face_data.get('ec2FaceId') and face_data.get('face_id') is None:
            message = {
                'face': face_data,
                **body,
            }
            try:
                response = self.group_face_sqs_handler.send_message(dumps(message))
                self.logger.info(f"GroupFace Queue message sent: {response['MessageId']}")
            except Exception as e:
                self.logger.error(f"GroupFace Queue send failed: {e}", exc_info=True)

    def process_single_face(self, face_data: Dict, body: Dict, idx: int, total: int) -> Dict:
        """Process a single face from the detected faces"""
        self.logger.debug(f"Processing face {idx}/{total}")
        face_start = datetime.datetime.now()
        
        try:
            ec2_face_id = self.mongodb_handler.create_face_document(face_data, body)
            if ec2_face_id:
                face_data['ec2FaceId'] = str(ec2_face_id)
                self.push_to_group_face_queue(face_data, body)
                self.logger.debug(f"Face {idx} processed successfully")
            else:
                self.logger.error(f"Failed to create face document for face {idx}")
        except Exception as e:
            self.logger.error(f"Error processing face {idx}: {str(e)}", exc_info=True)
            raise
        finally:
            face_time = (datetime.datetime.now() - face_start).total_seconds()
            self.logger.debug(f"Face {idx} processing took {face_time:.2f}s")
            
        return face_data

    def process_message(self, message: Dict) -> None:
        """Process a single message from the queue"""
        start_time = datetime.datetime.now()
        temp_image_path = None
        
        try:
            message_id = message['MessageId']
            receipt_handle = message['ReceiptHandle']
            counter = self.increment_counter()
            
            self.logger.info(f"=== Processing Message {counter} (ID: {message_id}) ===")
            
            body = json.loads(message['Body'])
            optimized_image_s3_key = body.get('optimizedImageS3Key')
            image_id = body.get('image_id')
            version_id = body.get('imageVersionId')

            self.logger.info(f"Processing image: {image_id}")
            self.logger.debug(f"S3 Key: {optimized_image_s3_key}")

            temp_image_path = self.s3_handler.download_image(optimized_image_s3_key)
            if not temp_image_path:
                self.logger.error(f"Failed to download image: {optimized_image_s3_key}")
                return

            faces_data = self.face_detector.process_image(temp_image_path, body)
            face_count = len(faces_data)
            self.logger.info(f"Detected {face_count} faces in image {image_id}")

            processed_faces = []
            if faces_data:
                with ThreadPoolExecutor(max_workers=min(face_count, self.max_face_workers)) as executor:
                    future_to_face = {
                        executor.submit(
                            self.process_single_face, 
                            face_data, 
                            body, 
                            idx + 1, 
                            face_count
                        ): idx 
                        for idx, face_data in enumerate(faces_data)
                    }

                    for future in as_completed(future_to_face):
                        try:
                            face_result = future.result()
                            processed_faces.append(face_result)
                        except Exception as e:
                            self.logger.error(f"Face processing failed: {str(e)}", exc_info=True)

                if processed_faces:
                    success = self.mongodb_handler.add_faces_to_image(image_id, version_id, processed_faces)
                    self.logger.info(f"Database update {'succeeded' if success else 'failed'} for image {image_id}")
                else:
                    self.logger.warning(f"No faces were successfully processed for image {image_id}")

            self.sqs_handler.delete_message(receipt_handle)
            self.logger.info(f"Message {message_id} deleted from queue")

        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}", exc_info=True)
        finally:
            if temp_image_path and os.path.exists(temp_image_path):
                try:
                    os.unlink(temp_image_path)
                    self.logger.debug("Temporary image file cleaned up")
                except Exception as e:
                    self.logger.error(f"Failed to clean up temp file: {str(e)}")
            
            total_time = (datetime.datetime.now() - start_time).total_seconds()
            self.logger.info(f"Total processing time: {total_time:.2f}s")

    def fetch_messages(self) -> None:
        """Continuously fetch messages and add them to the queue"""
        while not self.should_stop.is_set():
            try:
                messages = self.sqs_handler.receive_messages(
                    MaxNumberOfMessages=self.batch_size,
                    WaitTimeSeconds=config.POLLING_WAIT_TIME,
                    VisibilityTimeout=config.MESSAGE_VISIBILITY_TIMEOUT
                )
                
                if messages:
                    self.logger.info(f"Fetched batch of {len(messages)} messages")
                    for message in messages:
                        self.message_queue.put(message)
            except Exception as e:
                self.logger.error(f"Error fetching messages: {str(e)}", exc_info=True)
                time.sleep(1)

    def process_queue(self) -> None:
        """Process messages from the internal queue"""
        while not self.should_stop.is_set():
            try:
                message = self.message_queue.get(timeout=1)
                self.process_message(message)
                self.message_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in queue processing: {str(e)}", exc_info=True)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info("Shutdown signal received, stopping gracefully...")
        self.should_stop.set()

    def run(self) -> None:
        """Main run method for the processor"""
        self.logger.info("=== Starting Production Message Processor ===")
        
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        try:
            # Start message fetcher thread
            with ThreadPoolExecutor(max_workers=1) as fetcher_executor:
                fetcher_future = fetcher_executor.submit(self.fetch_messages)
                
                # Start worker threads
                with ThreadPoolExecutor(max_workers=self.num_workers) as worker_executor:
                    worker_futures = [
                        worker_executor.submit(self.process_queue)
                        for _ in range(self.num_workers)
                    ]
                    
                    # Wait for completion or interruption
                    while not self.should_stop.is_set():
                        time.sleep(1)
                        if any(future.done() for future in worker_futures):
                            self.logger.warning("Some worker threads have stopped unexpectedly")

        except Exception as e:
            self.logger.error(f"Error in main processing loop: {str(e)}", exc_info=True)
        finally:
            self.logger.info("Initiating shutdown sequence...")
            self.should_stop.set()
            
            try:
                self.message_queue.join()
                self.logger.info("All pending messages processed")
            except Exception as e:
                self.logger.error(f"Error during shutdown: {str(e)}", exc_info=True)

def main() -> None:
    """Main entry point"""
    logger = get_logger(__name__)
    
    try:
        logger.info("=== Initializing Face Processing Service ===")
        logger.info(f"Starting at: {datetime.datetime.now()}")
        
        # Initialize handlers
        sqs_handler = SQSHandler(config.QUEUE_URL)
        s3_handler = S3Handler(config.BUCKET_NAME)
        mongodb_handler = MongoDBHandler(
            config.GALLERY_IMAGES_MONGO_URI,
            config.HUEMN_MONGO_URI
        )
        face_detector = FaceDetector(config.FACE_DETECTION_CONFIG)

        # Create and run processor
        processor = MessageProcessor(
            s3_handler,
            mongodb_handler,
            face_detector,
            sqs_handler
        )
        processor.run()
        
    except Exception as e:
        logger.critical(f"Critical error in main: {str(e)}", exc_info=True)
    finally:
        logger.info(f"Service stopped at: {datetime.datetime.now()}")

if __name__ == "__main__":
    main()