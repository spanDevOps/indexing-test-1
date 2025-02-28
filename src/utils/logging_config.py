import logging
import watchtower
import boto3
from datetime import datetime
import time
import threading
from botocore.exceptions import ClientError
from config import LOG_GROUP, AWS_REGION

STREAM_INTERVAL = 30  # 30 seconds

cloudwatch = boto3.client('logs', region_name=AWS_REGION)
current_stream_name = None
last_stream_creation_time = 0
stream_lock = threading.Lock()

class RotatingStreamHandler(watchtower.CloudWatchLogHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_rotation = time.time()

    def emit(self, record):
        global current_stream_name, last_stream_creation_time
        
        current_time = time.time()
        with stream_lock:
            if current_time - last_stream_creation_time >= STREAM_INTERVAL:
                new_stream_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                self.stream_name = new_stream_name
                current_stream_name = new_stream_name
                last_stream_creation_time = current_time
                
                # Force the handler to create a new stream
                self._stream = None
                
        super().emit(record)

def create_log_handler():
    global current_stream_name
    
    if current_stream_name is None:
        current_stream_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    try:
        handler = RotatingStreamHandler(
            log_group=LOG_GROUP,
            stream_name=current_stream_name,
            use_queues=False,  # Disable queuing for immediate pushing
            send_interval=1,  # Send logs every second
            create_log_group=True,
            boto3_client=cloudwatch
        )
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        return handler
    except ClientError as e:
        print(f"Failed to create CloudWatch log handler: {e}")
        return logging.StreamHandler()  # Fallback to console logging

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add our custom handler
    handler = create_log_handler()
    logger.addHandler(handler)
    
    return logger

# Create a global logger instance
logger = setup_logging()

def get_logger(name):
    return logging.getLogger(name)