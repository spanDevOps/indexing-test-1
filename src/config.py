# config.py
import os
from dotenv import load_dotenv
import multiprocessing

# Load environment variables from .env file
load_dotenv()


# AWS Configuration
AWS_REGION = os.getenv('AWS_REGION', 'ap-south-1')
QUEUE_URL = os.getenv('QUEUE_URL', 'https://sqs.ap-south-1.amazonaws.com/992382426603/ve-inhouse-indexing')
GROUP_FACE_QUEUE_URL = os.getenv('GROUP_FACE_QUEUE_URL', 'https://sqs.ap-south-1.amazonaws.com/566073089169/GroupFace')
BUCKET_NAME = os.getenv('BUCKET_NAME', "huego-gallery-processed")
LOG_GROUP = os.getenv('LOG_GROUP', '/ec2/ve-ai-gallery-images/index-face')
FACE_GROUP_BUCKET_NAME = os.getenv('FACE_GROUP_BUCKET_NAME', "ve-gallery-group-faces-use1")
# MongoDB Configuration
GALLERY_IMAGES_MONGO_URI = os.getenv('GALLERY_IMAGES_MONGO_URI')
HUEMN_MONGO_URI = os.getenv('HUEMN_MONGO_URI')

# Queue Processing Configuration
MESSAGE_VISIBILITY_TIMEOUT = int(os.getenv('MESSAGE_VISIBILITY_TIMEOUT', '1800'))  # 30 minutes default
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '5'))  # Reduced batch size to prevent memory overload
NUM_WORKERS = int(os.getenv('NUM_WORKERS', str(multiprocessing.cpu_count() * 2)))  # Default to 2 workers per CPU core
MAX_QUEUE_SIZE = int(os.getenv('MAX_QUEUE_SIZE', '50'))  # Internal queue buffer size
MAX_FACE_WORKERS = int(os.getenv('MAX_FACE_WORKERS', str(multiprocessing.cpu_count())))  # Default to 1 worker per CPU core
POLLING_WAIT_TIME = int(os.getenv('POLLING_WAIT_TIME', '20'))  # SQS long polling time

# Face detection configuration
FACE_DETECTION_CONFIG = {
    "sharpness_threshold": float(os.getenv('SHARPNESS_THRESHOLD', '6')),
    "brightness_threshold": float(os.getenv('BRIGHTNESS_THRESHOLD', '35')),
    "face_confidence_threshold": float(os.getenv('FACE_CONFIDENCE_THRESHOLD', '0.6954')),
    "blur_threshold": float(os.getenv('BLUR_THRESHOLD', '9')),
    "edge_density_threshold": float(os.getenv('EDGE_DENSITY_THRESHOLD', '3')),
    "max_image_size": tuple(map(int, os.getenv('MAX_IMAGE_SIZE', '1920,1080').split(',')))
}

# Validate required environment variables
required_vars = {
    'GALLERY_IMAGES_MONGO_URI': GALLERY_IMAGES_MONGO_URI,
    'HUEMN_MONGO_URI': HUEMN_MONGO_URI
}

for var_name, var_value in required_vars.items():
    if not var_value:
        raise ValueError(f"{var_name} must be set in the .env file")
# Worker Configuration
import multiprocessing

# Set workers based on CPU cores
CPU_COUNT = multiprocessing.cpu_count()
NUM_WORKERS = int(os.getenv('NUM_WORKERS', str(CPU_COUNT * 2)))  # Default to 2 workers per CPU core
MAX_FACE_WORKERS = int(os.getenv('MAX_FACE_WORKERS', str(CPU_COUNT)))  # Default to 1 worker per CPU core
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '5'))  # Reduced batch size to prevent memory overload
MAX_QUEUE_SIZE = int(os.getenv('MAX_QUEUE_SIZE', '50'))  # Internal queue buffer size

# Retry configuration
MAX_RETRIES = 3  # Maximum number of retry attempts
INITIAL_RETRY_DELAY = 30  # Initial retry delay in seconds
MAX_RETRY_DELAY = 900  # Maximum retry delay (15 minutes)
