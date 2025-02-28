#!/bin/bash

# AWS Region
export AWS_REGION=us-east-1

# S3 and SQS resources
export DEV_GALLERY_BUCKET="dev-gallery-processed-us-east-1"
export DEV_INDEXING_QUEUE_URL="https://sqs.us-east-1.amazonaws.com/242201295949/dev-indexing"
export DEV_GROUP_FACE_QUEUE_URL="https://sqs.us-east-1.amazonaws.com/242201295949/dev-group-face"

# Test images directory
export TEST_IMAGES_DIR="test/test-images"

# AWS Batch resources
export PRIMARY_BATCH_QUEUE="arn:aws:batch:us-east-1:242201295949:job-queue/face-recognition-primary-queue"
export SECONDARY_BATCH_QUEUE="arn:aws:batch:us-east-1:242201295949:job-queue/face-recognition-secondary-queue"
export FACE_RECOGNITION_JOB_DEFINITION="arn:aws:batch:us-east-1:242201295949:job-definition/face-recognition-job:2"
