#!/bin/bash

# Get Terraform outputs
cd ../terraform
export DEV_GALLERY_BUCKET=$(terraform output -raw s3_bucket_processed)
export DEV_INDEXING_QUEUE_URL=$(terraform output -raw sqs_indexing_queue_url)

# Create test images directory if it doesn't exist
cd ../test
mkdir -p test-images

echo "Ready to process images!"
echo "1. Place your test images in the test-images directory"
echo "2. Run: python submit_test_job.py"
echo
echo "Environment configured with:"
echo "Bucket: $DEV_GALLERY_BUCKET"
echo "Queue: $DEV_INDEXING_QUEUE_URL"
