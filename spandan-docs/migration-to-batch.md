# AWS Batch Migration Plan: Face Recognition Service (Console Steps)

## Phase 1: Initial Setup

### 1.1 Create ECR Repository
1. Go to Amazon ECR console
2. Click 'Create repository'
3. Name: `face-recognition`
4. Click 'Create repository'
5. Note the repository URI

### 1.2 Push Docker Image
```bash
# Get ECR login token
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.ap-south-1.amazonaws.com

# Build and push (using existing Dockerfile)
docker build -t face-recognition .
docker tag face-recognition:latest ${AWS_ACCOUNT_ID}.dkr.ecr.ap-south-1.amazonaws.com/face-recognition:latest
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.ap-south-1.amazonaws.com/face-recognition:latest
```

### 1.3 Create IAM Roles
1. Go to IAM console
2. Create service role for AWS Batch:
   - Click 'Create role'
   - Select 'Batch' service
   - Name: `BatchServiceRole`
   - Attach `AWSBatchServiceRole` policy

3. Create job role:
   - Click 'Create role'
   - Select 'EC2' service
   - Name: `BatchJobRole`
   - Attach policies:
     * `AmazonS3FullAccess`
     * `AmazonSQSFullAccess`
     * Custom policy for MongoDB access:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ec2:CreateNetworkInterface",
                "ec2:DescribeNetworkInterfaces",
                "ec2:DeleteNetworkInterface"
            ],
            "Resource": "*"
        }
    ]
}
```

## Phase 2: AWS Batch Setup

### 2.1 Create Primary Compute Environment
1. Go to AWS Batch console
2. Click 'Compute environments' → 'Create'
3. Configure:
   - Name: `face-recognition-primary`
   - Type: `Managed`
   - Provisioning model: `SPOT`
   - Maximum price percentage: 100%
   - Allowed instance types: `g4dn.xlarge`
   - Minimum vCPUs: 0
   - Maximum vCPUs: 16
   - VPC & subnets: Select your VPC with internet access
   - Security groups: Allow outbound internet access (for MongoDB)
   - Instance role: Select the EC2 role created earlier
   - Enable compute environment

### 2.2 Create Secondary Compute Environment
1. Same steps as primary, but with:
   - Name: `face-recognition-secondary`
   - Maximum vCPUs: 256
   - Allocation strategy: `SPOT_PRICE_CAPACITY_OPTIMIZED`

### 2.3 Create Job Queues
1. Go to 'Job queues' → 'Create'
2. Primary Queue:
   - Name: `face-recognition-primary-queue`
   - Priority: 100
   - Select primary compute environment
   - Enable queue

3. Secondary Queue:
   - Name: `face-recognition-secondary-queue`
   - Priority: 50
   - Select secondary compute environment
   - Enable queue

### 2.4 Create Job Definitions

#### Detection Service Job

1. Go to 'Job definitions' → 'Create'
2. Configure:
   - Name: `face-recognition-job`
   - Platform type: `EC2/Spot`
   - Container image: Your ECR image URI
   - Command: `python3 run_detection_service.py`
   - vCPUs: 4
   - Memory: 16384 MB
   - GPU: 1
   - Job role: Select BatchJobRole
   - Environment variables (default values from config.py):
     * `AWS_DEFAULT_REGION`: `ap-south-1`
     * `SERVICE_TYPE`: `detection`
     * `BUCKET_NAME`: `huego-gallery-processed`
     * `QUEUE_URL`: `https://sqs.ap-south-1.amazonaws.com/992382426603/ve-inhouse-indexing`  # From config.QUEUE_URL
     * `GALLERY_IMAGES_MONGO_URI`: Your MongoDB URI
     * `HUEMN_MONGO_URI`: Your MongoDB URI
     * `LOG_GROUP`: `/ec2/ve-ai-gallery-images/index-face`
     * `MESSAGE_VISIBILITY_TIMEOUT`: `1800`
     * `BATCH_SIZE`: `5`
     * `MAX_FACE_WORKERS`: `4`

#### Grouping Service Job
1. Go to 'Job definitions' → 'Create'
2. Configure:
   - Name: `face-grouping-job`
   - Platform type: `EC2/Spot`
   - Container image: Your ECR image URI
   - Command: `python3 run_grouping_service.py`
   - vCPUs: 4
   - Memory: 16384 MB
   - GPU: 1
   - Job role: Select BatchJobRole
   - Environment variables (default values from config.py):
     * `AWS_DEFAULT_REGION`: `ap-south-1`
     * `SERVICE_TYPE`: `grouping`
     * `BUCKET_NAME`: `huego-gallery-processed`
     * `GROUP_FACE_QUEUE_URL`: `https://sqs.ap-south-1.amazonaws.com/566073089169/GroupFace`  # From config.GROUP_FACE_QUEUE_URL
     * `GALLERY_IMAGES_MONGO_URI`: Your MongoDB URI
     * `HUEMN_MONGO_URI`: Your MongoDB URI
     * `FACE_GROUP_BUCKET_NAME`: `ve-gallery-group-faces-use1`
     * `LOG_GROUP`: `/ec2/ve-ai-gallery-images/index-face`
     * `MESSAGE_VISIBILITY_TIMEOUT`: `1800`
     * `BATCH_SIZE`: `5`
     * `MAX_FACE_WORKERS`: `4`

> Note: These are default values from config.py. You can override them by setting different values in the job definition environment variables.

### 2.5 Model Files Setup
1. Create an S3 bucket for model files
2. Upload required model files:
   ```bash
   aws s3 cp models/ s3://your-model-bucket/models/ --recursive
   ```
3. Add S3 copy commands to your Dockerfile:
   ```dockerfile
   RUN mkdir -p /app/models
   RUN aws s3 cp s3://your-model-bucket/models/ /app/models/ --recursive
   ```

## Phase 3: Lambda Integration

### 3.1 Create Lambda Function
```python
import boto3
import os

def lambda_handler(event, context):
    batch = boto3.client('batch')
    sqs = boto3.client('sqs')
    
    # Get queue depth
    queue_url = os.environ['SQS_QUEUE_URL']
    response = sqs.get_queue_attributes(
        QueueUrl=queue_url,
        AttributeNames=['ApproximateNumberOfMessages']
    )
    msg_count = int(response['Attributes']['ApproximateNumberOfMessages'])
    
    if msg_count > 0:
        # Submit to primary queue
        batch.submit_job(
            jobName='face-detection-primary',
            jobQueue='face-recognition-primary-queue',
            jobDefinition='face-detection-job'
        )
        
        # Scale out if needed
        if msg_count > 1000:
            batch.submit_job(
                jobName='face-detection-secondary',
                jobQueue='face-recognition-secondary-queue',
                jobDefinition='face-detection-job',
                arrayProperties={'size': min(msg_count // 1000, 10)}
            )
```

## Phase 4: Testing & Validation

### 4.1 Test Cases
1. Single image processing
2. Bulk upload handling
3. Spot instance interruption recovery
4. Queue depth-based scaling
5. Error handling and retries

### 4.2 Monitoring Setup
1. CloudWatch Dashboards:
   - SQS queue depth
   - Batch job success/failure rates
   - Processing time per image
   - Spot instance savings

2. Alerts:
   - Job failures
   - High queue depth
   - Spot interruptions

## Phase 5: Migration

> Note: Phase 5 can be executed after initial validation of Phases 1-4

### 5.1 Gradual Cutover
1. Start with 10% traffic to Batch
2. Monitor performance and costs
3. Gradually increase to 25%, 50%, 75%
4. Full cutover when stable

### 5.2 Rollback Plan
1. Keep ECS cluster running during migration
2. Maintain ability to route traffic back to ECS
3. Document trigger points for rollback

## Cost Optimization

### Current ECS Costs (Estimated)
- g4dn.xlarge running 24/7
- On-demand pricing
- Limited auto-scaling

### Expected Batch Savings
1. Spot Instance Usage:
   - Up to 70% cost reduction on compute
   - Capacity-optimized allocation

2. Better Scaling:
   - Scale to zero when idle
   - Efficient resource utilization

## Today's Action Items

1. **Setup & Configuration**
   - [ ] Request AWS account access
   - [ ] Create ECR repository and push image
   - [ ] Create IAM roles and policies

2. **Batch Environment**
   - [ ] Configure primary compute environment
   - [ ] Configure secondary compute environment
   - [ ] Set up job queues
   - [ ] Create job definitions

3. **Integration**
   - [ ] Create and configure Lambda function
   - [ ] Set up EventBridge trigger

4. **Testing**
   - [ ] Test single image processing
   - [ ] Test batch processing
   - [ ] Set up basic monitoring
   - [ ] Document test results
