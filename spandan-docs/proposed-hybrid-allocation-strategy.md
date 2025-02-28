# Hybrid Spot Instance Allocation Strategy for Face Recognition Service

## Overview
This document outlines a hybrid approach to AWS Batch compute environment configuration that combines both quick startup and cost optimization for the face recognition service.

## Architecture

### Compute Environments

#### 1. Primary Compute Environment
```json
{
  "computeEnvironmentName": "face-recognition-primary",
  "type": "MANAGED",
  "state": "ENABLED",
  "computeResources": {
    "type": "SPOT",
    "allocationStrategy": "SPOT_CAPACITY_OPTIMIZED",
    "minvCpus": 0,
    "maxvCpus": 16,
    "instanceTypes": ["g4dn.xlarge"],
    "subnets": ["subnet-xxx"],
    "securityGroupIds": ["sg-xxx"],
    "instanceRole": "ecsInstanceRole",
    "bidPercentage": 100
  }
}
```
Purpose:
- Quick startup for immediate processing
- Optimized for capacity availability
- Limited to smaller vCPU count for controlled scaling 

#### 2. Secondary Compute Environment
```json
{
  "computeEnvironmentName": "face-recognition-secondary",
  "type": "MANAGED",
  "state": "ENABLED",
  "computeResources": {
    "type": "SPOT",
    "allocationStrategy": "SPOT_PRICE_CAPACITY_OPTIMIZED",
    "minvCpus": 0,
    "maxvCpus": 256,
    "instanceTypes": ["g4dn.xlarge"],
    "subnets": ["subnet-xxx"],
    "securityGroupIds": ["sg-xxx"],
    "instanceRole": "ecsInstanceRole",
    "bidPercentage": 100
  }
}
```
Purpose:
- Cost-optimized scaling for bulk processing
- Higher vCPU limit for extensive scaling
- Balance between price and capacity

### Job Queues

#### 1. Primary Queue
```json
{
  "jobQueueName": "face-recognition-primary-queue",
  "state": "ENABLED",
  "priority": 100,
  "computeEnvironmentOrder": [
    {
      "order": 1,
      "computeEnvironment": "face-recognition-primary"
    }
  ]
}
```

#### 2. Secondary Queue
```json
{
  "jobQueueName": "face-recognition-secondary-queue",
  "state": "ENABLED",
  "priority": 50,
  "computeEnvironmentOrder": [
    {
      "order": 1,
      "computeEnvironment": "face-recognition-secondary"
    }
  ]
}
```

### Job Definition
```json
{
  "jobDefinitionName": "face-recognition-job",
  "type": "container",
  "containerProperties": {
    "image": "your-ecr-repo/face-recognition:latest",
    "vcpus": 4,
    "memory": 16384,
    "command": ["python", "run_detection_service.py"],
    "jobRoleArn": "arn:aws:iam::xxx:role/BatchJobRole",
    "environment": [
      {
        "name": "QUEUE_URL",
        "value": "https://sqs.ap-south-1.amazonaws.com/xxx/ve-inhouse-indexing"
      }
    ]
  }
}
```

## Workflow

### 1. Job Submission Logic
```python
def submit_batch_job(event, context):
    batch = boto3.client('batch')
    sqs = boto3.client('sqs')
    
    # Check queue depth
    response = sqs.get_queue_attributes(
        QueueUrl='your-queue-url',
        AttributeNames=['ApproximateNumberOfMessages']
    )
    msg_count = int(response['Attributes']['ApproximateNumberOfMessages'])
    
    if msg_count > 0:
        # Submit first job to primary queue
        batch.submit_job(
            jobName='face-recognition-primary',
            jobQueue='face-recognition-primary-queue',
            jobDefinition='face-recognition-job'
        )
        
        # If more messages, submit to secondary queue
        if msg_count > 1000:  # Threshold for scaling
            batch.submit_job(
                jobName='face-recognition-secondary',
                jobQueue='face-recognition-secondary-queue',
                jobDefinition='face-recognition-job',
                arrayProperties={
                    'size': min(msg_count // 1000, 10)  # Scale based on messages
                }
            )
```

### 2. Processing Flow
1. Images uploaded to S3 → SQS messages created
2. Lambda monitors SQS queue depth
3. First job submitted to primary queue (SPOT_CAPACITY_OPTIMIZED)
4. Primary compute environment starts quickly
5. If queue grows beyond threshold:
   - Secondary jobs submitted to secondary queue
   - Secondary environment scales out cost-effectively
   - Uses SPOT_PRICE_CAPACITY_OPTIMIZED for better pricing

## Benefits

1. **Quick Initial Response**
   - Primary environment optimized for capacity availability
   - Ensures fast startup for immediate processing

2. **Cost-Effective Scaling**
   - Secondary environment balances price and capacity
   - Better pricing for bulk processing

3. **Resource Efficiency**
   - Both environments scale to zero when idle
   - No ongoing costs when queue is empty

4. **Automatic Scaling**
   - Lambda monitors queue depth
   - Submits jobs based on workload

5. **Fault Tolerance**
   - SQS provides message persistence
   - Failed jobs can be retried
   - Messages have 30-minute visibility timeout

## Allocation Strategy Choices

AWS Batch offers three spot allocation strategies:

1. **SPOT_CAPACITY_OPTIMIZED** (Primary Environment)
   - Optimizes for instance availability
   - Reduces chance of interruptions
   - Best for immediate processing needs
   - Used in primary environment for quick startup

2. **SPOT_PRICE_CAPACITY_OPTIMIZED** (Secondary Environment)
   - Balances both price and availability
   - Good for bulk processing
   - Cost-effective for scaling operations
   - Used in secondary environment for efficient scaling

3. **BEST_FIT** (Not Used)
   - Simply picks instances based on resource requirements
   - Doesn't consider spot market conditions
   - May select instances with higher interruption rates
   - Could lead to more frequent job interruptions
   - Not suitable for our production workload
   - Avoided due to lack of spot market optimization

## Implementation Notes

1. **Configuration Requirements**
   - AWS Batch compute environments
   - Job queues with different priorities
   - Job definition
   - Lambda function for job submission
   - IAM roles and permissions

2. **Monitoring Considerations**
   - CloudWatch metrics for queue depth
   - Batch job status monitoring
   - Cost tracking across environments

3. **Scaling Parameters**
   - Queue depth threshold (1000 messages)
   - Maximum array size (10)
   - vCPU limits (16 primary, 256 secondary)

4. **Security**
   - VPC configuration
   - Security groups
   - IAM roles for services

## Next Steps

1. Create necessary IAM roles
2. Set up compute environments
3. Configure job queues
4. Deploy job definition
5. Implement Lambda function
6. Test with varying workloads
7. Monitor and adjust thresholds