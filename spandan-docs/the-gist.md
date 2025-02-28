# Face Recognition System: ECS to AWS Batch Migration

## Current System Architecture

### Core Services
1. **Face Detection Service** (`run_detection_service.py`)
   - Processes images from SQS queue
   - Performs face detection using InsightFace
   - Generates embeddings using ONNX models
   - Quality checks: blur, illumination, occlusion
   - Stores results in MongoDB and updates S3

2. **Face Grouping Service** (`run_grouping_service.py`)
   - Groups similar faces using FAISS
   - GPU-accelerated similarity search
   - Multi-worker processing
   - Stores group results in S3

### Infrastructure Components
- Running on ECS with GPU support
- SQS for job queuing
- S3 for image storage
- MongoDB (two databases):
  - gallery_images: stores face data
  - huemn: stores image metadata
- CloudWatch for logging

### Dependencies
- CUDA 11.8
- InsightFace
- ONNX Runtime
- PyTorch
- FAISS
- GPU-enabled instance

## Migration to AWS Batch

### Pre-Migration Tasks

1. **Create ECR Repositories**
```bash
aws ecr create-repository --repository-name face-detection
aws ecr create-repository --repository-name face-grouping
```

2. **Job Definitions**

Face Detection Job:
```json
{
  "jobDefinitionName": "face-detection-job",
  "type": "container",
  "containerProperties": {
    "image": "${ECR_REPO}/face-detection:latest",
    "vcpus": 4,
    "memory": 16384,
    "command": ["python", "run_detection_service.py"],
    "environment": [
      {"name": "CUDA_VISIBLE_DEVICES", "value": "0"},
      {"name": "NVIDIA_VISIBLE_DEVICES", "value": "all"}
    ],
    "resourceRequirements": [
      {"type": "GPU", "value": "1"}
    ]
  }
}
```

Face Grouping Job:
```json
{
  "jobDefinitionName": "face-grouping-job",
  "type": "container",
  "containerProperties": {
    "image": "${ECR_REPO}/face-grouping:latest",
    "vcpus": 8,
    "memory": 32768,
    "command": ["python", "run_grouping_service.py"],
    "environment": [
      {"name": "CUDA_VISIBLE_DEVICES", "value": "0"},
      {"name": "NVIDIA_VISIBLE_DEVICES", "value": "all"}
    ],
    "resourceRequirements": [
      {"type": "GPU", "value": "1"}
    ]
  }
}
```

3. **Compute Environment**
```json
{
  "computeEnvironmentName": "face-recognition-spot",
  "type": "MANAGED",
  "computeResources": {
    "type": "SPOT",
    "maxPrice": 60,
    "instanceTypes": [
      "g4dn.xlarge",
      "g4dn.2xlarge",
      "g4dn.4xlarge"
    ],
    "minvCpus": 0,
    "maxvCpus": 256,
    "subnets": ["subnet-xxx"],
    "securityGroupIds": ["sg-xxx"],
    "instanceRole": "ecsInstanceRole",
    "bidPercentage": 60,
    "allocationStrategy": "SPOT_CAPACITY_OPTIMIZED"
  }
}
```

### Special Considerations

1. **GPU Management**
   - Monitor GPU memory usage
   - Handle initialization errors
   - Current setup uses CUDA 11.8

2. **Error Handling**
   - GPU errors: Request new instance
   - Temporary errors: Rely on SQS visibility timeout (30 minutes)
   - Fatal errors: Log and exit

3. **Performance Optimization**
   - Package models in container image
   - Clean up temporary files
   - Monitor memory usage
   - Configure appropriate instance sizes

### Migration Benefits
1. Cost optimization with Spot Instances
2. Better scalability
3. Automatic job retries
4. Built-in instance management
5. Native GPU support

### Monitoring Setup
1. CloudWatch Metrics
   - GPU utilization
   - Memory usage
   - Processing time
   - Success/failure rates
2. CloudWatch Logs
   - Application logs
   - Error tracking
   - Performance metrics