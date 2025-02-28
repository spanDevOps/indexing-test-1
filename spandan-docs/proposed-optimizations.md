# Face Recognition Service: AWS Batch Migration Strategy

## Executive Summary

### Migration Overview
Transforming our 24/7 ECS-based face recognition service into an optimized AWS Batch system:

1. **AWS Batch Migration**
   - Implement hybrid spot instance strategy
   - Auto-scaling based on SQS queue depth

2. **Performance Optimization**
   - TensorRT model optimization
   - Smart batching with pre-filtering
   - Async job processing

## Current System Analysis

### Current Architecture (ECS)
- **Infrastructure**:
  - Always-on g4dn.xlarge instances
  - Fixed capacity ECS cluster
  - Direct SQS integration

### Performance Baseline
Current metrics from logging:
- Processing time tracked per face
- GPU utilization monitored via CUDA
- Worker configuration: MAX_FACE_WORKERS from config

## AWS Batch Implementation

### 1. Hybrid Allocation Strategy
```python
class HybridBatchProcessor:
    def __init__(self):
        # Primary queue for immediate processing
        self.primary_env = BatchComputeEnvironment(
            name='face-recognition-primary',
            instance_types=['g4dn.xlarge'],
            allocation_strategy='SPOT_CAPACITY_OPTIMIZED',
            max_vcpus=16  # Limited for quick scaling
        )
        
        # Secondary queue for cost-optimized bulk processing
        self.secondary_env = BatchComputeEnvironment(
            name='face-recognition-secondary',
            instance_types=['g4dn.xlarge', 'g4dn.2xlarge'],
            allocation_strategy='SPOT_PRICE_CAPACITY_OPTIMIZED',
            max_vcpus=256  # Larger for bulk processing
        )
    
    async def process_workload(self, queue_metrics):
        # Monitor SQS queue depth
        msg_count = queue_metrics.get_message_count()
        processing_time = queue_metrics.get_processing_time()
        
        # Submit to primary queue for immediate processing
        await self.primary_env.submit_job(
            job_name='face-recognition-primary',
            job_queue='primary-queue',
            job_definition='face-recognition-job'
        )
        
        # If backlog exists, use secondary queue
        if msg_count > self.BACKLOG_THRESHOLD:
            await self.secondary_env.submit_job(
                job_name='face-recognition-secondary',
                job_queue='secondary-queue',
                job_definition='face-recognition-job',
                array_properties={
                    'size': self._calculate_array_size(msg_count)
                }
            )
```

### 2. Infrastructure Configuration
```json
{
  "computeEnvironments": [
    {
      "name": "face-recognition-primary",
      "type": "MANAGED",
      "computeResources": {
        "type": "SPOT",
        "allocationStrategy": "SPOT_CAPACITY_OPTIMIZED",
        "minvCpus": 0,
        "maxvCpus": 16,
        "instanceTypes": ["g4dn.xlarge"],
        "subnets": ["subnet-xxx"],
        "securityGroupIds": ["sg-xxx"],
        "instanceRole": "face-recognition-ecs-role"
      }
    },
    {
      "name": "face-recognition-secondary",
      "type": "MANAGED",
      "computeResources": {
        "type": "SPOT",
        "allocationStrategy": "SPOT_PRICE_CAPACITY_OPTIMIZED",
        "minvCpus": 0,
        "maxvCpus": 256,
        "instanceTypes": ["g4dn.xlarge", "g4dn.2xlarge"],
        "subnets": ["subnet-xxx"],
        "securityGroupIds": ["sg-xxx"],
        "instanceRole": "face-recognition-ecs-role"
      }
    }
  ]
}
```

### 3. Job Queue Configuration
```json
{
  "jobQueues": [
    {
      "name": "face-recognition-primary-queue",
      "state": "ENABLED",
      "priority": 100,
      "computeEnvironmentOrder": [
        {
          "order": 1,
          "computeEnvironment": "face-recognition-primary"
        }
      ]
    },
    {
      "name": "face-recognition-secondary-queue",
      "state": "ENABLED",
      "priority": 50,
      "computeEnvironmentOrder": [
        {
          "order": 1,
          "computeEnvironment": "face-recognition-secondary"
        }
      ]
    }
  ]
}
```

### 4. Implementation Steps
1. **Infrastructure Setup**
   - Create compute environments
   - Configure job queues
   - Set up IAM roles and policies

2. **Application Updates**
   - Implement job submission logic
   - Add queue monitoring
   - Update logging and metrics

3. **Migration Process**
   - Deploy new infrastructure
   - Gradually shift workload
   - Monitor and optimize

4. **Validation**
   - Test scaling behavior
   - Verify cost optimization
   - Measure performance impact

## 1. GPU Batch Processing Optimization

### Performance Analysis
- **Memory Transfers**: Uses CUDA streams for CPU-GPU transfers
- **GPU Memory**: Managed through `GPUMultiWorkerGrouper` with configurable working memory
- **Model Loading**: Caches models in `.insightface/models` directory
- **Batch Processing**: Dynamic batch sizing based on available GPU memory

### Technical Solution
```python
class TensorRTProcessor:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        # Load TensorRT optimized engine
        self.engine = self.load_tensorrt_engine()
        self.context = self.engine.create_execution_context()
        # Pre-allocate device buffers
        self.input_buffer = cuda.mem_alloc(self.input_size)
        self.output_buffer = cuda.mem_alloc(self.output_size)
    
    def process_face_batch(self, faces_batch):
        # Preprocess batch using TensorRT optimized path
        preprocessed = self.preprocess_batch(faces_batch)
        
        # Execute TensorRT engine
        cuda.memcpy_htod(self.input_buffer, preprocessed)
        self.context.execute_async(
            batch_size=len(faces_batch),
            bindings=[int(self.input_buffer), int(self.output_buffer)],
            stream_handle=self.stream.handle
        )
        
        # Get results
        embeddings = cuda.memcpy_dtoh_async(self.output_buffer, self.stream)
        self.stream.synchronize()
        
        return self.postprocess_embeddings(embeddings)
        
        return embeddings, qualities
```

### Performance Improvements with TensorRT
1. **Inference Optimization**
   - Convert models to TensorRT format
   - Enable FP16 precision where accuracy allows
   - Optimize batch processing parameters

2. **Resource Utilization**
   - Profile GPU memory usage patterns
   - Monitor and optimize memory transfers
   - Analyze energy consumption

3. **Scaling Efficiency**
   - Measure cold start performance
   - Analyze batch processing overhead
   - Track dynamic batching efficiency

## 3. Smart Pre-filtering

### Current Implementation
```python
def process_image(self, image_path):
    img = cv2.imread(str(image_path))
    faces = self.face_app.get(img)
    for face in faces:
        quality = self._analyze_quality(face)
```

### Proposed Implementation
```python
class SmartPreFilter:
    def __init__(self):
        self.thumbnail_size = (224, 224)
        self.quality_threshold = 0.7
    
    async def process_image(self, s3_key):
        # Download thumbnail first
        thumbnail = await self.s3_handler.download_thumbnail(
            s3_key, 
            self.thumbnail_size
        )
        
        # Quick quality assessment
        quality_score = self.assess_image_quality(thumbnail)
        if quality_score < self.quality_threshold:
            return self.record_low_quality(s3_key, quality_score)
            
        # Download full image for processing
        full_image = await self.s3_handler.download_image(s3_key)
        return await self.process_full_image(full_image)
```

### Expected Benefits
- Early filtering of low-quality images
- Optimized bandwidth usage
- Streamlined processing pipeline
- Enhanced quality feedback

### Performance Monitoring
We will track:
- Quality assessment accuracy
- Bandwidth consumption
- Processing pipeline metrics
- User feedback metrics

### Technical Requirements
- Thumbnail generation system
- Quick quality assessment logic
- Quality score tracking
- Updated processing pipeline

## 4. Caching Layer Integration

### Current Implementation
```python
class S3Handler:
    def download_image(self, s3_key):
        return self.s3_client.download_file(
            self.bucket, 
            s3_key, 
            local_path
        )
```

### Proposed Implementation
```python
class CacheManager:
    def __init__(self):
        self.redis_client = redis.Redis()
        self.cache_ttl = 3600  # 1 hour
        
    async def get_image(self, s3_key):
        # Check cache first
        cached = await self.redis_client.get(s3_key)
        if cached:
            return self.deserialize_image(cached)
            
        # Download and cache if not found
        image = await self.s3_handler.download_image(s3_key)
        await self.redis_client.setex(
            s3_key,
            self.cache_ttl,
            self.serialize_image(image)
        )
        return image
        
    async def cache_embedding(self, face_id, embedding):
        await self.redis_client.setex(
            f'emb:{face_id}',
            self.cache_ttl,
            self.serialize_embedding(embedding)
        )
```

### Impact
- Reduced S3 operations
- Faster processing for duplicate faces
- Lower costs
- Better scalability

### Technical Requirements
- Redis/ElastiCache setup
- Caching logic implementation
- Cache invalidation mechanism
- Cache hit rate monitoring

## 5. MongoDB Optimization

### Current Implementation
```python
def create_face_document(self, face_doc):
    self.faces_collection.insert_one(face_doc)

def add_faces_to_image(self, image_id, faces):
    self.images_collection.update_one(
        {'_id': image_id},
        {'$push': {'faces': faces}}
    )
```

### Proposed Implementation
```python
class OptimizedMongoHandler:
    def __init__(self):
        self.batch_size = 1000
        self.write_buffer = []
        
    async def bulk_write_faces(self, faces_batch):
        operations = [
            UpdateOne(
                {'_id': face['image_id']},
                {'$push': {'faces': face}},
                upsert=True
            )
            for face in faces_batch
        ]
        
        result = await self.faces_collection.bulk_write(
            operations,
            ordered=False
        )
        return result
        
    def create_indexes(self):
        self.faces_collection.create_index([
            ('embedding', '2dsphere'),
            ('quality.score', -1)
        ])
```

### Impact
- Reduced database load
- Faster write operations
- Better query performance
- Improved scalability

### Technical Requirements
- Bulk operation handlers
- Optimized index structure
- Write buffering system
- Performance monitoring metrics

## 6. Enhanced Error Recovery

### Current Implementation
```python
try:
    process_message(message)
except RetryableError:
    # Simple retry
    retry_message(message)
```

### Proposed Implementation
```python
class SmartErrorHandler:
    def __init__(self):
        self.error_strategies = {
            GPUMemoryError: self.handle_gpu_error,
            NetworkError: self.handle_network_error,
            DatabaseError: self.handle_db_error
        }
        
    async def handle_error(self, error, context):
        strategy = self.error_strategies.get(
            type(error),
            self.handle_unknown_error
        )
        return await strategy(error, context)
        
    async def handle_gpu_error(self, error, context):
        # Reduce batch size and retry immediately
        new_batch_size = context.batch_size // 2
        return await self.retry_with_smaller_batch(
            context.messages,
            new_batch_size
        )
        
    async def handle_network_error(self, error, context):
        # Exponential backoff retry
        return await self.retry_with_backoff(
            context.message,
            initial_delay=1,
            max_retries=3
        )
```

### Impact
- Faster error recovery
- Reduced failed messages
- Better resource utilization
- Improved reliability

### Technical Requirements
- Error classification system
- Recovery strategy implementation
- Error tracking mechanism
- Recovery rate monitoring

## 7. Enhanced Monitoring

### Current Implementation
- Basic CloudWatch metrics
- Simple error logging

### Proposed Implementation
```python
class EnhancedMonitoring:
    def __init__(self):
        self.metrics = CloudWatchMetrics()
        self.traces = XRayTracer()
        
    def track_processing(self, context):
        # Track GPU metrics
        self.metrics.push_gpu_metrics({
            'memory_used': context.gpu.memory_used,
            'utilization': context.gpu.utilization,
            'temperature': context.gpu.temperature
        })
        
        # Track face quality
        self.metrics.push_quality_metrics({
            'blur_score': context.quality.blur,
            'illumination': context.quality.illumination,
            'orientation': context.quality.orientation
        })
        
        # Track processing time
        self.metrics.push_timing_metrics({
            'download_time': context.timings.download,
            'detection_time': context.timings.detection,
            'embedding_time': context.timings.embedding
        })
```

### Impact
- Better operational insights
- Proactive issue detection
- Easier troubleshooting
- Performance optimization opportunities

### Technical Requirements
- Enhanced metrics setup
- Distributed tracing implementation
- Dashboard creation
- Alert configuration

## 8. Model Optimization

### Current Implementation
```python
self.embedding_model = onnxruntime.InferenceSession(
    str(webface_path),
    providers=['CUDAExecutionProvider']
)
```

### Proposed Implementation
```python
class OptimizedModel:
    def __init__(self):
        self.model = self.build_optimized_model()
        
    def build_optimized_model(self):
        model = onnx.load(self.model_path)
        
        # Optimize graph
        model = optimize_graph(model)
        
        # Quantize weights
        model = quantize_model(
            model,
            per_channel=True,
            dtype='float16'
        )
        
        # TensorRT optimization
        if self.use_tensorrt:
            model = optimize_for_tensorrt(model)
            
        return model
        
    def optimize_graph(self, model):
        # Constant folding
        model = fold_constants(model)
        
        # Operator fusion
        model = fuse_operators(model)
        
        return model
```

### Impact
- 2-4x faster inference
- Reduced memory usage
- Better GPU utilization
- Lower latency

### Technical Requirements
- Model optimization implementation
- Quantization support
- TensorRT integration
- Performance benchmarking

## 9. Parallel Pipeline

### Current Implementation
Sequential processing pipeline

### Proposed Implementation
```python
class ParallelPipeline:
    def __init__(self, num_workers=3):
        self.download_queue = asyncio.Queue()
        self.detect_queue = asyncio.Queue()
        self.embed_queue = asyncio.Queue()

    async def start(self):
        # Start worker pools
        download_workers = [
            asyncio.create_task(self.download_worker())
            for _ in range(self.workers)
        ]
        
        detect_workers = [
            asyncio.create_task(self.detect_worker())
            for _ in range(self.workers)
        ]
        
        embed_workers = [
            asyncio.create_task(self.embed_worker())
            for _ in range(self.workers)
        ]
        
        # Monitor and scale workers
        await self.monitor_and_scale_workers()
        
    async def download_worker(self):
        while True:
            message = await self.download_queue.get()
            image = await self.download_image(message)
            await self.detect_queue.put((message, image))
            
    async def detect_worker(self):
        while True:
            message, image = await self.detect_queue.get()
            faces = await self.detect_faces(image)
            await self.embed_queue.put((message, faces))
```

### Impact
- Better hardware utilization
- Reduced end-to-end latency
- Improved throughput
- Better scalability

### Implementation Steps
1. Implement parallel pipeline
2. Add queue management
3. Create worker pools
4. Monitor pipeline performance

## 4. Face Grouping Service Optimization

### Current Implementation
- Face grouping requires loading all faces for a gallery
- Each new image triggers full reprocessing of gallery
- Uses MongoDB to fetch all face embeddings
- Processes 10,000 faces per batch
- Current caching limited to:
  - ML model files in `.insightface/models`
  - GPU memory management
  - CUDA stream processing

### Proposed Enhancement
1. **Distributed Caching Layer**
   - Cache frequently accessed embeddings
   - Incremental updates instead of full reprocessing
   - Optimize memory transfer between CPU and GPU

2. **Implementation Approach**
   ```python
   class CachedGroupingService:
       def __init__(self):
           self.cache = RedisClient()
           self.ttl = 24 * 60 * 60  # 24 hours
           
       async def process_new_faces(self, gallery_id, new_faces):
           # Get cached embeddings if available
           cached = await self.cache.get(f"gallery:{gallery_id}")
           
           # Only fetch missing faces from MongoDB
           if cached:
               delta = self.get_uncached_faces(new_faces, cached)
               await self.update_groups(delta, cached)
           else:
               # Initial load and cache
               all_faces = self.mongodb.get_faces(gallery_id)
               await self.cache.set(f"gallery:{gallery_id}", all_faces)
   ```

3. **Monitoring Strategy**
   - Track cache hit/miss rates
   - Measure memory usage patterns
   - Monitor processing time per batch
   - Compare full vs incremental updates

## Implementation Priority

1. GPU Batch Processing (Highest Impact/Effort Ratio)
2. Model Optimization
3. Smart Pre-filtering
4. MongoDB Optimization
5. Face Grouping Caching
6. Parallel Pipeline
7. Dynamic Queue Management
8. Enhanced Error Recovery
9. Enhanced Monitoring

## Conclusion

These optimizations together could provide:
- Improved throughput
- Reduced infrastructure costs
- Better reliability and maintainability
- Improved monitoring and observability