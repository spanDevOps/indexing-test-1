# AWS Provider configuration
provider "aws" {
  region = "us-east-1"  # Default region, can be overridden via AWS_REGION env var
}

# Get current AWS region
data "aws_region" "current" {}

# Get current AWS account ID
data "aws_caller_identity" "current" {}

# Get availability zones for current region
data "aws_availability_zones" "available" {
  state = "available"
}

# VPC and Network Setup
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"

  name = "face-recognition-vpc"
  cidr = "10.0.0.0/16"

  azs             = slice(data.aws_availability_zones.available.names, 0, 2)
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = true
}

# S3 Buckets
resource "aws_s3_bucket" "processed" {
  bucket = "dev-gallery-processed-${data.aws_region.current.name}"
}

resource "aws_s3_bucket" "faces" {
  bucket = "dev-gallery-faces-${data.aws_region.current.name}"
}

# Enable versioning for S3 buckets
resource "aws_s3_bucket_versioning" "processed" {
  bucket = aws_s3_bucket.processed.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_versioning" "faces" {
  bucket = aws_s3_bucket.faces.id
  versioning_configuration {
    status = "Enabled"
  }
}

# ECR Repository for development
resource "aws_ecr_repository" "face_recognition" {
  name = "dev-face-recognition"
}

# Primary Compute Environment - Quick startup, capacity optimized
resource "aws_batch_compute_environment" "face_recognition_primary" {
  compute_environment_name = "face-recognition-primary"
  type                    = "MANAGED"
  state                   = "ENABLED"

  compute_resources {
    max_vcpus = 16
    min_vcpus = 0

    instance_type = [
      "c6i.xlarge",   # 4 vCPU, Intel Ice Lake
      "c6i.2xlarge",  # 8 vCPU
      "c6i.4xlarge"   # 16 vCPU
    ]

    type                = "SPOT"
    allocation_strategy = "SPOT_CAPACITY_OPTIMIZED"
    bid_percentage      = 100

    subnets = module.vpc.private_subnets
    security_group_ids = [aws_security_group.batch.id]
    instance_role      = aws_iam_instance_profile.batch_instance_profile.arn
  }

  service_role = aws_iam_role.batch_service_role.arn
}

# Secondary Compute Environment - Cost-optimized scaling
resource "aws_batch_compute_environment" "face_recognition_secondary" {
  compute_environment_name = "face-recognition-secondary"
  type                    = "MANAGED"
  state                   = "ENABLED"

  compute_resources {
    max_vcpus = 32
    min_vcpus = 0

    instance_type = [
      "c6i.xlarge",   # 4 vCPU, Intel Ice Lake
      "c6i.2xlarge",  # 8 vCPU
      "c6i.4xlarge",  # 16 vCPU
      "c6i.8xlarge"   # 32 vCPU
    ]

    type                = "SPOT"
    allocation_strategy = "SPOT_PRICE_CAPACITY_OPTIMIZED"
    bid_percentage      = 100

    subnets = module.vpc.private_subnets
    security_group_ids = [aws_security_group.batch.id]
    instance_role      = aws_iam_instance_profile.batch_instance_profile.arn
  }

  service_role = aws_iam_role.batch_service_role.arn
}

# Primary Queue - Quick startup
resource "aws_batch_job_queue" "face_recognition_primary" {
  name     = "face-recognition-primary-queue"
  state    = "ENABLED"  # Enable the queue
  priority = 100
  compute_environment_order {
    order = 0
    compute_environment = aws_batch_compute_environment.face_recognition_primary.arn
  }
  depends_on = [aws_batch_compute_environment.face_recognition_primary]
}

# Secondary Queue - Cost-optimized bulk processing
resource "aws_batch_job_queue" "face_recognition_secondary" {
  name     = "face-recognition-secondary-queue"
  state    = "ENABLED"
  priority = 50
  compute_environment_order {
    order = 0
    compute_environment = aws_batch_compute_environment.face_recognition_secondary.arn
  }
  depends_on = [aws_batch_compute_environment.face_recognition_secondary]
}

# Batch Job Definition
resource "aws_batch_job_definition" "face_recognition" {
  name = "face-recognition-job"
  type = "container"
  platform_capabilities = ["EC2"]

  container_properties = jsonencode({
    image = "992382426603.dkr.ecr.us-east-1.amazonaws.com/ve-ai-face-recognition-face-index:latest"
    command = ["python", "/app/run_detection_service.py"]
    resourceRequirements = [
      {
        type  = "VCPU"
        value = "2"
      },
      {
        type  = "MEMORY"
        value = "4096"
      }
    ]
    privileged = true
    environment = [
      {
        name  = "AWS_REGION"
        value = "us-east-1"
      },
      {
        name  = "GALLERY_BUCKET"
        value = aws_s3_bucket.processed.id
      },
      {
        name  = "INDEXING_QUEUE_URL"
        value = aws_sqs_queue.face_indexing.url
      }
    ]
    mountPoints = []
    volumes = []
    ulimits = []
  })
}

# Security Group for Batch
resource "aws_security_group" "batch" {
  name        = "face-recognition-batch"
  description = "Security group for face recognition batch jobs"
  vpc_id      = module.vpc.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
