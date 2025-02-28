# IAM Role for Batch Service
resource "aws_iam_role" "batch_service_role" {
  name = "face-recognition-batch-service"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "batch.amazonaws.com"
        }
      }
    ]
  })
}

# IAM Role for Batch Jobs
resource "aws_iam_role" "batch_job_role" {
  name = "face-recognition-batch-job"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

# Attach AWS Batch Service Policy
resource "aws_iam_role_policy_attachment" "batch_service" {
  role       = aws_iam_role.batch_service_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}

# S3 Access Policy for Batch Jobs
resource "aws_iam_role_policy" "batch_job_s3" {
  name = "face-recognition-batch-job-s3"
  role = aws_iam_role.batch_job_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.processed.arn,
          "${aws_s3_bucket.processed.arn}/*",
          aws_s3_bucket.faces.arn,
          "${aws_s3_bucket.faces.arn}/*"
        ]
      }
    ]
  })
}

# ECR Access Policy for Batch Jobs
resource "aws_iam_role_policy" "batch_job_ecr" {
  name = "face-recognition-ecr-access"
  role = aws_iam_role.batch_job_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      }
    ]
  })
}

# SQS Access Policy for Batch Jobs
resource "aws_iam_role_policy" "batch_job_sqs" {
  name = "face-recognition-batch-job-sqs"
  role = aws_iam_role.batch_job_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sqs:GetQueueAttributes"
        ]
        Resource = [
          aws_sqs_queue.indexing.arn
        ]
      }
    ]
  })
}

# Instance Role for Batch Compute Environments
resource "aws_iam_role" "batch_instance_role" {
  name = "face-recognition-batch-instance"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "batch_instance_role" {
  role       = aws_iam_role.batch_instance_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_instance_profile" "batch_instance_profile" {
  name = "face-recognition-batch-instance"
  role = aws_iam_role.batch_instance_role.name
}

resource "aws_iam_role_policy" "batch_instance_ecr" {
  name = "face-recognition-batch-instance-ecr"
  role = aws_iam_role.batch_instance_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "arn:aws:ecr:us-east-1:992382426603:repository/ve-ai-face-recognition-face-index"
      },
      {
        Effect = "Allow"
        Action = "ecr:GetAuthorizationToken"
        Resource = "*"
      }
    ]
  })
}
